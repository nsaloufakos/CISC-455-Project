import numpy as np
import random
import json
import csv
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from algorithm import run_algorithm, BOUNDS
from setup import load_variants, COMPLETION_WEIGHT, AVERAGE_WAIT_WEIGHT, MAX_WAIT_WEIGHT

# The outer EA optimizes the inner EA's hyperparameters.
PARAMETERS = ["mu", "lamd", "lr", "tournament_size", "crossover_alpha", "mating_pool_size", "sigma_init", "sigma_bound"]
BOUNDS_OUTER = [(10, 60), (20, 150), (0.01, 0.5), (2, 10), (0.0, 1.0), (20, 300), (0.5, 15.0), (0.001, 1.0)]
INTEGER_GENES = (0, 1, 3, 5)

OUTER_MU = 6
OUTER_LAMD = 6
OUTER_GENERATIONS = 4
OUTER_TOURNAMENT_SIZE = 3
CONFIGS_IN_PARALLEL = 2
INNER_EVALUATION_BUDGET = 300
INNER_GENERATIONS = 25
INNER_REPEATS = 4
INNER_SEEDS = (101, 202, 303, 404, 505)
OUTER_SEED = 17
# Each run writes logs/outer_ea_<timestamp>.csv (one row per configuration, written as it
# finishes) and logs/outer_ea_<timestamp>_settings.json (the settings of that run).
LOG_DIR = Path(__file__).with_name("logs")


def decode(individual: list[float]) -> dict:
    settings = {}
    for i, name in enumerate(PARAMETERS):
        low, high = BOUNDS_OUTER[i]
        value = float(np.clip(individual[i], low, high))
        settings[name] = int(round(value)) if i in INTEGER_GENES else value

    settings["tournament_size"] = min(settings["tournament_size"], settings["mu"])
    settings["mating_pool_size"] = max(2, 2 * round(settings["mating_pool_size"] / 2))
    return settings


def run_inner(settings: dict, variants: list[dict], generations: int, seed: int) -> float:
    _, fitness = run_algorithm(
        mu = settings["mu"],
        lamd = settings["lamd"],
        generations = generations,
        bounds = BOUNDS,
        sigma_init = settings["sigma_init"],
        sigma_bound = settings["sigma_bound"],
        lr = settings["lr"],
        tournament_size = settings["tournament_size"],
        crossover_alpha = settings["crossover_alpha"],
        mating_pool_size = settings["mating_pool_size"],
        intersection_variants = variants,
        seed = seed,
        verbose = False
    )
    return fitness


def evaluate_outer(individual: list[float], variants: list[dict], executor: ProcessPoolExecutor, futures = None) -> list[float]:
    """Run the inner EA once per seed and return each run's best fitness (empty if it can't run)."""
    settings = decode(individual)
    generations = INNER_GENERATIONS #(INNER_EVALUATION_BUDGET - settings["mu"]) // settings["lamd"]
    if generations < 1:
        return []

    print(f"  Hyperparameters: {settings}", flush=True)
    print(f"  Inner generations: {generations}", flush=True)

    if futures is None:
        futures = [executor.submit(run_inner, settings, variants, generations, seed) for seed in INNER_SEEDS[:INNER_REPEATS]]

    results = []
    for i, seed in enumerate(INNER_SEEDS[:INNER_REPEATS]):
        print(f"  Inner run {i + 1}/{INNER_REPEATS} (seed = {seed})...", flush=True)

        fitness = futures[i].result()
        results.append(fitness)

        print(f"  Inner run {i + 1} complete: Fitness = {fitness:.2f}", flush=True)

    mean_fitness = float(np.mean(results))
    print(f"  Mean fitness = {mean_fitness:.2f}", flush=True)

    return results


def run_outer_algorithm():
    with ProcessPoolExecutor(max_workers = CONFIGS_IN_PARALLEL * INNER_REPEATS, mp_context = mp.get_context("spawn")) as executor:
        return _run_outer_algorithm(executor)


def _run_outer_algorithm(executor: ProcessPoolExecutor):
    rng = np.random.default_rng(OUTER_SEED)
    py_rng = random.Random(OUTER_SEED)
    variants = load_variants()
    cache = {}

    total_evaluations = OUTER_MU + OUTER_GENERATIONS * OUTER_LAMD
    evaluations_completed = 0
    best_fitness = float("inf")

    print("=" * 60, flush=True)
    print("STARTING OUTER EVOLUTIONARY ALGORITHM", flush=True)
    print(f"Outer population size: {OUTER_MU}", flush=True)
    print(f"Outer offspring per generation: {OUTER_LAMD}", flush=True)
    print(f"Outer generations: {OUTER_GENERATIONS}", flush=True)
    print(f"Inner evaluation budget: {INNER_EVALUATION_BUDGET}", flush=True)
    print(f"Inner repeats per configuration: {INNER_REPEATS}", flush=True)
    print(f"Traffic scenarios: {len(variants)}", flush=True)
    print("=" * 60, flush=True)

    # Log files for this run. Each row is flushed straight away, so a stopped or
    # crashed run still leaves everything evaluated up to that point.
    seeds = INNER_SEEDS[:INNER_REPEATS]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR.mkdir(exist_ok = True)
    log_path = LOG_DIR / f"outer_ea_{stamp}.csv"
    with (LOG_DIR / f"outer_ea_{stamp}_settings.json").open("w") as file:
        json.dump({
            "started": datetime.now().isoformat(timespec = "seconds"),
            "outer_mu": OUTER_MU, "outer_lamd": OUTER_LAMD, "outer_generations": OUTER_GENERATIONS,
            "outer_tournament_size": OUTER_TOURNAMENT_SIZE, "outer_seed": OUTER_SEED,
            "configs_in_parallel": CONFIGS_IN_PARALLEL,
            "inner_generations": INNER_GENERATIONS, "inner_repeats": INNER_REPEATS, "inner_seeds": list(seeds),
            "parameter_bounds": dict(zip(PARAMETERS, BOUNDS_OUTER)),
            "traffic_scenarios": len(variants),
            "fitness_weights": {"completion": COMPLETION_WEIGHT, "average_wait": AVERAGE_WAIT_WEIGHT, "max_wait": MAX_WAIT_WEIGHT},
        }, file, indent = 2)
    log_file = log_path.open("w", newline = "")
    log = csv.writer(log_file)
    log.writerow(["outer_generation", "configuration", *PARAMETERS, *[f"fitness_seed_{s}" for s in seeds],
                  "mean_fitness", "std_fitness", "cached", "best_so_far", "elapsed_seconds"])
    start_time = time.time()
    current_generation = 0  # 0 = initial population
    print(f"Logging to: {log_path}", flush=True)

    def fitness(individual, futures = None):
        nonlocal evaluations_completed, best_fitness

        evaluations_completed += 1
        settings = decode(individual)
        key = tuple(settings[name] for name in PARAMETERS)

        print(f"\nConfiguration {evaluations_completed}/{total_evaluations}", flush=True)

        cached = key in cache
        if not cached:
            cache[key] = evaluate_outer(individual, variants, executor, futures)
        else:
            print("  Using cached fitness", flush=True)

        runs = cache[key]
        score = float(np.mean(runs)) if runs else float("inf")

        if score < best_fitness:
            best_fitness = score
            print(f"  NEW BEST FITNESS: {best_fitness:.2f}", flush=True)

        print(f"  Configuration fitness: {score:.2f}", flush=True)
        print(f"  Best fitness so far: {best_fitness:.2f}", flush=True)

        log.writerow([current_generation, evaluations_completed, *(settings[name] for name in PARAMETERS),
                      *(runs if runs else [""] * len(seeds)), score, float(np.std(runs)) if runs else "",
                      cached, best_fitness, round(time.time() - start_time, 1)])
        log_file.flush()

        return score

    def evaluate_configurations(individuals):
        configuration_scores = []

        for start in range(0, len(individuals), CONFIGS_IN_PARALLEL):
            batch = individuals[start:start + CONFIGS_IN_PARALLEL]
            pending = {}

            # Submit both configurations before waiting for either one.
            for individual in batch:
                settings = decode(individual)
                key = tuple(settings[name] for name in PARAMETERS)

                if INNER_GENERATIONS >= 1 and key not in cache and key not in pending:
                    pending[key] = [executor.submit(run_inner, settings, variants, INNER_GENERATIONS, seed) for seed in INNER_SEEDS[:INNER_REPEATS]]

            # Update the cache and collect scores in the original order.
            for individual in batch:
                settings = decode(individual)
                key = tuple(settings[name] for name in PARAMETERS)
                configuration_scores.append(fitness(individual, pending.get(key)))

        return configuration_scores

    print("\nEvaluating initial population...", flush=True)

    population = [[float(rng.uniform(low, high)) for low, high in BOUNDS_OUTER] for _ in range(OUTER_MU)]
    scores = evaluate_configurations(population)

    for generation in range(OUTER_GENERATIONS):
        current_generation = generation + 1
        print("\n" + "=" * 60, flush=True)
        print(f"OUTER GENERATION {generation + 1}/{OUTER_GENERATIONS}", flush=True)
        print("=" * 60, flush=True)

        children = []
        for _ in range(OUTER_LAMD):
            tournament1 = py_rng.sample(range(len(population)), min(OUTER_TOURNAMENT_SIZE, len(population)))
            tournament2 = py_rng.sample(range(len(population)), min(OUTER_TOURNAMENT_SIZE, len(population)))
            parent1 = population[min(tournament1, key = lambda i: scores[i])]
            parent2 = population[min(tournament2, key = lambda i: scores[i])]
            child = [py_rng.choice([parent1[i], parent2[i]]) for i in range(len(PARAMETERS))]

            for i, (low, high) in enumerate(BOUNDS_OUTER):
                if rng.random() < 0.3:
                    child[i] = float(np.clip(child[i] + rng.normal(0, (high - low) * 0.1), low, high))
            children.append(child)

        child_scores = evaluate_configurations(children)
        candidates = list(zip(population + children, scores + child_scores))
        candidates.sort(key = lambda x: x[1])
        survivors = candidates[:OUTER_MU]
        population = [x[0] for x in survivors]
        scores = [x[1] for x in survivors]

        print(f"\nOuter generation {generation + 1} complete", flush=True)
        print(f"Best inner fitness = {scores[0]:.2f}", flush=True)
        print(f"Unique configurations evaluated = {len(cache)}", flush=True)

    best_settings = decode(population[0])

    results = {
        "best_hyperparameters": best_settings,
        "best_fitness": scores[0],
        "configurations_evaluated": len(cache),
        "inner_evaluation_budget": INNER_EVALUATION_BUDGET,
        "inner_repeats": INNER_REPEATS
    }

    results_file = Path(__file__).with_name("outer_results.json")
    with results_file.open("w") as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 60, flush = True)
    print("OUTER EVOLUTIONARY ALGORITHM COMPLETE", flush = True)
    print("=" * 60, flush=True)
    print("Best hyperparameters:", best_settings, flush = True)
    print(f"Best mean inner fitness: {scores[0]:.2f}", flush = True)
    print(f"Unique configurations evaluated: {len(cache)}", flush = True)
    print(f"Results saved to: {results_file}", flush = True)
    print(f"Log saved to: {log_path}", flush = True)
    log_file.close()

    return best_settings, scores[0]

if __name__ == "__main__":
    run_outer_algorithm()