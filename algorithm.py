import numpy as np
import random
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from simulation import simulate_traffic
from setup import load_variants, COMPLETION_WEIGHT, AVERAGE_WAIT_WEIGHT, MAX_WAIT_WEIGHT

NUM_GENES = 8
BOUNDS = [(0, 15), (5, 60)] * 4
GENE_NAMES = ["N_advanced", "N_green", "S_advanced", "S_green", "E_advanced", "E_green", "W_advanced", "W_green"]
LOG_DIR = Path(__file__).with_name("logs")
LOG_COLUMNS = ["generation", "best_fitness", "mean_fitness", "median_fitness", "worst_fitness", "std_fitness",
               *[f"best_{name}" for name in GENE_NAMES], "mean_sigma", "gene_diversity", "elapsed_seconds"]

def _log_row(population: list[list[float]], population_fitness: list[float], generation: int, elapsed: float) -> list:
    """One log row: fitness statistics, the best individual's genes, step sizes and how spread out the population is."""
    genes = np.array([individual[:NUM_GENES] for individual in population])
    sigmas = np.array([individual[NUM_GENES:] for individual in population])
    fitness = np.array(population_fitness)
    best = population[int(np.argmin(fitness))]
    return [generation, fitness.min(), fitness.mean(), float(np.median(fitness)), fitness.max(), fitness.std(),
            *best[:NUM_GENES], sigmas.mean(), genes.std(axis = 0).mean(), round(elapsed, 2)]

def evaluate_fitness(population: list[list[float]], intersection_variants: list[dict]) -> list[float]:
    fitness = []

    for individual in population:
        variant_fitness = []
        for variant in intersection_variants:
            result = simulate_traffic(variant, individual)
            score = (COMPLETION_WEIGHT * result["completion_time"] + AVERAGE_WAIT_WEIGHT * result["average_wait"] + MAX_WAIT_WEIGHT * result["max_wait"])
            variant_fitness.append(score)
        fitness.append(float(np.mean(variant_fitness)))

    return fitness

def initalize_population(mu: int, bounds: list[tuple[float, float]], sigma_init: float, rng = None) -> list[list[float]]:
    if rng is None:
        rng = np.random.default_rng()
    population = []

    for _ in range(mu):
        variables = [float(rng.uniform(low, high)) for low, high in bounds]
        sigmas = [sigma_init] * NUM_GENES
        individual = variables + sigmas
        population.append(individual)

    return population

def mutate(parent: list[float], bounds: list[tuple[float, float]], lr: float, sigma_bound: float, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    child_sigmas = []
    for i in range(NUM_GENES):
        child_sigma = parent[i + NUM_GENES] * np.exp(lr * rng.normal(0, 1))
        child_sigmas.append(max(float(child_sigma), sigma_bound))

    child = []
    for i in range(NUM_GENES):
        val = parent[i] + rng.normal(0, child_sigmas[i])
        child.append(float(np.clip(val, bounds[i][0], bounds[i][1])))

    return child + child_sigmas

def crossover(parent1: list[float], parent2: list[float], alpha: float = 0.5, rng = None):
    if rng is None:
        rng = random
    child = []

    for i in range(NUM_GENES):
        child.append(rng.choice([parent1[i], parent2[i]]))

    for i in range(NUM_GENES, 2 * NUM_GENES):
        sigma = alpha * parent1[i] + (1 - alpha) * parent2[i]
        child.append(sigma)

    return child

def parent_selection(population: list[list[float]], population_fitness: list[float], mating_pool_size: int, tournament_size: int, rng = None):
    if rng is None:
        rng = random
    selected_to_mate = []

    while len(selected_to_mate) < mating_pool_size:
        tournament_index = rng.sample(range(len(population)), tournament_size)
        best_index = min(tournament_index, key = lambda i: population_fitness[i])
        selected_to_mate.append(best_index)

    return selected_to_mate

def survivor_selection(population: list[list[float]], population_fitness: list[float], offspring: list[tuple], mu: int):
    # (mu + lambda) selection
    parents = list(zip(population, population_fitness))
    candidates = parents + offspring
    candidates.sort(key = lambda x: x[1])
    survivors = candidates[:mu]

    new_population = [x[0] for x in survivors]
    new_fitness = [x[1] for x in survivors]
    best_solution, best_fitness = survivors[0]
    return new_population, new_fitness, best_solution, best_fitness

def run_algorithm(mu: int, lamd: int, generations: int, bounds: list[tuple[float, float]], sigma_init: float, sigma_bound: float, lr: float, tournament_size: int = 3, variant_seed: int = 42, crossover_alpha: float = 0.5, mating_pool_size: int = None, intersection_variants: list[dict] = None, seed: int = None, verbose: bool = True, log_path: Path | str | None = None):
    """Run the inner EA. If log_path is given, write one CSV row per generation there (0 = initial population)."""
    if intersection_variants is None:
        intersection_variants = load_variants()
    if mating_pool_size is None:
        mating_pool_size = lamd * 2
    if mu < 2 or lamd < 1 or not 2 <= tournament_size <= mu or mating_pool_size < 2:
        raise ValueError("Invalid EA population or mating pool settings")
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    population = initalize_population(mu, bounds, sigma_init, rng)
    population_fitness = evaluate_fitness(population, intersection_variants)
    best_index = int(np.argmin(population_fitness))
    best_solution, best_fitness = population[best_index], population_fitness[best_index]

    log_file = None
    if log_path is not None:
        start_time = time.time()
        log_file = open(log_path, "w", newline = "")
        log = csv.writer(log_file)
        log.writerow(LOG_COLUMNS)
        log.writerow(_log_row(population, population_fitness, 0, time.time() - start_time))
        log_file.flush()

    for generation in range(generations):
        mating_pool = parent_selection(population, population_fitness, mating_pool_size = mating_pool_size, tournament_size = tournament_size, rng = py_rng)
        children = []
        for i in range(lamd):
            parent1 = population[py_rng.choice(mating_pool)]
            parent2 = population[py_rng.choice(mating_pool)]
            child = crossover(parent1, parent2, crossover_alpha, py_rng)
            child = mutate(child, bounds, lr, sigma_bound, rng)
            children.append(child)

        fitness = evaluate_fitness(children, intersection_variants)
        offspring = list(zip(children, fitness))
        population, population_fitness, best_solution, best_fitness = survivor_selection(
            population, population_fitness, offspring, mu
        )
        if verbose:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
        if log_file is not None:
            log.writerow(_log_row(population, population_fitness, generation + 1, time.time() - start_time))
            log_file.flush()

    if log_file is not None:
        log_file.close()
    return best_solution, best_fitness

if __name__ == "__main__":
    results_file = Path(__file__).with_name("outer_results.json")

    if results_file.exists():
        with results_file.open("r") as file:
            settings = json.load(file)["best_hyperparameters"]

        print("Using optimized hyperparameters from outer_results.json")
        
    else:
        settings = {
            "mu": 20,
            "lamd": 100,
            "lr": 0.1,
            "tournament_size": 3,
            "crossover_alpha": 0.5,
            "mating_pool_size": 200,
            "sigma_init": 5.0,
            "sigma_bound": 0.01
        }

        print("Using default hyperparameters")

    generations = 100
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR.mkdir(exist_ok = True)
    log_path = LOG_DIR / f"inner_ea_{stamp}.csv"
    settings_path = LOG_DIR / f"inner_ea_{stamp}_settings.json"
    run_info = {
        "started": datetime.now().isoformat(timespec = "seconds"),
        "hyperparameters": settings,
        "hyperparameters_from": "outer_results.json" if results_file.exists() else "built-in defaults",
        "generations": generations,
        "gene_bounds": dict(zip(GENE_NAMES, BOUNDS)),
        "traffic_scenarios": len(load_variants()),
        "fitness_weights": {"completion": COMPLETION_WEIGHT, "average_wait": AVERAGE_WAIT_WEIGHT, "max_wait": MAX_WAIT_WEIGHT},
    }
    with settings_path.open("w") as file:
        json.dump(run_info, file, indent = 2)
    print(f"Logging to: {log_path}")

    best_solution, best_fitness = run_algorithm(
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
        log_path = log_path
    )

    print("Best timing genes:", best_solution[:NUM_GENES])
    print("Best mutation sigmas:", best_solution[NUM_GENES:])
    print("Best fitness:", best_fitness)

    run_info["finished"] = datetime.now().isoformat(timespec = "seconds")
    run_info["best_fitness"] = best_fitness
    run_info["best_genes"] = dict(zip(GENE_NAMES, best_solution[:NUM_GENES]))
    run_info["best_sigmas"] = best_solution[NUM_GENES:]
    with settings_path.open("w") as file:
        json.dump(run_info, file, indent = 2)
    print(f"Log saved to: {log_path}")