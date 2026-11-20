import numpy as np
import random
import json
from pathlib import Path
from simulation import simulate_traffic
from setup import load_variants, COMPLETION_WEIGHT, AVERAGE_WAIT_WEIGHT, MAX_WAIT_WEIGHT

NUM_GENES = 8
BOUNDS = [(0, 15), (5, 60)] * 4

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

def run_algorithm(mu: int, lamd: int, generations: int, bounds: list[tuple[float, float]], sigma_init: float, sigma_bound: float, lr: float, tournament_size: int = 3, variant_seed: int = 42, crossover_alpha: float = 0.5, mating_pool_size: int = None, intersection_variants: list[dict] = None, seed: int = None, verbose: bool = True):
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

    best_solution, best_fitness = run_algorithm(
        mu = settings["mu"],
        lamd = settings["lamd"],
        generations = 100,
        bounds = BOUNDS,
        sigma_init = settings["sigma_init"],
        sigma_bound = settings["sigma_bound"],
        lr = settings["lr"],
        tournament_size = settings["tournament_size"],
        crossover_alpha = settings["crossover_alpha"],
        mating_pool_size = settings["mating_pool_size"]
    )

    print("Best timing genes:", best_solution[:NUM_GENES])
    print("Best mutation sigmas:", best_solution[NUM_GENES:])
    print("Best fitness:", best_fitness)