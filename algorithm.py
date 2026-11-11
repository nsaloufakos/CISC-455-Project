import numpy as np
import random
from intersection import Intersection

intersection_variants = []
for seed in [10, 20, 30]:
    intersection = Intersection(seed = seed)
    intersection.populate_random()
    intersection_variants.append(intersection)

def evaluate_fitness(population: list[list[float]]) -> list[float]:
    fitness = []

    for individual in population:
        variant_fitness = []
        advanced_green, green, red = individual[:3]

        for variant in intersection_variants:

            intersection = variant.copy()
            waiting_time = simulate_traffic(
                intersection,
                advanced_green,
                green,
                red
            )
            variant_fitness.append(waiting_time)

        fitness.append(float(np.mean(variant_fitness)))

    return fitness

def initalize_population(mu: int, bounds: list[float], sigma_init: float) -> list[list[float]]:
    rng = np.random.default_rng()
    population = []

    for _ in range(mu):
        variables = rng.uniform(bounds[0], bounds[1], size = 3)
        sigmas = [sigma_init] * 3
        individual = list(variables) + sigmas
        population.append(individual)

    return population

def mutate(parent: list[float], bounds: list[float], lr: float, sigma_bound: float):
    child_sigmas = []
    for i in range(3):
        child_sigma = parent[i + 3] * np.exp(lr * np.random.normal(0, 1))
        child_sigmas.append(max(child_sigma, sigma_bound))

    child = []
    for i in range(3):
        val = parent[i] + np.random.normal(0, child_sigmas[i])
        child.append(np.clip(val, bounds[0], bounds[1]))

    return child + child_sigmas

def crossover(parent1: list[float], parent2: list[float], alpha: float = 0.5):
    child = []

    for i in range(3):
        child.append(random.choice([parent1[i], parent2[i]]))

    for i in range(3, 6):
        sigma = alpha * parent1[i] + (1 - alpha) * parent2[i]
        child.append(sigma)

    return child

def parent_selection(population: list[list[float]], population_fitness: list[float], mating_pool_size: int, tournament_size: int):
    #tournament selelection

    selected_to_mate = []

    while len(selected_to_mate) < mating_pool_size:
        tournament_index = random.sample(range(len(population)), tournament_size)
        best_index = min(tournament_index, key = lambda i: population_fitness[i])

        selected_to_mate.append(best_index)

    return selected_to_mate

def survivor_selection(population: list[list[float]], population_fitness: list[float], offspring: list[list[float]], mu):
    # (mu + lambda) selection

    parents = list(zip(population, population_fitness))
    candidates = parents + offspring
    candidates.sort(key = lambda x: x[1])
    survivors = candidates[:mu]

    new_population = [x[0] for x in survivors]
    new_fitness = [x[1] for x in survivors]

    best_solution, best_fitness = survivors[0]
    return new_population, new_fitness, best_solution, best_fitness

def run_algorithm(mu: int, lamd: int, generations: int, bounds: list[float], sigma_init: float, sigma_bound: float, lr: float):
    population = initalize_population(mu, bounds, sigma_init)
    population_fitness = evaluate_fitness(population)

    for generation in range(generations):
        mating_pool = parent_selection(
            population,
            population_fitness,
            mating_pool_size = lamd * 2,
            tournament_size = 3
        )

        children = []
        for i in range(0, len(mating_pool), 2):
            parent1 = population[mating_pool[i]]
            parent2 = population[mating_pool[i + 1]]

            child = crossover(parent1, parent2)
            child = mutate(child, bounds, lr, sigma_bound)

            children.append(child)

        fitness = evaluate_fitness(children)
        offspring = list(zip(children, fitness))
        population, population_fitness, best_solution, best_fitness = survivor_selection(
            population,
            population_fitness,
            offspring,
            mu
        )

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

if __name__ == "__main__":
    best_solution, best_fitness = run_algorithm(
        mu = 20,
        lamd = 100,
        generations = 100,
        bounds = [0, 100],
        sigma_init = 5.0,
        sigma_bound = 0.01,
        lr = 0.1
    )