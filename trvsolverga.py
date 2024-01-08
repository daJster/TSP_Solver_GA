import numpy as np
import random

# Define the TSP problem parameters
num_cities = 10
cities = np.random.rand(num_cities, 2)  # Randomly generate city coordinates

# Define genetic algorithm parameters
population_size = 50
generations = 1000
mutation_rate = 0.01

# Define helper functions
def calculate_distance(order):
    total_distance = 0
    for i in range(len(order) - 1):
        city1, city2 = order[i], order[i + 1]
        total_distance += np.linalg.norm(cities[city1] - cities[city2])
    return total_distance

def initialize_population(population_size):
    return [np.random.permutation(num_cities) for _ in range(population_size)]

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = np.hstack((parent1[:crossover_point], np.setdiff1d(parent2, parent1[:crossover_point])))
    return child

def mutate(order):
    if random.random() < mutation_rate:
        indices = random.sample(range(num_cities), 2)
        order[indices[0]], order[indices[1]] = order[indices[1]], order[indices[0]]
    return order

# Main genetic algorithm loop
population = initialize_population(population_size)

for generation in range(generations):
    population = sorted(population, key=lambda x: calculate_distance(x))
    
    # Select the top 50% of the population (based on fitness) as parents
    parents = population[:population_size // 2]
    
    # Generate offspring through crossover and mutation
    offspring = []
    while len(offspring) < population_size - len(parents):
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        offspring.append(child)
    
    # Replace the old population with the new generation
    population = parents + offspring

# Get the best solution from the final population
best_solution = min(population, key=lambda x: calculate_distance(x))

# Print the best solution and its total distance
print("Best solution:", best_solution)
print("Total distance:", calculate_distance(best_solution))


import numpy as np
import random

# ... (Previous code remains the same)

# Additional optimization: Elitism
elite_size = 2

# Main genetic algorithm loop with enhancements
population = initialize_population(population_size)

for generation in range(generations):
    population = sorted(population, key=lambda x: calculate_distance(x))

    # Elitism: Preserve the best solutions without modification
    elites = population[:elite_size]

    # Select the top 50% of the population (based on fitness) as parents
    parents = population[:population_size // 2]

    # Generate offspring through crossover and mutation
    offspring = []
    while len(offspring) < population_size - len(elites):
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        offspring.append(child)

    # Local Search (2-opt) on offspring
    offspring = [local_search(solution) for solution in offspring]

    # Elitism: Combine elites with the new generation
    population = elites + offspring

    # Termination Criteria: Stop if no improvement in best solution for a certain number of generations
    if generation > 10 and calculate_distance(population[0]) == calculate_distance(population[1]):
        break

# ... (Rest of the code remains the same)


# ... (Previous code remains the same)

def local_search(solution):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(solution) - 2):
            for j in range(i + 1, len(solution)):
                if j - i == 1:
                    continue  # No reversal for two consecutive edges
                new_solution = solution.copy()
                new_solution[i:j] = reversed(new_solution[i:j])
                if calculate_distance(new_solution) < calculate_distance(solution):
                    solution = new_solution
                    improved = True
    return solution

# ... (Continue from where the previous code left off)

# Main genetic algorithm loop with enhancements
population = initialize_population(population_size)

for generation in range(generations):
    population = sorted(population, key=lambda x: calculate_distance(x))

    # Elitism: Preserve the best solutions without modification
    elites = population[:elite_size]

    # Select the top 50% of the population (based on fitness) as parents
    parents = population[:population_size // 2]

    # Generate offspring through crossover and mutation
    offspring = []
    while len(offspring) < population_size - len(elites):
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        offspring.append(child)

    # Local Search (2-opt) on offspring
    offspring = [local_search(solution) for solution in offspring]

    # Elitism: Combine elites with the new generation
    population = elites + offspring

    # Termination Criteria: Stop if no improvement in best solution for a certain number of generations
    if generation > 10 and calculate_distance(population[0]) == calculate_distance(population[1]):
        break

# Get the best solution from the final population
best_solution = min(population, key=lambda x: calculate_distance(x))

# Print the best solution and its total distance
print("Best solution:", best_solution)
print("Total distance:", calculate_distance(best_solution))
