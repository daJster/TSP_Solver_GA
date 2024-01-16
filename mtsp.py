import numpy as np
import random
from tsp import GA, parse_arguments

np.random.seed(123)

class MultiTravelerGA(GA):
    def __init__(self, num_travelers, num_cities, population_size, generations, mutation_rate, elite_size):
        super().__init__(num_cities, population_size, generations, mutation_rate, elite_size)
        self.num_travelers = num_travelers
        self.travelers_paths = {}

    def initialize_multi_traveler_population(self):
        return {traveler_id: np.random.permutation(self.num_cities) for traveler_id in range(self.num_travelers)}

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(parent1) - 1)
        child = np.hstack((parent1[:crossover_point], np.setdiff1d(parent2, parent1[:crossover_point])))
        return child

    def mutate(self, order):
        if random.random() < self.mutation_rate:
            indices = random.sample(range(self.num_cities), 2)
            order[indices[0]], order[indices[1]] = order[indices[1]], order[indices[0]]
        return order

    def local_search(self, solution):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(solution) - 2):
                for j in range(i + 2, len(solution)):
                    if j - i == 1:
                        continue  # No reversal for two consecutive edges
                    new_solution = solution.copy()
                    new_solution[i:j] = list(reversed(new_solution[i:j]))
                    if self.calculate_distance(new_solution) < self.calculate_distance(solution):
                        solution = new_solution
                        improved = True
        return solution

    def calculate_total_distance(self, paths):
        total_distance = 0
        for traveler_id, path in paths.items():
            total_distance += self.calculate_distance(path)
        return total_distance

    def run(self, check_generation=10):
        population = self.initialize_multi_traveler_population()
        for generation in range(self.generations):
            population = dict(sorted(population.items(), key=lambda x: self.calculate_distance(x[1])))

            elites = dict(list(population.items())[:self.elite_size])
            parents = dict(list(population.items())[:self.population_size // 2])

            offspring = {}
            while len(offspring) < self.population_size - len(elites):
                parent1, parent2 = random.sample(list(parents.keys()), 2)
                child = self.crossover(parents[parent1], parents[parent2])
                child = self.mutate(child)
                offspring[len(offspring)] = child

            offspring = {k: self.local_search(v) for k, v in offspring.items()}

            population = {**elites, **offspring}

            if generation > check_generation and self.calculate_total_distance(elites) == self.calculate_total_distance(population):
                break

        best_solution_id = min(population.items(), key=lambda x: self.calculate_total_distance({x[0]: x[1]}))[0]
        best_solution = population[best_solution_id]

        for traveler_id, path in population.items():
            print(f"Optimal path for Traveler {traveler_id}: {path}")
        
        print("Best solution (Traveler {}): {}".format(best_solution_id, best_solution))
        print("Mean distance:", (self.calculate_total_distance(population) / self.num_travelers).round(4))



if __name__ == "__main__":
    args = parse_arguments()
    ga_solver = MultiTravelerGA(args.num_travelers, args.num_cities, args.population_size, args.generations, args.mutation_rate, args.elite_size)
    ga_solver.run()
