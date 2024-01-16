import numpy as np
import random
import argparse

np.random.seed(123)

class GA:
    def __init__(self, num_cities, population_size, generations, mutation_rate, elite_size):
        """
        Initializes the GeneticAlgorithm class.

        Args:
            num_cities (int): The number of cities in the problem.
            population_size (int): The size of each population in the genetic algorithm.
            generations (int): The number of generations the algorithm will run.
            mutation_rate (float): The probability of mutation in the genetic algorithm.
            elite_size (int): The number of elite individuals to be preserved in each generation.

        Returns:
            None
        """
        self.num_cities = num_cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.cities = np.random.rand(self.num_cities, 2)

    def calculate_distance(self, order):
        """
        Calculate the total distance traveled for a given order of cities.

        Parameters:
            order (list): A list of cities representing the order in which they are visited.

        Returns:
            float: The total distance traveled.
        """
        total_distance = 0
        for i in range(len(order) - 1):
            city1, city2 = order[i], order[i + 1]
            total_distance += np.linalg.norm(self.cities[city1] - self.cities[city2])
        return total_distance

    def initialize_population(self):
        """
        Generate and return an initial population for the genetic algorithm.

        Returns:
            list: A list of randomly generated permutations of the cities.
        """
        return [np.random.permutation(self.num_cities) for _ in range(self.population_size)]

    def crossover(self, parent1, parent2):
        """
        Generates a new child by performing crossover between two parents.

        Parameters:
            parent1 (numpy.ndarray): The first parent array.
            parent2 (numpy.ndarray): The second parent array.

        Returns:
            numpy.ndarray: The child array generated through crossover.
        """
        crossover_point = random.randint(0, len(parent1) - 1)
        child = np.hstack((parent1[:crossover_point], np.setdiff1d(parent2, parent1[:crossover_point])))
        return child

    def mutate(self, order):
        """
        Mutates the given order list with a certain probability.

        Parameters:
            order (list): The order list representing the current sequence of cities.

        Returns:
            list: The mutated order list.
        """
        if random.random() < self.mutation_rate:
            indices = random.sample(range(self.num_cities), 2)
            order[indices[0]], order[indices[1]] = order[indices[1]], order[indices[0]]
        return order

    def local_search(self, solution):
        """
        Performs a local search algorithm to improve the given solution.

        Parameters:
            solution (list): The initial solution to be improved.

        Returns:
            list: The improved solution.
        """
        improved = True
        while improved:
            improved = False
            for i in range(1, len(solution) - 2):
                for j in range(i + 1, len(solution)):
                    if j - i == 1:
                        continue
                    new_solution = solution.copy()
                    new_solution[i:j] = list(reversed(new_solution[i:j]))
                    if self.calculate_distance(new_solution) < self.calculate_distance(solution):
                        solution = new_solution
                        improved = True
        return solution

    def run(self, check_generation=10):
        """
        Runs the genetic algorithm to find the best solution for the given problem.
        
        Parameters:
            None
        
        Returns:
            None
        """
        population = self.initialize_population()

        for generation in range(self.generations):
            population = sorted(population, key=lambda x: self.calculate_distance(x))

            elites = population[:self.elite_size]
            parents = population[:self.population_size // 2]

            offspring = []
            while len(offspring) < self.population_size - len(elites):
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                offspring.append(child)

            offspring = [self.local_search(solution) for solution in offspring]

            population = elites + offspring

            if generation > check_generation and self.calculate_distance(population[0]) == self.calculate_distance(population[1]):
                break

        best_solution = min(population, key=lambda x: self.calculate_distance(x))

        print("Best solution:", best_solution)
        print("Total distance:", self.calculate_distance(best_solution).round(4))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Traveling Salesman Problem Solver using Genetic Algorithm')
    parser.add_argument('--num_travelers', type=int, default=2, help='Number of travelers (optional)')
    parser.add_argument('--num_cities', type=int, default=10, help='Number of cities')
    parser.add_argument('--population_size', type=int, default=50, help='Population size per traveler')
    parser.add_argument('--generations', type=int, default=1000, help='Number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.01, help='Mutation rate')
    parser.add_argument('--elite_size', type=int, default=2, help='Number of elites')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    ga_solver = GA(args.num_cities, args.population_size, args.generations, args.mutation_rate, args.elite_size)
    ga_solver.run()
