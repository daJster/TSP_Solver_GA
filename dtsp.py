import numpy as np
from keras import layers, models
import tensorflow as tf
import sys, os

# Disable
def blockPrint():
    """
    Function to block print output by redirecting stdout to /dev/null.
    """
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    """
    Enable standard output.
    """
    sys.stdout = sys.__stdout__
    
# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU device found:', tf.test.gpu_device_name())
else:
    print("No GPU device found. Make sure you have installed TensorFlow with GPU support.")

class TSPGeneticAlgorithm:
    def __init__(self, num_cities, population_size):
        """
        Initialize the genetic algorithm with the number of cities and population size.
        """
        self.num_cities = num_cities
        self.population_size = population_size
        self.exploration_model = self.build_exploration_model()
        self.population = np.array([np.random.permutation(num_cities) for _ in range(population_size)])
        self.best_individual = []
        self.best_fitness = 9000.0
        self.stagnation_count = 0
        self.stagnation_threshold = 20  # Number of generations to tolerate stagnation

    def build_exploration_model(self):
        """
        Builds and compiles an exploration model.

        :return: The compiled exploration model.
        """
        model = models.Sequential([
            layers.Input(shape=(self.num_cities**2,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Output exploration factor between 0 and 1
        ])
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def distance_matrix(self, cities):
        """
        Calculate the distance matrix between cities.

        :param cities: An array of city coordinates.
        :return: The distance matrix between cities.
        """
        return np.linalg.norm(cities[:, None, :] - cities, axis=-1)

    def fitness_function(self, individual, distance_matrix):
        """
        Calculate the fitness of an individual based on the given distance matrix.

        :param individual: The individual for which to calculate the fitness.
        :type individual: list or numpy array

        :param distance_matrix: The matrix containing the distances between locations.
        :type distance_matrix: numpy array

        :return: The fitness value of the individual.
        :rtype: float
        """
        total_distance = np.sum(distance_matrix[individual, np.roll(individual, 1)])
        return 1 / total_distance # maximize fitness value to minimize total distance

    def explore_mutate(self, individual, exploration_factor):
        """
        Mutates the individual based on the exploration factor.

        :param individual: The individual to be mutated.
        :param exploration_factor: The probability of mutation.
        :return: The mutated individual.
        """
        mutated_individual = individual.copy()

        if np.random.rand() < exploration_factor:
            mutation_points = np.random.choice(self.num_cities, size=int(exploration_factor * self.num_cities), replace=False)
            mutation_points = np.random.shuffle(mutation_points)
            mutated_individual[mutation_points] = np.random.permutation(self.num_cities)

        return mutated_individual

    def evolve(self, generations):
        """
        Evolves the population over a given number of generations using genetic algorithm.
        
        Parameters:
        - generations: int, the number of generations to evolve the population
        
        Returns:
        - None
        """
        cities = np.random.rand(self.num_cities, 2)
        distance_matrix = self.distance_matrix(cities)
        current_best_fitness = self.best_fitness
        previous_fitness = []
        
        for generation in range(generations):
            blockPrint()
            fitness_values = np.array([self.fitness_function(ind, distance_matrix) for ind in self.population])

            parents = self.tournament_selection(fitness_values, tournament_size=3)
            offspring = self.crossover(parents)

            # Train the deep learning model using the current distance matrix and total distance as the loss
            features = distance_matrix.flatten().reshape(1, -1)
            target_distance = self.fitness_function(self.population[0], distance_matrix)  # Target is the total distance of the best individual
            self.exploration_model.fit(features, np.array([(current_best_fitness - self.best_fitness) / current_best_fitness]))
            
            # Use deep learning model to predict exploration factor based on the flattened distance matrix
            flattened_distance_matrix = distance_matrix.flatten().reshape(1, -1)
            exploration_factor = self.exploration_model.predict(flattened_distance_matrix)[0][0]
            if exploration_factor < 0.01: # put a threshold to keep exploration factor
                exploration_factor = 0.01
            # Introduce exploration through mutation
            for i in range(len(offspring)):
                offspring[i] = self.explore_mutate(offspring[i], exploration_factor)

                
            combined_population = np.vstack((self.population, offspring))
            indices = np.argsort(-np.array([self.fitness_function(ind, distance_matrix) for ind in combined_population]))
            self.population = combined_population[indices[:self.population_size]]

            current_best_individual = self.population[0]
            current_best_fitness = self.fitness_function(current_best_individual, distance_matrix)

            
            
            if current_best_fitness < self.best_fitness:
                self.best_individual = current_best_individual
                self.best_fitness = current_best_fitness
                self.stagnation_count = 0
            else:
                for idx in range(len(previous_fitness)):
                    if previous_fitness[idx] != current_best_individual[idx]:
                        self.stagnation_count = 0
                        break
                self.stagnation_count += 1
                
                enablePrint()
                print(f"repeated, {self.stagnation_count}")
                blockPrint()

            previous_fitness = current_best_individual
            enablePrint()
            print(f"{generation + 1}: distance : {round(1/current_best_fitness, 6)}, \npath : {current_best_individual}, \nmutation rate : {exploration_factor*100}%\n\n")

            if self.stagnation_count >= self.stagnation_threshold:
                print(f"Stopping evolution due to stagnation for {self.stagnation_threshold} consecutive generations.")
                break

    def tournament_selection(self, fitness_values, tournament_size):
        """
        Perform tournament selection to select parents from the population.

        Args:
            fitness_values: A list of fitness values for each individual in the population.
            tournament_size: The size of the tournament for selection.

        Returns:
            np.array: An array of indices representing the selected parents.
        """
        selected_parents = []

        for _ in range(self.population_size):
            tournament_indices = np.random.choice(len(fitness_values), tournament_size, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            selected_parents.append(tournament_indices[np.argmax(tournament_fitness)])

        return np.array(selected_parents)

    def crossover(self, parents):
        """
        Performs crossover on a population of parents to generate offspring.

        Args:
            parents: A list of parent indices in the population.

        Returns:
            numpy array: An array of offspring resulting from the crossover operation.
        """
        offspring = []

        for i in range(0, len(parents), 2):
            parent1 = self.population[parents[i]]
            parent2 = self.population[parents[i + 1]]

            # Perform ordered crossover
            crossover_point1 = np.random.randint(self.num_cities)
            crossover_point2 = np.random.randint(crossover_point1 + 1, self.num_cities + 1)
            child1 = self.ordered_crossover(parent1, parent2, crossover_point1, crossover_point2)
            child2 = self.ordered_crossover(parent2, parent1, crossover_point1, crossover_point2)

            offspring.extend([child1, child2])

        return np.array(offspring)

    def ordered_crossover(self, parent1, parent2, crossover_point1, crossover_point2):
        """
        Perform ordered crossover on the given parents at the specified crossover points.
        
        :param parent1: The first parent array
        :param parent2: The second parent array
        :param crossover_point1: The starting index for crossover
        :param crossover_point2: The ending index for crossover
        :return: The child array after crossover
        """
        child = -1 * np.ones_like(parent1)
        child[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]

        remaining_indices = np.setdiff1d(parent2, child, assume_unique=True)
        child[:crossover_point1] = remaining_indices[:crossover_point1]
        child[crossover_point2:] = remaining_indices[crossover_point1:]

        return child


if __name__ == "__main__":
    # Example usage
    num_cities = 86
    population_size = 1000
    generations = 500

    tsp_genetic_algorithm = TSPGeneticAlgorithm(num_cities, population_size)
    tsp_genetic_algorithm.evolve(generations)
