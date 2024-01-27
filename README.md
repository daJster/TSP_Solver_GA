# Traveling Salesperson Problem Solvers
# Author : Jad El Karchi
## tsp.py
This file contains a simple implementation of a genetic algorithm (GA) to solve the Traveling Salesperson Problem (TSP). The GA class is designed with elitism and local search features. Here is an overview of the most important functionalities:

### GA Class
#### __init__(self, num_cities, population_size, generations, mutation_rate, elite_size)
    Initializes the GeneticAlgorithm class with the specified parameters.
Parameters:
- num_cities (int): The number of cities in the TSP.
- population_size (int): The size of each population in the genetic algorithm.
- generations (int): The number of generations the algorithm will run.
- mutation_rate (float): The probability of mutation in the genetic algorithm.
- elite_size (int): The number of elite individuals to be preserved in each generation.


#### local_search(self, solution)
    Performs a local search algorithm to improve the given solution.
Parameters:
- solution (list): The initial solution to be improved.

Returns:
- list: The improved solution.
#### run(self, check_generation=10)
    Runs the genetic algorithm to find the best solution for the given TSP.
Parameters:
- check_generation (int): The generation at which to check for convergence (default is 10).

## mtsp.py
This file extends the TSP solver for the Multi-Traveler Salesperson Problem (MTSP). The MultiTravelerGA class is a subclass of the GA class, and it introduces additional functionalities for handling multiple travelers. Here is an overview of the most important functionalities:

### MultiTravelerGA Class
#### __init__(self, num_travelers, num_cities, population_size, generations, mutation_rate, elite_size)
    Initializes the MultiTravelerGA class with the specified parameters, extending the GA class.
Additional Attributes:
- num_travelers (int): Number of travelers or individuals in the population.
- travelers_paths (dict): Dictionary to store paths for each traveler.

#### initialize_multi_traveler_population(self)
    Initializes the population for the multi-traveler problem.
Returns:
- dict: A dictionary representing the initial population where keys are traveler IDs and values are random permutations of city indices.

#### local_search(self, solution)
    Overrides the local_search method in the base class to accommodate multi-traveler local search.

#### run(self, check_generation=10)
    Overrides the run method in the base class to accommodate multi-traveler GA.
    Additional printing of optimal paths for each traveler and mean distance.

### Note 
This algorithm although it gives optimal solutions for each traveler it doesn't seem like working because sometimes multiple travelers visit the same cities using the same path.

## dtsp.py
This file utilizes deep learning for dynamic mutation rates in a TSP solver with elitism and local search. The TSPGeneticAlgorithm class integrates a neural network to predict mutation rates based on the current state of the population. Here is an overview of the most important functionalities:

### TSPGeneticAlgorithm Class
#### __init__(self, num_cities, population_size)
    Initializes the TSPGeneticAlgorithm class with the number of cities and population size.
Additional Attributes:
- exploration_model: Neural network model for predicting mutation rates.
- population: Numpy array representing the current population.
- best_individual: Best individual in the population.
- best_fitness: Fitness value of the best individual.
- stagnation_count: Counter for detecting stagnation.
- stagnation_threshold: Number of generations to tolerate stagnation.

#### build_exploration_model(self)
    Builds and compiles a neural network model for predicting exploration factor.

#### distance_matrix(self, cities)
    Calculates the distance matrix between cities.

#### tournament_selection(self, fitness_values, tournament_size)
    Performs tournament selection to select parents from the population.

## Features and Loss :
Features : Values of the distance matrix

Loss function : $$(current_{best\_fitness} - self.best\_fitness) / current_{best\_fitness}$$

$current_{best\_fitness}$ : best fitness value of the current iteration 

This way the further $current_{best\_fitness}$ is from $self.best\_fitness$ the more mutation rate increases. 