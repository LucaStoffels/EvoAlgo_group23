"""Module used for the execution of the evolutionary algorithm."""
import time
import math
import numpy as np

from evopy.individual import Individual
from evopy.progress_report import ProgressReport
from evopy.strategy import Strategy
from evopy.utils import random_with_seed


class EvoPy:
    """Main class of the EvoPy package."""

    def __init__(self, fitness_function, individual_length, warm_start=None, generations=100,
                 population_size=30, num_children=1, mean=0, std=1, maximize=False,
                 strategy=Strategy.SINGLE_VARIANCE, random_seed=None, reporter=None,
                 target_fitness_value=None, target_tolerance=1e-5, max_run_time=None,
                 max_evaluations=None, bounds=None):
        """Initializes an EvoPy instance.

        :param fitness_function: the fitness function on which the individuals are evaluated
        :param individual_length: the length of each individual
        :param warm_start: the individual to start from
        :param generations: the number of generations to execute
        :param population_size: the population size of each generation
        :param num_children: the number of children generated per parent individual
        :param mean: the mean for sampling the random offsets of the initial population
        :param std: the standard deviation for sampling the random offsets of the initial population
        :param maximize: whether the fitness function should be maximized or minimized
        :param strategy: the strategy used to generate offspring by individuals. For more
                         information, check the Strategy enum
        :param random_seed: the seed to use for the random number generator
        :param reporter: callback to be invoked at each generation with a ProgressReport as argument
        :param target_fitness_value: target fitness value for early stopping
        :param target_tolerance: tolerance to within target fitness value is to be acquired
        :param max_run_time: maximum time allowed to run in seconds
        :param max_evaluations: maximum allowed number of fitness function evaluations
        :param bounds: bounds for the sampling the parameters of individuals
        """
        self.fitness_function = fitness_function
        self.individual_length = individual_length
        self.warm_start = np.zeros(self.individual_length) if warm_start is None else warm_start
        self.generations = generations
        self.population_size = population_size
        self.num_children = num_children
        self.mean = mean
        self.std = std
        self.maximize = maximize
        self.strategy = strategy
        self.random_seed = random_seed
        self.random = random_with_seed(self.random_seed)
        self.reporter = reporter
        self.target_fitness_value = target_fitness_value
        self.target_tolerance = target_tolerance
        self.max_run_time = max_run_time
        self.max_evaluations = max_evaluations
        self.bounds = bounds
        self.evaluations = 0

    def _check_early_stop(self, start_time, best):
        """Check whether the algorithm can stop early, based on time and fitness target.

        :param start_time: the starting time to compare against
        :param best: the current best individual
        :return: whether the algorithm should be terminated early
        """
        return (self.max_run_time is not None
                and (time.time() - start_time) > self.max_run_time) \
               or \
               (self.target_fitness_value is not None
                and abs(best.fitness - self.target_fitness_value) < self.target_tolerance) \
               or (self.max_evaluations is not None
                and self.evaluations >= self.max_evaluations)

    def run(self):
        """Run the evolutionary strategy algorithm.

        :return: the best genotype found
        """
        if self.individual_length == 0:
            return None

        start_time = time.time()

        population = self._init_population()
        best = sorted(population, reverse=self.maximize,
                      key=lambda individual: individual.evaluate(self.fitness_function))[0]

        for generation in range(self.generations):
            children = [parent.reproduce() for _ in range(self.num_children)
                        for parent in population]
            population = sorted(children, reverse=self.maximize,
                                key=lambda individual: individual.evaluate(self.fitness_function))
            self.evaluations += len(population)
            population = population[:self.population_size]
            best = population[0]

            if self.reporter is not None:
                mean = np.mean([x.fitness for x in population])
                std = np.std([x.fitness for x in population])
                self.reporter(ProgressReport(generation, self.evaluations, best.genotype, best.fitness, mean, std))

            if self._check_early_stop(start_time, best):
                break

        return best.genotype

    def _init_population(self):
        if self.strategy == Strategy.SINGLE_VARIANCE:
            strategy_parameters = self.random.randn(1)
        elif self.strategy == Strategy.MULTIPLE_VARIANCE:
            strategy_parameters = self.random.randn(self.individual_length)
        elif self.strategy == Strategy.FULL_VARIANCE:
            strategy_parameters = self.random.randn(
                int((self.individual_length + 1) * self.individual_length / 2))
        else:
            raise ValueError("Provided strategy parameter was not an instance of Strategy")
        population_parameters = np.asarray([
            self.warm_start + self.random.normal(loc=self.mean, scale=self.std, size=self.individual_length)
            for _ in range(self.population_size)
        ])

        # print(population_parameters)
        # print("...")
        self.initialise_population_paremeters()

        # Make sure parameters are within bounds
        if self.bounds is not None:
            oob_indices = (population_parameters < self.bounds[0]) | (population_parameters > self.bounds[1])
            population_parameters[oob_indices] = self.random.uniform(self.bounds[0], self.bounds[1], size=np.count_nonzero(oob_indices))
            #population_parameters = self.initialise_population_paremeters()
            population_parameters = []
            for i in range(self.population_size):
                population_parameters.append(self.initialise_population_paremeters(True if i == 0 else False))
            #print("population_parameters: " + str(population_parameters))
           

       
        return [
            Individual(
                # Initialize genotype within possible bounds
                parameters,
                # Set strategy parameters
                self.strategy, strategy_parameters,
                # Set seed and bounds for reproduction
                random_seed=self.random,
                bounds=self.bounds
            ) for parameters in population_parameters
        ]

    def initialise_population_paremeters(self, debug=False):
        genotype = []

        nr_points_in_square = int(self.individual_length/2)
        nr_points_in_circle = 0;
        square_size = 0;
        while math.sqrt(nr_points_in_square).is_integer() == False:
            nr_points_in_square -= 1;
            nr_points_in_circle += 1;
            if (nr_points_in_square < 1):
                break;
        square_size = int(math.sqrt(nr_points_in_square))

        nr_points_in_center = 0;
        if  nr_points_in_circle > 1 and math.sqrt(nr_points_in_square) % 2 == 0:
            nr_points_in_center = 1;
            nr_points_in_circle -= 1;

        square_factor = 0.9
        square_stepsize = square_factor/(square_size-1)
        for i in range(square_size):
            for j in range(square_size):
                x = ((1-square_factor)/2) + (j)*square_stepsize;
                y = ((1-square_factor)/2) + (i)*square_stepsize;
                genotype.append(x)
                genotype.append(y)
                
        if(nr_points_in_circle > 0):
            circle_step = (2*math.pi)/nr_points_in_circle;
            scale_factor = square_stepsize/2.2;
            for i in range(nr_points_in_circle):
                x = 0.5+math.sin(i*circle_step) * scale_factor;
                y = 0.5+math.cos(i*circle_step) * scale_factor;
                genotype.append(x)
                genotype.append(y)
            
        if nr_points_in_center == 1:
            x = 0.5;
            y = 0.5;
            genotype.append(x)
            genotype.append(y)
        
        if(debug):
            print()
            print("Starting Grid Generation")
            print(" * Nr circles in square pattern: " + str(nr_points_in_square))
            print(" * Nr circles in circle pattern: " + str(nr_points_in_circle))
            print(" * Nr circles in square pattern: " + str(nr_points_in_center))
            print(" * Square size: " + str(square_size))
            print()

        return genotype
