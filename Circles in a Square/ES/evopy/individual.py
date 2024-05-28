"""Module containing the individuals of the evolutionary strategy algorithm."""
import numpy as np

from evopy.strategy import Strategy
from evopy.repair import Repair
from evopy.utils import random_with_seed


class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the genotype and the specified
    strategy.

    For the full variance reproduction strategy, we adopt the implementation as described in:
    [1] Schwefel, Hans-Paul. (1995). Evolution Strategies I: Variants and their computational
        implementation. G. Winter, J. Perieaux, M. Gala, P. Cuesta (Eds.), Proceedings of Genetic
        Algorithms in Engineering and Computer Science, John Wiley & Sons.
    """
    _BETA = 0.0873
    _EPSILON = 0.01

    def __init__(self, genotype, strategy, strategy_parameters, repair, bounds=None, random_seed=None):
        """Initialize the Individual.

        :param genotype: the genotype of the individual
        :param strategy: the strategy chosen to reproduce. See the Strategy enum for more
                         information
        :param strategy_parameters: the parameters required for the given strategy, as a list
        """
        self.genotype = genotype
        self.length = len(genotype)
        self.random_seed = random_seed
        self.random = random_with_seed(self.random_seed)
        self.fitness = None
        self.constraint = None
        self.bounds = bounds
        self.strategy = strategy
        self.strategy_parameters = strategy_parameters
        self.repair = repair
        if not isinstance(strategy, Strategy):
            raise ValueError("Provided strategy parameter was not an instance of Strategy.")
        if strategy == Strategy.SINGLE_VARIANCE and len(strategy_parameters) == 1:
            self.reproduce = self._reproduce_single_variance
        elif strategy == Strategy.MULTIPLE_VARIANCE and len(strategy_parameters) == self.length:
            self.reproduce = self._reproduce_multiple_variance
        elif strategy == Strategy.FULL_VARIANCE and len(strategy_parameters) == self.length * (
                self.length + 1) / 2:
            self.reproduce = self._reproduce_full_variance
        else:
            raise ValueError("The length of the strategy parameters was not correct.")

    def evaluate(self, fitness_function):
        """Evaluate the genotype of the individual using the provided fitness function.

        :param fitness_function: the fitness function to evaluate the individual with
        :return: the value of the fitness function using the individuals genotype
        """
        if self.repair == Repair.CONSTRAINT_DOMINATION:
            if self.bounds != None:
                oob_indices = (self.genotype < self.bounds[0]) | (self.genotype > self.bounds[1])
                left_bound_indices = (self.genotype < self.bounds[0])
                right_bound_indices = (self.genotype < self.bounds[1])
                left_oob_values = np.multiply(np.subtract(self.genotype, self.bounds[0]), left_bound_indices)
                right_oob_values = np.multiply(np.subtract(self.genotype, self.bounds[1]), right_bound_indices)
                oob_values = np.add(left_oob_values, right_oob_values)
                penalty = np.sum(np.square(oob_values))
                self.fitness = fitness_function(self.genotype) - penalty
            else:
                self.fitness = fitness_function(self.genotype)
        else:  
            self.fitness = fitness_function(self.genotype)

        return self.fitness

    def _reproduce_single_variance(self):
        """Create a single offspring individual from the set genotype and strategy parameters.

        This function uses the single variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_genotype = self.genotype + self.strategy_parameters[0] * self.random.randn(self.length)
        # Randomly sample out of bounds indices
        oob_indices = (new_genotype < self.bounds[0]) | (new_genotype > self.bounds[1])
        if self.repair == Repair.RANDOM_REPAIR:
            new_genotype[oob_indices] = self.random.uniform(self.bounds[0], self.bounds[1], size=np.count_nonzero(oob_indices))
        elif self.repair == Repair.BOUNDARY_REPAIR:
            dist_from_left_bound = np.absolute(np.subtract(new_genotype, np.full(new_genotype.shape, self.bounds[0])))
            dist_from_right_bound = np.absolute(np.subtract(new_genotype, np.full(new_genotype.shape, self.bounds[1])))
            take_left_bound = dist_from_left_bound[oob_indices] < dist_from_right_bound[oob_indices]
            take_right_bound = np.logical_not(take_left_bound)
            new_oob_values = np.add(np.multiply(take_left_bound, self.bounds[0]), np.multiply(take_right_bound, self.bounds[1]))
            new_genotype[oob_indices] = new_oob_values
        scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        new_parameters = [max(self.strategy_parameters[0] * np.exp(scale_factor), self._EPSILON)]
        return Individual(new_genotype, self.strategy, new_parameters, self.repair, bounds=self.bounds, random_seed=self.random)

    def _reproduce_multiple_variance(self):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the multiple variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_genotype = self.genotype + [self.strategy_parameters[i] * self.random.randn()
                                        for i in range(self.length)]
        # Randomly sample out of bounds indices
        oob_indices = (new_genotype < self.bounds[0]) | (new_genotype > self.bounds[1])
        new_genotype[oob_indices] = self.random.uniform(self.bounds[0], self.bounds[1], size=np.count_nonzero(oob_indices))
        global_scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [self.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
                         for _ in range(self.length)]
        new_parameters = [max(np.exp(global_scale_factor + scale_factors[i])
                              * self.strategy_parameters[i], self._EPSILON)
                          for i in range(self.length)]
        return Individual(new_genotype, self.strategy, new_parameters, self.repair, bounds=self.bounds)

    # pylint: disable=invalid-name
    def _reproduce_full_variance(self):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the full variance strategy, as described in [1]. To emphasize this, the
        variable names of [1] are used in this function.

        :return: an individual which is the offspring of the current instance
        """
        global_scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [self.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
                         for _ in range(self.length)]
        new_variances = [max(np.exp(global_scale_factor + scale_factors[i])
                             * self.strategy_parameters[i], self._EPSILON)
                         for i in range(self.length)]
        new_rotations = [self.strategy_parameters[i] + self.random.randn() * self._BETA
                         for i in range(self.length, len(self.strategy_parameters))]
        new_rotations = [rotation if abs(rotation) < np.pi
                         else rotation - np.sign(rotation) * 2 * np.pi
                         for rotation in new_rotations]
        T = np.identity(self.length)
        for p in range(self.length - 1):
            for q in range(p + 1, self.length):
                j = int((2 * self.length - p) * (p + 1) / 2 - 2 * self.length + q)
                T_pq = np.identity(self.length)
                T_pq[p][p] = T_pq[q][q] = np.cos(new_rotations[j])
                T_pq[p][q] = -np.sin(new_rotations[j])
                T_pq[q][p] = -T_pq[p][q]
                T = np.matmul(T, T_pq)
        new_genotype = self.genotype + T @ self.random.randn(self.length)
        # Randomly sample out of bounds indices
        oob_indices = (new_genotype < self.bounds[0]) | (new_genotype > self.bounds[1])
        new_genotype[oob_indices] = self.random.uniform(self.bounds[0], self.bounds[1], size=np.count_nonzero(oob_indices))
        return Individual(new_genotype, self.strategy, new_variances + new_rotations, self.repair, bounds=self.bounds)
