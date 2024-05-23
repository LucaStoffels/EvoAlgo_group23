import matplotlib
matplotlib.use('Qt5Agg')

import math
from evopy import EvoPy
from evopy import ProgressReport
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np

###########################################################
#                                                         #
# EvoPy framework from https://github.com/evopy/evopy     #
# Read documentation on github for further information.   #
#                                                         #
# Adjustments by Renzo Scholman:                          #
#       Added evaluation counter (also in ProgressReport) #
#       Added max evaluation stopping criterion           #
#       Added random repair for solution                  #
#       Added target fitness value tolerance              #
#                                                         #
# Original license stored in LICENSE file                 #
#                                                         #
# Install required dependencies with:                     #
#       pip install -r requirements.dev.txt               #
#                                                         #
###########################################################

# np/scipy CiaS implementation is faster for higher problem dimensions, i.e, more than 11 or 12 circles.
def circles_in_a_square_scipy(individual):
   points = np.reshape(individual, (-1, 2))
   dist = euclidean_distances(points)
   np.fill_diagonal(dist, 1e10)
   return np.min(dist)

# Pure python implementation is faster for lower problem dimensions
def circles_in_a_square(individual):
    n = len(individual)
    distances = []
    for i in range(0, n-1, 2):
        for j in range(i + 2, n, 2):
            distances.append(math.sqrt(math.pow((individual[i] - individual[j]), 2)
                              + math.pow((individual[i + 1] - individual[j + 1]), 2)))
    return min(distances)


class CirclesInASquare:
    def __init__(self, n_circles, output_statistics=True, plot_sols=False, print_sols=False):
        self.print_sols = print_sols
        self.output_statistics = output_statistics
        self.plot_best_sol = plot_sols
        self.n_circles = n_circles
        self.fig = None
        self.ax = None
        assert 2 <= n_circles <= 20

        if self.plot_best_sol:
            self.set_up_plot()

        if self.output_statistics:
            self.statistics_header()

    def set_up_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("$x_0$")
        self.ax.set_ylabel("$x_1$")
        self.ax.set_title("Best solution in generation 0")
        self.fig.show()

    def statistics_header(self):
        if self.print_sols:
            print("Generation Evaluations Best-fitness (Best individual..)")
        else:
            print("Generation Evaluations Best-fitness")

    def statistics_callback(self, report: ProgressReport):
        output = "{:>10d} {:>11d} {:>12.8f} {:>12.8f} {:>12.8f}".format(report.generation, report.evaluations,
                                                                        report.best_fitness, report.avg_fitness,
                                                                        report.std_fitness)

        if self.print_sols:
            output += " ({:s})".format(np.array2string(report.best_genotype))
        print(output)

        if self.plot_best_sol:
            points = np.reshape(report.best_genotype, (-1, 2))
            self.ax.clear()
            self.ax.scatter(points[:, 0], points[:, 1], clip_on=False, color="black")
            self.ax.set_xlim((0, 1))
            self.ax.set_ylim((0, 1))
            self.ax.set_title("Best solution in generation {:d}".format(report.generation))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def get_target(self):
        values_to_reach = [
            1.414213562373095048801688724220,  # 2
            1.035276180410083049395595350499,  # 3
            1.000000000000000000000000000000,  # 4
            0.707106781186547524400844362106,  # 5
            0.600925212577331548853203544579,  # 6
            0.535898384862245412945107316990,  # 7
            0.517638090205041524697797675248,  # 8
            0.500000000000000000000000000000,  # 9
            0.421279543983903432768821760651,  # 10
            0.398207310236844165221512929748,
            0.388730126323020031391610191835,
            0.366096007696425085295389370603,
            0.348915260374018877918854409001,
            0.341081377402108877637121191351,
            0.333333333333333333333333333333,
            0.306153985300332915214516914060,
            0.300462606288665774426601772290,
            0.289541991994981660261698764510,
            0.286611652351681559449894454738
        ]

        return values_to_reach[self.n_circles - 2]

    def run_evolution_strategies(self):
        callback = self.statistics_callback if self.output_statistics else None

        evopy = EvoPy(
            circles_in_a_square if self.n_circles < 12 else circles_in_a_square_scipy,  # Fitness function
            self.n_circles * 2,  # Number of parameters
            reporter=callback,  # Prints statistics at each generation
            maximize=True,
            generations=1000,
            bounds=(0, 1),
            target_fitness_value=self.get_target(),
            max_evaluations=1e5,
        )

        best_solution = evopy.run()

        if self.plot_best_sol:
            plt.close()

        return best_solution


if __name__ == "__main__":
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=True)
    best = runner.run_evolution_strategies()
