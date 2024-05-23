import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
import os

###########################################################
#                                                         #
# Install required dependencies with:                     #
#       pip install -r requirements.dev.txt               #
#                                                         #
###########################################################

class best_cias_solution_plots:
    def __init__(self, logdir):
        collection = None
        scatter = None

        self.logdir = logdir
        self.best_sols = []

        # Set up plot
        self.max_gens = 400
        self.fig, self.ax = plt.subplots()
        self.fig.set_tight_layout(True)

        self.ax.set_xlim((0,1))
        self.ax.set_ylim((0,1))

        self.ax.set_xlabel("$x_0$")
        self.ax.set_ylabel("$x_1$")

    def plot_final(self):
        filename = os.path.join(self.logdir, "best_final.dat")
        figname = os.path.join(self.logdir, "best_final.png")

        # Import best final solution and convert to numpy array
        data = pd.read_csv(filename, delim_whitespace=True, header=None).to_numpy().flatten()

        # Extract fitness and positions, ignore constraint value (last element)
        fitness = data[-2]
        positions = np.asarray(data[:len(data) - 2]).reshape((-1, 2))

        # Extract x and y positions
        x = positions[:,0]
        y = positions[:,1]

        # Make a scatter plot in the 0-1 range for both axis
        line = self.ax.plot(x, y, 'o', color="black", clip_on=False)
        ax.set_title("Best final solution of AMaLGaM with fitness {:.6f}".format(fitness))
        plt.show()
        plt.savefig(figname)
        plt.close()

    def update(self, i):
        print("At generation {:d}/{:d}".format(i, self.max_gens), end="\r")
        data = self.best_sols[i]
        fitness = data[1]
        positions = data[0]
        self.ax.set_title('Best solution of AMaLGaM with fitness {:.6f} in generation {:04d}'.format(fitness, i))
        self.scatter.set_offsets(positions)
        return self.scatter,

    # Only works if AMaLGaM has been run with -w option to write sols each generations
    def plot_evolution(self):
        best_files = [f for f in os.listdir(self.logdir) if os.path.isfile(os.path.join(self.logdir, f)) and f.startswith("best_generation")]
        self.best_sols = []
        for f in best_files:
            data = pd.read_csv(os.path.join(self.logdir, f), delim_whitespace=True, header=None).to_numpy().flatten()
            fitness = data[-2]
            positions = np.asarray(data[:len(data) - 2]).reshape((-1, 2))
            self.best_sols.append([positions, fitness])

            
        data = self.best_sols[0]
        positions = data[0]
        self.scatter = self.ax.scatter(positions[:,0], positions[:,1], marker='o', color="black", clip_on=False)
        ani = FuncAnimation(self.fig, self.update, range(min(len(self.best_sols), self.max_gens)))
        writer = PillowWriter(fps=10)
        filename = "best_sols_AMaLGaM.gif"
        ani.save(filename, writer=writer)


if __name__ == "__main__":
    plotter = best_cias_solution_plots("./logs/")
    #plotter.plot_evolution() # See function description for requirements
    plotter.plot_final()