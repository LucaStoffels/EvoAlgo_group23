from main import CirclesInASquare
from evopy.repair import Repair
import pickle
import numpy as np
import matplotlib.pyplot as plt

population_size = 30
def run_experiments(num_circles=10, times=10, plot_name="plot"):
    experiment_data_random = []
    experiment_data_boundary = []
    experiment_data_constraint = []
    baseline_value = 0
    
    for _ in range(times):
        runner = CirclesInASquare(num_circles, save_file="temp_random.pkl", plot_performance=False, dumb_version=False, custom_init=False, repair=Repair.RANDOM_REPAIR, population_size=population_size)
        baseline_value = runner.get_target()
        runner.run_evolution_strategies()
        with open('temp_random.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
            experiment_data_random.append(loaded_data)
        
        runner = CirclesInASquare(num_circles, save_file="temp_boundary.pkl", plot_performance=False, dumb_version=False, custom_init=False, repair=Repair.BOUNDARY_REPAIR, population_size=population_size)
        runner.run_evolution_strategies()
        with open('temp_boundary.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
            experiment_data_boundary.append(loaded_data)

        runner = CirclesInASquare(num_circles, save_file="temp_constraint.pkl", plot_performance=False, dumb_version=False, custom_init=False, repair=Repair.CONSTRAINT_DOMINATION, population_size=population_size)
        runner.run_evolution_strategies()
        with open('temp_constraint.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
            experiment_data_constraint.append(loaded_data)

    def process_experiment_data(experiment_data):
        generational_data = {}
        for data in experiment_data:
            for gen, best, avg, std, evals in data:
                if gen not in generational_data:
                    generational_data[gen] = {'best': [], 'avg': [], 'std': [], 'evals': 0}
                generational_data[gen]['best'].append(best)
                generational_data[gen]['avg'].append(avg)
                generational_data[gen]['std'].append(std)
                if generational_data[gen]['evals'] != evals and generational_data[gen]['evals'] != 0:
                    print("Evals mismatch")
                    print(generational_data[gen]['evals'])
                    print(evals)
                    exit(1)
                else:
                    generational_data[gen]['evals'] = evals
        
        avg_best_fitness = {}
        avg_avg_fitness = {}
        avg_std_fitness = {}
        max_best_fitness = {}
        max_avg_fitness = {}
        max_std_fitness = {}
        min_best_fitness = {}
        min_avg_fitness = {}
        min_std_fitness = {}
        evaluations = {}

        for gen in generational_data:
            avg_best_fitness[gen] = np.mean(generational_data[gen]['best'])
            avg_avg_fitness[gen] = np.mean(generational_data[gen]['avg'])
            avg_std_fitness[gen] = np.mean(generational_data[gen]['std'])

            max_best_fitness[gen] = np.max(generational_data[gen]['best'])
            max_avg_fitness[gen] = np.max(generational_data[gen]['avg'])
            max_std_fitness[gen] = np.max(generational_data[gen]['std'])

            min_best_fitness[gen] = np.min(generational_data[gen]['best'])
            min_avg_fitness[gen] = np.min(generational_data[gen]['avg'])
            min_std_fitness[gen] = np.min(generational_data[gen]['std'])

            evaluations[gen] = generational_data[gen]['evals']

        return avg_best_fitness, max_best_fitness, min_best_fitness, sorted(generational_data.keys()), evaluations

    avg_best_fitness_random, max_best_fitness_random, min_best_fitness_random, sorted_gens_random, _ = process_experiment_data(experiment_data_random)
    avg_best_fitness_boundary, max_best_fitness_boundary, min_best_fitness_boundary, sorted_gens_boundary, _ = process_experiment_data(experiment_data_boundary)
    avg_best_fitness_constraint, max_best_fitness_constraint, min_best_fitness_constraint, sorted_gens_constraint, evaluations = process_experiment_data(experiment_data_constraint)

    sorted_gens = sorted(set(sorted_gens_random).union(set(sorted_gens_boundary)).union(set(sorted_gens_constraint)))


    # Plotting
    avg_best_values_random = [avg_best_fitness_random.get(gen, np.nan) for gen in sorted_gens]
    max_best_values_random = [max_best_fitness_random.get(gen, np.nan) for gen in sorted_gens]
    min_best_values_random = [min_best_fitness_random.get(gen, np.nan) for gen in sorted_gens]

    avg_best_values_boundary = [avg_best_fitness_boundary.get(gen, np.nan) for gen in sorted_gens]
    max_best_values_boundary = [max_best_fitness_boundary.get(gen, np.nan) for gen in sorted_gens]
    min_best_values_boundary = [min_best_fitness_boundary.get(gen, np.nan) for gen in sorted_gens]

    avg_best_values_constraint = [avg_best_fitness_constraint.get(gen, np.nan) for gen in sorted_gens]
    max_best_values_constraint = [max_best_fitness_constraint.get(gen, np.nan) for gen in sorted_gens]
    min_best_values_constraint = [min_best_fitness_constraint.get(gen, np.nan) for gen in sorted_gens]

    evals = [evaluations[gen] for gen in sorted_gens]

    plt.figure(figsize=(10, 6))

    # Plot for random version
    plt.plot(evals, avg_best_values_random, label='Random - Average Best Fitness', color='blue')
    plt.fill_between(evals, min_best_values_random, max_best_values_random, color='blue', alpha=0.1, label='Random - Min/Max Best Fitness Range')

    # Plot for boundary repair version
    plt.plot(evals, avg_best_values_boundary, label='Boundary Repair - Average Best Fitness', color='green')
    plt.fill_between(evals, min_best_values_boundary, max_best_values_boundary, color='green', alpha=0.1, label='Boundary - Min/Max Best Fitness Range')

    # Plot for contraint domination version
    plt.plot(evals, avg_best_values_constraint, label='Constraint domination - Average Best Fitness', color='red')
    plt.fill_between(evals, min_best_values_constraint, max_best_values_constraint, color='red', alpha=0.1, label='Constraint - Min/Max Best Fitness Range')

    # Baseline
    plt.axhline(y=baseline_value, color='red', linestyle='--', label='Target')

    plt.xlabel('Evaluations')
    plt.ylabel('Fitness')
    plt.title(f'Comparison of different constraint handling methods')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'experiments/{plot_name}.png')
    plt.close()

for i in [20]:
    run_experiments(num_circles=i, times=10, plot_name=f'{i}_constraint_handling_comparison')

