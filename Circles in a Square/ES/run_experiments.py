from main import CirclesInASquare
from evopy.repair import Repair
import pickle
import numpy as np
import matplotlib.pyplot as plt

def run_experiments(num_circles=10, times=10, plot_name="plot"):
    # Run experiments for both smart and dumb versions
    experiment_data_smart = []
    experiment_data_dumb = []
    baseline_value = 0
    
    for _ in range(times):
        # Smart version
        runner = CirclesInASquare(num_circles, save_file="temp_smart.pkl", 
                                  plot_performance=False, dumb_version=False, population_size=70, num_children=13, 
                                  repair=Repair.BOUNDARY_REPAIR, init_mutation='random', init_alg='complex',
                                  custom_init=True)
        baseline_value = runner.get_target()
        runner.run_evolution_strategies()
        with open('temp_smart.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
            experiment_data_smart.append(loaded_data)
        
        # Dumb version
        runner = CirclesInASquare(num_circles, save_file="temp_dumb.pkl", plot_performance=False, dumb_version=True)
        runner.run_evolution_strategies()
        with open('temp_dumb.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
            experiment_data_dumb.append(loaded_data)

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

    def count_hits(experiment_data, baseline, threshold=0.99):
        hits_baseline = 0
        hits_threshold = 0
        for data in experiment_data:
            if any(best >= baseline for _, best, _, _, _ in data):
                hits_baseline += 1
            if any(best >= baseline * threshold for _, best, _, _, _ in data):
                hits_threshold += 1
        return hits_baseline, hits_threshold

    avg_best_fitness_smart, max_best_fitness_smart, min_best_fitness_smart, sorted_gens_smart, smart_evaluations = process_experiment_data(experiment_data_smart)
    avg_best_fitness_dumb, max_best_fitness_dumb, min_best_fitness_dumb, sorted_gens_dumb, dumb_evaluations = process_experiment_data(experiment_data_dumb)

    # Ensure both smart and dumb versions have the same generations for plotting
    # sorted_gens = sorted(set(sorted_gens_smart).union(set(sorted_gens_dumb)))

    # Count hits for smart and dumb versions
    hits_baseline_smart, hits_threshold_smart = count_hits(experiment_data_smart, baseline_value)
    hits_baseline_dumb, hits_threshold_dumb = count_hits(experiment_data_dumb, baseline_value)

    # Plotting
    avg_best_values_smart = [avg_best_fitness_smart.get(gen, np.nan) for gen in sorted_gens_smart]
    max_best_values_smart = [max_best_fitness_smart.get(gen, np.nan) for gen in sorted_gens_smart]
    min_best_values_smart = [min_best_fitness_smart.get(gen, np.nan) for gen in sorted_gens_smart]

    avg_best_values_dumb = [avg_best_fitness_dumb.get(gen, np.nan) for gen in sorted_gens_dumb]
    max_best_values_dumb = [max_best_fitness_dumb.get(gen, np.nan) for gen in sorted_gens_dumb]
    min_best_values_dumb = [min_best_fitness_dumb.get(gen, np.nan) for gen in sorted_gens_dumb]

    smart_evals = [smart_evaluations[gen] for gen in sorted_gens_smart]
    dumb_evals = [dumb_evaluations[gen] for gen in sorted_gens_dumb]

    plt.figure(figsize=(10, 6))

    # Plot for smart version
    plt.plot(smart_evals, avg_best_values_smart, label='Smart - Average Best Fitness', color='blue')
    plt.fill_between(smart_evals, min_best_values_smart, max_best_values_smart, color='blue', alpha=0.2, label='Smart - Min/Max Best Fitness Range')

    # Plot for dumb version
    plt.plot(dumb_evals, avg_best_values_dumb, label='Dumb - Average Best Fitness', color='green')
    plt.fill_between(dumb_evals, min_best_values_dumb, max_best_values_dumb, color='green', alpha=0.2, label='Dumb - Min/Max Best Fitness Range')

    # Baseline
    plt.axhline(y=baseline_value, color='red', linestyle='--', label='Target')

    plt.xlabel('Evaluations')
    plt.ylabel('Fitness')
    plt.title(f'Best Fitness Across Generations ({num_circles} circles)\nSmart: {hits_threshold_smart} hit 99% baseline\nDumb: {hits_threshold_dumb} hit 99% baseline')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'experiments/{plot_name}.png')
    plt.close()

for i in range(2, 21):
    run_experiments(num_circles=i, times=15, plot_name=f'{i}_circles_best_fitness_br')
