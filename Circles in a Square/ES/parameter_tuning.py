from ES.evopy.repair import Repair
from main import CirclesInASquare
from evopy.strategy import Strategy
import copy
import pickle

params = {
    'population_size': {
        'step':5,
        'lb':20,
        'ub':100
    },
    'num_children':{
        'step':2,
        'lb':1,
        'ub':20
    },
    'strategy':{
        'step':1,
        'lb':1,
        'ub':1
    }
}

ordered_params = ['population_size', 'num_children', 'strategy']

best_config = {
        'val': 100,
        'params': {}
    }

def tune_params(param_index, param_vals):
    global best_config
    if param_index >= len(ordered_params):
        best_fit = evaluate_params(param_vals)
        return {
            'val': best_fit,
            'params': param_vals
        }
    param_name = ordered_params[param_index]
    pd = params[param_name]
    curr_best = {
        'val': 100,
        'params': {}
    }
    for pv in range(pd['lb'],pd['ub']+1,pd['step']):
        param_vals_temp = copy.deepcopy(param_vals) 
        param_vals_temp[param_name] = pv
        if param_name == "strategy":
            param_vals_temp[param_name] = Repair(pv)
        best_params = tune_params(param_index + 1, param_vals_temp)
        if best_params['val'] < curr_best['val']:
            curr_best = {
                'val': best_params['val'],
                'params': best_params['params'].copy()
            }
        if best_params['val'] < best_config['val']:
            best_config = {
                'val': best_params['val'],
                'params': best_params['params'].copy()
            }
            print("NEW BEST CONFIG: ", curr_best)
    return best_params

def evaluate_params(param_vals):
    diff = 0
    for circles in range(2, 10):
        runner = CirclesInASquare(circles, save_file="temp_smart.pkl", plot_performance=False, dumb_version=False, **param_vals)
        runner.run_evolution_strategies()
        with open('temp_smart.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        best_fitness = 0
        for gen, best, avg, std in loaded_data:
            best_fitness = max(best_fitness, best)
        
        diff += runner.get_target() - best_fitness
    return diff

print(tune_params(0, {'population_size':-1, 'num_children': -1, 'strategy': -1}))




