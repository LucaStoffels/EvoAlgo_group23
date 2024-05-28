from main import CirclesInASquare
from evopy.strategy import Strategy
import copy

params = {
    'population_size': {
        'step':5,
        'lb':5,
        'ub':100
    },
    'num_children':{
        'step':1,
        'lb':1,
        'ub':10
    },
    'strategy':{
        'step':1,
        'lb':1,
        'ub':3
    }
}

ordered_params = ['population_size', 'num_children', 'strategy']

def tune_params(param_index, param_vals):
    if param_index >= len(ordered_params):
        best_fit = evaluate_params(param_vals)
        print("Evaluating:",param_vals)
        print("Result:", best_fit)
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
            param_vals_temp[param_name] = Strategy(pv)
        best_params = tune_params(param_index + 1, param_vals_temp)
        if best_params['val'] < curr_best['val']:
            curr_best = {
                'val': best_params['val'],
                'params': best_params['params'].copy()
            }
            print("NEW BEST CONFIG: ", curr_best)
    return best_params

def evaluate_params(param_vals):
    diff = 0
    for circles in range(10, 12):
        solver = CirclesInASquare(circles, output_statistics=False, **param_vals)
        diff += solver.get_target() - solver.run_evolution_strategies()
    return diff

print(tune_params(0, {'population_size':-1, 'num_children': -1, 'strategy': -1}))




