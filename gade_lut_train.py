import numpy as np
import random
from scipy import special
import os
from deap import base, creator, tools
import argparse
import json
from scipy.stats import cauchy, norm

ACT_FUNCS = {
    "exp": lambda x: np.where(x <= -5.5625, 32 * np.exp(x), np.exp(x)), # Parameter Scaling for EXP
    "reci": lambda x: 1 / x,
    "rsqrt": lambda x: 1 / np.sqrt(x),
    "gelu": lambda x: 0.5 * x * (1 + special.erf(x / np.sqrt(2))),
    "silu": lambda x: x / (1 + np.exp(-x)),
}

def round_to_nearest_bits(x, decimal_bits):
    scaled_value = x * (2 ** decimal_bits)
    rounded_value = np.round(scaled_value)
    result = rounded_value / (2 ** decimal_bits)
    return result

def save_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

def load_from_file(filename):
    with open(filename, "r") as file:
        return json.load(file)

def calculate_pwl_param(a1, a2, act_type, dynamic=False):
    func = ACT_FUNCS[act_type]
    # handle segment duplicate
    if a2 == a1:
        slope = 0
        intercept = 0
    else:
        # least square method for linear equation
        x = np.arange(a1, a2 + 1/128, 1/128)
        y = func(x)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    if dynamic:
        # dynamic fixed-point format (DFF)
        if slope == 0:
            slope_scale = 0
        else:
            slope_scale = np.floor(np.clip((1 + np.log2(np.abs(slope))), 0, 7))

        if intercept == 0:
            intercept_scale = 0
        else:
            intercept_scale = np.floor(np.clip((1 + np.log2(np.abs(intercept))), 0, 7))
    else:
        # fixed-point format
        slope_scale = 2
        intercept_scale = 2
    
    slope_dff = np.round(slope * 2**(7-slope_scale)).clip(-128, 127) / 2**(7-slope_scale)
    intercept_dff = np.round(intercept * 2**(7-intercept_scale)).clip(-128, 127) / 2**(7-intercept_scale)
    return slope_dff, intercept_dff

def get_list(split_point, act_type, dynamic=False):
    pwl_param = [calculate_pwl_param(a1, a2, act_type, dynamic=dynamic) for a1, a2 in zip(split_point[:-1], split_point[1:])]
    slope_dff, intercept_dff = zip(*pwl_param)
    return slope_dff, intercept_dff

def piecewise_linear_approximation(x, split_points, slope_dff, intercept_dff, dynamic=False):
    index = np.digitize(x, split_points) - 1
    index = min(index, len(slope_dff) - 1)
    if dynamic == True:
        # dynamic fixed-point format (DFF)
        if x == 0:
            x_scale = 0
        else:
            x_scale = np.floor(np.clip((1 + np.log2(np.abs(x))), 0, 7))
    else:
        # fixed-point format Q5.3
        x_scale = 2**4
    
    x_dff = np.round(x * 2**(7-x_scale)).clip(-128, 127) / 2**(7-x_scale)
    out = slope_dff[index] * x_dff + intercept_dff[index]
    return out

def create_float_point_attr(sp_range):
    rand_val = np.random.uniform(sp_range[0], sp_range[1])
    return rand_val

def best_split_points_finder(func_name, x_range, sp_range, num_splits, total_iters=1000, pop_size=50, neg_inf=-4, pos_inf=4, dynamic=False):
    func = ACT_FUNCS[func_name]
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", create_float_point_attr, sp_range)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_splits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        individual_sorted = sorted(individual)
        split_points = [neg_inf] + individual_sorted + [pos_inf]
        slope_dff, intercept_dff = get_list(split_points, func_name, dynamic=dynamic)
        error = 0.0
        x_values = np.arange(x_range[0], x_range[1], 0.01)
        y_values = np.array([func(x) for x in x_values])
        split_points = [neg_inf if split_point < neg_inf else pos_inf if split_point > pos_inf else split_point for split_point in split_points]
        split_points = [np.round(split_point * 2**4) / 2**4 for split_point in split_points]
        approx_values = [piecewise_linear_approximation(x, split_points, slope_dff, intercept_dff, dynamic=dynamic) for x in x_values]

        # Parameter Scaling for EXP
        if func_name == "exp":
            y_values = np.array([y_values[idx] / 32 if x <= -5.5625 else y_values[idx] for idx, x in enumerate(x_values)])
            approx_values = np.array([approx_values[idx] / 32 if x <= -5.5625 else approx_values[idx] for idx, x in enumerate(x_values)])

        error += np.mean(np.abs(y_values - approx_values))
        # minimum distance between breakpoints
        min_distance = 0.0078125  
        # large penalty for close or duplicate points
        for i in range(1, len(split_points) - 1):
            if i < len(split_points) - 1 and abs(split_points[i] - split_points[i - 1]) < min_distance:
                error += 1e5
        # large penalty for negative value in positive functions
        if func_name == 'exp' or func_name == 'reci' or func_name == 'rsqrt':
            negative_values = np.array(approx_values)[np.array(approx_values) < 0]
            penalty = len(negative_values) * 100 + np.sum(np.abs(negative_values)) * 1e+5
            error += penalty
        return error,

    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=pop_size)
    
    success_F = []
    success_CR = []
    strategy_success = [0, 0]
    
    pop_x = np.array([list(ind) for ind in population])
    pop_f = np.array([toolbox.evaluate(ind)[0] for ind in population])
    
    for gen in range(total_iters):
        new_pop_x = np.copy(pop_x)
        new_pop_f = np.copy(pop_f)
        progress = gen / total_iters

        max_prob = 0.6
        min_prob = 0.2
        # progressive strategy prob calculation
        strategy_prob = min_prob + (max_prob - min_prob) * (1 + np.cos(np.pi * progress)) / 2
        
        for i in range(pop_size):
            F = cauchy.rvs(loc=np.mean(success_F) if success_F else 0.5, scale=strategy_prob)
            CR = norm.rvs(loc=np.mean(success_CR) if success_CR else 0.5, scale=strategy_prob)

            F = np.clip(F, 0.1, 1.0)
            CR = np.clip(CR, 0.0, 1.0)
            
            strategy = 1 if np.random.rand() > strategy_prob else 0
        
            # Mutation
            mutant = [0] * num_splits
            if strategy == 0:  # DE/best/1
                r1, r2 = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                best_idx = np.argmin(pop_f) 
                for j in range(num_splits):
                    mutant[j] = pop_x[best_idx][j] + F * (pop_x[r1][j] - pop_x[r2][j])
            else:  # GA/Rand
                for j in range(num_splits):
                    mutant[j] = pop_x[i][j] + np.random.normal(0, 0.2)
            
            # clip to bounds
            for j in range(num_splits):
                mutant[j] = min(max(mutant[j], sp_range[0]), sp_range[1])
            
            # Crossover: Binomial
            trial = creator.Individual([0] * num_splits)
            j_rand = random.randint(0, num_splits - 1)
            for j in range(num_splits):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop_x[i][j]
            
            # Selection
            trial_f = toolbox.evaluate(trial)[0]
            if trial_f <= pop_f[i]:
                new_pop_x[i] = list(trial)
                new_pop_f[i] = trial_f
                success_F.append(F)
                success_CR.append(CR)
                strategy_success[strategy] += 1
        
        # Update population
        pop_x = new_pop_x
        pop_f = new_pop_f
        
        if gen % 10 == 0:
            best_idx = np.argmin(pop_f)
            best_individual = pop_x[best_idx]
            print(f"[{func_name}_Entry{num_splits}][{gen}] Best MAE: {pop_f[best_idx]:.9f}, Strategy success: {strategy_success}, Strategty Prob: {strategy_prob}")

    # Select the best individual
    best_idx = np.argmin(pop_f)
    best_individual = pop_x[best_idx]
    best_splits = [neg_inf] + sorted(best_individual) + [pos_inf]
    best_fitness = pop_f[best_idx]
    print(f"Final MAE for {func_name}: {best_fitness:.9f}")
    
    return best_splits, best_fitness

def autopwl(activation_function_name, x_range=(-4, 4), sp_range=(-4, 4), num_splits=10, total_iters=100, neg_inf=-4, pos_inf=4, dynamic=False):
    if activation_function_name not in ACT_FUNCS:
        print("Invalid activation function name. Valid names are:", ", ".join(ACT_FUNCS.keys()))
        return
    print("x_range:", x_range, "sp_range:", sp_range)
    split_points, mae = best_split_points_finder(activation_function_name, x_range, sp_range, num_splits, total_iters, neg_inf=neg_inf, pos_inf=pos_inf, dynamic=dynamic)
    slope_dff, intercept_dff = get_list(split_points, activation_function_name, dynamic=dynamic)
    return split_points, slope_dff, intercept_dff, mae

def gade_lut_trainer(act_func='exp', x_range=(-4, 4), sp_range=(-4, 4), num_splits=7, total_iters=100, dynamic=False, seed=42):
    if act_func == 'gelu' or act_func == 'silu':
        neg_inf = -4.5
        pos_inf = 10000
    elif act_func == 'exp':
        neg_inf = -9.0
        pos_inf = 0.0
    elif act_func == 'reci':
        neg_inf = 0.5
        pos_inf = 2.5
    elif act_func == 'rsqrt':
        neg_inf = 0.5
        pos_inf = 4.5
    else:
        raise NotImplementedError('Not support')
    
    results = {}
    results[act_func] = {}
    split_points, slope_dff, intercept_dff, mae = autopwl(act_func, x_range=x_range, sp_range=sp_range, num_splits=num_splits, total_iters=total_iters, neg_inf=neg_inf, pos_inf=pos_inf, dynamic=dynamic)
    split_points_tmp = [round_to_nearest_bits(split_point, 4) for split_point in split_points]

    results[act_func] = {
        "breakpoints": split_points_tmp[1:-1],
        "slope_dff": slope_dff,
        "intercept_dff": intercept_dff,
    }

    save_to_file(results, f"./gade_pwl_param/{act_func}/entry_{num_splits+1}/{act_func}_{num_splits}_dff_{dynamic}_seed_{seed}_{mae:.9f}.json")
    return results, mae

def config_parser():
    parser = argparse.ArgumentParser(description='LUT-APP GADE Algorithm')
    parser.add_argument("--act_func", type=str, default='exp', help="Activation function name")
    parser.add_argument("--num_splits", type=int, default=7, help="Number of split points")
    parser.add_argument("--total_iters", type=int, default=500, help="Total iterations for DE")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--x_range", nargs='+', type=float, default=[-9.0, 0.0], help="List of x range")
    parser.add_argument("--sp_range", nargs='+', type=float, default=[-9.0, 0.0], help="List of split points range")
    parser.add_argument("--dynamic", action='store_true', default=True, help="Activate Dynamic Fixed-point Format(DFF)")
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()
    
    best_mae = float('inf')
    best_results = None
    
    for run in range(args.num_runs):
        # set random seed for each run
        seed = random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Run {run + 1}/{args.num_runs} with seed {seed}")
        
        results, mae = gade_lut_trainer(
            act_func=args.act_func,
            x_range=tuple(args.x_range),
            sp_range=tuple(args.sp_range),
            num_splits=args.num_splits,
            total_iters=args.total_iters,
            dynamic=args.dynamic,
            seed=seed
        )
        
        if mae < best_mae:
            best_mae = mae
            best_results = results

if __name__ == "__main__":
    main()