import json
import logging
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from skopt import Optimizer
from skopt.space import Integer, Real

from src.executeinitial import prepare_simulation_config, prepare_workloads, flatten_workloads, execute_simulation
from src.motivational.constants import PROACTIVE_RECONCILE_INTERVAL, REACTIVE_RECONCILE_INTERVAL, \
    PREPARE_PREDICTION_WINDOW_SIZE
from src.preprocessing import create_inputs_outputs_seperated_per_app_windowed
from src.train import save_models

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStep:
    params: Dict[str, float]
    X: np.ndarray
    y: Dict[str, float]  # Task-specific targets
    penalty: float
    improvement: bool


def run_reactive_simulation(state, params):
    apps = state['apps']
    mapping = state['mapping']
    infra_config = state['infra_config']
    workload_base = state['workload_base']
    sim_inputs = state['sim_inputs']
    cache_policy = state['cache_policy']
    task_priority = state['task_priority']
    keep_alive = state['keep_alive']
    queue_length = state['queue_length']

    sample = [0] * len(params)
    for idx, param in mapping.items():
        sample[int(idx)] = params[param]

    sample = np.array(sample)

    # Prepare infrastructure configuration
    sim_config = prepare_simulation_config(sample, mapping, infra_config)

    # Prepare workloads
    workloads = prepare_workloads(sample, mapping, workload_base, apps)
    # Flatten workloads into single sorted list
    flattened_workloads = flatten_workloads(workloads)

    # Combine infrastructure and workload configurations
    full_config = {
        "infrastructure": sim_config,
        "workload": flattened_workloads
    }

    try:
        scheduling_strategy = 'kn_kn'
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy, cache_policy=cache_policy,
                                    task_priority=task_priority,
                                    keep_alive=keep_alive,
                                    queue_length=queue_length, reconcile_interval=REACTIVE_RECONCILE_INTERVAL)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length
        }

        return result
    except Exception as e:
        logger.error(f"Error in simulation")
        logger.exception(e)


def run_proactive_simulation(state, models, params):
    apps = state['apps']
    mapping = state['mapping']
    infra_config = state['infra_config']
    workload_base = state['workload_base']
    sim_inputs = state['sim_inputs']
    cache_policy = state['cache_policy']
    task_priority = state['task_priority']
    keep_alive = state['keep_alive']
    queue_length = state['queue_length']

    sample = [0] * len(params)
    for idx, param in mapping.items():
        sample[int(idx)] = params[param]

    sample = np.array(sample)

    # Prepare infrastructure configuration
    sim_config = prepare_simulation_config(sample, mapping, infra_config)

    # Prepare workloads
    workloads = prepare_workloads(sample, mapping, workload_base, apps)
    # Flatten workloads into single sorted list
    flattened_workloads = flatten_workloads(workloads)

    # Combine infrastructure and workload configurations
    full_config = {
        "infrastructure": sim_config,
        "workload": flattened_workloads
    }

    try:
        scheduling_strategy = 'prokn_prokn'
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy, cache_policy=cache_policy,
                                    task_priority=task_priority,
                                    keep_alive=keep_alive,
                                    queue_length=queue_length, models=models,
                                    reconcile_interval=PROACTIVE_RECONCILE_INTERVAL)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length
        }

        return result
    except Exception as e:
        logger.error(f"Error in simulation")
        logger.exception(e)


class TabularPrintCallback:
    def __init__(self, param_names):
        self.param_names = param_names
        self.iteration = 0
        self.header_printed = False

    def __call__(self, res):
        # Print header on first call
        if not self.header_printed:
            # Create header row with parameter names (truncated if needed)
            param_headers = [name[:8] + "..." if len(name) > 8 else name for name in self.param_names]
            header = f"|   iter    |  target   | " + " | ".join(param_headers) + " |"

            # Print separator line
            separator = "=" * len(header)
            print(separator)
            print(header)
            print("-" * len(header))

            self.header_printed = True

        # Get best result so far
        best_idx = np.argmin(res.func_vals)
        best_value = res.func_vals[best_idx]
        best_x = res.x_iters[best_idx]

        # Format the row
        row = f"| {self.iteration:<9} | {-best_value:<9.2f} | " + " | ".join(f"{val:<9.3f}" for val in best_x) + " |"
        print(row)

        self.iteration += 1
        return False  # Continue optimization


def create_device_constraint_function(dimensions):
    """
    Creates a constraint function that ensures all parameters starting with 'device_' sum to 1

    Args:
        dimensions: List of dimension objects with names
    """
    # Create a mapping of parameter names to indices
    param_indices = {dim.name: i for i, dim in enumerate(dimensions)}

    # Filter for parameters that start with 'device_'
    device_params = [name for name in param_indices.keys() if name.startswith('device_')]

    def constraint(x):
        # If no device parameters, constraint is satisfied
        if not device_params:
            return True

        # Sum the values of the device parameters
        device_sum = sum(x[param_indices[param]] for param in device_params)

        # Check if sum equals 1 (with small tolerance for floating point precision)
        return abs(device_sum - 1.0) < 1e-6

    return constraint


# Function to preprocess points before evaluation
def preprocess_points(points, param_names):
    processed_points = []

    for point in points:
        # Create a copy of the point
        new_point = point.copy()

        # Find all device parameters and their indices
        device_indices = [i for i, name in enumerate(param_names) if name.startswith('device_')]

        if device_indices:
            # Get the current sum of device parameters
            device_sum = sum(point[i] for i in device_indices)

            # Only scale if the sum is not zero (to avoid division by zero)
            if device_sum > 0:
                # Scale the device parameters to sum to 1
                for i in device_indices:
                    new_point[i] = point[i] / device_sum
            else:
                # If all device parameters are zero, distribute evenly
                for i in device_indices:
                    new_point[i] = 1.0 / len(device_indices)

        processed_points.append(new_point)

    return processed_points


class ProactiveParallelOptimizer:
    def __init__(self, initial_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]], target_penalty=0.1,
                 param_bounds_range_factor=0.5, n_iterations=10, n_parallel=4):
        self.target_penalty = target_penalty
        self.initial_datasets = initial_datasets  # {task: (X, y)} format
        # Track best penalty
        self.best_proactive_penalty = float('inf')
        # Store optimization steps that led to improvements
        self.improvement_history = {task: [] for task in initial_datasets.keys()}
        # Determines the bounds of the parameters
        self.param_bounds_range_factor = param_bounds_range_factor
        self.n_iterations = n_iterations
        self.n_parallel = n_parallel

    def create_model(self):
        """Creates a new model instance. Override this method to change the model type."""
        return make_pipeline(
            PolynomialFeatures(degree=3),
            StandardScaler(),
            SGDRegressor(
                loss='huber',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                tol=1e-4,
                epsilon=0.1,
                random_state=42
            )
        )

    def optimize_sample(self, initial_sample, state, space):
        # Create the search space based on initial sample
        dimensions, param_names = self.create_space(initial_sample, state, space)

        # Create constraint function for device parameters
        constraint = create_device_constraint_function(dimensions)

        # Initialize the optimizer
        opt = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",  # Gaussian Process
            acq_func="EI",  # Expected Improvement
            acq_optimizer="sampling",
            initial_point_generator="lhs",
            random_state=42
        )

        iterations = []
        # Print header for tabular output
        param_headers = [name[:8] + "..." if len(name) > 8 else name for name in param_names]
        header = f"|   iter    |  target   | " + " | ".join(param_headers) + " |"
        separator = "=" * len(header)
        print(separator)
        print(header)
        print("-" * len(header))
        max_i = self.n_iterations
        i = 0
        while i < max_i:
            # Ask for points to evaluate in parallel
            points = opt.ask(n_points=self.n_parallel)
            print(points)
            valid_points = preprocess_points(points, param_names)
            print(valid_points)
            # Evaluate points in parallel
            eval_results = Parallel(n_jobs=self.n_parallel)(
                delayed(self.evaluate_parameters_wrapper)(x, state['sample'], self.initial_datasets, param_names)
                for x in valid_points
            )

            # Extract penalties for optimizer
            penalties = [result['penalty'] for result in eval_results if result['penalty']]

            # Tell optimizer the results
            opt.tell(points, penalties)

            # Find the best result from this batch
            best_batch_result = None
            best_batch_penalty = float('inf')
            for result in eval_results:
                if 'proactive_penalty' in result and result['proactive_penalty'] < best_batch_penalty:
                    logger.info("Found best result in batch")
                    best_batch_penalty = result['proactive_penalty']
                    best_batch_result = result

            # Only update if the batch's best result is better than our overall best
            if best_batch_result is not None and best_batch_penalty < self.best_proactive_penalty:
                print("Best in batch is global best for now")
                self.best_proactive_penalty = best_batch_penalty

                # Add new data to improvement history
                for task in self.initial_datasets.keys():
                    if task in best_batch_result['X_new'] and len(best_batch_result['X_new'][task]) > 0:
                        self.improvement_history[task].append(
                            OptimizationStep(
                                params=best_batch_result['params'].copy(),
                                X=best_batch_result['X_new'][task].copy(),
                                y={task: np.array(best_batch_result['y_new'][task])},
                                penalty=best_batch_penalty,
                                improvement=True
                            )
                        )

                        # Update initial datasets with new data
                        X_initial, y_initial = self.initial_datasets[task]
                        X_new = best_batch_result['X_new'][task].reshape(-1, 1)
                        y_new = best_batch_result['y_new'][task]

                        self.initial_datasets[task] = (
                            np.concatenate([X_initial, X_new]),
                            np.concatenate([y_initial, y_new])
                        )

            # Track iteration
            best_idx = np.argmin(opt.yi)
            best_value = opt.yi[best_idx]
            best_params = {param_names[j]: opt.Xi[best_idx][j] for j in range(len(param_names))}
            best_x = opt.Xi[best_idx]

            if -best_value != -1000000.00:
                iterations.append({
                    'iteration': i,
                    'best_penalty': -best_value,
                    'best_params': best_params
                })

                row = f"| {i:<9} | {-best_value:<9.2f} | " + " | ".join(f"{val:<9.3f}" for val in best_x) + " |"
                print(row)
                i = i + 1
            else:
                print("invalid config")

        # Return best parameters found
        best_idx = np.argmin(opt.yi)
        best_params = {param_names[j]: opt.Xi[best_idx][j] for j in range(len(param_names))}

        return best_params, iterations

    def create_space(self, initial_sample, state, space):
        """Convert parameter bounds to skopt space definition"""
        param_bounds = self.get_bounds(initial_sample, state, space)

        dimensions = []
        param_names = []

        for param_name, (lower, upper) in param_bounds.items():
            param_names.append(param_name)
            if param_name == 'cluster_size':
                dimensions.append(Integer(lower, upper, name=param_name))
            else:
                dimensions.append(Real(lower, upper, name=param_name))

        return dimensions, param_names

    def evaluate_parameters_wrapper(self, x, state, datasets, param_names):
        """Convert list of parameters to dictionary for evaluation"""
        params = {param_names[i]: x[i] for i in range(len(x))}
        return self.evaluate_parameters(state, datasets, **params)

    def get_bounds(self, initial_sample, state, space):
        mapping = state['sample']['mapping']

        param_bounds = {}
        for idx, param in mapping.items():
            param_value = initial_sample[int(idx)]
            # TODO decide whether we need to clamp/abort on specific values (i.e., out of "range")
            param_up = param_value + (param_value * self.param_bounds_range_factor)
            param_down = param_value - (param_value * self.param_bounds_range_factor)
            if param == 'cluster_size':
                param_down = int(param_down)
                param_up = int(param_up)
                cluste_max = space['csc']['max']
                cluster_min = space['csc']['min']

                if param_up > cluste_max:
                    param_up = cluste_max
                if param_down < cluster_min:
                    param_down = cluster_min
                param_up = int(cluste_max)
                param_down = int(cluster_min)

            if 'device' in param:
                device = param.split('_')[-1]
                device_max_proportion = space['pci'][device]['max']
                device_min_proportion = space['pci'][device]['min']
                if param_up > device_max_proportion:
                    param_up = device_max_proportion
                if param_down < device_min_proportion:
                    param_down = device_min_proportion

                param_up = device_max_proportion
                param_down = device_min_proportion
            if param == 'network_bandwidth':
                network_max_bandwidth = space['nwc']['max']
                network_min_bandwidth = space['nwc']['min']
                if param_up > network_max_bandwidth:
                    param_up = network_max_bandwidth
                if param_down < network_min_bandwidth:
                    param_down = network_min_bandwidth

                param_up = network_max_bandwidth
                param_down = network_min_bandwidth
            if 'workload' in param:
                app = param.split('_')[-1]
                workload_min = space['wsc'][app]['min']
                workload_max = space['wsc'][app]['max']
                if param_up > workload_max:
                    param_up = workload_max
                if param_down < workload_min:
                    param_down = workload_min

                param_up = workload_max
                param_down = workload_min
            if param != 'cluster_size' and param_up == param_down:
                param_up += 0.0000001

            if param == 'cluster_size' and param_up == param_down:
                param_up += 1

            param_bounds[param] = (param_down, param_up)
        print(param_bounds)
        return param_bounds

    def evaluate_parameters(self, state, datasets, **params):
        # We want to avoid parameters which sum is higher than 1
        total_prop = 0
        for k in params.keys():
            if k.startswith('device_'):
                total_prop += params[k]

        # Cluster size parameter must be discrete
        params['cluster_size'] = round(params['cluster_size'])

        if not np.isclose(total_prop, 1.0, atol=0.1):
            return {
                'penalty': 1e6,
                'proactive_penalty': 1e6,
                'params': params,
            }

        # Run reactive simulation
        reactive_result = run_reactive_simulation(state, params)
        app_definitions = {}
        for task in reactive_result['stats']['taskResults']:
            app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())

        # Prepare data for all tasks
        X_new, y_new = create_inputs_outputs_seperated_per_app_windowed(reactive_result['stats'],
                                                                        PREPARE_PREDICTION_WINDOW_SIZE,
                                                                        app_definitions)

        # Create new models and train them with all data (initial + all improvements so far)
        current_models = {}
        for task in datasets.keys():
            if task in X_new and len(X_new[task]) > 0:
                # Get current dataset (which includes all previous improvements)
                X_initial, y_initial = datasets[task]
                print(X_initial.shape)
                print(X_new[task].shape)
                # Create a new model and train from scratch with all data
                model = self.create_model()
                X_formatted = np.concatenate([X_initial, X_new[task].reshape(-1, 1)], axis=0)
                y_formatted = np.concatenate([y_initial, y_new[task]], axis=0)

                model.fit(X_formatted, y_formatted)

                current_models[task] = model
            else:
                # If no new data for this task, still train with existing data
                model = self.create_model()
                X_current, y_current = datasets[task]
                model.fit(X_current, y_current)
                current_models[task] = model

        # Run proactive simulation with newly trained models
        proactive_result = run_proactive_simulation(state, current_models, params)

        # Get penalty from proactive simulation
        proactive_penalty = proactive_result['stats']['penaltyProportion']

        # Return all the data needed to track improvements
        return {
            'penalty': -proactive_penalty,
            'proactive_penalty': proactive_penalty,
            'saturated': proactive_result['saturated'],
            'params': params,
            'X_new': X_new,
            'y_new': y_new
        }

    def save_optimization_results(self, output_dir: Path):
        """Save models and their improvement datasets."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for task, (X_data, y_data) in self.initial_datasets.items():
            task_dir = output_dir / task
            task_dir.mkdir(exist_ok=True)

            # Train final model with all data
            final_model = self.create_model()
            X_formatted = [[x] for x in X_data]
            final_model.fit(X_formatted, y_data)

            # Save model
            final_model.save_model(task_dir / "model.json")

            # Save datasets
            np.save(task_dir / "X_improvements.npy", X_data)
            np.save(task_dir / "y_improvements.npy", y_data)

            # Save metadata
            metadata = {
                "n_improvements": len(self.improvement_history[task]),
                "final_penalty": self.best_proactive_penalty,
                "improvement_steps": [
                    {
                        "params": step.params,
                        "penalty": step.penalty
                    }
                    for step in self.improvement_history[task]
                ]
            }

            with open(task_dir / "optimization_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)


def load_optimization_results(input_dir: Path) -> Dict[str, Any]:
    """Load models and their improvement datasets."""
    results = {}

    for task_dir in input_dir.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name
            results[task_name] = {
                "model": xgb.XGBRegressor(),
                "X_improvements": np.load(task_dir / "X_improvements.npy"),
                "y_improvements": np.load(task_dir / "y_improvements.npy")
            }
            results[task_name]["model"].load_model(task_dir / "model.json")

            with open(task_dir / "optimization_metadata.json", "r") as f:
                results[task_name]["metadata"] = json.load(f)

    return results


def read_and_finetune_opt_results(opt_path: pathlib.Path):
    with open(opt_path / 'optimization_summary.json', 'r') as fd:
        optimization_summary = json.load(fd)
        n_samples = len(optimization_summary['optimization_results'])
        X_improvements_by_task = defaultdict(list)
        y_improvements_by_task = defaultdict(list)
        for i in range(n_samples):
            input_dir = opt_path / str(i) / "models"
            for task_dir in input_dir.iterdir():
                if task_dir.is_dir():
                    task_name = task_dir.name
                    X_improvements_by_task[task_name].extend(np.load(task_dir / "X_improvements.npy"))
                    y_improvements_by_task[task_name].extend(np.load(task_dir / "y_improvements.npy").reshape(-1, 1))
    return X_improvements_by_task, y_improvements_by_task


def finetune_initial_models(models_path: pathlib.Path, opt_path: pathlib.Path):
    X_improvements_by_task, y_improvements_by_task = read_and_finetune_opt_results(opt_path)
    fine_tuned_models = {}

    for model_file in models_path.glob("*_model.json"):
        task_name = model_file.stem.replace("_model", "")
        model = xgb.XGBRegressor()
        model.load_model(str(model_file))
        X_new = X_improvements_by_task[task_name]
        y_new = y_improvements_by_task[task_name]
        model.fit(
            X_new, y_new,
            xgb_model=model
        )
        fine_tuned_models[task_name] = model
    return fine_tuned_models


def main():
    models_path = Path("simulation_data/initial_results_simple")
    opt_path = Path("simulation_data/optimization_simple_results/20250309_203222")
    fine_tuned_models = finetune_initial_models(models_path=models_path, opt_path=opt_path)
    path_fine_tuned_models = opt_path / "fine_tuned_models"
    os.makedirs(path_fine_tuned_models, exist_ok=True)
    save_models(fine_tuned_models, path_fine_tuned_models)


if __name__ == '__main__':
    main()
