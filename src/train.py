import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from src.models.abdullah import load_baas_model
# Assuming increase_events is imported from your previous script
from src.preprocessing import train_gpr_per_task, evaluate_model_per_task, \
    create_inputs_outputs_seperated_per_app_windowed, create_train_test_split_per_windowed, \
    create_inputs_outputs_seperated_per_app_windowed_system_events, \
    create_train_test_split_per_windowed_per_device_type, \
    create_inputs_outputs_seperated_per_app_windowed_per_device_type, evaluate_gpr_per_task, \
    create_inputs_outputs_based_on_metrics_seperated_per_app_windowed_per_device_type, \
    create_inputs_outputs_based_on_metrics_seperated_per_app_windowed_from_until, train_sgr_per_task

def prepare_data(output_dir, n, include_queue_length: bool, window_size=5):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for i in range(n):
        with open(output_dir / f"simulation_{i + 1}.json", 'r') as fd:
            obj = json.load(fd)['stats']
            app_definitions = {}
            for task in obj['taskResults']:
                app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
            if not include_queue_length:
                train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                    create_inputs_outputs_seperated_per_app_windowed(obj, window_size, app_definitions),test_size=0.01)
            else:
                train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                    create_inputs_outputs_seperated_per_app_windowed_system_events(obj, window_size, app_definitions))
            for fn, data in train_data_sample.items():
                all_train_data[fn].extend(data)
            for fn, data in test_data_sample.items():
                all_test_data[fn].extend(data)

    X_train_all = defaultdict(list)
    y_train_all = defaultdict(list)
    for task_type, data in all_train_data.items():
        # Extract features (timestamp, workload_count) and target (pod_count)
        X_train = [item[0] for item in data]  # Input
        y_train = [item[1] for item in data]  # Pod count

        # Convert lists to NumPy arrays
        X_train_all[task_type].extend(X_train)
        y_train_all[task_type].extend(y_train)

    data = {}

    for task_type in all_train_data.keys():
        data[task_type] = (np.array(X_train_all[task_type]), np.array(y_train_all[task_type]))

    return data

def train_model(output_dir, samples, include_queue_length: bool, window_size=5):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for i, sample in enumerate(samples):
        with open(output_dir / f"simulation_{i + 1}.json", 'r') as fd:
            obj = json.load(fd)['stats']
            app_definitions = {}
            for task in obj['taskResults']:
                app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
            if not include_queue_length:
                train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                    create_inputs_outputs_seperated_per_app_windowed(obj, window_size, app_definitions))
            else:
                train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                    create_inputs_outputs_seperated_per_app_windowed_system_events(obj, window_size, app_definitions))
            for fn, data in train_data_sample.items():
                all_train_data[fn].extend(data)
            for fn, data in test_data_sample.items():
                all_test_data[fn].extend(data)
    # for fn, data in all_train_data.items():
    #     all_train_data[fn] = np.array(data).reshape(-1, 2)
    # for fn, data in all_test_data.items():
    #     all_test_data[fn] = np.array(data).reshape(-1, 2)
    models = train_sgr_per_task(all_train_data)
    return models, evaluate_model_per_task(models, all_test_data)


def train_model_reactive_then_proactive(output_files, include_queue_length: bool, window_size=5, test_size=0.2,
                                        until=None):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for out_file in output_files:
        with open(out_file, 'r') as fd:
            obj = json.load(fd)
            app_definitions = {}
            for task in obj['taskResults']:
                app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
            if not include_queue_length:
                train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                    create_inputs_outputs_seperated_per_app_windowed(obj, window_size, app_definitions, until),
                    test_size)
            else:
                train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                    create_inputs_outputs_seperated_per_app_windowed_system_events(obj, window_size, app_definitions),
                    test_size)
            for fn, data in train_data_sample.items():
                all_train_data[fn].extend(data)
            for fn, data in test_data_sample.items():
                all_test_data[fn].extend(data)
    # for fn, data in all_train_data.items():
    #     all_train_data[fn] = np.array(data).reshape(-1, 2)
    # for fn, data in all_test_data.items():
    #     all_test_data[fn] = np.array(data).reshape(-1, 2)
    models = train_sgr_per_task(all_train_data)
    return models, evaluate_model_per_task(models, all_test_data)


def train_xgr_model_reactive_then_proactive_per_device_type(output_files, include_queue_length: bool, window_size=5,
                                                            test_size=0.2, until=None, encoder=None):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for out_file in output_files:
        with open(out_file, 'r') as fd:
            obj = json.load(fd)
            app_definitions = {}
            for task in obj['taskResults']:
                app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
            if not include_queue_length:
                train_data_sample, test_data_sample, encoder = create_train_test_split_per_windowed_per_device_type(
                    create_inputs_outputs_seperated_per_app_windowed_per_device_type(obj, window_size, app_definitions,
                                                                                     until), test_size, encoder=encoder)
            else:
                assert False
                train_data_sample, test_data_sample = create_train_test_split_per_windowed_per_device_type(
                    create_inputs_outputs_seperated_per_app_windowed_system_events(obj, window_size, app_definitions),
                    test_size)
            for fn, data in train_data_sample.items():
                all_train_data[fn].extend(data)
            for fn, data in test_data_sample.items():
                all_test_data[fn].extend(data)
    # for fn, data in all_train_data.items():
    #     all_train_data[fn] = np.array(data).reshape(-1, 2)
    # for fn, data in all_test_data.items():
    #     all_test_data[fn] = np.array(data).reshape(-1, 2)
    models = train_sgr_per_task(all_train_data)
    return models


def train_gpr_model_reactive_then_proactive_per_device_type(output_files, include_queue_length: bool, window_size=5,
                                                            test_size=0.2, until=None, encoder=None):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for out_file in output_files:
        with open(out_file, 'r') as fd:
            obj = json.load(fd)
            app_definitions = {}
            for task in obj['taskResults']:
                app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
            if not include_queue_length:
                train_data_sample, test_data_sample, encoder = create_train_test_split_per_windowed_per_device_type(
                    create_inputs_outputs_seperated_per_app_windowed_per_device_type(obj, window_size, app_definitions,
                                                                                     until), test_size, encoder=encoder)
                pass
            else:
                assert False
                train_data_sample, test_data_sample = create_train_test_split_per_windowed_per_device_type(
                    create_inputs_outputs_seperated_per_app_windowed_system_events(obj, window_size, app_definitions),
                    test_size)
            for fn, data in train_data_sample.items():
                all_train_data[fn].extend(data)
            for fn, data in test_data_sample.items():
                all_test_data[fn].extend(data)
    # for fn, data in all_train_data.items():
    #     all_train_data[fn] = np.array(data).reshape(-1, 2)
    # for fn, data in all_test_data.items():
    #     all_test_data[fn] = np.array(data).reshape(-1, 2)
    models, X_scalers, y_scalers = train_gpr_per_task(all_train_data)
    evaluate_gpr_per_task(models, all_test_data, X_scalers, y_scalers)
    return models


def train_gpr_model_reactive_post_experiment_then_proactive_per_device_type(output_folders, window_size=5,
                                                                            test_size=0.2, until=None, encoder=None):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for out_folder in output_folders:
        with open(f'{out_folder}/metrics.json', 'r') as fd:
            metrics = json.load(fd)
        with open(f'{out_folder}/app_definitions.json', 'r') as fd:
            app_definitions = json.load(fd)

        train_data_sample, test_data_sample, encoder = create_train_test_split_per_windowed_per_device_type(
            create_inputs_outputs_based_on_metrics_seperated_per_app_windowed_per_device_type(
                metrics, app_definitions), test_size, encoder=encoder)
        for fn, data in train_data_sample.items():
            all_train_data[fn].extend(data)
        for fn, data in test_data_sample.items():
            all_test_data[fn].extend(data)

    models, X_scalers, y_scalers = train_gpr_per_task(all_train_data)
    evaluate_gpr_per_task(models, all_test_data, X_scalers, y_scalers)
    return models

def train_xgr_model_reactive_post_experiment_then_proactive_per_device_type(output_folders, window_size=5,
                                                                            test_size=0.2, until=None, encoder=None):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for out_folder in output_folders:
        with open(f'{out_folder}/metrics.json', 'r') as fd:
            metrics = json.load(fd)
        with open(f'{out_folder}/app_definitions.json', 'r') as fd:
            app_definitions = json.load(fd)

            train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                create_inputs_outputs_based_on_metrics_seperated_per_app_windowed_from_until(metrics, app_definitions, until=until),
                test_size)

        for fn, data in train_data_sample.items():
            all_train_data[fn].extend(data)
        for fn, data in test_data_sample.items():
            all_test_data[fn].extend(data)

    models = train_sgr_per_task(all_train_data)
    print('here')
    evaluate_model_per_task(models, all_test_data)
    return models



def train_model_in_memory(output_dir, all_results, window_size=5):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    for obj in all_results:
        app_definitions = {}
        for task in obj['taskResults']:
            app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
        train_data_sample, test_data_sample = create_train_test_split_per_windowed(
            create_inputs_outputs_seperated_per_app_windowed(obj, window_size, app_definitions))
        for fn, data in train_data_sample.items():
            all_train_data[fn].append(data)
        for fn, data in test_data_sample.items():
            all_test_data[fn].append(data)
    for fn, data in all_train_data.items():
        all_train_data[fn] = np.array(data).reshape(-1, 2)
    for fn, data in all_test_data.items():
        all_test_data[fn] = np.array(data).reshape(-1, 2)
    models = train_sgr_per_task(all_train_data)
    return models, evaluate_model_per_task(models, all_test_data)


def load_models(model_locations: Dict[str, str]):
    models = {}
    if type(model_locations) is str:
        return load_baas_model(model_locations)
    try:
        for fn, model_location in model_locations.items():
            model_location = f'{model_location}'
            # loaded_model = xgb.XGBRegressor()
            # loaded_model.load_model(model_location)
            loaded_model = joblib.load(model_location)
            models[fn] = loaded_model
        return models
    except Exception as e:
        print(e)


def save_models(models, output_dir):
    model_paths = {}
    for fn, model in models.items():
        model_path = output_dir / f"{fn}_model.json"
        joblib.dump(model, model_path)
        model_paths[fn] = str(model_path)
    return model_paths


if __name__ == '__main__':
    base_dir = Path("simulation_data")
    sim_input_path = Path("data/ids")  # Base path for simulation input files
    samples_file = base_dir / "lhs_samples.npy"
    mapping_file = base_dir / "lhs_samples_mapping.pkl"
    config_file = base_dir / "infrastructure_config.json"
    workload_base_file = "data/ids/traces/workload-83-100.json"
    output_dir = base_dir / "results"
    samples = np.load(samples_file)

    print(train_model(output_dir, samples)[0])
