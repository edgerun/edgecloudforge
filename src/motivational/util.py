import json
import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.executeinitial import load_simulation_inputs
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH, REACTIVE_RECONCILE_INTERVAL, \
    PROACTIVE_RECONCILE_INTERVAL
from src.motivationalhetero.encoders import PLATFORM_TYPES
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.preprocessing import calculate_metrics_combined


def save_results(results_folder, stats):
    scenario_statistics = {
        "averageQueueTime": stats["averageQueueTime"],
        "penaltyProportion": stats["penaltyProportion"],
        "averageExecutionTime": stats["averageExecutionTime"],
        "averageComputerTime": stats["averageComputeTime"],
        "averageWaitTime": stats["averageWaitTime"],
        "endTime": stats["endTime"],
        "applicationResponseTimeDistribution": stats["applicationResponseTimeDistribution"],
        "energy": stats["energy"]
    }

    with open(os.path.join(results_folder, "results.json"), "w") as results_file:
        json.dump(scenario_statistics, results_file)

    list1, list2 = zip(*stats['penaltyDistributionOverTime'])
    penalty_over_time = pd.DataFrame({'time': list1, 'penalty': list2})
    system_events_df = pd.DataFrame(stats['systemEvents'])
    system_events_df.to_csv(os.path.join(results_folder, "system_events.csv"))
    penalty_over_time.to_csv(os.path.join(results_folder, "penalty_over_time.csv"))

    sns.scatterplot(x='time', y='penalty', data=penalty_over_time)
    plt.savefig(os.path.join(results_folder, "penalties.pdf"))
    plt.close()
    sns.scatterplot(x='timestamp', y='count', hue='name', data=system_events_df)
    plt.savefig(os.path.join(results_folder, "system_events.pdf"))
    plt.close()
    melted_df = system_events_df.melt(id_vars=['timestamp'],
                                      value_vars=[x for x in PLATFORM_TYPES if x in system_events_df.columns],
                                      var_name='platform', value_name='pods_count')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted_df, x='timestamp', y='pods_count', hue='platform', marker='o')
    plt.title('Number of Pods per Platform Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Pods')
    plt.legend(title='Platform')
    plt.savefig(os.path.join(results_folder, "system_events_by_platform.pdf"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=system_events_df, x='timestamp', y='count', marker='o')
    plt.title('Number of Pods Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Pods')
    plt.savefig(os.path.join(results_folder, "system_events.pdf"))
    plt.close()

    app_definitions = {}
    for task in stats['taskResults']:
        app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())

    metrics = calculate_metrics_combined(stats['applicationResults'], stats['systemEvents'], window_size=5,
                                         application_to_task_map=app_definitions)
    with open(os.path.join(results_folder, "metrics.json"), "w") as outfile:
        json.dump(metrics, outfile, indent=2, cls=DataclassJSONEncoder)

    with open(os.path.join(results_folder, "app_definitions.json"), "w") as outfile:
        json.dump(app_definitions, outfile, indent=2, cls=DataclassJSONEncoder)

    rows = []
    for function, windows in metrics.items():
        for window, stats in windows.items():
            row = {
                'function': function,
                'window_start': stats.get('window_start'),
                'window_end': stats.get('window_end'),
                'avg_pods': stats.get('avg_pods'),
                'avg_queue_length': stats.get('avg_queue_length'),
                'avg_throughput': stats.get('avg_throughput'),
                'penalty_rate': stats.get('penalty_rate'),
                'total_requests': stats.get('total_requests')
            }
            rows.append(row)

    # Create DataFrame
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(results_folder, "performance_metrics_over_time.csv"))

    sns.lineplot(x='window_start', y='avg_throughput', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "throughput_by_platform.pdf"))
    plt.close()
    sns.lineplot(x='window_start', y='penalty_rate', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "penalty_rate_by_platform.pdf"))
    plt.close()
    sns.lineplot(x='window_start', y='total_requests', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "total_requests_by_platform.pdf"))
    plt.close()
    sns.lineplot(x='window_start', y='avg_queue_length', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "queue_length_by_platform.pdf"))
    plt.close()


def save_stats(output_dir, rps, stats, infra, results_postfix):
    results_folder = os.path.join(output_dir, f"infra-{infra}", f"results-{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    save_results(results_folder, stats)
    with open(os.path.join(results_folder, f"peak-config.json"), "w") as outfile:
        json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)
    return results_folder


def get_model_locations_direct(model_dir: pathlib.Path):
    model_locations = {}
    for file in model_dir.iterdir():
        if not '_model' in pathlib.Path(file).stem:
            continue
        fn = pathlib.Path(file).stem.replace('_model', '')
        model_locations[fn] = file
    return model_locations


def get_model_locations(input_dir, infra, model_dir):
    dir = pathlib.Path(str(os.path.join(input_dir, f"infra-{infra}", 'models', model_dir)))
    model_locations = {}
    for file in dir.iterdir():
        fn = pathlib.Path(file).stem.replace('_model', '')
        model_locations[fn] = file
    return model_locations


def save_single_stats(results_dir, rps, stats, output_infra, results_postfix, save_raw_results: bool = True):
    results_folder = os.path.join(results_dir, f"infra-{output_infra}", f"results-{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    save_results(results_folder, stats)
    if save_raw_results:
        with open(os.path.join(results_folder, f"peak-config.json"), "w") as outfile:
            json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)
    return results_folder


def execute_reactive(base_dir, infra, workload_config, sim_input_path):
    sim_inputs = load_simulation_inputs(sim_input_path)
    with open(base_dir / f'motivational-infrastructures/{infra}.json', 'r') as fd:
        infrastructure = json.load(fd)

    with open(workload_config, 'r') as fd:
        workload = json.load(fd)
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'kn_kn'
        simulation_data = SimulationData(
            platform_types=sim_inputs['platform_types'],
            storage_types=sim_inputs['storage_types'],
            qos_types=sim_inputs['qos_types'],
            application_types=sim_inputs['application_types'],
            task_types=sim_inputs['task_types'],
        )
        print(f'Start simulation: peak-config - {infra}')
        stats = execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, task_priority,
                            queue_length,
                            scheduling_strategy, workload, 'workload-mine',
                            reconcile_interval=REACTIVE_RECONCILE_INTERVAL)
        print(f'End simulation: peak-config - {infra}')
        return stats


def reactive_worker_function(args):
    rep_idx, workload_config, config_idx, base_dir, infra, sim_input_path, output_dir, start_time = args
    stats = execute_reactive(base_dir, infra, workload_config, sim_input_path)
    return save_stats(output_dir, workload_config, stats, infra, f'reactive/{start_time}/{str(rep_idx)}/{str(config_idx)}')


def execute_proactive(base_dir, infra, workload_config, sim_input_path, model_locations, scheduling_strategy):
    sim_inputs = load_simulation_inputs(sim_input_path)
    with open(base_dir / f'motivational-infrastructures/{infra}.json', 'r') as fd:
        infrastructure = json.load(fd)
    with open(workload_config, 'r') as fd:
        workload = json.load(fd)
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        simulation_data = SimulationData(
            platform_types=sim_inputs['platform_types'],
            storage_types=sim_inputs['storage_types'],
            qos_types=sim_inputs['qos_types'],
            application_types=sim_inputs['application_types'],
            task_types=sim_inputs['task_types'],
        )
        print(f'Start simulation: peak_config - {infra}, {model_locations}')
        stats = execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, task_priority,
                            queue_length,
                            scheduling_strategy, workload, 'workload-mine', model_locations=model_locations,
                            reconcile_interval=PROACTIVE_RECONCILE_INTERVAL)
        print(f'End simulation: peak_config - {infra}')
        return stats


def proactive_worker_function(args):
    rep_idx, workload_config, config_idx, base_dir, output_infra, sim_input_path, model_locations, results_dir, start_time, model_dir, model_infra, scheduling_strategy = args
    stats = execute_proactive(base_dir, output_infra, workload_config, sim_input_path, model_locations,
                              scheduling_strategy=scheduling_strategy)
    results_postfix = f'proactive/{model_dir}-origin-{model_infra}-target-{output_infra}/{start_time}/{str(rep_idx)}/{config_idx}'
    save_single_stats(results_dir, workload_config, stats, output_infra, results_postfix, save_raw_results=False)
