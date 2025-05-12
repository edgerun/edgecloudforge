import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from datetime import datetime
from pathlib import Path

from src.executeinitial import setup_logging
from src.motivational.util import reactive_worker_function
from src.train import train_model_reactive_then_proactive, save_models


def main():

    output_dir = sys.argv[1]
    infra = sys.argv[2]
    workload_config_file = sys.argv[3]
    repetitions = int(sys.argv[4])
    num_cores = int(sys.argv[5])
    region = sys.argv[6]
    fn = sys.argv[7]
    os.makedirs(output_dir, exist_ok=True)

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    # Limit number of cores to available cores
    num_cores = min(num_cores, mp.cpu_count())

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")

    print(f'Loading infra {infra}, executing with {workload_config_file} using {num_cores} cores')
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Create all combinations of repetitions and RPS values
    work_items = [
        (rep_idx, config, config_idx, base_dir, infra, sim_input_path, output_dir, start_time)
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate(workload_configs)
    ]

    # Create a pool of workers and map the work items
    start_ts = time.time()
    reactive_pool = mp.Pool(num_cores)
    try:
        result_folders = reactive_pool.map(reactive_worker_function, work_items)
    finally:
        reactive_pool.close()
        reactive_pool.join()

    test_size = 0.01

    # 172 * 2 => 40 minutes simulation duration
    one_day_until = 172 * 2
    result_files = [f'{x}/peak-config.json' for x in result_folders[:1]]
    models, eval_results = train_model_reactive_then_proactive(result_files, include_queue_length=False, test_size=test_size, until=one_day_until)
    dir_first_second = pathlib.Path(output_dir) / 'one_day'
    os.makedirs(dir_first_second, exist_ok=True)
    model_paths = save_models(models, dir_first_second)

    # first week training
    result_files = [f'{x}/peak-config.json' for x in result_folders[:1]]
    models, eval_results = train_model_reactive_then_proactive(result_files, include_queue_length=False, test_size=test_size)
    dir_first_second = pathlib.Path(output_dir) / 'first'
    os.makedirs(dir_first_second, exist_ok=True)
    model_paths = save_models(models, dir_first_second)

    # first & second week training
    result_files = [f'{x}/peak-config.json' for x in result_folders[:2]]
    models, eval_results = train_model_reactive_then_proactive(result_files, include_queue_length=False, test_size=test_size)
    dir_first_second = pathlib.Path(output_dir) / 'first_second'
    os.makedirs(dir_first_second, exist_ok=True)
    model_paths = save_models(models, dir_first_second)

    # first & second & third week training
    result_files = [f'{x}/peak-config.json' for x in result_folders[:3]]
    models, eval_results = train_model_reactive_then_proactive(result_files, include_queue_length=False, test_size=test_size)
    dir_all = pathlib.Path(output_dir) / 'first_second_third'
    os.makedirs(dir_all, exist_ok=True)
    model_paths = save_models(models, dir_all)

    for result_folder in result_folders:
        print(f'Delete {result_folder}/peak-config.json')
        os.remove(f'{result_folder}/peak-config.json')


if __name__ == '__main__':
    main()
