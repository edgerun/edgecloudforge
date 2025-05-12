import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from datetime import datetime
from pathlib import Path

from src.executeinitial import setup_logging
from src.motivational.util import proactive_worker_function, get_model_locations_direct


def main():
    output_dir = sys.argv[1]
    infra = sys.argv[2]
    workload_config_file = sys.argv[3]
    repetitions = int(sys.argv[4])
    num_cores = int(sys.argv[5])
    region = sys.argv[6]
    fn = sys.argv[7]
    model_folder = sys.argv[8]
    os.makedirs(output_dir, exist_ok=True)

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    # Limit number of cores to available cores
    num_cores = min(num_cores, mp.cpu_count())

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")

    model_locations = get_model_locations_direct(pathlib.Path(output_dir) / model_folder)
    print(f"Model locations: {model_locations}")
    print(
        f"Starting proactive simulation with infra-{infra} workload_config: {workload_config_file} using {num_cores} cores")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    scheduling_strategy = 'prokn_prokn'
    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, config, config_idx, base_dir, infra, sim_input_path, model_locations, f'{output_dir}/{model_folder}',
         start_time,
         f'{region}-{fn}', infra, scheduling_strategy)
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate([workload_configs[-1]])
    ]

    start_ts = time.time()
    # Create a pool of workers and map the work items
    print(f"Start time: {start_ts}")
    with mp.Pool(num_cores) as proactive_pool:
        proactive_pool.map(proactive_worker_function, work_items)
    end_ts = time.time()
    print(f'{end_ts - start_ts} seconds passed')
    print(f"Finished simulation")
    print(f"Results saved under {os.path.join(output_dir, f'infra-{infra}', 'results-proactive', model_folder)}")

    end_ts = time.time()
    print(f'Duration: {end_ts - start_ts} seconds')


if __name__ == '__main__':
    main()
