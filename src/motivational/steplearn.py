import json
import os.path
import sys
import time
from datetime import datetime
from pathlib import Path

from src.executeinitial import setup_logging
from src.motivational.util import proactive_worker_function


def main():
    import multiprocessing as mp
    # Set this at the beginning of your script
    mp.set_start_method('spawn')
    if len(sys.argv) != 8:
        print(
            "Usage: script.py <results_dir> <model_infra> <output_infra> <model_path> <workload_config_file> <repetitions> <num_cores>")
        sys.exit(1)

    results_dir = sys.argv[1]
    model_infra = sys.argv[2]
    output_infra = sys.argv[3]
    model_path = sys.argv[4]
    workload_config_file = sys.argv[5]
    repetitions = int(sys.argv[6])
    num_cores = int(sys.argv[7])

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    # Limit number of cores to available cores
    num_cores = min(num_cores, mp.cpu_count())

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")

    print("Fetching model locations")
    print(
        f"Starting proactive simulation with infra-{output_infra} workload_config: {workload_config_file} using {num_cores} cores")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, config, config_idx, base_dir, output_infra, sim_input_path, model_path, results_dir, start_time,
         model_path, model_infra, 'steplearn_steplearn')
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate(workload_configs)
    ]

    start_ts = time.time()
    # Create a pool of workers and map the work items
    print(f"Start time: {start_ts}")
    with mp.Pool(num_cores) as pool:
        pool.map(proactive_worker_function, work_items)
    end_ts = time.time()
    print(f'{end_ts - start_ts} seconds passed')
    print(f"Finished simulation")
    print(f"Results saved under {os.path.join(results_dir, f'infra-{output_infra}', 'results-proactive')}")


if __name__ == '__main__':
    main()
