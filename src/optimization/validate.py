import json
import multiprocessing as mp
import pathlib
import sys
import time

from src.motivational.util import proactive_worker_function, get_model_locations_direct


def main():
    out_dir = pathlib.Path(sys.argv[1])
    exp_id = sys.argv[2]
    opt_path = out_dir / "optimization_results" / exp_id
    path_fine_tuned_models = opt_path / "fine_tuned_models"
    model_locations = get_model_locations_direct(path_fine_tuned_models)
    assert (len(model_locations) > 0)
    sim_input_path = pathlib.Path("data/nofs-ids")  # Base path for simulation input files
    base_dir = pathlib.Path("data/nofs-ids")  # Base path for simulation input files

    workload_config_file = sys.argv[3]
    num_cores = int(sys.argv[4])

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    result_dir = out_dir / "validation_results"
    repetitions = 1
    infra = '1'
    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, config, config_idx, base_dir, infra, sim_input_path, model_locations, result_dir, exp_id,
         "fine_tuned_models",
         infra, 'prokn_prokn')
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate(workload_configs)
    ]

    start_ts = time.time()
    # Create a pool of workers and map the work items
    print(f"Start time: {start_ts}")
    with mp.Pool(num_cores) as proactive_pool:
        proactive_pool.map(proactive_worker_function, work_items)
    end_ts = time.time()
    print(f'{end_ts - start_ts} seconds passed')
    print(f"Finished simulation")

    end_ts = time.time()
    print(f'Duration: {end_ts - start_ts} seconds')


if __name__ == '__main__':
    main()
