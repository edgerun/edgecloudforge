import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import deque


def exponential_model(x, a, b, c):
    """Model the response time using an exponential curve as described in the paper."""
    return a * np.exp(-b * x) + c


def simulate_workload(requests_per_second, duration_seconds, execution_time_ms):
    """
    Simulate a workload using Poisson process for arrivals.

    Parameters:
    requests_per_second: Average arrival rate (λ)
    duration_seconds: Duration of simulation
    execution_time_ms: Time to process one request
    """
    # Convert execution time to seconds
    execution_time = execution_time_ms / 1000.0

    # Initialize simulation variables
    queue = deque()
    response_times = []
    queue_lengths = []
    throughput_per_second = []

    # Generate arrival times using Poisson process
    # In a Poisson process, inter-arrival times follow exponential distribution
    arrival_times = []
    current_time = 0

    # Generate arrivals for the entire simulation duration
    while current_time < duration_seconds:
        # Generate next inter-arrival time (exponentially distributed)
        inter_arrival = np.random.exponential(1.0 / requests_per_second)
        current_time += inter_arrival

        if current_time < duration_seconds:
            arrival_times.append(current_time)

    # Time when the device will be free
    device_free_at = 0

    # Track metrics per second
    for second in range(duration_seconds):
        requests_completed = 0
        queue_lengths.append(len(queue))

        # Add new arrivals for this second to the queue
        while arrival_times and arrival_times[0] < second + 1:
            queue.append(arrival_times.pop(0))

        # Process requests until we move to the next second
        while queue and device_free_at < second + 1:
            # Get the next request from queue
            arrival_time = queue.popleft()

            # Device starts processing at max(arrival_time, device_free_at)
            start_time = max(arrival_time, device_free_at)

            # Calculate when the device will finish this request
            device_free_at = start_time + execution_time

            # If the device finishes within this second, count it as completed
            if device_free_at <= second + 1:
                requests_completed += 1

                # Calculate response time (from arrival to completion)
                response_time = (device_free_at - arrival_time) * 1000  # convert to ms
                response_times.append(response_time)

        # Record throughput for this second
        throughput_per_second.append(requests_completed)

    # Calculate metrics
    p95_response_time = np.percentile(response_times, 95) if response_times else 0

    return {
        'response_times': response_times,
        'p95_response_time': p95_response_time,
        'avg_queue_length': np.mean(queue_lengths),
        'avg_throughput': np.mean(throughput_per_second),
        'requests_completed': sum(throughput_per_second),
        'requests_remaining': len(queue)
    }


def run_benchmark(min_rps=5000, max_rps=50000, step_size=2000,
                  duration_per_step=60, execution_time_ms=10,
                  reps_per_rps=5, p95_threshold=1000):
    """
    Run a step-up benchmark to model response time behavior with multiple repetitions per RPS level.

    Parameters:
    min_rps: Starting requests per second (5000 in the paper)
    max_rps: Maximum requests per second to test
    step_size: Increment in requests per second between steps (2000 in the paper)
    duration_per_step: Duration in seconds for each load level (60 in the paper)
    execution_time_ms: Time needed to execute one request
    reps_per_rps: Number of repetitions for each RPS level
    p95_threshold: Threshold in ms for 95th percentile response time to stop benchmark (default 1000ms)

    Returns:
    DataFrame with benchmark results
    """
    results = []

    # Test different request rates in a step-up fashion
    for rps in range(min_rps, max_rps + 1, step_size):
        print(f"Benchmarking at {rps} requests per second...")

        # Run multiple repetitions for each RPS level
        for rep in range(reps_per_rps):
            print(f"  Repetition {rep + 1}/{reps_per_rps}")

            # Run the simulation for the specified duration
            sim_result = simulate_workload(rps, duration_per_step, execution_time_ms)

            # Record results
            results.append({
                'requests_per_second': rps,
                'repetition': rep + 1,
                'response_time_95p_ms': sim_result['p95_response_time'],
                'avg_queue_length': sim_result['avg_queue_length'],
                'avg_throughput': sim_result['avg_throughput'],
                'completion_rate': sim_result['requests_completed'] / (rps * duration_per_step)
            })

            # If response time is extremely high, we may have saturated the system
            if sim_result['p95_response_time'] > p95_threshold:
                print(f"Response time exceeded {p95_threshold}ms at {rps} RPS. Stopping benchmark.")
                return pd.DataFrame(results)

    return pd.DataFrame(results)


def model_response_time(benchmark_data, plot_path='response_time_model.png',
                        params_path='model_params.json', reps_per_rps=1):
    """Fit response time data to exponential curve, showing mean and std when multiple repetitions exist."""
    if len(benchmark_data) < 3:
        raise ValueError("Not enough data points for curve fitting. Need at least 3 points.")

    # If we have multiple repetitions, calculate mean and std for each RPS level
    if reps_per_rps > 1:
        # Group by RPS and calculate statistics
        grouped = benchmark_data.groupby('requests_per_second')
        avg_data = grouped['response_time_95p_ms'].mean().reset_index()
        std_data = grouped['response_time_95p_ms'].std().reset_index()

        requests = avg_data['requests_per_second'].values
        response_times = avg_data['response_time_95p_ms'].values
        std_values = std_data['response_time_95p_ms'].values
    else:
        requests = benchmark_data['requests_per_second'].values
        response_times = benchmark_data['response_time_95p_ms'].values
        std_values = None

    # Initial parameter guesses
    p0 = [8.0, 0.0003, -45.0]

    try:
        # Fit the curve
        params, _ = curve_fit(exponential_model, requests, response_times, p0=p0)
        a, b, c = params

        print(f"Response time model: r_t = {a:.5f} * exp(-{b:.7f} * n_t) + {c:.5f}")

        # Plot the results
        plt.figure(figsize=(10, 6))

        if reps_per_rps > 1:
            # Plot all individual data points with lower opacity
            plt.scatter(benchmark_data['requests_per_second'],
                        benchmark_data['response_time_95p_ms'],
                        alpha=0.3, label='Individual measurements')

            # Plot the means with error bars showing standard deviation
            plt.errorbar(requests, response_times, yerr=std_values,
                         fmt='o', color='blue', ecolor='blue',
                         elinewidth=2, capsize=5, label='Mean ± Std Dev')
        else:
            plt.scatter(requests, response_times, label='Observed data')

        # Generate points for the fitted curve
        x_fit = np.linspace(min(requests), max(requests), 100)
        y_fit = exponential_model(x_fit, a, b, c)
        plt.plot(x_fit, y_fit, 'r-', label='Fitted curve')

        plt.xlabel('Requests per second')
        plt.ylabel('Response time (ms)')
        plt.title('Response Time Model')
        plt.legend()
        plt.grid(True)

        # Create directory for plot if it doesn't exist
        os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else '.', exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Save parameters to JSON file with additional statistics
        params_dict = {
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'model_type': 'exponential',
            'equation': 'r_t = a * exp(-b * n_t) + c',
            'reps_per_rps': reps_per_rps,
            'timestamp': str(pd.Timestamp.now())
        }

        # Add standard deviation data if available
        if reps_per_rps > 1:
            std_dict = {str(int(rps)): float(std) for rps, std in zip(requests, std_values)}
            params_dict['std_by_rps'] = std_dict

        with open(params_path, 'w') as f:
            json.dump(params_dict, f, indent=4)

        return params
    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        return (8.24018, 0.0003632, -47.27967)  # Default parameters from paper


def load_model_params(params_path='model_params.json'):
    """
    Load model parameters from a JSON file.

    Parameters:
    -----------
    params_path : str
        Path to the JSON file containing model parameters

    Returns:
    --------
    params : tuple
        Loaded parameters (a, b, c) for the exponential model
    """
    with open(params_path, 'r') as f:
        params_dict = json.load(f)

    a = params_dict['a']
    b = params_dict['b']
    c = params_dict['c']

    print(f"Loaded response time model: r_t = {a:.5f} * exp(-{b:.7f} * n_t) + {c:.5f}")
    return (a, b, c)


def default_main():
    # Run the benchmark with parameters matching the paper
    benchmark_results = run_benchmark(
        min_rps=10,  # Start at 5000 as mentioned in the paper
        max_rps=5000,
        step_size=5,  # Increase by 2000 as in the paper
        duration_per_step=60,  # 1 minute per step as in the paper
        execution_time_ms=10  # Example execution time per request
    )
    # Save raw results
    benchmark_results.to_csv('benchmark_results.csv', index=False)
    # Model the response time
    try:
        rt_params = model_response_time(benchmark_results)
        print("Benchmark complete. Response time model parameters:", rt_params)
    except ValueError as e:
        print(f"Error: {e}")


def dnn2_rpiCpu():
    execution_time_ms = 0.16842 * 1000  # rpicpu, dnn2, in ms
    folder = './abdullasynth'
    device = 'rpicpu'
    fn = 'dnn2'
    # medium
    slo_threshold = execution_time_ms * 15
    # Run the benchmark with parameters matching the paper
    benchmark_results = run_benchmark(
        min_rps=1,  # Start at 5000 as mentioned in the paper
        max_rps=5000,
        step_size=1,  # Increase by 2000 as in the paper
        duration_per_step=60,  # 1 minute per step as in the paper
        execution_time_ms=execution_time_ms,  # Example execution time per request
        p95_threshold=slo_threshold,  # in ms,
        reps_per_rps=5
    )

    # Save raw results
    benchmark_results.to_csv(f'{folder}/benchmark_results_rpicpu_dnn2.csv', index=False)

    # Model the response time
    try:

        plot_path = f'{folder}/response_times_{device}_{fn}.png'
        params_path = f'{folder}/response_times_{device}_{fn}.json'
        rt_params = model_response_time(benchmark_results, plot_path, params_path, reps_per_rps=5)
        print("Benchmark complete. Response time model parameters:", rt_params)
    except ValueError as e:
        print(f"Error: {e}")

def dnn1_xavierGpu():
    execution_time_ms = 0.020835 * 1000  # rpicpu, dnn2, in ms
    folder = './abdullasynth'
    device = 'xaviergpu'
    fn = 'dnn1'
    # medium
    slo_threshold = execution_time_ms * 15
    # Run the benchmark with parameters matching the paper
    benchmark_results = run_benchmark(
        min_rps=1,  # Start at 5000 as mentioned in the paper
        max_rps=5000,
        step_size=1,  # Increase by 2000 as in the paper
        duration_per_step=60,  # 1 minute per step as in the paper
        execution_time_ms=execution_time_ms,  # Example execution time per request
        p95_threshold=slo_threshold,  # in ms,
        reps_per_rps=5
    )

    # Save raw results
    benchmark_results.to_csv(f'{folder}/benchmark_results_rpicpu_dnn2.csv', index=False)

    # Model the response time
    try:

        plot_path = f'{folder}/response_times_{device}_{fn}.png'
        params_path = f'{folder}/response_times_{device}_{fn}.json'
        rt_params = model_response_time(benchmark_results, plot_path, params_path, reps_per_rps=5)
        print("Benchmark complete. Response time model parameters:", rt_params)
    except ValueError as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    dnn2_rpiCpu()
    dnn1_xavierGpu()
