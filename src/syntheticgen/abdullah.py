from collections import defaultdict

import numpy as np
import xgboost as xgb
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.syntheticgen.stepsim import load_model_params


def exponential_model(x, a, b, c):
    """Model the response time using an exponential curve."""
    return a * np.exp(-b * x) + c


def model_response_time(requests, response_times):
    """Fit the response time data to an exponential curve."""
    # Initial parameter guesses
    p0 = [10, 0.0001, -50]

    # Fit the curve
    params, _ = curve_fit(exponential_model, requests, response_times, p0=p0)
    a, b, c = params

    print(f"Response time model: r_t = {a:.5f} * exp(-{b:.7f} * n_t) + {c:.5f}")
    return params


def simulate_reactive_autoscaling(workload, response_time_params, slo_threshold=200, duration=198):
    """Simulate reactive autoscaling to collect training data."""
    a, b, c = response_time_params

    # Initialize data collection
    intervals = []
    requests = []
    response_times = []
    vm_counts = []
    all_slo_thresholds = []
    # Start with 1 VM
    current_vms = 1

    # Run simulation for specified duration
    for t in range(duration):
        # Get current workload
        current_requests = workload[t] if t < len(workload) else workload[-1]

        # Calculate response time based on current VMs and workload
        # Assuming response time decreases with more VMs (simplified model)
        rt_per_vm = exponential_model(current_requests / current_vms, a, b, c)

        # Apply reactive scaling policy
        if rt_per_vm > slo_threshold:
            # Scale out - add one VM
            current_vms += 1
        elif t >= 2 and all(rt < slo_threshold / 2 for rt in response_times[-3:]):
            # Scale in - remove one VM if possible
            current_vms = max(1, current_vms - 1)

        # Record data
        intervals.append(t)
        requests.append(current_requests)
        response_times.append(rt_per_vm)
        vm_counts.append(current_vms)
        all_slo_thresholds.append(slo_threshold)

    return {
        'intervals': intervals,
        'requests': requests,
        'response_times': response_times,
        'vm_counts': vm_counts,
        'slo_thresholds': all_slo_thresholds
    }


def forecast_workload(historical_workload, window_size=20, method='xgb'):
    """Forecast workload for the next interval based on historical data."""
    if len(historical_workload) < window_size:
        # Not enough data, return the last known workload
        return historical_workload[-1] if historical_workload else 0

    # Prepare training data
    X = []
    y = []

    for i in range(len(historical_workload) - window_size):
        X.append(historical_workload[i:i + window_size])
        y.append(historical_workload[i + window_size])

    # Select forecasting method based on workload type
    if method == 'lr':
        model = LinearRegression()
    elif method == 'en':
        model = ElasticNet(alpha=0.1, l1_ratio=0.7)
    elif method == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    else:
        raise ValueError(f"Unknown forecasting method: {method}")

    # Train model
    model.fit(X, y)

    # Forecast next workload
    next_workload = model.predict([historical_workload[-window_size:]])[0]

    return max(0, next_workload)  # Ensure non-negative workload


def predictive_autoscaling(workload, resource_model, response_time_params,
                           slo_threshold=200, window_size=20, forecast_method='xgb'):
    """Implement predictive autoscaling based on workload forecasting and resource model."""
    a, b, c = response_time_params

    # Initialize data collection
    intervals = []
    requests = []
    response_times = []
    vm_counts = []
    slo_violations = []

    # Start with 1 VM
    current_vms = 1

    # Run simulation
    for t in range(len(workload)):
        # Current actual workload
        current_requests = workload[t]

        # Forecast workload for next interval
        if t >= window_size:
            forecasted_workload = forecast_workload(
                workload[max(0, t - window_size):t],
                window_size,
                forecast_method
            )
        else:
            forecasted_workload = current_requests

        # Predict required VMs using the resource model
        predicted_vms = resource_model.predict([[forecasted_workload, slo_threshold]])[0]
        predicted_vms = max(1, round(predicted_vms))  # Ensure at least 1 VM

        # Update VM count based on prediction
        current_vms = predicted_vms

        # Calculate actual response time with current VMs
        rt_per_vm = exponential_model(current_requests / current_vms, a, b, c)

        # Record data
        intervals.append(t)
        requests.append(current_requests)
        response_times.append(rt_per_vm)
        vm_counts.append(current_vms)
        slo_violations.append(1 if rt_per_vm > slo_threshold else 0)

    return {
        'intervals': intervals,
        'requests': requests,
        'response_times': response_times,
        'vm_counts': vm_counts,
        'slo_violations': slo_violations,
        'total_violations': sum(slo_violations),
        'processed_requests': sum(requests),
        'total_vms': sum(vm_counts)  # Proxy for cost
    }


def simulate_reactive_autoscaling_multi(workloads, response_time_params, slo_thresholds, duration=198):
    """Simulate reactive autoscaling with multiple workloads and SLO thresholds."""
    a, b, c = response_time_params

    # Initialize combined data collection
    all_intervals = []
    all_requests = []
    all_response_times = []
    all_vm_counts = []
    all_slo_thresholds = []
    all_workload_types = []

    # Process dictionary of workloads
    if isinstance(workloads, dict):
        workload_items = list(workloads.items())
    else:
        workload_items = [("unknown", w) for w in workloads]

    # Run simulation for each workload and SLO threshold combination
    for workload_name, workload in workload_items:
        for slo_threshold in slo_thresholds:
            print(f"Simulating {workload_name} workload with SLO threshold {slo_threshold}ms")

            # Start with 1 VM
            current_vms = 1

            # Run simulation for specified duration
            for t in range(duration):
                # Get current workload
                current_requests = workload[t] if t < len(workload) else workload[-1]

                # Calculate response time
                rt_per_vm = exponential_model(current_requests / current_vms, a, b, c)

                # Apply reactive scaling policy
                if rt_per_vm > slo_threshold:
                    # Scale out - add one VM
                    current_vms += 1
                elif (t >= 2 and len(all_response_times) >= 3 and
                      all(rt < slo_threshold / 2 for rt in all_response_times[-3:])):
                    # Scale in - remove one VM if possible
                    current_vms = max(1, current_vms - 1)

                # Record data
                all_intervals.append(t)
                all_requests.append(current_requests)
                all_response_times.append(rt_per_vm)
                all_vm_counts.append(current_vms)
                all_slo_thresholds.append(slo_threshold)
                all_workload_types.append(workload_name)

    return {
        'intervals': all_intervals,
        'requests': all_requests,
        'response_times': all_response_times,
        'vm_counts': all_vm_counts,
        'slo_thresholds': all_slo_thresholds,
        'workload_types': all_workload_types
    }


def train_resource_provisioning_model(simulation_data, save_dir=None):
    """
    Train a model to predict required VMs based on workload and SLO.

    Parameters:
    simulation_data: Dict containing simulation results
    save_dir: Directory to save model and results (optional)

    Returns: Trained DecisionTreeRegressor model
    """
    # Prepare training data
    X = []
    y = []

    for i in range(len(simulation_data['intervals'])):
        # Only use data points that satisfy SLO
        if simulation_data['response_times'][i] <= simulation_data['slo_thresholds'][i]:
            # Features: requests and desired response time (SLO threshold)
            X.append([simulation_data['requests'][i], simulation_data['slo_thresholds'][i]])
            # Target: number of VMs
            y.append(simulation_data['vm_counts'][i])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    score = model.score(X_test, y_test)
    print(f"Model R² score: {score:.4f}")

    # Save model and results if directory is provided
    if save_dir:
        import os
        import joblib
        import pandas as pd

        os.makedirs(save_dir, exist_ok=True)

        # Save model
        joblib.dump(model, os.path.join(save_dir, 'resource_model.joblib'))

        # Save training data
        pd.DataFrame(X_train, columns=['requests', 'slo_threshold']).to_csv(
            os.path.join(save_dir, 'X_train.csv'), index=False)
        pd.DataFrame(y_train, columns=['vm_count']).to_csv(
            os.path.join(save_dir, 'y_train.csv'), index=False)

        # Save test results
        test_results = pd.DataFrame({
            'requests': X_test[:, 0],
            'slo_threshold': X_test[:, 1],
            'actual_vms': y_test,
            'predicted_vms': model.predict(X_test)
        })
        test_results.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)

    return model


# Linear increasing workload
def linear_workload(initial=5000, increment=2000, duration=198):
    return [initial + t * increment for t in range(duration)]


def generate_workloads():
    """Generate synthetic workloads for testing."""

    # Periodic workload
    def periodic_workload(base=20000, amplitude=15000, period=24, duration=198):
        return [base + amplitude * np.sin(2 * np.pi * t / period) for t in range(duration)]

    # Wikipedia-like workload (synthetic approximation)
    def wikipedia_workload(duration=198):
        base = 15000
        trend = np.linspace(0, 10000, duration)
        daily = 5000 * np.sin(2 * np.pi * np.arange(duration) / 24)
        weekly = 3000 * np.sin(2 * np.pi * np.arange(duration) / (24 * 7))
        noise = np.random.normal(0, 1000, duration)
        return [max(1000, int(base + t + d + w + n))
                for t, d, w, n in zip(trend, daily, weekly, noise)]

    return {
        'linear': linear_workload(),
        'periodic': periodic_workload(),
        'wikipedia': wikipedia_workload()
    }


def simulate_dnn2_rpicpu():
    execution_time_ms = 0.16842 * 1000  # rpicpu, dnn2, in ms
    slo_threshold = execution_time_ms * 15

    folder = './abdullasynth'
    device = 'rpicpu'
    fn = 'dnn2'
    params_path = f'{folder}/response_times_{device}_{fn}.json'
    max_rps = 200
    rt_params = load_model_params(params_path)

    # 3. Generate synthetic workloads
    linear_workloads = linear_workload(initial=1, increment=2, duration=max_rps)

    # 4. Run trace-driven simulation with reactive autoscaling
    simulation_data = simulate_reactive_autoscaling(
        linear_workloads,
        rt_params,
        slo_threshold=slo_threshold,
        duration=max_rps
    )

    return simulation_data


def simulate_dnn1_xaviergpu():
    folder = './abdullasynth'
    device = 'xaviergpu'
    fn = 'dnn1'
    params_path = f'{folder}/response_times_{device}_{fn}.json'
    max_rps = 200
    execution_time_ms = 0.020835 * 1000  # rpicpu, dnn2, in ms
    slo_threshold = execution_time_ms * 15
    rt_params = load_model_params(params_path)

    # 3. Generate synthetic workloads
    linear_workloads = linear_workload(initial=1, increment=2, duration=max_rps)

    # 4. Run trace-driven simulation with reactive autoscaling
    simulation_data = simulate_reactive_autoscaling(
        linear_workloads,
        rt_params,
        slo_threshold=slo_threshold,
        duration=max_rps
    )

    return simulation_data


def merge_simulation_data(items):
    data = defaultdict(list)

    for k in items[0].keys():
        for item in items:
            data[k].extend(item[k])
    return data


def main():
    dnn2_rpicpu_simulation_data = simulate_dnn2_rpicpu()
    dnn1_xaviergpu_simulation_data = simulate_dnn1_xaviergpu()

    simulation_data = merge_simulation_data([dnn2_rpicpu_simulation_data, dnn1_xaviergpu_simulation_data])

    # 5. Train resource provisioning model
    resource_model = train_resource_provisioning_model(simulation_data, save_dir='./abdullasynth')

    return

    # 6. Evaluate predictive autoscaling on different workloads
    results = {}
    for name, workload in workloads.items():
        # Select best forecasting method for each workload
        if name == 'wikipedia':
            forecast_method = 'xgb'
            window_size = 20
        elif name == 'periodic':
            forecast_method = 'xgb'
            window_size = 10
        else:  # linear
            forecast_method = 'lr'
            window_size = 40

        # Run predictive autoscaling
        result = predictive_autoscaling(
            workload,
            resource_model,
            rt_params,
            window_size=window_size,
            forecast_method=forecast_method
        )

        results[name] = result

        print(f"\nResults for {name} workload:")
        print(f"Total requests processed: {result['processed_requests']}")
        print(f"SLO violations: {result['total_violations']} ({result['total_violations'] / len(workload) * 100:.2f}%)")
        print(f"Total VM-minutes (cost): {result['total_vms']}")


if __name__ == "__main__":
    main()
