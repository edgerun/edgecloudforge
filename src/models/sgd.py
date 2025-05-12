import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import joblib

from src.models.abdullah import load_baas_model

# Load data

# Function to find minimum containers for each request level
def find_min_containers(df):
    # Group by request count and find minimum containers for each
    min_containers = df.groupby('requests')['containers'].min().reset_index()
    return min_containers



# Create prediction function with safety factor
def predict_min_containers(request_count, model, safety_factor=1.1):
    prediction = model.predict([[request_count]])[0] * safety_factor
    return max(0, np.ceil(prediction))




def main():
    X_path = '/home/philipp/Documents/code/python/my-herosim/spaces/globus_endpoint_02-endpoint/optimization_results/pipeline-globus_endpoint_02-endpoint-20250506222123/0/models/dnn1/X_improvements.npy'
    y_path = '/home/philipp/Documents/code/python/my-herosim/spaces/globus_endpoint_02-endpoint/optimization_results/pipeline-globus_endpoint_02-endpoint-20250506222123/0/models/dnn1/y_improvements.npy'

    X = np.load(X_path).reshape(-1, 1)
    y = np.load(y_path).reshape(-1)
    df = pd.DataFrame({'requests': X.flatten(), 'containers': y})


    # Get minimum container data points
    min_df = find_min_containers(df)

    # Create SGD regressor with polynomial features
    model = make_pipeline(
        PolynomialFeatures(degree=2),
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

    # Fit model on the minimum container data points
    model.fit(min_df['requests'].values.reshape(-1, 1), min_df['containers'])

    # Visualize results
    request_range = np.linspace(df['requests'].min(), df['requests'].max(), 100)
    predicted_containers = [predict_min_containers(r, model) for r in request_range]

    plt.figure(figsize=(12, 7))
    plt.scatter(df['requests'], df['containers'], alpha=0.3, label='Original Data')
    plt.scatter(min_df['requests'], min_df['containers'], color='green', s=50, label='Minimum Containers per Request')
    plt.plot(request_range, predicted_containers, 'r-', linewidth=2, label='Predicted Minimum with Safety Buffer')
    plt.title('Request Throughput vs. Minimum Containers Required')
    plt.xlabel('Number of Requests')
    plt.ylabel('Number of Containers')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print some example predictions
    print("Predictions for different request levels:")
    test_requests = [50, 100, 200, 300, 500]
    for req in test_requests:
        print(f"Requests: {req}, Predicted minimum containers: {predict_min_containers(req, model)}")

    # Optional: Save the model
    from joblib import dump

    dump(model, 'min_container_model.joblib')

if __name__ == '__main__':
    main()
# Function to apply the model for capacity planning
def plan_capacity(request_forecast, model, safety_factor=1.1):
    """
    Plan container capacity based on request forecast

    Parameters:
    - request_forecast: List or array of forecasted request counts
    - model: Trained regression model
    - safety_factor: Buffer to ensure sufficient capacity

    Returns:
    - List of recommended container counts
    """
    return [predict_min_containers(r, model, safety_factor) for r in request_forecast]
