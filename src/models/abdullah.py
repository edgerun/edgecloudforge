import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

def prepare_data(df, response_time_slo=200):
    """
    Prepare data for training/testing the resource prediction model.

    Args:
        df: DataFrame with workload data
        response_time_slo: SLO threshold for response time in ms

    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
    """
    # Extract features and convert to numpy arrays
    X = np.column_stack((
        df['avg_throughput'].values,
        np.ones(len(df)) * response_time_slo  # Add response_time_slo as a feature
    ))

    # Use the existing avg_pods as the target variable
    y = df['avg_pods'].values

    return X, y

def train_resource_prediction_model(df, response_time_slo=200, test_size=0.2, random_state=42):
    """
    Train the resource prediction model using Decision Tree Regression.

    Args:
        df: DataFrame with workload data
        response_time_slo: SLO threshold for response time in ms
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        model: Trained model
        X_train, X_test, y_train, y_test: Split data for evaluation
    """
    # Prepare data
    X, y = prepare_data(df, response_time_slo)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train Decision Tree Regression model
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained model
        X_test: Test features (numpy array)
        y_test: Test targets (numpy array)

    Returns:
        metrics: Dictionary of evaluation metrics
        y_pred: Predicted values
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate percentage of predictions that would lead to under-provisioning
    under_provision = np.sum(y_pred < y_test) / len(y_test) * 100

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': np.sqrt(mse),
        'Under-provisioning %': under_provision
    }

    return metrics, y_pred

def save_model(model, filename='resource_prediction_model.joblib'):
    """
    Save the trained model to disk.

    Args:
        model: Trained model
        filename: Path to save the model
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_baas_model(filename='resource_prediction_model.joblib'):
    """
    Load a trained model from disk.

    Args:
        filename: Path to the saved model

    Returns:
        model: Loaded model
    """
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def predict_required_resources(model, throughput, response_time_slo):
    """
    Predict the required number of containers for a given throughput.

    Args:
        model: Trained resource prediction model
        throughput: Average throughput
        response_time_slo: Response time SLO

    Returns:
        n: Predicted number of containers required
    """
    # Create input array with features
    input_data = np.array([[throughput, response_time_slo]])

    # Make prediction
    n = model.predict(input_data)[0]

    # Round up to nearest integer
    n = np.ceil(n)

    return int(n)

def plot_predictions(y_test, y_pred):
    """
    Plot actual vs predicted container counts.

    Args:
        y_test: Actual container counts (numpy array)
        y_pred: Predicted container counts (numpy array)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Container Count')
    plt.ylabel('Predicted Container Count')
    plt.title('Resource Prediction Model: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample data (using the provided example)
    import pandas as pd

    data = {
        'function': ['dnn1', 'dnn1', 'dnn1'],
        'window_start': [125, 130, 135],
        'window_end': [130, 135, 140],
        'avg_pods': [45.0, 45.0, 45.0],
        'avg_queue_length': [3.088888889, 0.111111111, 0.066666667],
        'avg_throughput': [301.6, 188.4, 222.8],
        'penalty_rate': [0.085543767, 0.0, 0.0],
        'total_requests': [1508, 942, 1114]
    }

    df = pd.DataFrame(data)

    # Train the model
    model, X_train, X_test, y_train, y_test = train_resource_prediction_model(df, response_time_slo=200)

    # Evaluate the model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Save the model
    save_model(model)

    # Load the model
    loaded_model = load_baas_model()

    # Make a prediction for a new workload
    new_throughput = 240
    response_time_slo = 200
    required_containers = predict_required_resources(loaded_model, new_throughput, response_time_slo)
    print(f"For a throughput of {new_throughput}, and response time SLO of {response_time_slo}ms, {required_containers} containers are required.")

    # Plot predictions
    plot_predictions(y_test, y_pred)
