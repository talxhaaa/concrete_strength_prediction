import torch
import pandas as pd
import joblib
from model_definition import ConcreteStrengthModel

# Function to make predictions on unseen data
def predict_unseen_data(unseen_data_path):
    # Load the trained model and scaler
    print("Loading the model and scaler for evaluation...")
    model = ConcreteStrengthModel()  # Adjust parameters if necessary
    model.load_state_dict(torch.load("results/concrete_strength_model.pt"))
    model.eval()  # Set the model to evaluation mode
    scaler = joblib.load("results/scaler.joblib")
    print("Model and scaler loaded successfully.")

    # Load unseen data
    unseen_data = pd.read_csv(unseen_data_path)
    
    # Prepare features (X) by dropping the target variable if it exists
    X_unseen = unseen_data.drop("Concrete compressive strength(MPa, megapascals) ", axis=1, errors='ignore')

    # Preprocess the unseen data by standardizing it
    X_unseen_scaled = scaler.transform(X_unseen)  # Standardize using the fitted scaler
    print("Unseen data standardized using the saved scaler.")

    # Convert unseen data to PyTorch tensor
    X_unseen_tensor = torch.tensor(X_unseen_scaled, dtype=torch.float32)

    # Make predictions
    print("Evaluating model on unseen data...")
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(X_unseen_tensor).squeeze()  # Get the predictions

    # Display predictions
    print("Predictions on unseen data (MPa):", predictions.numpy())

# Example usage
unseen_data_path = "data/unseen_test_data.csv"  # Replace with your unseen data path
predict_unseen_data(unseen_data_path)