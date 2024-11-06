import torch
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model_definition import ConcreteStrengthModel
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter

# Load the best hyperparameters
best_hyperparams = joblib.load("results/optuna_study.pkl").best_trial.params

# Load the trained model and scaler
print("Loading the model and scaler for evaluation...")
model = ConcreteStrengthModel(
    hidden_size=best_hyperparams["hidden_size"],
    dropout_rate=best_hyperparams["dropout_rate"]
)
model.load_state_dict(torch.load("results/concrete_strength_model.pt"))
model.eval()
scaler = joblib.load("results/scaler.joblib")
print("Model and scaler loaded successfully.")

# Load and preprocess test data
print("Loading and preprocessing test data...")
test_data = pd.read_csv("data/test.csv")
X_test = test_data.drop("Concrete compressive strength(MPa, megapascals) ", axis=1)
y_test = test_data["Concrete compressive strength(MPa, megapascals) "]

# Apply the same scaling to the test data
X_test_scaled = scaler.transform(X_test)
print("Test data standardized using the saved scaler.")

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Predict and evaluate the model
print("Evaluating model on test data...")
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()

# Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = mean_absolute_error(y_test, predictions.numpy())
mse = mean_squared_error(y_test, predictions.numpy())

# Log the test performance to TensorBoard
writer = SummaryWriter("runs/experiment_1")  # Ensure this matches the training logdir
writer.add_scalar('Test MAE', mae)
writer.add_scalar('Test MSE', mse)

# Print the results
print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")

# Close the TensorBoard writer
writer.close()
