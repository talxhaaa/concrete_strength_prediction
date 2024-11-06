import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import joblib
import optuna  # Import Optuna for hyperparameter tuning
from model_definition import ConcreteStrengthModel
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter

# Constants
epochs = 100
k_folds = 5

# Load and standardize the data
print("Loading and preprocessing data...")
data = pd.read_csv("data/train-val.csv")
X = data.drop("Concrete compressive strength(MPa, megapascals) ", axis=1).values
y = data["Concrete compressive strength(MPa, megapascals) "].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "results/scaler.joblib")
print("Data standardized and scaler saved.")

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter("runs/experiment_1")

# Define Optuna objective function for tuning
def objective(trial):
    # Hyperparameter suggestions from Optuna
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Initialize KFold
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_performance = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        print(f"\nStarting Fold {fold + 1}/{k_folds}")

        # Split into training and validation sets
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Define model, loss, and optimizer with dynamic hyperparameters
        model = ConcreteStrengthModel(hidden_size=hidden_size, dropout_rate=dropout_rate)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop for each fold
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Evaluate on validation set every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor)
                    val_mae = torch.mean(torch.abs(val_predictions - y_val_tensor))

                # Log both MAE and MSE to TensorBoard
                writer.add_scalar(f'Fold {fold + 1}/Training Loss', loss.item(), epoch)
                writer.add_scalar(f'Fold {fold + 1}/Validation Loss', val_loss.item(), epoch)
                writer.add_scalar(f'Fold {fold + 1}/Validation MAE', val_mae.item(), epoch)

                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation MAE: {val_mae.item():.4f}")

        # Store validation performance for this fold
        fold_performance.append(val_loss.item())

    # Return the average validation loss across folds as the objective metric
    average_val_loss = np.mean(fold_performance)
    return average_val_loss

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  # Adjust number of trials as needed

# Print and save best parameters
print("Best hyperparameters found:", study.best_trial.params)
joblib.dump(study, "results/optuna_study.pkl")

# Final training with best parameters
best_params = study.best_trial.params
print("\nTraining final model with optimal parameters...")
final_model = ConcreteStrengthModel(hidden_size=best_params["hidden_size"], dropout_rate=best_params["dropout_rate"])
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

for epoch in range(epochs):
    final_model.train()
    optimizer.zero_grad()
    predictions = final_model(X_tensor)
    loss = criterion(predictions, y_tensor)
    loss.backward()
    optimizer.step()

    # Log final training loss to TensorBoard
    writer.add_scalar('Final Training Loss', loss.item(), epoch)

    # Print training progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Save the final trained model
torch.save(final_model.state_dict(), "results/concrete_strength_model.pt")
joblib.dump(final_model, "results/concrete_strength_model.joblib")
print("Final model training complete and saved.")

# Close the TensorBoard writer
writer.close()
