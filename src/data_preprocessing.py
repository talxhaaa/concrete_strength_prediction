import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
print("Loading data for preprocessing...")
data = pd.read_csv("data/train-val.csv")

# Split data into features (X) and target (y) with corrected column name
X = data.drop("Concrete compressive strength(MPa, megapascals) ", axis=1)
y = data["Concrete compressive strength(MPa, megapascals) "]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}.")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the scaler
joblib.dump(scaler, "results/scaler.joblib")
print("Scaler saved to results/scaler.joblib for future use.")

# Convert data to tensors and save them
train_data = {
    "X_train": torch.tensor(X_train_scaled, dtype=torch.float32),
    "y_train": torch.tensor(y_train.values, dtype=torch.float32)
}
val_data = {
    "X_val": torch.tensor(X_val_scaled, dtype=torch.float32),
    "y_val": torch.tensor(y_val.values, dtype=torch.float32)
}

# Save preprocessed data as .pt files for easy loading later
torch.save(train_data, "data/preprocessed_train_data.pt")
torch.save(val_data, "data/preprocessed_val_data.pt")
print("Preprocessed data saved to data/preprocessed_train_data.pt and data/preprocessed_val_data.pt.")
