# Project Overview
This project involves building a neural network model to predict the compressive strength of concrete based on various input parameters such as cement, water, aggregate types, and curing age. The goal is to leverage machine learning to assist in predicting concrete properties for construction applications.

## Dataset
The dataset consists of concrete mixture properties and their corresponding compressive strength values:
- **Features**: Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age (days).
- **Target**: Concrete compressive strength (in MPa).

## Model and Approach
The model uses a feed-forward neural network built with PyTorch:
1. **Layers**: Multiple hidden layers with ReLU activation and dropout for regularization.
2. **Optimization**: Hyperparameter tuning with Optuna, utilizing k-fold cross-validation to evaluate and reduce overfitting.
3. **Metrics**: Mean Absolute Error (MAE) and Mean Squared Error (MSE) to evaluate prediction accuracy.

## Installation
1. Clone this repository.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt

# Usage

##      1. Training: Run the training script to train the model:

        python train_model.py

This will use k-fold cross-validation and save the best model and hyperparameters.

##      2. Testing: Use the evaluation script to test the model on unseen data:

        python evaluate_model.py

The script outputs performance metrics (MAE, MSE) on the test dataset.

##      3. Prediction on New Data: Use the provided scripts to make predictions on custom data by modifying input features in the predict.py script.


# Additional Info

## Preprocessing Techniques
The model performs standardization and uses k-fold cross-validation. Additional steps like feature scaling and normalization are applied to optimize model input.

## Acknowledgments
This project is based on the "Modeling of strength of high-performance concrete using artificial neural networks" dataset from the work of I.C. Yeh (1998).

## Results and Performance
After tuning, the model achieved:

    Mean Absolute Error (MAE): final MAE value here
    Mean Squared Error (MSE): final MSE value here