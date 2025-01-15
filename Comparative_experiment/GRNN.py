import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import load_and_preprocess_data, plot_loss_curves, plot_metric_curve
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
SIGMA = 0.5        # GRNN smoothing parameter, can be adjusted as needed
BATCH_SIZE = 32
NUM_FOLDS = 5

# Define GRNN model
class GRNNRegressor(nn.Module):
    def __init__(self, sigma=1.0):
        super(GRNNRegressor, self).__init__()
        self.sigma = sigma
        self.training_data = None
        self.training_targets = None

    def store_training_data(self, X_train, y_train):
        """
        Store training data and target values
        """
        self.training_data = X_train  # [num_train_samples, features]
        self.training_targets = y_train  # [num_train_samples]

    def forward(self, x):
        """
        Make predictions for input x
        x: [batch_size, features]
        Returns: [batch_size]
        """
        if self.training_data is None or self.training_targets is None:
            raise ValueError("Training data not stored. Call 'store_training_data' first.")

        # Calculate Euclidean distance between input and all training samples
        # x: [batch_size, features]
        # training_data: [num_train, features]
        # Compute differences
        diff = x.unsqueeze(1) - self.training_data.unsqueeze(0)  # [batch_size, num_train, features]
        dist_sq = torch.sum(diff ** 2, dim=2)  # [batch_size, num_train]
        # Calculate Gaussian kernel
        weights = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # [batch_size, num_train]
        # Compute weighted sum
        numerator = torch.matmul(weights, self.training_targets)  # [batch_size]
        denominator = torch.sum(weights, dim=1)  # [batch_size]
        y_pred = numerator / denominator  # [batch_size]
        return y_pred

def main():
    # Set output directory
    output_dir = 'GRNN_result'
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    # Please ensure the data path is correct
    data_file = '处理后的结果.xlsx'
    X, y = load_and_preprocess_data(data_file)
    INPUT_SIZE = X.shape[1]

    # Convert to tensors
    # Assume each feature is independent and no reshaping is needed
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Create result storage dictionary
    results = {
        'Fold': [],
        'RMSE': [],
        'MAE': [],
        'MSE': [],
        'R2': []
    }

    # Initialize model
    model = GRNNRegressor(sigma=SIGMA)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'Fold {fold + 1}/{NUM_FOLDS}')
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        # Store training data
        model.store_training_data(X_train, y_train)

        # Create validation data loader
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Prediction history (since GRNN does not require training, history is only for visualization)
        history = {
            'val_loss': [],
            'val_mae': [],
            'val_r2': []
        }

        model.eval()
        val_running_loss = 0.0
        val_running_mae = 0.0
        val_running_r2 = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                all_predictions.append(outputs.numpy())
                all_targets.append(targets.numpy())

        # Concatenate all batch predictions
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        # Record current fold's results
        results['Fold'].append(fold + 1)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['MSE'].append(mse)
        results['R2'].append(r2)

        # Plot and save prediction scatter plot
        plt_path = os.path.join(output_dir, f'GRNN_Predictions_Fold_{fold + 1}.png')
        plt.figure(figsize=(6,6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'GRNN Predictions - Fold {fold + 1}')
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')  # Diagonal line
        plt.savefig(plt_path)
        plt.close()
        print(f'Fold {fold + 1} Predictions plot saved to {plt_path}')

    # Save results as DataFrame and export to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'GRNN_Results.csv')
    results_df.to_csv(csv_path, index=False)

    print("\nCross-Validation Results:")
    print(results_df)

    # Optional: Plot boxplots for each metric
    metrics = ['RMSE', 'MAE', 'MSE', 'R2']
    for metric in metrics:
        plt.figure(figsize=(8,6))
        plt.boxplot(results_df[metric], vert=True, patch_artist=True)
        plt.title(f'GRNN {metric} across {NUM_FOLDS} Folds')
        plt.ylabel(metric)
        plt.savefig(os.path.join(output_dir, f'GRNN_{metric}_Boxplot.png'))
        plt.close()
        print(f'{metric} boxplot saved.')

if __name__ == '__main__':
    # Import libraries required for plotting
    import matplotlib.pyplot as plt
    main()
