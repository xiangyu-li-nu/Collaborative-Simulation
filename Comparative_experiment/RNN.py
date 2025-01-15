import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_FOLDS = 5

# Define RNN model
class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [batch, seq, hidden]
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'train_r2': [], 'val_r2': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_r2 = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_mae += mean_absolute_error(targets.numpy(), outputs.squeeze().detach().numpy()) * inputs.size(0)
            running_r2 += r2_score(targets.numpy(), outputs.squeeze().detach().numpy()) * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_mae = running_mae / len(train_loader.dataset)
        train_r2 = running_r2 / len(train_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_r2'].append(train_r2)

        model.eval()
        val_running_loss = 0.0
        val_running_mae = 0.0
        val_running_r2 = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_mae += mean_absolute_error(targets.numpy(), outputs.squeeze().detach().numpy()) * inputs.size(0)
                val_running_r2 += r2_score(targets.numpy(), outputs.squeeze().detach().numpy()) * inputs.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_mae = val_running_mae / len(val_loader.dataset)
        val_r2 = val_running_r2 / len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        # Optional: Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                  f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f} | "
                  f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

    return history

def main():
    # Set output directory
    output_dir = 'RNN_result'
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    # data_file = '../回归任务data/回归问题原始数据集1.xlsx'  # Please ensure the data path is correct
    # data_file = '../回归任务data/渗透率回归问题原始数据集2.xlsx'  # Please ensure the data path is correct
    data_file = '处理后的结果.xlsx'  # Please ensure the data path is correct
    X, y = load_and_preprocess_data(data_file)
    INPUT_SIZE = X.shape[1]

    # Convert to tensor
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # [samples, seq, features]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # K-Fold Cross Validation
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Create result storage dictionary
    results = {
        'Fold': [],
        'RMSE': [],
        'MAE': [],
        'MSE': [],
        'R2': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'Fold {fold + 1}/{NUM_FOLDS}')
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model
        model = RNNRegressor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train model
        history = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS)

        # Plot and save loss curves
        plot_loss_curves(history, 'RNN', fold + 1, output_dir)

        # Plot and save MAE curves
        plot_metric_curve(history, 'mae', 'RNN', fold + 1, output_dir)

        # Plot and save R2 curves
        plot_metric_curve(history, 'r2', 'RNN', fold + 1, output_dir)

        # Compute evaluation metrics
        model.eval()
        with torch.no_grad():
            predictions = model(X_val).squeeze().numpy()
        rmse = np.sqrt(mean_squared_error(y_val.numpy(), predictions))
        mae = mean_absolute_error(y_val.numpy(), predictions)
        mse = mean_squared_error(y_val.numpy(), predictions)
        r2 = r2_score(y_val.numpy(), predictions)

        # Record the results of the current fold
        results['Fold'].append(fold + 1)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['MSE'].append(mse)
        results['R2'].append(r2)

    # Save the results as a DataFrame and then as CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'RNN_Results.csv')
    results_df.to_csv(csv_path, index=False)

    print("\nCross-Validation Results:")
    print(results_df)

if __name__ == '__main__':
    main()
