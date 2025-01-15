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
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_FOLDS = 5
DROPOUT = 0.1  # Dropout rate for regularization

class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Define the Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs):
        """
        encoder_outputs: [batch_size, seq_length, hidden_size]
        """
        # Compute attention scores
        scores = self.attention(encoder_outputs)  # [batch_size, seq_length, 1]
        scores = torch.softmax(scores, dim=1)  # [batch_size, seq_length, 1]

        # Compute the context vector
        context = torch.sum(scores * encoder_outputs, dim=1)  # [batch_size, hidden_size]
        return context


# Define the Attention-Based Regression Model
class AttentionRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(AttentionRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define a simple feedforward network with attention
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Attention layer
        self.attention = Attention(hidden_size)

        # Fully connected layers after attention
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # Apply first linear layer
        out = self.fc1(x)  # [batch_size, seq_length, hidden_size]
        out = self.relu(out)
        out = self.dropout(out)

        # Apply attention
        context = self.attention(out)  # [batch_size, hidden_size]

        # Fully connected layers
        out = self.fc2(context)  # [batch_size, 64]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_r2': [],
        'val_r2': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_r2 = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)  # [batch_size, 1, input_size]
            targets = targets.to(device)  # [batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_size]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_mae += mean_absolute_error(targets.cpu().numpy(), outputs.detach().cpu().numpy()) * inputs.size(0)
            running_r2 += r2_score(targets.cpu().numpy(), outputs.detach().cpu().numpy()) * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_mae = running_mae / len(train_loader.dataset)
        train_r2 = running_r2 / len(train_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_r2'].append(train_r2)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_mae = 0.0
        val_running_r2 = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_mae += mean_absolute_error(targets.cpu().numpy(),
                                                       outputs.detach().cpu().numpy()) * inputs.size(0)
                val_running_r2 += r2_score(targets.cpu().numpy(), outputs.detach().cpu().numpy()) * inputs.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_mae = val_running_mae / len(val_loader.dataset)
        val_r2 = val_running_r2 / len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                  f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f} | "
                  f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

    return history


def main():
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set output directory
    output_dir = 'AttentionNet_result'
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    # Ensure the data path is correct
    data_file = '处理后的结果.xlsx'
    X, y = load_and_preprocess_data(data_file)
    INPUT_SIZE = X.shape[1]

    # Convert to tensors
    # Assuming each sample is a feature vector; reshape for attention if necessary
    # For simplicity, treating each sample as a sequence of length 1 with INPUT_SIZE features
    # If you have sequence data, adjust the reshaping accordingly
    # Here, we'll assume that the model expects [batch_size, seq_length, input_size]
    # Let's treat features as sequence elements for demonstration

    # Example: Treat each feature as a time step in a sequence
    # Reshape X to [samples, seq_length, input_size] where seq_length = number of features
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [samples, 1, input_size]
    y_tensor = torch.tensor(y, dtype=torch.float32)  # [samples]

    # K-Fold Cross-Validation
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Initialize results dictionary
    results = {
        'Fold': [],
        'RMSE': [],
        'MAE': [],
        'MSE': [],
        'R2': []
    }

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
        print(f'\nFold {fold + 1}/{NUM_FOLDS}')
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        # Create datasets and loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize the model
        model = AttentionRegressor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                                   num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train the model
        history = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS, device)

        # Plot and save loss curves
        plot_loss_curves(history, 'AttentionNet', fold + 1, output_dir)

        # Plot and save MAE curve
        plot_metric_curve(history, 'mae', 'AttentionNet', fold + 1, output_dir)

        # Plot and save R2 curve
        plot_metric_curve(history, 'r2', 'AttentionNet', fold + 1, output_dir)

        # Evaluate on validation set
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        # Store results
        results['Fold'].append(fold + 1)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['MSE'].append(mse)
        results['R2'].append(r2)

        # Plot and save Predictions vs True Values
        plt_path = os.path.join(output_dir, f'AttentionNet_Predictions_Fold_{fold + 1}.png')
        plt.figure(figsize=(6, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'AttentionNet Predictions - Fold {fold + 1}')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')  # Diagonal line
        plt.savefig(plt_path)
        plt.close()
        print(f'Fold {fold + 1} Predictions plot saved to {plt_path}')

    # Save all results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'AttentionNet_Results.csv')
    results_df.to_csv(csv_path, index=False)

    print("\nCross-Validation Results:")
    print(results_df)

    # Optional: Plot boxplots for each metric
    metrics = ['RMSE', 'MAE', 'MSE', 'R2']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.boxplot(results_df[metric], vert=True, patch_artist=True)
        plt.title(f'AttentionNet {metric} across {NUM_FOLDS} Folds')
        plt.ylabel(metric)
        plt.savefig(os.path.join(output_dir, f'AttentionNet_{metric}_Boxplot.png'))
        plt.close()
        print(f'{metric} boxplot saved.')


if __name__ == '__main__':
    main()
