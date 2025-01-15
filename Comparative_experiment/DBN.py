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
import torch.nn.functional as F  # Needed for RBM sample_h etc.

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
HIDDEN_SIZES = [256, 128]  # Sizes for each RBM layer
EPOCHS_PRETRAIN = 50
EPOCHS_FINETUNE = 100
BATCH_SIZE = 32
LEARNING_RATE_PRETRAIN = 0.01
LEARNING_RATE_FINETUNE = 0.001
NUM_FOLDS = 5
CD_K = 1  # Contrastive Divergence steps

# Define RBM model
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Initialize weights
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # Hidden layer bias
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # Visible layer bias

    def sample_h(self, v):
        # Compute probabilities of hidden units
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        # Compute probabilities of visible units
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, torch.bernoulli(p_v)

    def forward(self, v):
        p_h, h = self.sample_h(v)
        p_v, v = self.sample_v(h)
        return v

    def contrastive_divergence(self, v):
        # Positive phase
        p_h0, h0 = self.sample_h(v)

        # Gibbs sampling (negative phase)
        v1, _ = self.sample_v(h0)
        p_h1, h1 = self.sample_h(v1)

        # Compute gradients
        positive_grad = torch.matmul(h0.t(), v)
        negative_grad = torch.matmul(h1.t(), v1)

        # Update parameters
        self.W.grad = -(positive_grad - negative_grad) / v.size(0)
        self.v_bias.grad = -torch.mean(v - v1, dim=0)
        self.h_bias.grad = -torch.mean(p_h0 - p_h1, dim=0)

        # Return reconstruction error
        loss = torch.mean((v - v1) ** 2)
        return loss

# Define DBN model
class DBN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(DBN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.rbm_layers = []
        previous_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            rbm = RBM(previous_size, hidden_size)
            self.rbm_layers.append(rbm)
            previous_size = hidden_size
        self.rbm_layers = nn.ModuleList(self.rbm_layers)

        # Define the fine-tuning network (MLP)
        mlp_layers = []
        input_mlp_size = hidden_sizes[-1]  # Start with last RBM's hidden size
        for hidden_size in hidden_sizes:
            mlp_layers.append(nn.Linear(input_mlp_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            input_mlp_size = hidden_size
        mlp_layers.append(nn.Linear(hidden_sizes[-1], 1))  # Regression output
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # Pass through RBM layers
        for rbm in self.rbm_layers:
            p_h, h = rbm.sample_h(x)
            x = p_h
        # Pass through MLP
        out = self.mlp(x)
        return out

    def pretrain(self, train_loader, epochs, learning_rate, device):
        for i, rbm in enumerate(self.rbm_layers):
            print(f'Pretraining RBM Layer {i+1}/{len(self.rbm_layers)}')
            optimizer = optim.SGD(rbm.parameters(), lr=learning_rate)
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch in train_loader:
                    v, _ = batch
                    v = v.to(device)
                    loss = rbm.contrastive_divergence(v)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f'RBM Layer {i+1}, Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            # Transform data for the next RBM layer
            transformed_data = []
            with torch.no_grad():
                for batch in train_loader:
                    v, _ = batch
                    v = v.to(device)
                    p_h, h = rbm.sample_h(v)
                    transformed_data.append(p_h)
            transformed_tensor = torch.cat(transformed_data)
            train_loader = DataLoader(TensorDataset(transformed_tensor, transformed_tensor), batch_size=BATCH_SIZE, shuffle=True)
        print('Pretraining completed.')

def train_finetune_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'train_r2': [], 'val_r2': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_r2 = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_mae += mean_absolute_error(targets.cpu().numpy(), outputs.squeeze().detach().cpu().numpy()) * inputs.size(0)
            running_r2 += r2_score(targets.cpu().numpy(), outputs.squeeze().detach().cpu().numpy()) * inputs.size(0)

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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_running_loss += loss.item() * inputs.size(0)
                val_running_mae += mean_absolute_error(targets.cpu().numpy(), outputs.squeeze().detach().cpu().numpy()) * inputs.size(0)
                val_running_r2 += r2_score(targets.cpu().numpy(), outputs.squeeze().detach().cpu().numpy()) * inputs.size(0)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set output directory
    output_dir = 'DBN_result'
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    data_file = '处理后的结果.xlsx'  # Please ensure the data path is correct
    X, y = load_and_preprocess_data(data_file)
    INPUT_SIZE = X.shape[1]

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Make it [samples, 1]

    # K-Fold cross-validation
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
        print(f'\nFold {fold + 1}/{NUM_FOLDS}')
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize DBN model
        dbn = DBN(INPUT_SIZE, HIDDEN_SIZES).to(device)

        # Pretraining
        dbn.pretrain(train_loader, EPOCHS_PRETRAIN, LEARNING_RATE_PRETRAIN, device)

        # Fine-tuning
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dbn.mlp.parameters(), lr=LEARNING_RATE_FINETUNE)

        history = train_finetune_model(dbn, criterion, optimizer, train_loader, val_loader, EPOCHS_FINETUNE, device)

        # Plot and save loss curves
        plot_loss_curves(history, 'DBN', fold + 1, output_dir)

        # Plot and save MAE curves
        plot_metric_curve(history, 'mae', 'DBN', fold + 1, output_dir)

        # Plot and save R2 curves
        plot_metric_curve(history, 'r2', 'DBN', fold + 1, output_dir)

        # Compute evaluation metrics
        dbn.eval()
        with torch.no_grad():
            predictions = dbn(X_val.to(device)).squeeze().cpu().numpy()
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

    # Save the results as a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'DBN_Results.csv')
    results_df.to_csv(csv_path, index=False)

    print("\nCross-Validation Results:")
    print(results_df)

if __name__ == '__main__':
    main()
