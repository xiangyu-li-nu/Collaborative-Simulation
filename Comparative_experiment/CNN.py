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
import torch
import torch.nn as nn
from einops import rearrange

# Paper: RFAConv: Innovating Spatial Attention and Standard Convolutional Operation
# Paper URL: https://arxiv.org/pdf/2304.03198
# Most comprehensive 100+ plug-and-play modules on GitHub: https://github.com/ai-dawang/PlugNPlay-Modules

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel, bias=False)
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(
                in_channel, 
                in_channel * (kernel_size ** 2), 
                kernel_size=kernel_size, 
                padding=kernel_size // 2,
                stride=stride, 
                groups=in_channel, 
                bias=False
            ),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
        )

        # self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
        #                           nn.BatchNorm2d(out_channel),
        #                           nn.ReLU())
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)  # b c*kernel**2,h,w ->  b c k**2 h w
        weighted_data = feature * weighted
        conv_data = rearrange(
            weighted_data, 
            'b c (n1 n2) h w -> b c (h n1) (w n2)', 
            n1=self.kernel_size,
            # b c k**2 h w ->  b c h*k w*k
            n2=self.kernel_size
        )
        return self.conv(conv_data)


class SE(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # from c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # from c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class RFCBAMConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        if kernel_size % 2 == 0:
            assert ("the kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(
                in_channel, 
                in_channel * (kernel_size ** 2), 
                kernel_size, 
                padding=kernel_size // 2,
                stride=stride, 
                groups=in_channel, 
                bias=False
            ),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
        )
        self.get_weight = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), 
            nn.Sigmoid()
        )
        self.se = SE(in_channel)

        # self.conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride=kernel_size),nn.BatchNorm2d(out_channel),nn.ReLu())
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        channel_attention = self.se(x)
        generate_feature = self.generate(x)

        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(
            generate_feature, 
            'b c (n1 n2) h w -> b c (h n1) (w n2)', 
            n1=self.kernel_size,
            n2=self.kernel_size
        )

        unfold_feature = generate_feature * channel_attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))
        conv_data = unfold_feature * receptive_field_attention
        return self.conv(conv_data)


class RFCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(
                inp, 
                inp * (kernel_size ** 2), 
                kernel_size, 
                padding=kernel_size // 2,
                stride=stride, 
                groups=inp,
                bias=False
            ),
            nn.BatchNorm2d(inp * (kernel_size ** 2)),
            nn.ReLU()
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(
            generate_feature, 
            'b c (n1 n2) h w -> b c (h n1) (w n2)', 
            n1=self.kernel_size,
            n2=self.kernel_size
        )

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)



# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
HIDDEN_SIZE = 64  # For CNN, this can be interpreted as the number of filters
NUM_LAYERS = 1    # Not used in CNN, kept for consistency
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_FOLDS = 5

# Define CNN model
class CNNRegressor(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear((hidden_size*2) * (1 if input_channels == 1 else 2), 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch_size, channels, seq_length]
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
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
    output_dir = 'CNN_result'
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    # data_file = '../Regression_Task_Data/Original_Regression_Dataset1.xlsx'  # Ensure data path is correct
    # data_file = '../Regression_Task_Data/Original_Penetration_Regression_Dataset2.xlsx'  # Ensure data path is correct
    data_file = 'processed_results.xlsx'  # Ensure data path is correct
    X, y = load_and_preprocess_data(data_file)
    INPUT_SIZE = X.shape[1]

    # Convert to tensors
    # Reshape for CNN: [batch_size, channels, seq_length]
    # Assuming each feature is a channel and sequence length is 1
    # Alternatively, if sequence length >1, adjust accordingly
    # Here, treating features as sequence length and channels as 1
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # [samples, channels, seq_length]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # K-Fold Cross Validation
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Create results storage dictionary
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
        input_channels = X_train.shape[1]  # Should be 1
        model = CNNRegressor(input_channels=input_channels, hidden_size=HIDDEN_SIZE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train model
        history = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS)

        # Plot and save loss curves
        plot_loss_curves(history, 'CNN', fold + 1, output_dir)

        # Plot and save MAE curves
        plot_metric_curve(history, 'mae', 'CNN', fold + 1, output_dir)

        # Plot and save R2 curves
        plot_metric_curve(history, 'r2', 'CNN', fold + 1, output_dir)

        # Calculate evaluation metrics
        model.eval()
        with torch.no_grad():
            predictions = model(X_val).squeeze().numpy()
        rmse = np.sqrt(mean_squared_error(y_val.numpy(), predictions))
        mae = mean_absolute_error(y_val.numpy(), predictions)
        mse = mean_squared_error(y_val.numpy(), predictions)
        r2 = r2_score(y_val.numpy(), predictions)

        # Record current fold's results
        results['Fold'].append(fold + 1)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['MSE'].append(mse)
        results['R2'].append(r2)

    # Save results as DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'CNN_Results.csv')
    results_df.to_csv(csv_path, index=False)

    print("\nCross-Validation Results:")
    print(results_df)

if __name__ == '__main__':
    main()
