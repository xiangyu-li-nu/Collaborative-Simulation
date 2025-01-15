# utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_and_preprocess_data(file_path):
    # Choose the reading method based on the file extension
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.csv':
        data = pd.read_csv(file_path, encoding='utf-8')  # If there is an encoding issue, try other encodings like 'gbk'
    elif file_extension.lower() in ['.xls', '.xlsx']:
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    # Separate features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Feature standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def plot_loss_curves(history, model_name, fold, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{model_name} Loss Curve (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_Loss_Fold_{fold}.png')
    plt.savefig(plot_path)
    plt.close()


def plot_metric_curve(history, metric, model_name, fold, output_dir):
    plt.figure(figsize=(10, 6))
    if f'train_{metric}' in history and f'train_{metric}' != 'train_val_r2':
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.upper()}', color='blue')
    plt.plot(history[f'val_{metric}'], label=f'Validation {metric.upper()}', color='orange')
    plt.title(f'{model_name} {metric.upper()} Curve (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_{metric.capitalize()}_Fold_{fold}.png')
    plt.savefig(plot_path)
    plt.close()
