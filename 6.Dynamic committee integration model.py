import matplotlib
# 1) If you do not need to display images in a window, you can use the "Agg" backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ============== 1. Read data and split into training and testing sets ==============
data_path = 'processed_results.xlsx'
df = pd.read_excel(data_path)

# Features / Labels
X = df.drop('pm25_pre', axis=1)
y = df['pm25_pre']

# Convert uniformly to NumPy to avoid feature name mismatch warnings
X_np = X.to_numpy()
y_np = y.to_numpy()

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

# ============== 2. Train three base models (RF, GBDT, XGB) ==============
print("Training base models...")

# 2.1 Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_np, y_train_np)

# 2.2 GBDT
gbdt = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbdt.fit(X_train_np, y_train_np)

# 2.3 XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_np, y_train_np)

print("Base models trained.\n")

# ============== 3. Static Ensemble ==============
class StaticEnsemble:
    """
    Static Ensemble - Simple average ensemble example
    """
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = []
        for model in self.models:
            pred = model.predict(X)
            preds.append(pred)
        preds = np.array(preds)            # shape: (n_models, n_samples)
        avg_preds = np.mean(preds, axis=0) # shape: (n_samples,)
        return avg_preds

print("Performing Static Ensemble prediction...")
static_ensemble_model = StaticEnsemble(models=[rf, gbdt, xgb])
y_pred_static = static_ensemble_model.predict(X_test_np)

# ============== 4. Dynamic Ensemble with FCM ==============
class DynamicEnsembleFCM:
    """
    Dynamic Ensemble - Uses Fuzzy C-Means for clustering and weights models based on cluster results
    """
    def __init__(self, models, n_clusters=3):
        self.models = models
        self.n_clusters = n_clusters
        self.cluster_centers_ = None  # (n_clusters, n_features)
        self.cluster_weights_ = None  # (n_clusters, n_models)

    def fit_fcm(self, X_train, y_train, m=2, error=0.005, maxiter=1000):
        """
        Uses FCM to cluster X_train and calculates weighted coefficients for each model in each cluster.
        """
        # 1) FCM clustering: input (features, samples) needs to be transposed
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_train.T, c=self.n_clusters, m=m,
            error=error, maxiter=maxiter, init=None
        )
        self.cluster_centers_ = cntr  # (n_clusters, n_features)

        # 2) memberships: shape = (n_samples, n_clusters)
        memberships = u.T

        self.cluster_weights_ = np.zeros((self.n_clusters, len(self.models)))

        # Iterate through each cluster to calculate weighted MSE for each model
        for cluster_idx in range(self.n_clusters):
            membership_col = memberships[:, cluster_idx]  # (n_samples,)

            # Calculate model predictions on the training set
            model_preds = []
            for model in self.models:
                pred = model.predict(X_train)
                model_preds.append(pred)
            model_preds = np.array(model_preds)  # (n_models, n_samples)

            # Calculate weighted MSE
            weighted_mses = []
            for i in range(len(self.models)):
                mse_val = mean_squared_error(
                    y_train, model_preds[i],
                    sample_weight=membership_col
                )
                weighted_mses.append(mse_val)
            weighted_mses = np.array(weighted_mses)

            # MSE -> weights (smaller MSE means larger weight)
            inv_mse = 1.0 / (weighted_mses + 1e-9)
            weights = inv_mse / np.sum(inv_mse)
            self.cluster_weights_[cluster_idx] = weights

            print(f"Cluster {cluster_idx}: Weighted MSE = {weighted_mses}, Weights = {weights}")

        print("\nFCM fitting done.\n")

    def predict(self, X):
        """
        For new samples, first compute distances to each cluster center -> memberships -> weighted predictions
        """
        dist = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            diff = X - self.cluster_centers_[i]
            dist[:, i] = np.linalg.norm(diff, axis=1)

        # Avoid division by zero
        dist = np.where(dist == 0, 1e-9, dist)
        inv_dist = 1.0 / dist
        memberships = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        # (n_samples, n_clusters)

        # Predictions from each model on X
        model_preds = []
        for model in self.models:
            pred = model.predict(X)
            model_preds.append(pred)
        model_preds = np.array(model_preds).T  # (n_samples, n_models)

        # Get final weights
        # final_weights: (n_samples, n_models) = memberships @ cluster_weights_
        final_weights = np.dot(memberships, self.cluster_weights_)

        # Weighted sum
        weighted_preds = np.sum(model_preds * final_weights, axis=1)
        return weighted_preds

print("Training Dynamic Ensemble (FCM)...")
dynamic_ensemble_fcm = DynamicEnsembleFCM(models=[rf, gbdt, xgb], n_clusters=3)
dynamic_ensemble_fcm.fit_fcm(X_train_np, y_train_np)

print("Performing Dynamic Ensemble prediction...")
y_pred_dynamic = dynamic_ensemble_fcm.predict(X_test_np)

# ============== 5. Evaluation and Visualization ==============
def evaluate_and_print_metrics(y_true, y_pred, ensemble_type="Static"):
    mse_val = mean_squared_error(y_true, y_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    print(f"{ensemble_type} Ensemble:")
    print(f"  MSE : {mse_val:.4f}")
    print(f"  RMSE: {rmse_val:.4f}")
    print(f"  MAE : {mae_val:.4f}")
    print(f"  R2  : {r2_val:.4f}\n")

evaluate_and_print_metrics(y_test_np, y_pred_static, "Static")
evaluate_and_print_metrics(y_test_np, y_pred_dynamic, "Dynamic")

# ============== 6. Visualization Scatter Plots (True Values vs. Predicted Values) ==============
def plot_predictions(y_true, y_pred, title="Ensemble Predictions", save_fig=False, fig_name="plot.png"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(title)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    if save_fig:
        plt.savefig(fig_name, dpi=300)
        print(f"Figure saved to: {fig_name}")
    else:
        plt.show()

# If using the "Agg" backend, plt.show() cannot display a window, so use plt.savefig() instead
plot_predictions(y_test_np, y_pred_static,  "Static Ensemble Predictions",  save_fig=True, fig_name="static_ensemble.png")
plot_predictions(y_test_np, y_pred_dynamic, "Dynamic Ensemble Predictions", save_fig=True, fig_name="dynamic_ensemble.png")

print("All done.")
