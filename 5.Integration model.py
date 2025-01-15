import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Read data
df = pd.read_excel('处理后的结果.xlsx')

# 2. Feature and label separation
X = df.drop('pm25_pre', axis=1)
y = df['pm25_pre']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
base_models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Define the meta-model
meta_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define the stacking ensemble model
stacking_regressor = StackingRegressor(
    estimators=[
        ('rf1', base_models['Random Forest']),  # First random forest
        ('gb', base_models['Gradient Boosting']),  # Gradient boosting
        ('dt', base_models['Decision Tree']),  # Decision tree
    ],
    final_estimator=meta_model,  # Use random forest as the final model
    cv=5,
    n_jobs=-1
)

# Train the stacking model
print("\nTraining and evaluating model: Stacking Regressor")
try:
    stacking_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred_stack = stacking_regressor.predict(X_test)

    # Compute performance metrics
    mse_stack = mean_squared_error(y_test, y_pred_stack)
    mae_stack = mean_absolute_error(y_test, y_pred_stack)
    rmse_stack = np.sqrt(mse_stack)
    r2_stack = r2_score(y_test, y_pred_stack)

    # Print performance metrics of the stacking model
    print(f"Mean Squared Error (MSE): {mse_stack:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_stack:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_stack:.4f}")
    print(f"R-squared (R²): {r2_stack:.4f}")

except Exception as e:
    print(f"Error occurred while processing Stacking Regressor: {e}")
