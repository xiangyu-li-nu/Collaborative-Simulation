import pandas as pd
import numpy as np
import matplotlib

# Set the Matplotlib backend to 'TkAgg', make sure Tkinter is installed
matplotlib.use('TkAgg')  # Try using 'TkAgg' backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# 1. Read the data
df = pd.read_excel('processed_results.xlsx').iloc[0:50000]

# 2. Split the features and labels
X = df.drop('pm25_pre', axis=1)
y = df['pm25_pre']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor()
}

# Initialize a list to store the performance metrics of all models
metrics_list = []

# 5. Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining and evaluating model: {name}")
    try:
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Add the current model's performance metrics to the list
        metrics_list.append({
            'Model': name,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

        # Print the current model's performance metrics
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² (R-squared): {r2:.4f}")
    except Exception as e:
        print(f"Error occurred while processing model {name}: {e}")

# Convert the performance metrics list to a DataFrame
metrics_df = pd.DataFrame(metrics_list)
metrics_df.set_index('Model', inplace=True)

# 6. Plot the comparison bar chart for performance metrics
# Define the color palette for each model
palette = sns.color_palette("Set2", len(models))

# Create 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('Comparison of Regression Models', fontsize=20)

# Plot MSE chart
sns.barplot(
    x=metrics_df.index, y='MSE', data=metrics_df, palette=palette, ax=axes[0, 0]
)
axes[0, 0].set_title('Mean Squared Error (MSE)')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot MAE chart
sns.barplot(
    x=metrics_df.index, y='MAE', data=metrics_df, palette=palette, ax=axes[0, 1]
)
axes[0, 1].set_title('Mean Absolute Error (MAE)')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot RMSE chart
sns.barplot(
    x=metrics_df.index, y='RMSE', data=metrics_df, palette=palette, ax=axes[1, 0]
)
axes[1, 0].set_title('Root Mean Squared Error (RMSE)')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot R² Score chart
sns.barplot(
    x=metrics_df.index, y='R2', data=metrics_df, palette=palette, ax=axes[1, 1]
)
axes[1, 1].set_title('R² Score')
axes[1, 1].set_ylabel('R²')
axes[1, 1].tick_params(axis='x', rotation=45)

# Adjust the layout to avoid overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()


