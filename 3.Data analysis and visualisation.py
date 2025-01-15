# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os

# Set Matplotlib backend to avoid displaying plots
matplotlib.use('Agg')  # Use non-interactive backend

# Check versions
print(f"Pandas version: {pd.__version__}")
print(f"Seaborn version: {sns.__version__}")

# Set plot styles for better aesthetics
sns.set(style="whitegrid", palette="muted")
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly
plt.rcParams['font.size'] = 12

# 1. Read the dataset
df = pd.read_excel('merged_results.xlsx')  # Translated filename from '合并后的结果.xlsx' to 'merged_results.xlsx'

# 2. Data Preprocessing

# Convert applicable columns to numeric types
non_numeric_columns = ['Period', 'Mode', 'LATITUDE N/S', 'LONGITUDE E/W']
numeric_columns = df.columns.drop(non_numeric_columns)
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill missing values with the median
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Confirm there are no missing values
print("\nMissing values after filling:")
print(df.isnull().sum())

# 3. Data Type Conversions and Feature Engineering

# Convert 'LATITUDE N/S' and 'LONGITUDE E/W' to numeric values
df['Latitude'] = df['LATITUDE N/S'].str[:-1].astype(float)
df['Longitude'] = df['LONGITUDE E/W'].str[:-1].astype(float)

# Convert 'Period' and 'Mode' to categorical data
df['Period'] = df['Period'].astype('category')
df['Mode'] = df['Mode'].astype('category')

# Rename 'Minute_x' to 'Minute' if necessary
if 'Minute_x' in df.columns:
    df.rename(columns={'Minute_x': 'Minute'}, inplace=True)

# Create a datetime column for time series analysis
df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])

# Redefine numeric_columns after preprocessing
non_numeric_columns = ['Period', 'Mode', 'LATITUDE N/S', 'LONGITUDE E/W', 'Datetime']
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# 4. Statistical Summary of Numerical Variables
print("\nStatistical summary of numerical variables:")
print(df.describe())

# Create a folder to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# 5. Data Visualization

## 5.1 Visualizing All Numerical Features

### 5.1.1 Histograms and KDE Plots for Numerical Features
numerical_features = ['pm25_pre', 'PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2', 'AQI',
                      'TEM', 'RHU', 'PRS', 'WIN_S_Avg_2mi', 'average_speed',
                      'average_delay', 'stops_numbers', 'windpower', 'tigan']

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'plots/{feature}_distribution.png')
    plt.close()  # Close the figure to free up memory

### 5.1.2 Boxplots for Numerical Features
for feature in numerical_features:
    plt.figure(figsize=(6, 8))
    sns.boxplot(y=df[feature], color='lightgreen')
    plt.title(f'Boxplot of {feature}')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f'plots/{feature}_boxplot.png')
    plt.close()

## 5.2 Correlation Analysis

### 5.2.1 Correlation Heatmap
plt.figure(figsize=(16, 14))
corr_matrix = df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

### 5.2.2 Correlation with pm25_pre
pm25_corr = corr_matrix['pm25_pre'].sort_values(ascending=False)
print("\nCorrelation of variables with pm25_pre:")
print(pm25_corr)

## 5.3 Relationships Between pm25_pre and Other Variables

### 5.3.1 Scatter Plots with Regression Lines
variables_to_plot = ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'TEM', 'RHU', 'PRS',
                     'WIN_S_Avg_2mi', 'average_speed', 'average_delay', 'stops_numbers', 'windpower', 'tigan']

for var in variables_to_plot:
    plt.figure(figsize=(8, 6))
    sns.regplot(x=var, y='pm25_pre', data=df, scatter_kws={'alpha':0.5})
    plt.title(f'pm25_pre vs {var}')
    plt.xlabel(var)
    plt.ylabel('pm25_pre')
    plt.tight_layout()
    plt.savefig(f'plots/pm25_pre_vs_{var}.png')
    plt.close()

## 5.4 Categorical Analysis

### 5.4.1 pm25_pre by Period
plt.figure(figsize=(8, 6))
sns.boxplot(x='Period', y='pm25_pre', data=df, palette='Set2')
plt.title('pm25_pre by Period')
plt.xlabel('Period')
plt.ylabel('pm25_pre')
plt.tight_layout()
plt.savefig('plots/pm25_pre_by_Period.png')
plt.close()

### 5.4.2 pm25_pre by Mode
plt.figure(figsize=(8, 6))
sns.boxplot(x='Mode', y='pm25_pre', data=df, palette='Set3')
plt.title('pm25_pre by Mode')
plt.xlabel('Mode of Transportation')
plt.ylabel('pm25_pre')
plt.tight_layout()
plt.savefig('plots/pm25_pre_by_Mode.png')
plt.close()

## 5.5 Time Series Analysis

### 5.5.1 Time Series Plot of pm25_pre
plt.figure(figsize=(12, 6))
sns.lineplot(x='Datetime', y='pm25_pre', data=df, color='blue')
plt.title('Time Series of pm25_pre')
plt.xlabel('Datetime')
plt.ylabel('pm25_pre')
plt.tight_layout()
plt.savefig('plots/pm25_pre_timeseries.png')
plt.close()

## 5.6 Geographical Analysis

### 5.6.1 Geographical Scatter Plot of pm25_pre
plt.figure(figsize=(10, 8))
sc = plt.scatter(df['Longitude'], df['Latitude'], c=df['pm25_pre'], cmap='coolwarm', alpha=0.7)
plt.colorbar(sc, label='pm25_pre')
plt.title('Geographical Distribution of pm25_pre')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('plots/pm25_pre_geographical.png')
plt.close()

### 5.6.2 Geographical Scatter Plot of PM2.5
plt.figure(figsize=(10, 8))
sc = plt.scatter(df['Longitude'], df['Latitude'], c=df['PM2.5'], cmap='viridis', alpha=0.7)
plt.colorbar(sc, label='PM2.5')
plt.title('Geographical Distribution of PM2.5')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('plots/PM2.5_geographical.png')
plt.close()

## 5.7 Additional Analysis

### 5.7.1 Grouped Statistics by Hour
hourly_stats = df.groupby('Hour')[numerical_features].mean()
print("\nHourly statistics of numerical features:")
print(hourly_stats)

### 5.7.2 Grouped Statistics by Day
daily_stats = df.groupby('Day')[numerical_features].mean()
print("\nDaily statistics of numerical features:")
print(daily_stats)

### 5.7.3 Pairplot of Selected Variables
selected_vars = ['pm25_pre', 'PM2.5', 'PM10', 'NO2', 'O3', 'TEM', 'RHU']
sns.pairplot(df[selected_vars], diag_kind='kde', corner=True)
plt.suptitle('Pairplot of Selected Variables', y=1.02)
plt.tight_layout()
plt.savefig('plots/pairplot_selected_variables.png')
plt.close()

# 7. Save the Cleaned Dataset (Optional)
df.to_excel('Cleaned_Data.xlsx', index=False)
