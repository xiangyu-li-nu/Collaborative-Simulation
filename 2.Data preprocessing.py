import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the data
df = pd.read_excel('合并后的结果.xlsx')

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Numeric columns and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns (using mean)
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing values for categorical variables (using mode)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Check if there are still missing values
print("Missing values after processing:")
print(df.isnull().sum())

# Export the processed data
df.to_excel('处理后的结果.xlsx', index=False)
