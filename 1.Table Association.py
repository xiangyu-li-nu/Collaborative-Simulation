import pandas as pd

# 1. Read and process the main table
df_main = pd.read_excel('data/需要预测的车内污染物浓度.xlsx')

# Add a 'Year' column, assuming the year is 2022. If the year is different, please modify according to the actual situation.
df_main['Year'] = 2022

# 2. Process '表1.xlsx'
df1 = pd.read_excel('data/表1.xlsx')

# Extract 'Year', 'Month', 'Day'
df1['Year'] = df1['date'].astype(str).str[:4].astype(int)
df1['Month'] = df1['date'].astype(str).str[4:6].astype(int)
df1['Day'] = df1['date'].astype(str).str[6:8].astype(int)

# Pivot the table, expand the 'type' column
df1_pivot = df1.pivot_table(index=['Year', 'Month', 'Day', 'hour'], columns='type', values='值', aggfunc='first').reset_index()

# Rename 'hour' column to 'Hour' for consistency
df1_pivot.rename(columns={'hour': 'Hour'}, inplace=True)

# 3. Process '表2.xlsx'
df2 = pd.read_excel('data/表2.xlsx')

# Extract 'Year', 'Month', 'Day'
df2['Year'] = df2['date'].astype(str).str[:4].astype(int)
df2['Month'] = df2['date'].astype(str).str[4:6].astype(int)
df2['Day'] = df2['date'].astype(str).str[6:8].astype(int)

# Pivot the table, expand the 'type' column
df2_pivot = df2.pivot_table(index=['Year', 'Month', 'Day', 'hour'], columns='type', values='海淀万柳', aggfunc='first').reset_index()

# Rename 'hour' column to 'Hour'
df2_pivot.rename(columns={'hour': 'Hour'}, inplace=True)

# 4. Process '天气背景数据.xlsx'
df_weather = pd.read_excel('data/天气背景数据.xlsx')

# Rename 'Mon' to 'Month' for consistency in column names
df_weather.rename(columns={'Mon': 'Month'}, inplace=True)

# 5. Merge all data

# First, merge the data from '表1'
df_merge1 = pd.merge(df_main, df1_pivot, on=['Year', 'Month', 'Day', 'Hour'], how='left')

# Then, merge the data from '表2'
df_merge2 = pd.merge(df_merge1, df2_pivot, on=['Year', 'Month', 'Day', 'Hour'], how='left')

# Finally, merge the weather data
df_final = pd.merge(df_merge2, df_weather, on=['Year', 'Month', 'Day', 'Hour'], how='left')

# 6. Export the result
df_final.to_excel('合并后的结果.xlsx', index=False)
