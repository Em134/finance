import helper
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Heiti TC']

data = pd.read_csv('data/收盘价.csv')
df = pd.DataFrame(data)


# # Standardize the dataframe
# scaler = StandardScaler()
# for col in df.columns[1:]:
#     df[col] = scaler.fit_transform(df[[col]])

# 日期格式 convert to datetime format
df['date'] = pd.to_datetime(df['date'])

df_copy = df

# 保留每月的最后一天 only save the last date's data of each month
df = df.groupby(df['date'].dt.to_period('M')).apply(lambda x: x.iloc[-1]).reset_index(drop=True)

for col in df.columns[1:]:
    df[f'{col}_Return'] = df[col].pct_change() * 100  # Calculate percentage change

# Drop NaN values resulting from the pct_change operation
nan_counts = df.isna().sum()
# print(nan_counts)
df.dropna(inplace=True)

# print("Filtered DataFrame:")
# print(df)

# # date and df[]
# plt.figure(figsize=(8, 5))
# plt.plot(df['date'], df['农林牧渔'], marker='o', linestyle='-', label='农林牧渔')
# plt.title('Plot of Feature1')
# plt.xlabel('date')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()

df_copy['农林牧渔_Return'] = df_copy['农林牧渔'].pct_change()
RSTR_example = helper.calculate_RSTR(df_copy['农林牧渔_Return'], np.zeros_like(np.array(df_copy['农林牧渔_Return'])))
RSTR_example = helper.standardize_and_trim(RSTR_example)
print(RSTR_example)
RSTR_example = pd.DataFrame(RSTR_example)
RSTR_example.to_csv('RSTR_example.csv')

# plt.figure(figsize=(8, 5))
# plt.plot(df_copy['date'], RSTR_example, linestyle='-', label='RSTR')
# plt.plot(df_copy['date'], df_copy['农林牧渔'], linestyle='-', label='RSTR')
# plt.title('Plot of RSTR')
# plt.xlabel('date')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()
x = df_copy['date']
y1 = RSTR_example
y2 = df_copy['农林牧渔']
fig, ax1 = plt.subplots()

ax1.plot(x, y1, 'g-', label='RSTR')
ax1.set_xlabel('X data')
ax1.set_ylabel('RSTR', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-', label='农林牧渔')
ax2.set_ylabel('农林牧渔', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Add legends for both lines
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Plot with Two Different Y-Axes')
plt.show()