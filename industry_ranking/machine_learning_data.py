import numpy as np
import pandas as pd
import matplotlib
import warnings
# ML packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

matplotlib.rcParams['font.family'] = ['Heiti TC']
# warnings.filterwarnings("ignore")


'''数据处理 *——————————————————————————————————————————————————————————————'''
# # 把因子值改成rank，并标准化：
# def rank_std_df(data: pd.DataFrame):
#     scaler = StandardScaler()
#     a = np.array(data)
#     a_scaled = scaler.fit_transform(a.T)
#     data.loc[:, :] = a_scaled.T
#
#     return data
#
#
# close = pd.read_csv('data/申万31大行业_close.csv', index_col=0, parse_dates=True)
# returns = close.pct_change(periods=22)  # 月收益率
# returns = returns.shift(-22)  # 接下来一个月的收益率
#
#
# sample1 = pd.read_csv('data/barra/factor_values/barra_交通运输(申万).csv', index_col=0, parse_dates=True)
# barra_factors = list(sample1.columns)
#
# other_factors_normal = ['index_growth', 'tech_concentric', 'big_break', 'herd_behavior']
# other_factors_ranked = ['profit_growth']
# all_other_factors = other_factors_normal.copy()
# all_other_factors.extend(other_factors_ranked)
#
# factors_list = barra_factors.copy().extend(all_other_factors)
#
# dataframes = []
#
# returns.columns = returns.columns.str.replace(r'\(.*', '', regex=True).str.strip()
# returns.drop(columns=['银行', '非银金融'], inplace=True)
# melted_df = returns.melt(ignore_index=False, var_name='ind_name', value_name='return_m')
# melted_df.reset_index(inplace=True)
# melted_df.columns = ['Date', 'ind_name', 'return_m']
# dataframes.append(melted_df)
#
# for fac in other_factors_normal:
#     df = pd.read_csv('data/其他因子数据/处理后/' + fac + '.csv', index_col=0, parse_dates=True)
#     df.columns = df.columns.str.replace(r'\(.*', '', regex=True).str.strip()
#     melted_df = df.melt(ignore_index=False, var_name='ind_name', value_name=fac)
#     melted_df.reset_index(inplace=True)
#     melted_df.columns = ['Date', 'ind_name', fac]
#     dataframes.append(melted_df)
#
# for fac in other_factors_ranked:
#     df = pd.read_csv('data/其他因子数据/处理后/' + fac + '.csv', index_col=0, parse_dates=True)
#     df.columns = df.columns.str.replace(r'\(.*', '', regex=True).str.strip()
#     df = rank_std_df(df)
#     melted_df = df.melt(ignore_index=False, var_name='ind_name', value_name=fac)
#     melted_df.reset_index(inplace=True)
#     melted_df.columns = ['Date', 'ind_name', fac]
#     dataframes.append(melted_df)
#
# print(barra_factors)
# for fac in barra_factors:
#     if fac == 'leverage':
#         df = pd.read_csv('data/杠杆因子/申万一级月频杠杆因子.csv', index_col=0, parse_dates=True)
#         df.drop(columns=['银行(申万)', '非银金融(申万)'], inplace=True)
#     else:
#         df = pd.read_csv('data/barra/standardized_factors/barra_' + fac + '.csv', index_col=0, parse_dates=True)
#     df.columns = df.columns.str.replace(r'\(.*', '', regex=True).str.strip()
#     melted_df = df.melt(ignore_index=False, var_name='ind_name', value_name=fac)
#     melted_df.reset_index(inplace=True)
#     melted_df.columns = ['Date', 'ind_name', fac]
#     dataframes.append(melted_df)
#
# merged_df = dataframes[0]
# for df in dataframes[1:]:
#     merged_df = pd.merge(merged_df, df, on=['Date', 'ind_name'], how='left')
#
# # merged_df.drop(columns='Date', inplace=True)
# # merged_df.reset_index(drop=True, inplace=True)
#
# # Display the final merged DataFrame
# print(merged_df)
# print(merged_df['Date'].iloc[0], merged_df['Date'].iloc[-1])
# merged_df = merged_df.ffill().dropna()
# merged_df = merged_df[merged_df['Date'] <= pd.to_datetime('2024-03-31')]
# print(merged_df)
# print(merged_df['Date'].iloc[0], merged_df['Date'].iloc[-1])
#
# print(merged_df['ind_name'].unique())
#
# merged_df.to_csv('ML_merged_df.csv')
'''数据处理结束 *——————————————————————————————————————————————————————————————'''

''' ML *——————————————————————————————————————————————————————————————'''
original_date = pd.read_csv('ML_merged_df.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
data = original_date.drop(columns=['Date'])
data = pd.get_dummies(data, columns=['ind_name'], drop_first=True)

X = data.drop(columns=['return_m'])
y = data['return_m']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# NN
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")

from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {test_mse:.4f}")
import numpy as np
test_rmse = np.sqrt(test_mse)
print(f"Test RMSE: {test_rmse:.4f}")
from sklearn.metrics import r2_score
test_r2 = r2_score(y_test, y_pred)
print(f"Test R-squared: {test_r2:.4f}")
import matplotlib.pyplot as plt
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

history = model.history.history
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_list = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    mse_list.append(mse)

print(f"Average MSE from Cross-Validation: {np.mean(mse_list):.4f}")

''' ML *——————————————————————————————————————————————————————————————'''
