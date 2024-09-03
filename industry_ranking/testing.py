import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import barra残差动量 as barra
import tushare as ts
import statsmodels.api as sm
import matplotlib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier

# tstoken = ""
# ts.set_token(token)
# pro = ts.pro_api()
#
# code = '000059.SZ'
# df = pro.balancesheet(ts_code=code, start_date='20140101', end_date='20240701',\
#                       fields='ts_code,f_ann_date,total_ncl,total_assets,total_liab,oth_eqt_tools_p_shr')
# df = df.drop_duplicates()
# df.index = pd.to_datetime(df['f_ann_date'])
# print(df)

'''行业轮动和多头策略/空头策略的收益率的关系'''


# 探究行业轮动速度，与多头策略，空头策略收益率之间的关系
'''线性回归'''
# # short
# y = output['short_value_return']
# X = output[['all_ind', 'speed']]
# X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# print('short value return: ')
# print(model.summary())
# # long
# y = output['long_value_return']
# X = output[['all_ind', 'speed']]
# X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# print('long value return')
# print(model.summary())

'''决策树'''
# # short
# y = np.where(output['short_value_return'] > 0, 1, 0)
# X = output[['all_ind', 'speed']]
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# # 创建并训练决策树回归模型
# tree_model = DecisionTreeRegressor(random_state=42, max_depth=1)
# tree_model.fit(X_train, y_train)
# # 预测测试集
# y_pred = tree_model.predict(X_test)
# # 计算评价指标
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print('short_value_return: ')
# print("Mean Squared Error (MSE):", mse)
# print("R-squared (R²):", r2)
# plt.figure(figsize=(12, 8))
# plot_tree(tree_model, feature_names=['all_ind', 'speed'], filled=True)
# plt.show()
# plt.close()
# # long
# y = np.where(output['long_value_return'] > 0, 1, 0)
# X = output[['all_ind', 'speed']]
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# # 创建并训练决策树回归模型
# tree_model = DecisionTreeRegressor(random_state=42, max_depth=1)
# tree_model.fit(X_train, y_train)
# # 预测测试集
# y_pred = tree_model.predict(X_test)
# # 计算评价指标
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print('long_value_return: ')
# print("Mean Squared Error (MSE):", mse)
# print("R-squared (R²):", r2)
# plt.figure(figsize=(12, 8))
# plot_tree(tree_model, feature_names=['all_ind', 'speed'], filled=True)
# plt.show()

'''logistic regression'''
# # short
# y = np.where(output['short_value_return'] > 0, 1, 0)
# X = output[['all_ind', 'speed']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_train, y_train)
# y_pred = log_reg.predict(X_test)
#
# # 计算评估指标
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
#
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
#
# # 打印混淆矩阵和分类报告
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
#
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

'''KNN'''
# # short
# y = np.where(output['short_value_return'] > 0, 1, 0)
# X = output[['all_ind', 'speed']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# # 预测测试集
# y_pred = knn.predict(X_test)
# # 计算评估指标
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# # 打印混淆矩阵和分类报告
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

'''行业轮动速度和策略收益率之间的关系'''
def calculate_percentile(value, values):
    # Sort the list of values
    values = list(values)
    values.append(value)
    sorted_values = sorted(values)

    # Find the rank of the value in the sorted list
    rank = sorted_values.index(value) + 1

    # Calculate the percentile
    percentile = ((rank - 0.5) / len(sorted_values)) * 100

    return percentile


close = pd.read_csv('data/申万31大行业_close.csv', index_col=0, parse_dates=True)
close = close.resample('ME').last()
speed = barra.rotation_speed(close)

output = pd.read_csv('barra_output.csv', index_col=0, parse_dates=True)

result = pd.DataFrame(None, index=output.index[5:])

# result['short_value'] = output['short_value'].rolling(window=6).mean()
# result['long_value'] = output['long_value'].rolling(window=6).mean()
result['short_value'] = output['short_value']
result['long_value'] = output['long_value']
result['all_ind'] = output['all_ind'].rolling(window=6).mean()

result['short_return'] = result['short_value'].pct_change()
result['long_return'] = result['long_value'].pct_change()
result['all_return'] = result['all_ind'].pct_change()
result['speed'] = -1

result = result.iloc[1:, :]

# print(result['short_return'])
# print(result['long_return'])

# calculating dummy for speed, using historical data!
for date in result.index:
    prev_speed = speed[speed.index < date]
    percentile = calculate_percentile(output.loc[date, 'speed'], prev_speed)
    print(date, percentile)
    if percentile > 70:
        result.loc[date, 'speed'] = 1
    else:
        result.loc[date, 'speed'] = 0

result['short_return'] = (result['short_return'] > 0).astype(int)
result['long_return'] = (result['long_return'] > 0).astype(int)
result['all_return'] = (result['all_return'] > 0).astype(int)

result.to_csv('result.csv')

plt.figure(figsize=(10, 6))
# 创建主轴
fig, ax1 = plt.subplots(figsize=(10, 6))
# 绘制第一组数据（左轴）
ax1.plot(output.index, output['all_ind'], linestyle='-', label='全部行业', color='blue')
# ax1.plot(output.index, output[col_name], linestyle='-', label=col_name+'策略', color='green')
ax1.plot(output.index, output['long_value'], linestyle='-', label='long策略', color='pink')
ax1.plot(output.index, output['short_value'], linestyle='-', label='short策略', color='black')
# ax1.plot(output.index, output[col_name] - output['all_ind'], linestyle='-', label='累计超额', color='orange')
# ax1.plot(output.index, output_pct['超额'], linestyle='-', label='当月超额', color='red')
# 设置标题和标签
ax1.set_title('Barra行业轮动')
ax1.set_xlabel('date')
ax1.set_ylabel('net value')
# 添加左轴的图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
# 创建第二个y轴
ax2 = ax1.twinx()
# 绘制第二组数据（右轴）
ax2.plot(output.index, output['speed'], linestyle='-', label='轮动速度', color='purple')
# # 标数据点
# for i, (x, y) in enumerate(zip(output.index, output['speed'])):
#     ax2.annotate(f'{y:.2f}', xy=(x, y), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)
# 设置右轴的标签
ax2.set_ylabel('轮动速度')
# 添加右轴的图例
lines_2, labels_2 = ax2.get_legend_handles_labels()
# 合并图例
lines = lines_1 + lines_2
labels = labels_1 + labels_2
ax1.legend(lines, labels, loc='best')
ax1.grid(True)
plt.show()

result = pd.read_csv('result.csv', index_col=0, parse_dates=True)
table = pd.DataFrame(None, index=['all_0', 'all_1'], columns=['slow', 'fast'])

for i in table.index:
    for j in table.columns:
        table.loc[i, j] = [0, 0]

for date in result.index:
    temp = result.loc[date]

    row = int(temp.loc['all_return'])
    col = int(temp.loc['speed'])
    _return = int(temp.loc['short_return'])
    # print(row, col, _return)
    (table.iloc[row, col])[_return] += 1


print(table)




'''dataframe 示例'''
# # Create example DataFrames
# df1 = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4, 5, 6]
# }, index=['2023-06-19', '2023-06-20', '2023-06-21'])
#
# df2 = pd.DataFrame({
#     'C': [7, 8, 9, 10],
#     'D': [10, 11, 12, 13]
# }, index=['2023-06-19', '2023-06-20', '2023-06-21', '2023-06-22'])
#
# # Convert index to datetime
# df1.index = pd.to_datetime(df1.index)
# df2.index = pd.to_datetime(df2.index)
#
# # Join df2 to df1 by df1's index
# result = df1.join(df2)
# result = df1.iloc[1]
#
# print("DataFrame 1:")
# print(df1)
# print("\nDataFrame 2:")
# print(df2)
# print("\nJoined DataFrame:")
# print(df1.rank(axis=1, method='min', ascending=False))

'''行业轮动速度'''
# close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
# close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
# close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])
#
# speed = barra.rotation_speed(close)
# close = close.loc[speed.index]
# close['total'] = close.mean(axis=1)
# pct = close.pct_change()
# close['speed'] = np.array(speed)
#
# # pct = pct.rolling(3).mean()
# # close['speed'] = close['speed'].rolling(3).mean()
#
# print('correlation:', pct['total'].corr(close['speed']))
# print('correlation:', pct['total'][1:].corr(close['speed'][0:-1]))
# close.index = close.index.to_timestamp()
#
# plt.figure(figsize=(10, 6))
# # 创建主轴
# fig, ax1 = plt.subplots(figsize=(10, 6))
# # 绘制第一组数据（左轴）
# ax1.plot(close.index, close['total'], linestyle='-', label='全部行业', color='blue')
# # ax1.plot(output.index, output['value'], linestyle='-', label='策略', color='green')
# # ax1.plot(output.index, output['value'] - output['all_ind'], linestyle='-', label='累计超额', color='orange')
# # ax1.plot(output.index, output_pct['超额'], linestyle='-', label='当月超额', color='red')
# # 设置标题和标签
# ax1.set_title('Barra行业轮动')
# ax1.set_xlabel('date')
# ax1.set_ylabel('Value')
# # 添加左轴的图例
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# # 创建第二个y轴
# ax2 = ax1.twinx()
# # 绘制第二组数据（右轴）
# ax2.plot(close.index, close['speed'], linestyle='-', label='轮动速度', color='purple')
# # # 标数据点
# # for i, (x, y) in enumerate(zip(output.index, output['speed'])):
# #     ax2.annotate(f'{y:.2f}', xy=(x, y), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)
# # 设置右轴的标签
# ax2.set_ylabel('轮动速度')
# # 添加右轴的图例
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# # 合并图例
# lines = lines_1 + lines_2
# labels = labels_1 + labels_2
# ax1.legend(lines, labels, loc='best')
# ax1.grid(True)
# plt.show()

'''debttoasset and leverage'''
# industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
# # Values to remove
# values_to_remove = ['银行(申万)', '非银金融(申万)']
# # Remove values using difference()
# industries = industries.difference(values_to_remove)
#
# for ind_name in industries:
#     sliced_ind_name = ind_name[:-4]
#     # print("calculating barra factors: " + sliced_ind_name)
#     # 建表
#     data = pd.read_csv('data/barra/sliced_csv/申万_' + sliced_ind_name + '.csv', index_col=0)
#     data = data.fillna(method='ffill')  # 把中间空的补上
#     data = data.fillna(0)  # 把最开始的空缺补上
#     data.index = pd.to_datetime(data.index)
#
#     data = data.loc[:, ['debttoassets']]
#     data = data.groupby(data.index.to_period('Q')).last()
#
#     leverage_data = pd.read_csv('data/风格因子/barra风格_' + ind_name + '.csv', index_col=0)
#     leverage_data = leverage_data.loc[:, ['杠杆']]
#     leverage_data.index = pd.to_datetime(leverage_data.index)
#
#     leverage_data = leverage_data.groupby(leverage_data.index.to_period('Q')).last()
#
#     data = data.loc[leverage_data.index]
#
#     print(ind_name, data['debttoassets'].corr(leverage_data['杠杆']))

'''季频因子与计算的因子的相关性'''
# industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
# # Values to remove
# values_to_remove = ['银行(申万)', '非银金融(申万)']
# # Remove values using difference()
# industries = industries.difference(values_to_remove)
# # ['beta', 'mom', 'size', 'earning', 'dastd', 'residualVolatility', 'bookToPrice', 'leverage', 'liquidity', 'nonLinearSize']
# factor_name1 = 'mom'
# factor_name2 = '动量'
#
# for ind_name in industries:
#     sliced_ind_name = ind_name[:-4]
#     # print("calculating barra factors: " + sliced_ind_name)
#     # 建表
#     data = pd.read_csv('data/barra/factor_values_stdized/barra_' + ind_name + '.csv', index_col=0)
#     data.index = pd.to_datetime(data.index)
#
#     data = data.loc[:, [factor_name1]]
#     data = data.groupby(data.index.to_period('Q')).last()
#
#     bara_data = pd.read_csv('data/风格因子/barra风格_' + ind_name + '.csv', index_col=0)
#     bara_data = bara_data.loc[:, [factor_name2]]
#     bara_data.index = pd.to_datetime(bara_data.index)
#
#     bara_data = bara_data.groupby(bara_data.index.to_period('Q')).last()
#
#     data = data.loc[bara_data.index]
#
#     print(ind_name, data[factor_name1].corr(bara_data[factor_name2]))
