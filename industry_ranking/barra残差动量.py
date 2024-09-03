""" Jun 19, 2024, Guangda Fei """
import helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier

matplotlib.rcParams['font.family'] = ['Heiti TC']
import warnings
warnings.filterwarnings("ignore")


# 计算行业轮动速度
# 上一个月的前10名，它们在本月的排名。两者相减取绝对值后求和，取上六个月（包括本月）平均值为本月的行业轮动速度。
# def rotation_speed(close):
#     close = close.pct_change().dropna()
#     pct = close.copy()  # 本月收益率
#     close = close.rank(axis=1, method='min', ascending=False)
#     top_stocks = pd.Series(get_top_stocks(pct, 10), index=close.index)  # 已验证
#     speed = pd.Series(None, index=close.index)
#     for i in range(len(close.index)):
#         # 上个月排名前几，在本月的名次。相减取绝对值再求和
#         if i == 0:
#             continue
#         last_date = close.index[i-1]
#         date = close.index[i]
#         stocks = top_stocks[last_date]  # 上个时间段的前几名
#         speed[date] = abs(close.loc[date, stocks] - close.loc[last_date, stocks]).sum()
#
#         # print(date, np.array(pct.loc[date, stocks]).sum()/10)  # 本月的top收益率之和
#
#     speed = speed.dropna().rolling(window=6).mean()
#     # speed = speed.dropna().rolling(window=6).apply(lambda x: helper.calculate_ewma(x, 2)[-1])
#     return speed

# 基于NDCG的行业轮动速度
# def rotation_speed(close):
#     close = close.pct_change().dropna()
#     # pct = close.copy()  # 本月收益率
#     # close = close.rank(axis=1)
#     close = close + 1
#     speed = pd.Series(None, index=close.index)
#     for i in range(len(close.index)):
#         if i == 0:
#             continue
#
#         speed[close.index[i]] = 1-ndcg_score([close.iloc[i - 1]], [close.iloc[i]], k=10)
#         # print(close.index[i], speed[close.index[i]])
#     speed = speed.dropna().rolling(window=3).mean()
#     # speed = speed.dropna().rolling(window=6).apply(lambda x: helper.calculate_ewma(x, 3)[-1])
#     return speed


# 基于RRG的行业轮动速度
def rotation_speed(close):
    scaler = StandardScaler()
    rs_ratio = pd.DataFrame(None, index=close.index, columns=close.columns)

    for date in rs_ratio.index:
        rs_ratio.loc[date] = close.loc[date] / close.loc[date].mean() * 100

    rs_mom = rs_ratio.pct_change().fillna(0) * 100 + 100

    # temp = np.array(rs_ratio)
    # temp_scaled = scaler.fit_transform(temp.T)
    # rs_ratio.loc[:, :] = temp_scaled.T
    #
    # temp = np.array(rs_mom)
    # temp_scaled = scaler.fit_transform(temp.T)
    # rs_mom.loc[:, :] = temp_scaled.T

    speed = pd.Series(0, index=rs_ratio.index)
    for i in range(len(rs_ratio.index)):
        if i == 0:
            continue
        date = rs_ratio.index[i]
        prev_date = rs_ratio.index[i-1]
        for stock in rs_ratio.columns:
            speed[date] += np.sqrt(np.square(rs_ratio.loc[date, stock] - rs_ratio.loc[prev_date, stock]) +
                                   np.square(rs_mom.loc[date, stock] - rs_mom.loc[prev_date, stock]))

    speed = speed.rolling(12).mean()

    '''Plotting'''
    # rs_ratio = rs_ratio.loc[rs_ratio.index[-12:], :]
    # rs_mom = rs_mom.loc[rs_ratio.index, :]
    # plt.figure(figsize=(10, 8))
    #
    # # Iterate over each stock to plot and connect dots
    # for stock in rs_ratio.columns[:6]:
    #     plt.plot(rs_ratio[stock], rs_mom[stock], marker='o', label=stock)
    #
    # # Add labels and legend
    # plt.xlabel('RS Ratio')
    # plt.ylabel('RS Momentum')
    # plt.title('RS Ratio vs RS Momentum for Stocks Over Time')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.show()

    return speed


# 基于收益率计算的行业分歧度
def secotr_deviation(close: pd.DataFrame):
    close = close.pct_change().fillna(0)

    return close.std(axis=1)


# np.random.seed(42)  # For reproducibility
# data = {
#     'Industry_A': np.random.rand(5) * 100,
#     'Industry_B': np.random.rand(5) * 100,
#     'Industry_C': np.random.rand(5) * 100
# }
#
# # Create a DataFrame
# close_prices = pd.DataFrame(data)
#
# # Print the test data
# print("Test Data (Closing Prices):")
# print(close_prices)
#
# # Call the rotation_speed function
# result = secotr_deviation(close_prices)
#
# # Print the output of the function
# print("\nRotation Speed (Standard Deviation of Returns):")
# print(result)


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


# 选择分数最高的
def get_top_stocks(scores_df, n):
    top_stocks = scores_df.apply(lambda x: x.nlargest(n).index.tolist(), axis=1)
    return top_stocks


# 选择分数最低的
def get_least_stocks(scores_df, n):
    least_stocks = scores_df.apply(lambda x: x.nsmallest(n).index.tolist(), axis=1)
    return least_stocks


# 各行业指数收盘价: close，各行业指标分数: score, 日期在index, 保证两个dataframe日期频率一致
# 每月选取排名前ranking_num个行业，等权做多。等权持有全部行业为基准
#        排名后ranking_num个行业，等权做空

def ranking_backtest(close, score, ranking_num=5):
    # 基准
    close_copy = close.copy()
    close['all_ind'] = close.mean(axis=1)
    # value
    # 计算加权月均score，用ewma半衰期10
    # for i in score.columns:
    #     score[i] = helper.calculate_ewma(score[i], 10)
    # 计算上个月平均分
    # score = score.rolling(window=21).mean().dropna()
    monthly_score = score.copy()

    # data 处理完毕
    long_stocks = get_top_stocks(monthly_score, ranking_num)  # 月底获取上一个月月排名前5，第二天买入（开盘价）直到下一个月月底（收盘价）
    short_stocks = get_least_stocks(monthly_score, ranking_num)
    monthly_score['long_value'] = 1
    monthly_score['long_value_return'] = 0  # 本月相比于上个月的增长
    monthly_score['short_value'] = 1
    monthly_score['short_value_return'] = 0  # 本月相比于上个月的增长
    monthly_score['long_short'] = 1  # 多空净值

    close_r6 = close['all_ind'].rolling(6).mean().pct_change()
    temp_close = pd.read_csv('data/申万31大行业_close.csv', index_col=0, parse_dates=True)
    temp_close = temp_close.groupby(temp_close.index.to_period('M')).apply(lambda x: x.iloc[-1])
    speed = rotation_speed(temp_close)

    for i in range(len(monthly_score)):
        date = monthly_score.index[i]
        # print(date)
        if i+1 < len(monthly_score):
            next_date = monthly_score.index[i+1]
        else:
            # next_date = monthly_score.index[-1]
            break
        picked_long_stocks = long_stocks[i]  # tested, picked_stocks is a list of column names
        picked_short_stocks = short_stocks[i]

        # long
        value_start = close.loc[date, picked_long_stocks].mean()  # 选中行业的 最后一天的 收盘价的平均
        value_end = close.loc[next_date, picked_long_stocks].mean()
        month_return = float(value_end-value_start)/value_start  # 下个月收益率
        monthly_score.loc[next_date, 'long_value'] = (1+month_return)*monthly_score.loc[date, 'long_value']  # 净值
        monthly_score.loc[next_date, 'long_value_return'] = month_return
        # print('long_return:', close.loc[next_date, picked_long_stocks] / close.loc[date, picked_long_stocks] - 1)
        # short
        value_start = close.loc[date, picked_short_stocks].mean()  # 选中行业的 最后一天的 收盘价的平均
        value_end = close.loc[next_date, picked_short_stocks].mean()
        month_return = -float(value_end - value_start) / value_start  # 下个月收益率
        # print('short_return:', close.loc[next_date, picked_short_stocks] / close.loc[date, picked_short_stocks] - 1)
        monthly_score.loc[next_date, 'short_value'] = (1 + month_return) * monthly_score.loc[date, 'short_value']  # 净值
        monthly_score.loc[next_date, 'short_value_return'] = month_return

        # long_short 计算多空策略收益，80%long, 80%short
        # 通过历史市场收益和行业轮动速度进行择时：
            # 市场上涨时：做多
            # 市场下跌：做空，如果速度快也做多50%
        prev_speed = speed[speed.index < date]
        percentile = calculate_percentile(speed.loc[date], prev_speed)
        weight_short = 1
        weight_long = 1
        # if close_r6.loc[date] <= 0:  # bear (通过当前all in收益率）
        #     if percentile <= 70:  # slow
        #         weight_long = 0
        # else:  # bull
        #     # if percentile <= 60:  # slow
        #     weight_short = 0
        # print(date, weight_short, weight_long)
        monthly_score.loc[next_date, 'long_short'] = (1 + 0.8 * (weight_short * monthly_score.loc[next_date, 'short_value_return'] +
                                                                 weight_long * monthly_score.loc[next_date, 'long_value_return'])) * \
                                                                 monthly_score.loc[date, 'long_short']


    # 全部行业
    monthly_score = monthly_score.join(close.loc[:, ['all_ind']])
    monthly_score['all_ind'] = monthly_score['all_ind']/float(monthly_score.all_ind[0])  # start from 1

    # 计算行业轮动速度
    speed = rotation_speed(close_copy).loc[monthly_score.index]
    monthly_score['speed'] = np.array(speed)

    # 计算行业分歧度
    monthly_score['deviation'] = secotr_deviation(close_copy).loc[monthly_score.index]



    return monthly_score.loc[:, ['all_ind', 'long_value', 'short_value', 'speed', 'long_short', 'short_value_return',
                                 'long_value_return', 'deviation']]


def extract_data(ind_name):
    # find file using ind_name
    directory = 'data/风格因子/'
    files = [f for f in os.listdir(directory) if ind_name in f and f.endswith('.xlsx')]
    if files:
        # Use the first result
        file_path = os.path.join(directory, files[0])

        # Read the third sheet (index 2), skip the second row (index 1)
        data = pd.read_excel(file_path, sheet_name=2, skiprows=[1], index_col=0)
        data = data.rename_axis('date')
        data.index = pd.to_datetime(data.index, format='%Y%m%d')
        # Display the DataFrame
        return data
    else:
        print("No files found matching the pattern：" + ind_name)
        return None


# 把工业指数收益率加到barra因子文件里，index是月频格式
def get_score_helper(ind_name):
    close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
    close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])  # 读收盘价然后调成月频

    score = pd.read_csv('data/barra/factor_values_stdized/barra_' + ind_name + '.csv', index_col=0)  # 读barra因子调成月频
    score.index = pd.to_datetime(score.index, format='%Y-%m-%d')
    score = score.groupby(score.index.to_period('M')).apply(lambda x: x.iloc[-1])
    # print(close)
    # print(score)

    close = close.loc[:, [ind_name]].pct_change().fillna(0)
    filtered_close = close.loc[score.index]
    score['return'] = np.array(filtered_close[ind_name])  # 把同期涨跌幅加到barra因子文件里

    return score


def get_score(industries):  # 过去12个月回归法预测收益率
    sample1 = pd.read_csv('data/barra/factor_values_stdized/barra_' + '交通运输(申万)' + '.csv', index_col=0)
    sample1.index = pd.to_datetime(sample1.index)
    sample1 = sample1.groupby(sample1.index.to_period('M')).apply(lambda x: x.iloc[-1])  # 读barra文件调成月频格式用于新建score df
    # print(sample1)
    score = pd.DataFrame(0, index=sample1.index, columns=industries)  # 新建score df
    # print(score)

    IC = pd.DataFrame()
    for i in industries:  # 循环工业
        data = get_score_helper(i)  # data是因子及收益率

        print(i)  # 计算barra的IC值, print and save
        for factor in data.columns:
            corr = data[factor][:-1].corr(data['return'][1:]) # 因子和下一期的收益率进行比较，结果不错
            print('\t', factor, 'IC:', corr)
            IC.loc[factor, i] = corr

        for j in range(11, len(data)):  # 开始循环每一天：回归并取残差，将残差放到score df里
            df_ne = data.iloc[j - 11:j+1, :]
            X = df_ne.drop(columns=['return'])
            # X = df_ne.drop(columns=['return'])
            y = df_ne['return']
            # Create results object and fit the model
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            score.loc[df_ne.index[-1], i] = y[-1] - y_pred[-1]

    IC.to_csv('data/barra/barra_IC.csv')  # 存储IC值
    return score
    # print(score)


def report(output):
    col_name = 'long_short'

    output_pct = output.pct_change().fillna(0)
    output_pct['超额'] = output_pct[col_name]-output_pct['all_ind']
    # print(output_pct)
    output.to_csv('data/output/barra残差动量/output_' + '12seasons' + '.csv')
    output.index = output.index.to_timestamp()
    # output = pd.read_csv('output_' + score_name + '.csv')

    yearly_return = helper.calculate_annualized_return_monthly(output[col_name])
    print('策略年化收益：', yearly_return)
    yearly_volatility = helper.calculate_annualized_volatility(output[col_name], 12)
    print('策略年化波动率：', yearly_volatility)
    max_draw = helper.calculate_max_drawdown(output[col_name])
    print('策略最大回撤：', max_draw)
    print('策略夏普：', yearly_return/yearly_volatility)

    print()
    yearly_return = helper.calculate_annualized_return_monthly(output['all_ind'])
    print('基准年化收益：', yearly_return)
    yearly_volatility = helper.calculate_annualized_volatility(output['all_ind'], 12)
    print('基准年化波动率：', yearly_volatility)
    max_draw = helper.calculate_max_drawdown(output['all_ind'])
    print('基准最大回撤：', max_draw)
    print('基准夏普：', yearly_return / yearly_volatility)

    print('\n超额统计：')
    count_greater_than_zero = np.sum(np.array(output_pct['超额']) > 0)
    count_smaller_than_zero = np.sum(np.array(output_pct['超额']) < 0)
    print('超额>0:', count_greater_than_zero)
    print('超额<0:', count_smaller_than_zero)
    print('胜率：', count_greater_than_zero/float(len(output_pct.index)-1))  # -1是为了除去第一天（都是1）

    # 探究行业轮动速度，与多头策略，空头策略收益率之间的关系
    '''线性回归'''
    # short
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

    plt.figure(figsize=(10, 6))
    # 创建主轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 绘制第一组数据（左轴）
    ax1.plot(output.index, output['all_ind'], linestyle='-', label='全部行业', color='blue')
    ax1.plot(output.index, output[col_name], linestyle='-', label=col_name+'策略', color='black')
    # ax1.plot(output.index, output['long_value'], linestyle='-', label='long策略', color='pink')
    # ax1.plot(output.index, output['short_value'], linestyle='-', label='short策略', color='green')
    ax1.plot(output.index, output[col_name] - output['all_ind'], linestyle='-', label='累计超额', color='orange')
    # ax1.plot(output.index, output_pct['超额'], linestyle='-', label='当月超额', color='red')
    # 设置标题和标签
    ax1.set_title('Barra行业轮动')
    ax1.set_xlabel('date')
    ax1.set_ylabel(col_name)
    # 添加左轴的图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    # 创建第二个y轴
    ax2 = ax1.twinx()
    # 绘制第二组数据（右轴）
    # ax2.plot(output.index, output['speed'], linestyle='-', label='轮动速度', color='purple')
    # # 标数据点
    # for i, (x, y) in enumerate(zip(output.index, output['speed'])):
    #     ax2.annotate(f'{y:.2f}', xy=(x, y), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)
    # 设置右轴的标签
    # ax2.set_ylabel('轮动速度')
    # 添加右轴的图例
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # 合并图例
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, loc='best')
    ax1.grid(True)
    plt.show()

    output.to_csv('barra_output.csv')


# industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
# # Values to remove
# values_to_remove = ['银行(申万)', '非银金融(申万)']
# # Remove values using difference()
# industries = industries.difference(values_to_remove)
# # # for i in industries:
# # #     data = extract_data(i)
# # #     data.to_csv('data/风格因子/barra风格_' + i + '.csv')
# #
# close = pd.read_csv('data/申万31大行业_close.csv', index_col=0, parse_dates=True)
# close = close.resample('ME').last()
#
# dev = secotr_deviation(close)
# speed = rotation_speed(close)
#
# df = pd.concat([dev, speed], axis=1)
# df.columns = ['dev', 'speed']
# # df['dev'] = df['dev'].rolling(6).mean()
# print('corr is', df['dev'].corr(df['speed']))
# df = df.iloc[5:, :]

# print(rotation_speed(close))
# score = get_score(industries)
# score = score.iloc[11:, :]
#
# std_scaler = StandardScaler()
# scaled_score = pd.DataFrame(std_scaler.fit_transform(score), columns=score.columns, index=score.index)
#
# output = ranking_backtest(close, score)
# report(output)

# plt.figure(figsize=(12, 6))
# # 创建主轴
# fig, ax1 = plt.subplots(figsize=(10, 6))
# # 绘制第一组数据（左轴）
# ax1.plot(df.index, df['speed'], linestyle='-', label='speed', color='blue')
# ax1.set_title('speed and deviation')
# ax1.set_xlabel('date')
# ax1.set_ylabel('speed')
# # 添加左轴的图例
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# # 创建第二个y轴
# ax2 = ax1.twinx()
# # 绘制第二组数据（右轴）
# ax2.plot(df.index, df['dev'], linestyle='-', label='dev', color='purple')
# # # 标数据点
# # for i, (x, y) in enumerate(zip(output.index, output['speed'])):
# #     ax2.annotate(f'{y:.2f}', xy=(x, y), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)
# # 设置右轴的标签
# ax2.set_ylabel('dev')
# # 添加右轴的图例
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# # 合并图例
# lines = lines_1 + lines_2
# labels = labels_1 + labels_2
# ax1.legend(lines, labels, loc='best')
# ax1.grid(True)
# plt.show()
