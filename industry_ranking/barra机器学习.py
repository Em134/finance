import helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
import os
matplotlib.rcParams['font.family'] = ['Heiti TC']
import warnings
import barra残差动量 as barra
warnings.filterwarnings("ignore")


# 把后一期的月收益率 添加到数据score里，频率为月M or 周W
def get_dataset(ind_name):
    close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
    close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])  # 读收盘价然后调成月频/周频

    score = pd.read_csv('data/barra/factor_values/barra_' + ind_name + '.csv', index_col=0)  # 读barra因子调成月频/周频
    score.index = pd.to_datetime(score.index, format='%Y-%m-%d')
    # score = score.groupby(score.index.to_period('M')).apply(lambda x: x.iloc[-1])
    score = score.groupby(score.index.to_period('M')).last()

    score = score.ffill().fillna(0)


    close = close.loc[:, [ind_name]].pct_change().fillna(0)
    close[ind_name] = close[ind_name].shift(-1)  # 收益率移动一期
    filtered_close = close.loc[score.index]
    score['return'] = np.array(filtered_close[ind_name])  # 把下一期涨跌幅加到barra因子文件里
    score = score.fillna(0)
    # print(score)
    # print('empty: ', score.isnull().sum())
    return score

def get_prediction(industries):
    sample1 = pd.read_csv('data/barra/factor_values/barra_' + '交通运输(申万)' + '.csv', index_col=0)
    sample1.index = pd.to_datetime(sample1.index)
    sample1 = sample1.groupby(sample1.index.to_period('M')).apply(lambda x: x.iloc[-1])  # 读barra文件调成月频格式用于新建score df
    # print(sample1)
    score = pd.DataFrame(0, index=sample1.index, columns=industries)  # 新建score df

    leverage_data = pd.read_csv('data/杠杆因子/申万一级月频杠杆因子.csv', index_col=0)
    leverage_data.index = pd.to_datetime(leverage_data.index)
    leverage_data = leverage_data.groupby(leverage_data.index.to_period('M')).last()
    # 月频降到季频再填充回月频
    leverage_data_copy = leverage_data.copy()
    leverage_data_copy['date'] = leverage_data_copy.index
    leverage_data_copy = leverage_data_copy.resample('Q').last()
    new_lev_data = pd.DataFrame(None, index=leverage_data.index, columns=leverage_data.columns)
    for lev_q in leverage_data_copy.index:
        lev_date = leverage_data_copy.loc[lev_q, 'date']
        new_lev_data.loc[lev_date] = leverage_data_copy.loc[lev_q]

    new_lev_data = new_lev_data.ffill().fillna(0)
    leverage_data = new_lev_data

    # 添加行业轮动速度
    close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
    close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])

    speed = barra.rotation_speed(close)
    for i in industries:  # 循环工业
        data = get_dataset(i)  # data是因子及收益率
        data['leverage'] = leverage_data.loc[data.index, i]  # 替换leverage data
        # data['speed'] = np.array(speed.loc[data.index])  # sector rotation speed
        # ['beta', 'mom', 'size', growth, 'residualVolatility', 'growth', 'bookToPrice', 'leverage', 'liquidity', 'nonLinearSize'])
        # data = data.drop(columns=['leverage'])
        # data['leverage'] = -data['leverage']
        # data['bookToPrice'] = -data['bookToPrice']
        # data['earning'] = -data['earning']

        # print(i)  # 计算barra的IC值, print and save
        # for factor in data.columns:
        #     corr = data[factor][:-1].corr(data['return'][1:]) # 因子和下一期的收益率进行比较，结果不错
        #     print('\t', factor, 'IC:', corr)
        #     IC.loc[factor, i] = corr

        for j in range(12, len(data)):  # 开始循环每一天：回归并预测
            df_ne = data.iloc[j - 12:j, :]
            X = df_ne.drop(columns=['return'])
            y = df_ne['return']
            # Create results object and fit the model
            model = LinearRegression()
            # degree = 2
            # model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            # model = BayesianRidge()
            model.fit(X, y)
            y_pred = model.predict(data.iloc[j:j+1, :].drop(columns=['return']))
            score.loc[data.index[j], i] = y_pred[0]  # 注意这里预估的下一期的收益率

            # score.loc[data.index[j], i] = -data.iloc[j].sum()  # 注意这里预估的下一期的收益率

    # IC.to_csv('data/barra/barra_IC.csv')  # 存储IC值
    return score


industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
# Values to remove
values_to_remove = ['银行(申万)', '非银金融(申万)']
# Remove values using difference()
industries = industries.difference(values_to_remove)

close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])

score = get_prediction(industries)
score = score.iloc[12:, :]

output = barra.ranking_backtest(close, score)
barra.report(output)

'''IC值检验'''
fac_values = {}
sample = pd.read_csv('data/barra/factor_values/barra_' + '交通运输(申万)' + '.csv', index_col=0)
factors = sample.columns

for f in factors:
    data = pd.DataFrame(None, columns=industries)
    fac_values[f] = data

leverage_data = pd.read_csv('data/杠杆因子/leverage_ind.csv', index_col=0)
leverage_data.index = pd.to_datetime(leverage_data.index)
leverage_data = leverage_data.groupby(leverage_data.index.to_period('M')).last()

for ind_name in industries:
    score = pd.read_csv('data/barra/factor_values/barra_' + ind_name + '.csv', index_col=0)  # 读barra因子调成月频
    score.index = pd.to_datetime(score.index, format='%Y-%m-%d')
    # score = score.groupby(score.index.to_period('M')).apply(lambda x: x.iloc[-1])
    score = score.groupby(score.index.to_period('M')).last()
    score['leverage'] = leverage_data.loc[score.index, ind_name]
    for f in factors:
        fac_values[f][ind_name] = score[f]

'''排名相加法'''
# # 根据排名计算score
# score = pd.DataFrame(0, index=sample.index, columns=industries)
# score.index = pd.to_datetime(score.index)
# score = score.groupby(score.index.to_period('M')).last()
# for f in factors:
#     data = fac_values[f]
#     rank_data = data.rank(axis=1)  # 获取排名
#     for ind_name in industries:
#         score[ind_name] = np.array(score[ind_name]) + np.array(rank_data[ind_name])


# output = barra.ranking_backtest(close, score)
# barra.report(output)
'''排名相加法代码结束'''

for f in factors:
    data = fac_values[f]
    close_return = close.pct_change().shift(-1)
    close_return = close_return.loc[data.index]
    close_return.index = data.index
    close_return['IC'] = None
    for i in data.index:
        close_return.loc[i, 'IC'] = data.loc[i].corr(close_return.loc[i])
        # close_return.loc[i, 'rank_IC'] = (data.loc[i].rank()).corr(close_return.loc[i].rank())
        # print(i, output.loc[i, 'IC'])

    print(f, 'IC: ', close_return.IC.mean(), close_return.IC.std(), close_return.IC.mean()/close_return.IC.std())
    # print(f, 'rank IC: ', close_return.rank_IC.mean(), close_return.rank_IC.std(), close_return.rank_IC.mean() / close_return.rank_IC.std())

