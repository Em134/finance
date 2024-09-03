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
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


# 把因子值改成rank，并标准化：
def rank_std_df(data: pd.DataFrame):
    scaler = StandardScaler()
    a = np.array(data)
    a_scaled = scaler.fit_transform(a.T)
    data.loc[:, :] = a_scaled.T

    return data

# 读取其他因子数据并调为月频
def read_resample_data(filename: str, ranked=False):
    data_temp = pd.read_csv(filename, index_col=0, parse_dates=True)
    data_temp = data_temp.groupby(data_temp.index.to_period('M')).last()
    if ranked:
        return rank_std_df(data_temp)
    return data_temp


# 把后一期的月收益率 添加到数据score里，频率为月M
def get_dataset(ind_name, start_date: str, end_date: str):
    close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
    close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])  # 读收盘价然后调成月频/周频

    score = pd.read_csv('data/barra/factor_values_stdized/barra_' + ind_name + '.csv', index_col=0)  # 读barra因子调成月频/周频
    score.index = pd.to_datetime(score.index, format='%Y-%m-%d')

    score = score[score.index > pd.to_datetime(start_date)]
    score = score[score.index < pd.to_datetime(end_date)]

    # score = score.groupby(score.index.to_period('M')).apply(lambda x: x.iloc[-1])
    score = score.groupby(score.index.to_period('M')).last()

    close = close.loc[:, [ind_name]].pct_change().fillna(0)
    close[ind_name] = close[ind_name].shift(-1)  # 收益率移动一期
    filtered_close = close.loc[score.index]
    score['return'] = np.array(filtered_close[ind_name])  # 把下一期涨跌幅加到barra因子文件里
    score = score.fillna(0)
    # print(score)
    # print('empty: ', score.isnull().sum())
    return score

def get_prediction(industries, start_date: str, end_date: str):
    sample1 = pd.read_csv('data/barra/factor_values_stdized/barra_' + '交通运输(申万)' + '.csv', index_col=0, parse_dates=True)
    sample1 = sample1[sample1.index > pd.to_datetime(start_date)]
    sample1 = sample1[sample1.index < pd.to_datetime(end_date)]
    sample1 = sample1.groupby(sample1.index.to_period('M')).apply(lambda x: x.iloc[-1])  # 读barra文件调成月频格式用于新建score df
    # print(sample1)
    score = pd.DataFrame(0, index=sample1.index, columns=industries)  # 新建score df

    leverage_data = pd.read_csv('data/杠杆因子/申万一级月频杠杆因子.csv', index_col=0, parse_dates=True)
    leverage_data = leverage_data.groupby(leverage_data.index.to_period('M')).last()

    # 其他因子
    # other_factors_normal = ['tech_vol_std', 'tech_change_lsc', 'tech_corr', 'tech_concentric',
    #                         'big_break', 'herd_behavior']
    # other_factors_ranked = ['profit_growth', 'tech_amt_std', 'tech_second_order', 'tech_term_spread']
    other_factors_normal = ['index_growth', 'tech_concentric', 'big_break', 'herd_behavior']
    other_factors_ranked = ['profit_growth']
    all_other_factors = other_factors_normal.copy()
    all_other_factors.extend(other_factors_ranked)

    other_factors_data = {}
    for other_f in other_factors_normal:
        other_factors_data[other_f] = read_resample_data('data/其他因子数据/处理后/' + other_f + '.csv')
    for other_f in other_factors_ranked:
        other_factors_data[other_f] = read_resample_data('data/其他因子数据/处理后/' + other_f + '.csv', True)

    # 收盘价
    close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
    close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])

    # 添加行业轮动速度
    speed = barra.rotation_speed(close)

    '''往期收益法'''

    # for i in industries:  # 循环工业
    #     data = get_dataset(i)  # data是因子及收益率
    #     data['leverage'] = leverage_data.loc[data.index, i]
    #     data['speed'] = np.array(speed.loc[data.index])  # 添加行业轮动速度
    #     data = data.rank(axis=1)
    #
    #     for j in range(12, len(data)):  # 开始循环每一天
    #         df_ne = data.iloc[j - 12:j, :]  # 过去一年的数据,计算因子值IC
    #         X = df_ne.drop(columns=['return'])
    #         y = df_ne['return']
    #         # 计算各因子的'IC'值
    #         ICs = pd.DataFrame(None, columns=X.columns, index=[0,1])  # index 0 is ICs, index 1 is weighted factor
    #         for col in X.columns:
    #             ICs.loc[0, col] = X[col].corr(y)
    #             ICs.loc[1, col] = ICs.loc[0, col] * data.loc[data.index[j], col]  # 加权后的因子值
    #         ICs.loc[1] = ICs.loc[1]/abs(ICs.loc[0]).sum()  # 除以绝对值之和
    #
    #         score.loc[data.index[j], i] = ICs.loc[1].sum()

    '''#IC法'''
    factors = list(sample1.columns)
    # factors.append('speed')
    factors.extend(all_other_factors)

    dfs = {}  # 用来存所有因子数据，每个因子一个df，index为date， columns为行业名称
    for fac_name in factors:
        dfs[fac_name] = pd.DataFrame(None, index=sample1.index, columns=industries)
    # dfs['speed'] = pd.DataFrame(None, index=sample1.index, columns=industries)  # 行业轮动速度,没有意义，所有行业的轮动速度和因子值和权重都一样
    dfs['return'] = pd.DataFrame(None, index=sample1.index, columns=industries)  # 下期收益率

    for ind in industries:  # 循环工业
        data = get_dataset(ind, start_date, end_date)  # data是因子及对应的下期收益率

        data['leverage'] = leverage_data.loc[data.index, ind]  # 更改杠杆因子值为自己计算的杠杆因子值
        # data['speed'] = np.array(speed.loc[data.index])  # 添加行业轮动速度
        sliced_ind = ind[:-4]
        # data['mfd_netbuy_amt'] = mfd_netbuy.loc[data.index, sliced_ind]
        # data['valuation_growth'] = valuation_growth.loc[data.index, sliced_ind]
        for other_f in all_other_factors:
            data[other_f] = other_factors_data[other_f].loc[data.index, sliced_ind]

        for fac_name in data.columns:  # 循环所有因子和下期收益率
            dfs[fac_name].loc[:, ind] = data[fac_name]  # data里就是现在的ind的因子值

    returns = dfs['return']  # 把returns单拎出来

    for index_temp in range(12, len(returns.index)):  # 循环日期

        date_now = returns.index[index_temp]  # 现在的日期
        ICs = pd.Series(None, index=factors)  # 创建存储所有factors的IC值的Series
        for f_temp in factors:  # 循环所有因子
            data_temp = dfs[f_temp].iloc[index_temp-12: index_temp]  # 前12个月的数据，不包括本月
            ICs_temp = pd.Series(None, index=data_temp.index)  # 每个日期的IC值，存到这个Series里
            for each_date in data_temp.index:  # 循环这前12个月
                ICs_temp.loc[each_date] = data_temp.loc[each_date].corr(returns.loc[each_date])  # 计算IC
            ICs[f_temp] = ICs_temp.mean()  # 存储这个因子f_temp的过去12月的因子值的平均值

        for ind_temp in industries:  # 循环行业
            # 提取这个行业因子数据，存在一个series里
            factors_value = pd.Series(None, index=factors)
            # 把因子存到series里
            for fac_temp in factors:
                factors_value.loc[fac_temp] = dfs[fac_temp].loc[date_now, ind_temp]

            # 计算加权后的因子值，相加后存在score里
            score.loc[date_now, ind_temp] = ( factors_value * ICs.loc[factors_value.index] ).sum() / abs(ICs).sum()

    return score


industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
# Values to remove
values_to_remove = ['银行(申万)', '非银金融(申万)']
# Remove values using difference()
industries = industries.difference(values_to_remove)

close = pd.read_csv('data/申万31大行业_close.csv', index_col=0, parse_dates=True)
# close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])

score = get_prediction(industries, '2016-06-30', '2024-03-27')
score = score.iloc[12:, :]

output = barra.ranking_backtest(close, score)
barra.report(output)

'''IC值检验'''
# fac_values = {}
# sample = pd.read_csv('data/barra/factor_values_stdized/barra_' + '交通运输(申万)' + '.csv', index_col=0)
# factors = list(sample.columns)
#
# for f in factors:
#     data = pd.DataFrame(None, columns=industries)
#     fac_values[f] = data
#
# leverage_data = pd.read_csv('data/杠杆因子/申万一级月频杠杆因子.csv', index_col=0)
# leverage_data.index = pd.to_datetime(leverage_data.index)
# leverage_data = leverage_data.groupby(leverage_data.index.to_period('M')).last()
#
# for ind_name in industries:
#     score = pd.read_csv('data/barra/factor_values_stdized/barra_' + ind_name + '.csv', index_col=0)  # 读barra因子调成月频
#     score.index = pd.to_datetime(score.index, format='%Y-%m-%d')
#     # score = score.groupby(score.index.to_period('M')).apply(lambda x: x.iloc[-1])
#     score = score.groupby(score.index.to_period('M')).last()
#     score['leverage'] = leverage_data.loc[score.index, ind_name]
#     for f in factors:
#         fac_values[f][ind_name] = score[f]

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

# for f in factors:
#     data = fac_values[f]
#     close_return = close.pct_change().shift(-1)
#     close_return = close_return.loc[data.index]
#     close_return.index = data.index
#     close_return['IC'] = None
#     for i in data.index:
#         close_return.loc[i, 'IC'] = data.loc[i].corr(close_return.loc[i])
#         # close_return.loc[i, 'rank_IC'] = (data.loc[i].rank()).corr(close_return.loc[i].rank())
#         # print(i, output.loc[i, 'IC'])
#
#     print(f, 'IC: ', close_return.IC.mean(), close_return.IC.std(), close_return.IC.mean()/close_return.IC.std())
#     # print(f, 'rank IC: ', close_return.rank_IC.mean(), close_return.rank_IC.std(), close_return.rank_IC.mean() / close_return.rank_IC.std())

