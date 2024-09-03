'''
光大证券研究报告《基于阻力支撑相对强度（RSRS）的市场择时》前四种择时方法的复现
文章来源： https://mp.weixin.qq.com/s/LehwGXe_6JFZEZ8ekC4dCg
'''

import pandas as pd
import os
import datetime
import numpy as np
import statsmodels.formula.api as sml
import matplotlib.pyplot as plt
import scipy.stats as scs
import matplotlib.mlab as mla
import warnings
warnings.filterwarnings("ignore")


def rsrs_calculator(date, close, high, low, M=18, N=600, M2=16, N2=300):
    date = np.array(date)
    close = np.array(close)
    high = np.array(high)
    low = np.array(low)

    data = pd.DataFrame({
        'date': date,
        'close': close,
        'high': high,
        'low': low
    })
    data = data.reset_index(drop=True)

    '''先计算右偏标准分，M和N数值不一样 calculate beta_right first since it needs different M and N'''
    # 斜率
    data['beta'] = 0
    data['R2'] = 0
    for i in range(1, len(data) - 1):
        df_ne = data.loc[i - N2 + 1:i, :]
        model = sml.ols(formula='high~low', data=df_ne)
        result = model.fit()

        data.loc[i + 1, 'beta'] = result.params[1]
        data.loc[i + 1, 'R2'] = result.rsquared
    # 日收益率
    data['ret'] = data.close.pct_change(1)

    # 标准分
    data['beta_norm'] = (data['beta'] - data.beta.rolling(M2).mean().shift(1)) / data.beta.rolling(M2).std().shift(1)
    for i in range(M2):
        data.loc[i, 'beta_norm'] = (data.loc[i, 'beta'] - data.loc[:i - 1, 'beta'].mean()) / data.loc[:i - 1,
                                                                                             'beta'].std()
    data.loc[2, 'beta_norm'] = 0
    # 修正标准分
    data['RSRS_R2'] = data.beta_norm * data.R2
    data = data.fillna(0)

    # 右偏标准分
    data['beta_right'] = data.RSRS_R2 * data.beta

    '''右偏计算完毕,开始计算其它指标 finished beta_right'''

    # 斜率
    data['beta'] = 0
    data['R2'] = 0
    for i in range(1, len(data) - 1):
        df_ne = data.loc[i - N + 1:i, :]
        model = sml.ols(formula='high~low', data=df_ne)
        result = model.fit()

        data.loc[i + 1, 'beta'] = result.params[1]
        data.loc[i + 1, 'R2'] = result.rsquared
    # 日收益率
    data['ret'] = data.close.pct_change(1)

    # 标准分
    data['beta_norm'] = (data['beta'] - data.beta.rolling(M).mean().shift(1)) / data.beta.rolling(M).std().shift(1)
    for i in range(M):
        data.loc[i, 'beta_norm'] = (data.loc[i, 'beta'] - data.loc[:i - 1, 'beta'].mean()) / data.loc[:i - 1, 'beta'].std()
    data.loc[2, 'beta_norm'] = 0
    # 修正标准分
    data['RSRS_R2'] = data.beta_norm * data.R2
    data = data.fillna(0)


    '''生成买卖信号 Generating signals'''

    return data


close = pd.read_csv('data/申万31大行业_close.csv')
high = pd.read_csv('data/申万31大行业_high.csv')
low = pd.read_csv('data/申万31大行业_low.csv')
for i in close.columns[1:]:
    print(i)

    df = rsrs_calculator(date=close.iloc[:, 0], close=close[i], high=high[i], low=low[i])
    df.to_csv('data/rsrs_data/rsrs_'+i+'.csv')

