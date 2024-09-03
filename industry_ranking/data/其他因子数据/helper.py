import numpy as np
import pandas as pd
import statsmodels.api as sm
import time

# 万得因子处理
# 中位数去极值法
def mad_outlier(data: pd.Series, n=5):
    # data 是时间截面上股票池的所有数据, 格式为Series, n是偏离中位数的mad倍数
    data_no_zero = data.copy().dropna()  # 忽略空值
    md = data_no_zero.median()  # 计算中位数
    mad = abs(data_no_zero - md).median()  #计算中位差
    top = md + n*mad
    low = md - n*mad
    for i in data.index:
        # print('type', type(data[i]), i, ':', data[i])
        if pd.isna(data[i]):  # 忽略空值
            continue
        if data[i] > top:
            data[i] = top
        elif data[i] < low:
            data[i] = low

    # print('blank havenot filled:', data.isnull().sum() / len(data))
    return data


# 填补缺失
# 组合行业中位数
def md_fill_black(data: pd.Series, industries: pd.Series):
    # data为股票池因子值，ind为wind行业代码, 它们的index都是股票代码
    industries = industries[data.index]  # 筛选掉不需要的股票wind行业数据
    for ind in industries.unique():  # 循环每个wind行业
        codes = (industries[industries == ind]).index  # 同行业的股票的代码[]
        same_ind_data = data[codes]
        same_ind_data_nozero = same_ind_data.dropna()
        md = same_ind_data_nozero.median()
        for i in codes:  # i 为股票代码
            # print('type', type(same_ind_data[i]), same_ind_data[i])
            if pd.isna(same_ind_data[i]):  # 如果是空值
                data[i] = md

    # print(data.isnull().sum())
    return data


# 行业市值中性化
def ind_mkt_neutralization(data: pd.Series, mkt: pd.Series, ind: pd.Series):
    # data: 因子值（全部参与计算的股票）(已经过前两步）， mkt：市值， ind：所属wind行业. 他们三个的index都是股票代码
    # Combine them into a DataFrame
    mkt = mkt.loc[data.index]  # 过滤不需要的市值数据
    ind = ind.loc[data.index]  # 过滤不需要的wind行业数据
    df = pd.concat([data, mkt, ind], axis=1)  # 合成一个df
    # rename the columns for clarity
    df.columns = ['factor', 'mkt', 'ind']
    df['mkt'] = np.log(df['mkt'])  # 对数市值
    # Convert the third column to dummy variables
    dummies = pd.get_dummies(df['ind'], prefix='ind').astype(int)
    # Drop the original third column and concatenate the dummies
    df = df.drop('ind', axis=1)
    df = pd.concat([df, dummies], axis=1)

    X = df.drop('factor', axis=1)
    X = sm.add_constant(X)
    y = df['factor']

    X = np.asarray(X)
    y = np.asarray(y)
    # Perform the linear regression
    model = sm.OLS(y, X).fit()
    # Calculate residuals
    df['residual'] = model.resid

    # num_nulls = df['residual'].isnull().sum()
    # if num_nulls != 0:
    #     print('Residual: ', num_nulls)
    return df['residual']


# wind市值加权标准化
def wind_standardize(data: pd.Series, mkt: pd.Series):
    # data为因子值， mkt为市值，他们的index都是股票代码
    mkt = mkt.loc[data.index]
    df = pd.concat([data, mkt], axis=1)
    df.columns = ['factor', 'mkt']
    df['mkt'] = np.log(df['mkt'])  # 对数市值

    mean = (df['factor'] * df['mkt']).sum() / df['mkt'].sum()

    std = np.sqrt( np.square(df['factor'] - mean).sum() / (len(df['factor'])-1) )

    df['new_factor'] = (df['factor'] - mean) / std
    # print('std:', std)
    # print('standardized: ', df['new_factor'].isnull().sum())
    return df['new_factor']
