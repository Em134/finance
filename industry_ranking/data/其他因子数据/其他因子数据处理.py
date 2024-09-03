import pandas as pd
import numpy as np
import helper
from sklearn.preprocessing import StandardScaler


# 去极值
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


# 中位数填补缺失
def fill_null_with_median(factors: pd.DataFrame):
    factors = factors.apply(lambda row: row.fillna(row.median()), axis=1)

    return factors


def data_std(filename):

    # data_raw = pd.read_csv(filename, index_col=0, parse_dates=True)
    fs = ['amt_std', 'vol_std', 'turnover', 'second_order', 'term_spread', 'total_lsc', 'change_lsc', 'deviate_cov',
          'corr', 'first_order', 'concentric']

    for fac_name in fs:
        print(fac_name)
        data_raw = pd.read_excel(filename, sheet_name=fac_name, index_col=0, parse_dates=True)
        data_raw.drop(columns=['非银金融'], inplace=True)

        mkt_data = pd.read_csv('申万行业mkt_cap_ard.csv', index_col=0)
        mkt_data.index = pd.to_datetime(mkt_data.index)
        mkt_data = mkt_data.loc[data_raw.index]

        new_data = data_raw.copy()

        scaler = StandardScaler()
        for date in new_data.index:
            new_data.loc[date] = mad_outlier(new_data.loc[date])
            new_data.loc[date].fillna(new_data.loc[date].median())
            new_data.loc[date] = helper.wind_standardize(data_raw.loc[date], mkt_data.loc[date])

        # a = np.array(new_data)
        # a_scaled = scaler.fit_transform(a.T)
        # new_data.loc[:, :] = a_scaled.T

        # new_data = new_data.loc[new_data.index > pd.to_datetime('2016-01-28')]
        new_data.to_csv('处理后/tech_' + fac_name + '.csv')
        # fac_name = 'herd_behavior'
        # new_data.to_csv('处理后/' + fac_name + '.csv')

filename = 'tech.xlsx'
data_std(filename)

