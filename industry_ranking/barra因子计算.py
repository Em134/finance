"""
https://zhuanlan.zhihu.com/p/31412967?from_voters_page=true
"""
import helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
matplotlib.rcParams['font.family'] = ['Heiti TC']
import warnings
warnings.filterwarnings("ignore")


def get_previous_workday(df, target_date):
    if target_date in df.index:
        return target_date
    else:
        # 找到上一个最近的工作日
        previous_workday = df.index.asof(target_date)
        return previous_workday


def slice_data(): # used
    # 处理一下 申万一级初期因子.csv
    header_df = pd.read_csv('data/申万一级初期因子.csv', nrows=2, header=None, index_col=0)
    multi_index_columns = pd.MultiIndex.from_arrays([header_df.iloc[0], header_df.iloc[1]])
    df = pd.read_csv('data/申万一级初期因子.csv', skiprows=2, header=None, index_col=0)
    df.columns = multi_index_columns

    for i in industries:
        name = i[:-4]
        data = df.loc[:, name]
        print(name)
        # print(data.head())
        data.index = pd.to_datetime(data.index)
        data.to_csv('data/barra/sliced_csv/申万_' + name + '.csv')


def calculate_factors(ind_name):
    """用日频数据计算"""
    sliced_ind_name = ind_name[:-4]
    print("calculating barra factors: " + sliced_ind_name)
    # 建表
    data = pd.read_csv('data/barra/sliced_csv/申万_' + sliced_ind_name + '.csv', index_col=0)
    data = data.fillna(method='ffill') # 把中间空的补上
    data = data.fillna(0) # 把最开始的空缺补上
    data.index = pd.to_datetime(data.index)
    output = pd.DataFrame(0, index=data.index, columns=['beta', 'mom', 'size', 'earning', 'dastd', 'cmra',
                                                        'growth', 'bookToPrice', 'leverage', 'liquidity', 'nonLinearSize'])
    # output = pd.DataFrame(0, index=data.index, columns=['beta', 'mom', 'size', 'earning', 'dastd', 'cmra',
    #                                                     'growth', 'bookToPrice', 'liquidity', 'nonLinearSize'])
    output = output[output.index >= '2014-06-29']
    # output = output[output.index >= '2015-06-1']
    close_data = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close_data.index = pd.to_datetime(close_data.index)
    close_data = close_data.loc[:, [ind_name]]

    """ 1 beta """
    beta_data = pd.read_csv('data/barra/beta.csv', index_col=0)
    beta_data.index = pd.to_datetime(beta_data.index)
    beta_data = beta_data.loc[output.index, sliced_ind_name]
    output['beta'] = np.array(beta_data)
    del beta_data

    """ 2 momentum = 1 * RSTR """
    mom_data = close_data.copy()
    mom_data['rstr'] = helper.calculate_RSTR(mom_data.pct_change().loc[:, ind_name], np.zeros_like(np.array(mom_data[ind_name])))
    output['mom'] = np.array(mom_data.loc[output.index, 'rstr'])
    del mom_data

    """ 3 size = 1 * LNCAP """
    size_data = data.loc[:, ['mkt_cap_ard']]
    size_data['mkt_cap_ard'] = np.log(size_data['mkt_cap_ard'])
    output['size'] = np.array(size_data.loc[output.index, 'mkt_cap_ard'])

    """ 4 earning = 0.68 * EPIBS + 0.11 * ETOP + 0.21 * CETOP """
    # 仅有 ETOP: netprofit_ttm/mkt_cap_ard
    earning_data = data.loc[:, ['netprofit_ttm', 'mkt_cap_ard']]
    earning_data['earning'] = earning_data['netprofit_ttm'] / earning_data['mkt_cap_ard']
    output['earning'] = np.array(earning_data.loc[output.index, 'earning'])

    """ 5 residualVolatility = 0.74 * DASTD + 0.16 * CMRA + 0.10 * HSIGMA 没有HSIGMA """
    resiVola_data = pd.read_csv('data/barra/volatility.csv', index_col=0)
    resiVola_data.index = pd.to_datetime(resiVola_data.index)
    resiVola_data = resiVola_data.loc[:, [sliced_ind_name]] # column 0 is sliced_ind_name
    resiVola_data['dastd'] = None # column 1

    close_pct = close_data.pct_change()
    # Calculate cumulative returns
    rolling_cumulative_returns = pd.DataFrame(close_pct[ind_name].rolling(window=21).apply(lambda x: (1 + x).prod() - 1, raw=True)) # 月收益
    rolling_cumulative_returns['index'] = rolling_cumulative_returns.index
    rolling_cumulative_returns = rolling_cumulative_returns.groupby(rolling_cumulative_returns.index.to_period('M')).apply(lambda x: x.iloc[-1])
    rolling_cumulative_returns.set_index('index', inplace=True)  # 月均收益率
    rolling_cumulative_returns.columns = ['return_m']
    # print(rolling_cumulative_returns)

    # 计算累计 SUM of ( ln(1+r) - ln(1+ rf) ), assume rf = 0
    rolling_cumulative_returns['ln_sum'] = np.log(1 + rolling_cumulative_returns['return_m']).cumsum()

    for i in range(251, len(resiVola_data.index)):
        resiVola_data.iloc[i, 1] = helper.calculate_ewma(resiVola_data.iloc[i-251: i+1, 0], 42)[-1]

    resiVola_data['cmra'] = None  # column 2

    for i in range(11, len(rolling_cumulative_returns.index)):
        temp = rolling_cumulative_returns.iloc[i-11: i+1, :] # 近12月数据
        resiVola_data.loc[rolling_cumulative_returns.index[i], 'cmra'] = temp['ln_sum'].max() - temp['ln_sum'].min()

    output['dastd'] = np.array(resiVola_data.loc[output.index, 'dastd'])
    output['cmra']  = np.array(resiVola_data.loc[output.index, 'cmra'])
    output['dastd'] = output['dastd'].fillna(method='ffill')
    output['cmra'] = output['cmra'].fillna(method='ffill')

    """ 6 growth = 0.47 * SGRO + 0.24 * EGRO + 0.18 * EGIBS + 0.11 * EGIBS_s  只有EGRO：五年净利润增长（回归法） """
    growth_data = data.loc[:, ['netprofit_ttm']]
    growth_data['egro'] = None
    for i in range(252*5, len(growth_data.index)):
        given_date = growth_data.index[i]
        years = [given_date - pd.DateOffset(years=i) for i in range(5)]
        previous_workdays = [get_previous_workday(growth_data, year) for year in years]
        annual_netprofits = growth_data.loc[previous_workdays, 'netprofit_ttm'] # 过去五年的每年净利润
        # 将日期转换为年数（例如：2019-01-01 -> 2019.0）
        X = np.array([date.year for date in previous_workdays]).reshape(-1, 1)
        y = annual_netprofits.values
        # 线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        # 获取回归系数并计算增长率
        growth_data.loc[given_date, 'egro'] = model.coef_[0]/y.mean()

    output['growth']  = np.array(growth_data.loc[output.index, 'egro'])
    output['growth'] = output['growth'].fillna(method='ffill')

    """ 7 bookToPrice """
    bp_data = data.loc[:, ['pb_lf']]
    output['bookToPrice'] = 1/np.array(bp_data.loc[output.index, 'pb_lf'])
    output['bookToPrice'] = output['bookToPrice'].fillna(method='ffill')

    """ 8 leverage 用季频数据"""
    output['leverage'] = None
    leverage_data = pd.read_csv('data/风格因子/barra风格_' + ind_name + '.csv', index_col=0)
    leverage_data = leverage_data.loc[:, ['杠杆']]
    leverage_data.index = pd.to_datetime(leverage_data.index)
    for i in range(len(leverage_data.index)):
        given_date = leverage_data.index[i]
        workday = get_previous_workday(output, given_date)
        output.loc[workday, 'leverage'] = leverage_data.loc[given_date, '杠杆']

    output['leverage'] = output['leverage'].ffill()

    """ 9 liquidity = 0.35 * STOM + 0.35 * STOQ + 0.30 * STOA """
    liquid_data = data.loc[:, ['turn']]
    liquid_data['stom'] = np.log(np.array(liquid_data['turn'].rolling(21, min_periods=1).sum()))
    liquid_data['stoq'] = np.log(1.0/3 * np.array(liquid_data['turn'].rolling(21*3, min_periods=1).sum()))
    liquid_data['stoa'] = np.log(1.0/12 * np.array(liquid_data['turn'].rolling(21*12, min_periods=1).sum()))
    liquid_data['liquidity'] = 0.35 * liquid_data['stom'] + 0.35 * liquid_data['stoq'] + 0.30 * liquid_data['stoa']
    # print(liquid_data)

    output['liquidity'] = np.array(liquid_data.loc[output.index, 'liquidity'])
    output['liquidity'] = output['liquidity'].fillna(method='ffill')

    """ 10 nonLinearSize"""
    # Step 1: 计算 nonLinearSize 列
    output['nonLinearSize'] = output['size'] ** 3
    # Step 2: 回归分析并获取残差
    X = sm.add_constant(output['size'])  # 添加截距项
    y = output['nonLinearSize']
    model = sm.OLS(y, X)
    results = model.fit()
    residuals = results.resid  # 获取回归模型的残差
    # Step 3: Winsorize 处理
    winsorized_residuals = winsorize(residuals, limits=[0.05, 0.05])  # 缩尾处理，限定超出范围的值
    # Step 4: 标准化
    # scaler = StandardScaler()
    # scaled_residuals = scaler.fit_transform(winsorized_residuals.reshape(-1, 1))  # 标准化残差
    # flat_list = [item for sublist in scaled_residuals for item in sublist]

    # 将处理后的残差添加到数据框中
    output['nonLinearSize'] = winsorized_residuals


    """全部因子标准化处理"""
    # col = output.columns
    # ind = output.index
    # output = pd.DataFrame(scaler.fit_transform(output), columns=col, index=ind)
    # output['residualVolatility'] = 0.8 * output['dastd'] + 0.2 * output['cmra']
    # output = output.drop(columns=['dastd', 'cmra'])

    output.to_csv('data/barra/factor_values/barra_' + ind_name + '.csv')


def dataframe_standardize(data):
    scaler = StandardScaler()
    a = np.array(data)
    a_scaled = scaler.fit_transform(a.T)
    data.loc[:, :] = a_scaled.T

    return data


def barra_standardize(industries):
    sample = pd.read_csv('data/barra/factor_values/barra_交通运输(申万).csv', index_col=0, parse_dates=True)
    factor_list = sample.columns
    all_ind_data = {}
    all_fac_data = {}
    for ind_name in industries:
        all_ind_data[ind_name] = pd.read_csv('data/barra/factor_values/barra_' + ind_name + '.csv', index_col=0,
                                             parse_dates=True)
    for fac in factor_list:
        data = pd.DataFrame(None, index=sample.index, columns=industries)
        for ind_name in industries:
            data.loc[:, ind_name] = all_ind_data[ind_name].loc[:, fac]

        all_fac_data[fac] = dataframe_standardize(data)
        all_fac_data[fac].to_csv('data/barra/standardized_factors/barra_' + fac + '.csv')

    # 把标准化完成后的 存回一个行业一个.csv的形式
    for ind_name in industries:
        for fac in factor_list:
            all_ind_data[ind_name].loc[:, fac] = all_fac_data[fac].loc[:, ind_name]

        all_ind_data[ind_name]['residualVolatility'] = 0.8 * all_ind_data[ind_name]['dastd'] + 0.2 * all_ind_data[ind_name]['cmra']
        all_ind_data[ind_name] = all_ind_data[ind_name].drop(columns=['dastd', 'cmra'])
        all_ind_data[ind_name].to_csv('data/barra/factor_values_stdized/barra_' + ind_name + '.csv')


    return


industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
# Values to remove
values_to_remove = ['银行(申万)', '非银金融(申万)']
# Remove values using difference()
industries = industries.difference(values_to_remove)

# slice_data() # 已提取完毕
# for i in industries: # 已计算完毕
#     calculate_factors(i)

# barra_standardize(industries)

