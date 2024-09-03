import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import barra残差动量 as barra
import matplotlib.pyplot as plt


# 得到一个行业的因子dataframe
def get_score_helper(ind_name, factors):
    output = pd.DataFrame()
    sliced_ind_name = ind_name[:-4]
    data = pd.read_csv('data/barra/sliced_csv/申万_' + sliced_ind_name + '.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.loc[data.index > pd.to_datetime('2010-01-03'), factors]  # 2017年以后的
    print(ind_name)
    # print(data.isnull().sum())
    # data = data.groupby(data.index.to_period('M')).apply(lambda x: x.iloc[-1])
    data = data.groupby(data.index.to_period('M')).mean()  # 改成月频
    for factor_name in factors:
        sc = StandardScaler()
        data[factor_name] = data[factor_name].ffill()
        data[factor_name] = data[factor_name].fillna(0)
        data[factor_name] = np.array(sc.fit_transform(data.loc[:, [factor_name]]))  # 标准化

        for i in range(11, len(data.index)):  # 回归法
            X = np.array(range(12)).reshape(-1, 1)
            y = data[factor_name][i - 11:i + 1]
            model = LinearRegression()
            model.fit(X, y)
            output.loc[data.index[i], factor_name] = model.coef_[0] / y.mean()
    return output


# 根据因子，获取所有行业的综合排名
def speedup_score():
    industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
    # Values to remove
    values_to_remove = ['银行(申万)', '非银金融(申万)']
    # Remove values using difference()
    industries = industries.difference(values_to_remove)
    # west_avgoperatingprofit_YOY 11+5, west_avgroe_YOY 10+6, west_netprofit_YOY 13+9
    # netprofit_ttm 4+16, mfd_netbuyamt
    # factors = pd.read_csv('data/barra/sliced_csv/申万_' + industries[0][:-4] + '.csv', index_col=0)
    # for factor_name in factors.columns:
    #     break_bool = False
    # factors = ['dividendyield2', 'west_avgoc_YOY', 'yoynetprofit', 'netprofitmargin', 'tech_downpct']  # 2 or them are correlated
    # factors = ['dividendyield2', 'west_avgoc_YOY', 'netprofitmargin', 'tech_downpct']
    factors = ['dividendyield2', 'netprofitmargin', 'tech_downpct']
    # 股息率 一直预测的运营成本同比 净利润边际增长 下跌成分股占比

    sample = get_score_helper(industries[0], factors)
    # 创建score
    score = pd.DataFrame(0, index=sample.index, columns=industries)

    '''回归预测收益法'''
    close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close.index = pd.to_datetime(close.index)
    close = close.groupby(close.index.to_period('M')).last()
    close_pct = close.pct_change().shift(-1)  # 下一期收益率

    # for ind_name in industries:
    #     data = get_score_helper(ind_name, factors)
    #     data['return'] = np.array(close_pct.loc[data.index, ind_name])  #将下一期收益率添加到data里
    #     for j in range(12, len(data)):  # 开始循环每一天：回归并预测
    #         df_ne = data.iloc[j - 12:j, :]
    #         X = df_ne.drop(columns=['return'])
    #         y = df_ne['return']
    #         # Create results object and fit the model
    #         model = LinearRegression()
    #         # degree = 2
    #         # model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    #         # model = BayesianRidge()
    #         model.fit(X, y)
    #         y_pred = model.predict(data.iloc[j:j+1, :].drop(columns=['return']))
    #         score.loc[data.index[j], ind_name] = y_pred[0]  # 注意这里预估的下一期的收益率

    '''排名相加法'''
    # # 创建factor values dataframes
    # factor_values = {}
    # for f in factors:
    #     factor_values[f] = pd.DataFrame(None, index=sample.index, columns=industries)
    #
    # for ind_name in industries:
    #     output = get_score_helper(ind_name, factors)  # 月频因子值
    #     for f in factors:
    #         factor_values[f][ind_name] = output[f]
    # # 计算排名和
    # for f in factors:
    #     data = factor_values[f]  # 因子值
    #     # 打分（排名）
    #     ranked_df = data.rank(axis=1, ascending=False)
    #     # 对每一行的排名求和
    #     for ind_name in industries:
    #         score[ind_name] += ranked_df[ind_name]
    '''预测排名法 ndcg'''
    rank_close_pct = close_pct.rank(axis=1)
    for ind_name in industries:
        data = get_score_helper(ind_name, factors)
        data['return'] = np.array(close_pct.loc[data.index, ind_name])  #将下一期收益率添加到data里
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
            score.loc[data.index[j], ind_name] = y_pred[0]  # 注意这里预估的下一期的收益率
    return score


'''测试加速后的模型'''
score = speedup_score()
score = score.iloc[12:]
# print(score)

close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
close.index = pd.to_datetime(close.index)
close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])
# close_pct = close.pct_change()  # 同一期收益率
close_pct = close.pct_change().shift(-1)  # 下一期收益率
close_pct = close_pct.loc[score.index]

# print(close_pct)

output = barra.ranking_backtest(close, score)
barra.report(output)

'''IC 值检测'''
# score['IC'] = None
# for i in score.index:
#     score.loc[i, 'IC'] = score.loc[i].corr(close_pct.loc[i])
#     # print(i, output.loc[i, 'IC'])
#
# score.index = score.index.to_timestamp()
#
# score_copy = score.copy()
# score = score.loc[score.index > pd.to_datetime('2021-06-06')]
# print('2021-6之后IC: ', score.IC.mean(), score.IC.std(), score.IC.mean()/score.IC.std())
#
# score = score_copy.loc[score_copy.index < pd.to_datetime('2021-06-06')]
# print('2021-6之前IC: ', score.IC.mean(), score.IC.std(), score.IC.mean()/score.IC.std())

# plt.figure(figsize=(8, 5))
# plt.plot(score.index, score['IC'], linestyle='-', label='IC')
# plt.title('4 factors' + '\nIC mean & std: ' + str(score.IC.mean()) + ', ' + str(score.IC.std()))
# plt.xlabel('date')
# plt.ylabel('IC')
# plt.legend()
# plt.grid(True)
# plt.show()

