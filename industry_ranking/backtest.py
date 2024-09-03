
import helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Heiti TC']
import warnings
warnings.filterwarnings("ignore")


def get_top_stocks(scores_df, n):
    top_stocks = scores_df.apply(lambda x: x.nlargest(n).index.tolist(), axis=1)
    return top_stocks


'''各行业指数收盘价: close（日频），各行业指标分数: score(日频）, 日期在index
每月选取排名前ranking_num个行业，等权。等权持有全部行业为基准'''
def ranking_backtest(close, score, ranking_num=5):
    # 基准
    close['all_ind'] = close.mean(axis=1)
    # value
    # 计算加权月均score，用ewma半衰期10
    # for i in score.columns:
    #     score[i] = helper.calculate_ewma(score[i], 10)
    # 计算上个月平均分
    # score = score.rolling(window=21).mean().dropna()

    score['index'] = score.index
    # 选出每月最后一天的数值去计算
    monthly_score = score.groupby(score.index.to_period('M'), as_index=False).apply(lambda x: x.iloc[-1])
    monthly_score.set_index('index', inplace=True)

    # data 处理完毕
    stocks = get_top_stocks(monthly_score, ranking_num) # 月底获取上一个月月排名前5，第二天买入（开盘价）直到下一个月月底（收盘价）

    monthly_score['value'] = 1
    for i in range(len(monthly_score)):
        date = monthly_score.index[i]
        if i+1 < len(monthly_score):
            next_date = monthly_score.index[i+1]
        else:
            next_date = monthly_score.index[-1]
        picked_stocks = stocks[i] # tested, picked_stocks is a list of column names
        value_start = close.loc[date, picked_stocks].mean()  # 选中行业的 最后一天的 收盘价的平均
        value_end = close.loc[next_date, picked_stocks].mean()
        month_return = float(value_end-value_start)/value_start  # 本月收益率
        monthly_score.loc[next_date, 'value'] = (1+month_return)*monthly_score.loc[date, 'value']  # 净值

    monthly_score = monthly_score.join(close.loc[:, ['all_ind']])
    monthly_score['all_ind'] = monthly_score['all_ind']/float(monthly_score.all_ind[0])  # start from 1

    return monthly_score.loc[:, ['all_ind', 'value']]


# 构建close和score
def get_scores(score_name):
    close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
    close.index = pd.to_datetime(close.index)

    score = close.copy()
    for i in score.columns:
        data = pd.read_csv('data/rsrs_data/rsrs_' + i + '.csv')
        # print(data[score_name])
        score[i] = np.array(data[score_name])

    score.to_csv('data/rsrs_data/score_' + score_name + '.csv')


# get_scores('beta_right')  # done
# get_scores('beta_norm')
# get_scores('RSRS_R2')

# back testing
score_name = 'RSRS_R2'
close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
score = pd.read_csv('data/rsrs_data/score_' + score_name + '.csv', index_col=0)
close.index = pd.to_datetime(close.index)
score.index = pd.to_datetime(score.index)

output = ranking_backtest(close, score)
output_pct = output.pct_change().fillna(0)
output_pct['超额'] = output_pct['value']-output_pct['all_ind']
# print(output_pct)
output.to_csv('data/output/RSRS/output_' + score_name + '.csv')

# output = pd.read_csv('output_' + score_name + '.csv')

yearly_return = helper.calculate_annualized_return_monthly(output['value'])
print('策略年化收益：', yearly_return)
yearly_volatility = helper.calculate_annualized_volatility(output['value'], 12)
print('策略年化波动率：', yearly_volatility)
max_draw = helper.calculate_max_drawdown(output['value'])
print('策略最大回撤：', max_draw)

print()
yearly_return = helper.calculate_annualized_return_monthly(output['all_ind'])
print('基准年化收益：', yearly_return)
yearly_volatility = helper.calculate_annualized_volatility(output['all_ind'], 12)
print('基准年化波动率：', yearly_volatility)
max_draw = helper.calculate_max_drawdown(output['all_ind'])
print('基准最大回撤：', max_draw)

print('超额统计：')
count_greater_than_zero = np.sum(np.array(output_pct['超额']) > 0)
count_smaller_than_zero = np.sum(np.array(output_pct['超额']) < 0)
print('超额>0:', count_greater_than_zero)
print('超额<0:', count_smaller_than_zero)
print('胜率：', count_greater_than_zero/float(len(output_pct.index)))

plt.figure(figsize=(8, 5))
plt.plot(output.index, output['all_ind'], linestyle='-', label='全部行业')
plt.plot(output.index, output['value'], linestyle='-', label='策略')
plt.plot(output.index, output['value']-output['all_ind'], linestyle='-', label='累计超额')
plt.plot(output.index, output_pct['超额'], linestyle='-', label='当月超额')
plt.title('RSRS Plot of ' + score_name)
plt.xlabel('date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

