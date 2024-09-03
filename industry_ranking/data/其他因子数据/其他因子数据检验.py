import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# IC 检验
def ic_testing(factor: pd.DataFrame, close=None, returns=None):
    """
    :param factor: factor values
    :param close: close prices
    :param returns: returns of next period
    for all inputs: index is date, columns are stock_codes
    :return: None
    """
    close = close.resample('ME').last()
    factor = factor.resample('ME').last()

    # if not already have next date's returns
    if returns is None:
        if close is None:
            print('need close prices or returns')
            return
        returns = close.pct_change().shift(-1)

    # make the index and column of the two dfs be the same
    returns = returns.loc[factor.index, factor.columns]

    # a Series to save ICs for all date
    ICs = pd.Series(None, index=factor.index)
    rank_ICs = pd.Series(None, index=factor.index)

    ranked_factor = factor.rank(axis=1)
    temp = np.array(ranked_factor)
    scaler = StandardScaler()
    temp = scaler.fit_transform(temp.T)
    ranked_factor.loc[:, :] = temp.T

    # calculation
    for date in factor.index:
        ICs.loc[date] = factor.loc[date].corr(returns.loc[date])
        # rank_ICs[date] = factor.loc[date].rank().corr(returns.loc[date].rank())
        rank_ICs.loc[date] = ranked_factor.loc[date].corr(returns.loc[date])
    # results
    print('IC mean:', ICs.mean())
    print('IC std:', ICs.std())
    print('IC_IR:', ICs.mean()/ICs.std())

    print('rank IC mean:', rank_ICs.mean())
    print('rank IC std:', rank_ICs.std())
    print('rank IC_IR:', rank_ICs.mean() / rank_ICs.std())

    # Plot the series
    plt.figure(figsize=[12, 5])
    ax = ICs.plot(label='Pearson IC')
    # rank_ICs.plot(ax=ax, label='Rank IC')

    plt.title('ICs over Time')
    plt.xlabel('Date')
    plt.ylabel('IC')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()


def factors_corr(factors_name: list, start_date: str, end_date: str):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    factors_data = {}
    for f in factors_name:
        print(f)
        data = pd.read_csv('处理后/' + f + '.csv', index_col=0, parse_dates=True)
        data = data[data.index > start_date]
        data = data[data.index < end_date]
        data = data.resample('ME').last()
        factors_data[f] = data.ffill().fillna(0)

    index_date = factors_data[factors_name[0]].index
    stocks = factors_data[factors_name[0]].columns  # 统一排序
    list_of_corr_matrix = []
    for date in index_date:
        temp_df = pd.DataFrame(None, index=stocks, columns=factors_name)
        for f in factors_name:
            temp_df.loc[:, f] = factors_data[f].loc[date, stocks]

        list_of_corr_matrix.append(temp_df.corr())

    corr_matrices_array = np.array([corr.values for corr in list_of_corr_matrix])
    avg_corr_matrix = np.mean(corr_matrices_array, axis=0)

    average_corr_matrix_df = pd.DataFrame(avg_corr_matrix,
                                          index=list_of_corr_matrix[0].index,
                                          columns=list_of_corr_matrix[0].columns)
    print(average_corr_matrix_df)
    average_corr_matrix_df.to_csv('factor_correlations.csv')


# close = pd.read_csv('申万31大行业_close.csv', index_col=0, parse_dates=True)
# new_columns = []
# for i in range(len(close.columns)):
#     new_columns.append(close.columns[i][:-4])
# close.columns = new_columns

# tech.xlsx
# fs = ['amt_std', 'vol_std', 'turnover', 'second_order', 'term_spread', 'total_lsc', 'change_lsc', 'deviate_cov',
#       'corr', 'first_order', 'concentric']
#
# for f in fs:
#     factor = pd.read_csv('处理后tech/tech_' + f + '.csv', index_col=0, parse_dates=True)
#     print(f)
#     ic_testing(factor, close)
#     print()

# .csv文件
# factor = pd.read_csv('处理后羊群效应因子.csv', index_col=0, parse_dates=True)
# ic_testing(factor, close)

factor_names = ['index_growth', 'profit_growth', 'tech_vol_std', 'tech_amt_std', 'tech_second_order',
                'tech_term_spread', 'tech_change_lsc', 'tech_corr', 'tech_concentric',
                'big_break', 'herd_behavior']
# factor_names = ['index_growth', 'tech_concentric', 'big_break', 'herd_behavior', 'profit_growth']

factors_corr(factor_names, '2016-06-30', '2024-03-27')

