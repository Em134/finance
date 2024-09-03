import math
import time #计算运行时间
import pandas as pd
import numpy as np
import requests #同花顺
from datetime import datetime, timedelta # 获取当前日期
import warnings
warnings.filterwarnings("ignore")
# import tushare as ts
# token = ""
# ts.set_token(token)
# pro = ts.pro_api()

'''指数名称及代码'''
# data = pro.index_classify(level='L1', src='SW2021')
# print(data)
# data.to_csv('classification_L1.csv')

'''指数成分'''
# codes = pd.read_csv('classification_L1.csv', index_col=0).loc[:, ['index_code', 'industry_name']]
# # print(codes)
#
# for i in range(len(codes.index)):
#     ind_num= i
#     ind_index = codes.index_code[ind_num]
#     ind_name = codes.industry_name[ind_num]
#
#     print(ind_name)
#     df = pro.index_member(index_code=ind_index)
#     df.to_csv('raw/weights_' + ind_name + '.csv')

'''权重计算'''
# circ_mv = pd.read_csv('../杠杆因子/circ_mv.csv', index_col=0)
# circ_mv.index = pd.to_datetime(circ_mv.index)
# circ_mv = circ_mv.dropna(how='all').ffill()  # 删除非工作日，填补数据
# circ_mv = circ_mv.resample('ME').last()  # 选取每月最后一个交易日
#
# codes = pd.read_csv('classification_L1.csv', index_col=0).loc[:, ['index_code', 'industry_name']]
# for ind_name in codes.industry_name:
#     print(ind_name)
#     raw_weights = pd.read_csv('raw/weights_' + ind_name +'.csv', index_col=0)  # read in-out file
#     # 调整日期格式
#     raw_weights['in_date'] = pd.to_datetime(raw_weights['in_date'])
#     raw_weights['out_date'] = pd.to_datetime(raw_weights['out_date'])
#     # create empty df 月频
#     weights = pd.DataFrame(0.0, index=circ_mv.index, columns=raw_weights['con_code'])
#     for date in circ_mv.index:  # 遍历日期
#         for i in raw_weights.index:
#             con_code = raw_weights.loc[i, 'con_code']
#             if date < raw_weights.loc[i, 'in_date'] or date >= raw_weights.loc[i, 'out_date']: # 如果不在纳入日期内
#                 weights.loc[date, con_code] = 0  # 当日这个成分股的权重（流动市值）为0
#             else:  # 如果在的话，它的权重就是它的流动市值
#                 if pd.isnull(circ_mv.loc[date, con_code]):
#                     continue
#                 else:
#                     weights.loc[date, con_code] = circ_mv.loc[date, con_code]
#
#     for i in weights.index:
#         total = sum(weights.loc[i])
#         if total ==0:
#             continue
#         weights.loc[i] = weights.loc[i]/total
#
#     weights.to_csv('weights/weights_' + ind_name + '.csv')
'''权重计算结束'''

'''万得权重数据提取'''
# from WindPy import *
#
#
# def weights_extractor(sliced_ind_name, windcode):
#     print('In weights extractor:', sliced_ind_name, windcode)
#     sample = pd.read_excel('../杠杆因子/个股分类因子/' + sliced_ind_name + '.xlsx', skiprows=3, index_col=0)
#     sample.index = pd.to_datetime(sample.index)
#     weights = pd.DataFrame(0.0, index=sample.index, columns=sample.columns)  # index are date(s), columns are codes
#
#     for date in weights.index:
#         date_string = date.strftime('%Y%m%d')
#         try:
#             ret = w.wset("indexconstituent", "date=" + date_string + ";windcode=" + windcode + ";field=wind_code,i_weight")
#             if ret.ErrorCode == -40520017:
#                 print('\tdate is', date_string)
#                 print(ret)
#                 continue
#             df = pd.DataFrame(ret.Data)
#             df.columns = df.iloc[0]
#             df = df[1:]
#             df.reset_index(drop=True, inplace=True)
#             for col in df.columns:
#                 weights.loc[date, col] = df.loc[0, col]
#         except Exception as e:  # 如果失败了：
#             print('oops, saving current process')
#             # weights.to_csv('./wind_weights/' + sliced_ind_name + '_weights.csv')
#             print(ret)
#             print(date_string)
#             print(f"An error occurred: {e}")
#             exit(1)
#
#     # ends for loop
#
#     weights.to_csv('./wind_weights/' + sliced_ind_name + '_weights.csv')
#
# ret = w.start()
# ret = w.isconnected()
#
# industries = pd.read_excel('申银万国一级行业指数.xlsx')
# for i in industries.index:
#     ind_name = industries.loc[i, '名称']
#     code = industries.loc[i, '代码']
#     sliced_ind_name = ind_name[:-4]
#     weights_extractor(sliced_ind_name, code)

'''testing, 增加已被剔除的股票因子数据'''
sliced_ind_name = '建筑装饰'
weights = pd.read_csv('./wind_weights/' + sliced_ind_name + '_weights.csv', index_col=0)

columns_with_nan = weights.columns[weights.isnull().any()].tolist()


weights = weights.fillna(0)
sum_w = weights.sum(axis=1)
not_equal_to_one_count1 = (sum_w < 99.999).sum()
not_equal_to_one_count2 = (sum_w > 100.001).sum()

print(sum_w)
print('not = 1:', str(not_equal_to_one_count1 + not_equal_to_one_count2))

print(columns_with_nan)

