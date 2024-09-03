import pandas as pd
import time
import numpy as np
import glob

# factors = ['total_ncl', 'total_assets', 'total_liab', 'oth_eqt_tools_p_shr']  # 资产负债表包含数据
# datas = {}

# for f in factors:
#     datas[f] = pd.read_csv(f+'.csv', index_col=0)
#     datas[f].index = pd.to_datetime(datas[f].index)
#     print(f)
#     print(datas[f].isnull().sum())
#     print(len(datas[f].index))
#
'''创建csv'''
# sample = pd.read_csv(factors[0]+'.csv', index_col=0)
# data = pd.DataFrame(None, index=sample.index, columns=sample.columns)
# data.index = pd.to_datetime(data.index)
# data.to_csv('circ_mv.csv')

# print(data)

'''获取 stock_codes.csv'''
# codes_list = []
# codes = pd.read_csv('../申万一级权重/classification_L1.csv', index_col=0).loc[:, ['index_code', 'industry_name']]
# for i in range(len(codes.index)):
#     ind_num= i
#     ind_index = codes.index_code[ind_num]
#     ind_name = codes.industry_name[ind_num]
#     stocks = pd.read_csv('../申万一级权重/weights_' + ind_name + '.csv', index_col=0).loc[:, 'con_code']
#     for code in stocks:
#         codes_list.append(code)
#
# codes_list = pd.Series(codes_list)
#
# codes_list.to_csv('stock_codes.csv')

'''安装wind api'''
# import os
# import site
# os.system('mkdir -p ' + site.getusersitepackages())
# os.system('ln -sf "/Applications/Wind API.app/Contents/python/WindPy.py"' + ' ' + site.getusersitepackages())
# os.system('ln -sf ~/Library/Containers/com.wind.mac.api/Data/.Wind ~/.Wind')

'''测试wind api'''
# from WindPy import *
#
# ret = w.start()
# # print(ret)
# ret = w.isconnected()
# # print(ret)
# #test WSD function
# ret = w.wset("indexconstituent", "date=20240628;windcode=801950.SI;field=wind_code,i_weight")
# print(ret.Data)
# df1 = pd.DataFrame(ret.Data)
# print(df1)
#
# ret = w.wset("indexconstituent", "date=20240531;windcode=801950.SI;field=wind_code,i_weight")
# df2 = pd.DataFrame(ret.Data)
# print(df2)

'''scratch'''


'''计算后行业风格因子，与提取后的风格因子数据对比'''

cal_lev = pd.read_csv('申万一级月频杠杆因子.csv', index_col=0)
cal_lev.index = pd.to_datetime(cal_lev.index)
cal_lev = cal_lev.loc[cal_lev.index > pd.to_datetime('20220701')]
cal_lev = cal_lev.groupby(cal_lev.index.to_period('M')).last()

for ind in cal_lev.columns:
    file_pattern = 'wind行业提取因子/' + ind + '*.xlsx'
    matching_files = glob.glob(file_pattern)
    if matching_files:
        file_to_read = matching_files[0]
    else:
        print('no matching files')
        continue

    data = pd.read_excel(file_to_read, sheet_name='历史风格走势', index_col=0, skiprows=[1])
    data.index = pd.to_datetime(data.index, format='%Y%m%d')
    data = data.groupby(data.index.to_period('M')).last()

    print(ind + ':', data['杠杆'].corr(cal_lev[ind]))




'''3306 mySQL数据库'''
# import pymysql
#
# dbName = 'barra_cne6'
# try:
#     db =  pymysql.connect(host='192.168.10.2',
#                       port=3306,
#                       user='quant_data',
#                       passwd='quant_pass',
#                       )
#     print('successfully connected')
# except pymysql.MySQLError as e:
#     print('erro occur:', e)
#
# try:
#     cur = db.cursor()
#     # sql = f"SELECT TABLE_NAME \
#     #         FROM INFORMATION_SCHEMA.TABLES \
#     #         WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA='{dbName}';"
#     # sql = "SELECT * FROM barra_liquidity LIMIT 5;"
#     sql = "SHOW DATABASES"
#     cur.execute(sql)
#     # Fetch all the rows
#     tables = cur.fetchall()
#     print(tables)
#
#     # columns = [col[0] for col in cur.description]
#
#     # # Create DataFrame
#     # df = pd.DataFrame(tables, columns=columns)
#     # print(df)
#     # print(df.columns)
#
#     db.commit()
# except pymysql.MySQLError as e:
#     print('erro occur:', e)
#     db.rollback()
# finally:
#     cur.close()
#     db.close()

'''mongodb 数据库'''
# from pymongo import MongoClient
#
# # Replace with your MongoDB connection string
# uri = 'mongodb://user1:123456@192.168.10.2:27017/wind_data_history'
#
# # Create a MongoClient
# client = MongoClient(uri)
#
# # Access the database
# db = client['wind_data_history']
#
# # Print the list of collections
# # col_list = db.list_collection_names()
# # for c in col_list:
# #     print(c)
#
# # Example: Access a collection
# # collection = db['ASHAREFINANCIALINDICATOR']
# collection = db['ASHAREEODDERIVATIVEINDICATOR']
#
# dates = pd.read_csv('../申万一级权重/wind_weights/农林牧渔_weights.csv', index_col=0).index
# date = pd.to_datetime(dates[1]).strftime('%Y%m%d')
# print(date)
# query = {'TRADE_DT': date}
#
# document = list(collection.find(query))
#
#
# # Close the connection
# client.close()
#
# df = pd.DataFrame(document)
# print(df.loc[:, ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_MV']])
# # print(df.columns)


