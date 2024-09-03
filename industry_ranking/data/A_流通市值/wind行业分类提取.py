import pandas as pd
import time
import numpy as np
from pymongo import MongoClient


sample = pd.read_csv('A股流通市值.csv', index_col=0)
sample_index = sample.index
sample_col = sample.columns

data = pd.DataFrame(None, index=sample_index, columns=sample_col)
# MongoDB connection string
uri = 'mongodb://user1:123456@192.168.10.2:27017/wind_data_history'

# Create a MongoClient
client = MongoClient(uri)
db = client['wind_data_history']
collection = db['ASHAREINDUSTRIESCLASS']

for code in sample_col:
    print(code)
    # query = {'REMOVE_DT': {'$eq': None}}
    query = {'S_INFO_WINDCODE': {'$eq': code}}
    document = list(collection.find(query))
    if len(document) == 0:
        continue
    df = pd.DataFrame(document).loc[:, ['S_INFO_WINDCODE', 'WIND_IND_CODE', 'ENTRY_DT', 'REMOVE_DT', 'CUR_SIGN']]
    print(df)
    if len(df.index) == 1:  # 没有更改记录
        dates = sample_index[sample_index > int(df.loc[0, 'ENTRY_DT'])]
        data.loc[dates, code] = int(df.loc[0, 'WIND_IND_CODE'])
    else:  # 有更改记录
        for i in df.index:
            if int(df.loc[i, 'CUR_SIGN']) == 1:  # current industry
                dates = sample_index[sample_index > int(df.loc[i, 'ENTRY_DT'])]
            else:  # past industry
                dates = sample_index[sample_index < int(df.loc[i, 'REMOVE_DT'])]
            data.loc[dates, code] = int(df.loc[0, 'WIND_IND_CODE'])

data.to_csv('A股wind行业分类.csv')
