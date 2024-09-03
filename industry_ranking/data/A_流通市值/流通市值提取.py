import pandas as pd
import time
import numpy as np
from pymongo import MongoClient

# Replace with your MongoDB connection string
uri = 'mongodb://user1:123456@192.168.10.2:27017/wind_data_history'

# Create a MongoClient
client = MongoClient(uri)
db = client['wind_data_history']
collection = db['ASHAREEODDERIVATIVEINDICATOR']

dates = pd.read_csv('../申万一级权重/wind_weights/农林牧渔_weights.csv', index_col=0).index
dates = pd.to_datetime(dates).strftime('%Y%m%d')[:-4]

col = pd.read_excel('temp.xlsx')['证券代码']

data = pd.DataFrame(None, index=dates, columns=col)

for date in dates:
    print(date)
    query = {'TRADE_DT': date}
    document = list(collection.find(query))

    df = pd.DataFrame(document).loc[:, ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_MV']]
    data.loc[date, list(df['S_INFO_WINDCODE'])] = np.array(df['S_DQ_MV'])
    # for i in df.index:
    #     code = df.loc[i, 'S_INFO_WINDCODE']
    #     date = df.loc[i, 'TRADE_DT']
    #     value = df.loc[i, 'S_DQ_MV']
    #     data.loc[date, code] = value

# Close the connection
client.close()
data.to_csv('A股流通市值.csv')
