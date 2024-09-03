import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import 行业加速 as speedup
import barra机器学习 as ml
import matplotlib.pyplot as plt
import barra残差动量 as barra


industries = pd.read_csv('data/申万31大行业_close.csv').columns[1:]
# Values to remove
values_to_remove = ['银行(申万)', '非银金融(申万)']
# Remove values using difference()
industries = industries.difference(values_to_remove)

close = pd.read_csv('data/申万31大行业_close.csv', index_col=0)
close.index = pd.to_datetime(close.index, format='%Y-%m-%d')
close = close.groupby(close.index.to_period('M')).apply(lambda x: x.iloc[-1])


score1 = ml.get_prediction(industries)
score1 = score1.iloc[12:, :]

score2 = speedup.speedup_score()
score2 = score2.loc[score1.index, score1.columns]

speed = barra.rotation_speed(close).rolling(3).mean()
speed = speed.loc[score1.index]

score = pd.DataFrame(0, score1.index, score1.columns)

for date in score1.index:
    if speed.loc[date] > 130:
        # print(date)
        score.loc[date] = score2.loc[date]
    else:
        score.loc[date] = score1.loc[date]

output = barra.ranking_backtest(close, score)
barra.report(output)

