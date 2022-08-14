# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import datetime as dt
import time

plt.style.use('ggplot')

# %%
path = os.getcwd()
print(path)

# %%
df = pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/TWF_Futures_Minute_Trade.txt')
df
# %%
df.index = pd.to_datetime(df['Date']+' '+df['Time'])
df = df.drop(columns = ['Date', 'Time'])
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df['Hour'] = df.index.map(lambda x: x.hour)
df.head()
# %%
plt.figure(figsize = (15, 7))
plt.plot(df['Close'], color = 'blue', alpha = 0.5, label = 'Close Price')
plt.plot(df['Volume'], color = '#01889f', alpha = 0.5, label = 'Volume')
plt.legend()
plt.show()
# %%
rule = '3T'
Morning = df[(df['Hour'] >= 8) & (df['Hour'] <= 13)] ## 日盤
Morning.index = Morning.index + dt.timedelta(minutes = 15)
Morning.iloc[0:8]
# %%
Morning.resample(rule = rule, closed = 'right', label = 'left').first()['Open'].iloc[0:3]
# %%
rule = '60T'
Morning = df[(df['Hour'] >= 8) & (df['Hour'] <= 13)] ## 日盤
Morning.index = Morning.index + dt.timedelta(minutes = 15)

d1 = Morning.resample(rule = rule, closed = 'right', label = 'left').first()['Open']
d2 = Morning.resample(rule = rule, closed = 'right', label = 'left').max()['High']
d3 = Morning.resample(rule = rule, closed = 'right', label = 'left').min()['Low']
d4 = Morning.resample(rule = rule, closed = 'right', label = 'left').last()['Close']
d5 = Morning.resample(rule = rule, closed = 'right', label = 'left').sum()['Volume']

df_Morning = pd.concat([d1, d2, d3, d4, d5], axis = 1)
df_Morning = df_Morning.dropna()
df_Morning.index = df_Morning.index - dt.timedelta(minutes = 15)
df_Morning.head()

# %%
rule = '60T'
Night = df[(df['Hour'] < 8) | (df['Hour'] > 13)] ## 夜盤

d1 = Night.resample(rule = rule, closed = 'right', label = 'left').first()['Open']
d2 = Night.resample(rule = rule, closed = 'right', label = 'left').max()['High']
d3 = Night.resample(rule = rule, closed = 'right', label = 'left').min()['Low']
d4 = Night.resample(rule = rule, closed = 'right', label = 'left').last()['Close']
d5 = Night.resample(rule = rule, closed = 'right', label = 'left').sum()['Volume']

df_Night = pd.concat([d1, d2, d3, d4, d5], axis = 1)
df_Night = df_Night.dropna()
df_Night.head()

# %%
df_Day = pd.concat([df_Morning, df_Night], axis = 0)
df_Day = df_Day.sort_index(ascending = True)
df_Day.head()
# %% [markdown]
# ## 日K 處理

# %%
df_day = df.loc['2017-05-15':].copy()
df_day.index = df_day.index - dt.timedelta(hours = 8)

d1 = df_day.resample(rule = '1D', closed = 'right', label = 'left').first()['Open']
d2 = df_day.resample(rule = '1D', closed = 'right', label = 'left').max()['High']
d3 = df_day.resample(rule = '1D', closed = 'right', label = 'left').min()['Low']
d4 = df_day.resample(rule = '1D', closed = 'right', label = 'left').last()['Close']
d5 = df_day.resample(rule = '1D', closed = 'right', label = 'left').sum()['Volume']

df_day = pd.concat([d1, d2, d3, d4, d5], axis = 1)
df_day = df_day.dropna()
df_day.index = df_day.index + dt.timedelta(days = 1)
df_day.tail()

# %%
df_Morning['Hour'] = df_Morning.index.map(lambda x: x.hour)

trainData = df_Morning[(df_Morning.index >= '2011-01-01 00:00:00') & (df_Morning.index <= '2019-12-31 00:00:00')].copy()
testData = df_Morning[(df_Morning.index >= '2020-01-01 00:00:00') & (df_Morning.index <= '2022-05-22 00:00:00')].copy()
# %%
trainData.head()
# %%
testData.tail()
# %%
settlementDate = pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/settlementDate.csv')
settlementDate.columns = ['settlementDate', 'futures', 'settlementPrice']
settlementDate
# %%
bool = [False if 'W' in i else True for i in settlementDate['futures']]
# %%
settlementDate = [i.replace('/','-') for i in list(settlementDate[bool]['settlementDate'])]
settlementDate = [pd.to_datetime(i).date() for i in settlementDate]
settlementDate = settlementDate
# %%
fund = 1000000
fee = 600
length = 20
time_std = 2
slpoint = 0.05

trainData['SMA'] = trainData['Close'].rolling(window = length, center = False).mean()
trainData['std'] = trainData['Close'].rolling(window = length, center = False).std()
trainData['BBU'] = trainData['SMA'] + time_std * trainData['std']
trainData['BBL'] = trainData['SMA'] - time_std * trainData['std']
trainData['BBW'] = (trainData['BBU'] - trainData['BBL']) / trainData['SMA']
trainData['shortSMA'] = trainData['Close'].rolling(window = 5, center = False).mean()
trainData['longSMA'] = trainData['Close'].rolling(window = 21, center = False).mean()

trainData.tail()

# %%
df_arr = np.array(trainData)
time_arr = np.array(trainData.index)
date_arr = [pd.to_datetime(i).date() for i in time_arr]
# %%
signal = 0
buy = []
sell = []
short = []
cover = []
profit_list = [0]
profit_with_fee_list = [0]
profit_with_fee_list_realized = [0]

for i in range(len(df_arr)):

    if i == len(df_arr):
        break

    entryLong = (df_arr[i-1, 10] < 0.015 and df_arr[i, 10] >= 0.015) and df_arr[i, 11] > df_arr[i, 12]
    entryShort = (df_arr[i-1, 10] < 0.015 and df_arr[i, 10] >= 0.015) and df_arr[i, 11] < df_arr[i, 12]
    entryCondition = date_arr[i] not in settlementDate

    exitShort = (df_arr[i-1, 10] > 0.015 and df_arr[i, 10] <= 0.015)
    exitCover = (df_arr[i-1, 10] > 0.015 and df_arr[i, 10] <= 0.015) 
    exitCondition = date_arr[i] in settlementDate and df_arr[i, 5] >= 11

    if signal == 1:
        stopLoss = df_arr[i, 3] <= df_arr[t, 0] * (1-slpoint)
    elif signal == -1:
        stopLoss = df_arr[i, 3] >= df_arr[t, 0] * (1+slpoint)
    
    if signal == 0:
        profit_list.append(0)
        profit_with_fee_list.append(0)

        if entryLong and entryCondition:
            signal = 1
            t = i + 1
            buy.append(t)

        elif entryShort and entryCondition:
            signal = -1
            t = i + 1
            short.append(t)
    
    elif signal == 1:
        profit = 200 * (df_arr[i+1, 0] - df_arr[i, 0])
        profit_list.append(profit)

        if exitShort or i == len(df_arr)-2 or exitCondition or stopLoss:
            Pnl = 200 * (df_arr[i+1, 0] - df_arr[i, 0])
            profit_with_fee = profit - fee * 2
            profit_with_fee_list.append(profit_with_fee)
            sell.append(i+1)
            signal = 0

            profit_with_fee_realized = Pnl - fee * 2
            profit_with_fee_list_realized.append(profit_with_fee_realized)

        else:
            profit_with_fee = profit
            profit_with_fee_list.append(profit_with_fee)

    elif signal == -1:
        profit = 200 * (df_arr[i, 0] - df_arr[i+1, 0])
        profit_list.append(profit)

        if exitCover or i == len(df_arr)-2 or exitCondition or stopLoss:
            Pnl = 200 * (df_arr[i, 0] - df_arr[i+1, 0])
            profit_with_fee = profit - 2 * fee
            profit_with_fee_list.append(profit_with_fee)
            cover.append(i+1)
            signal = 0

            profit_with_fee_realized = Pnl - fee * 2
            profit_with_fee_list_realized.append(profit_with_fee_realized)

        else:
            profit_with_fee = profit
            profit_with_fee_list.append(profit_with_fee)

equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profitfee': np.cumsum(profit_with_fee_list)})
equity = equity.drop(0)
equity = equity.set_index(trainData.index)
print(equity)
equity.plot(grid = True, figsize = (12, 8))
# %%
