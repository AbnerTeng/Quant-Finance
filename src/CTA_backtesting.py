# %% [markdown]
# ## CTA backtesting framework with BTCUSD and bollingerband strategy

# %% [markdown]
# ## import packages

# %%
from ast import parse
import os
from re import S
from tkinter import Grid
from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import datetime as dt
import time

path = os.getcwd()
print(path)
# %% [markdown]
# ## Data crawling

# %%
symbol = 'BTC/USD'

startTime = '2021-1-1'
endTime = '2022-7-22'
resolution = 60

startTimeStamp = dt.datetime.strptime(startTime, '%Y-%m-%d').timestamp()
endTimeStamp = dt.datetime.strptime(endTime, '%Y-%m-%d').timestamp()
data = []

import tqdm
for i in tqdm.tqdm(range(100)):
    while True:
        if startTimeStamp < endTimeStamp:
            t1 = startTimeStamp
            t2 = startTimeStamp + resolution * 1440

            url = f'https://ftx.com/api//markets/{symbol}/candles?resolution={resolution}&limit=1440&start_time={t1}&end_time={t2}'
            response = requests.get(url)
            if response.status_code == 200:
                result = response.json()['result'][:-1]
                data += result
            else:
                print('error: {}, {} ~ {}'.format(symbol, t1, t2))

            startTimeStamp += resolution * 1440
        else:
            break

data = pd.DataFrame(data)
data.index = pd.to_datetime(data['startTime'])
# %% [markdown]
# tidy data

# %%
data = data[['open', 'high', 'low', 'close', 'volume']]
data.to_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTCUSD.csv')
pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTCUSD.csv', parse_dates = True, index_col = 'startTime')
# %% [markdown]
# ## Strategy Backtesting
# data resample
# %%
rule = '1D'
d1 = data.resample(rule = rule, closed = 'right', label = 'right').first()[['open']]
d2 = data.resample(rule = rule, closed = 'right', label = 'right').max()[['high']]
d3 = data.resample(rule = rule, closed = 'right', label = 'right').min()[['low']]
d4 = data.resample(rule = rule, closed = 'right', label = 'right').last()[['close']]
d5 = data.resample(rule = rule, closed = 'right', label = 'right').sum()[['volume']]

df = pd.concat([d1, d2, d3, d4, d5], axis = 1)
df.iloc[0:3]
# %% [markdown]
# ## strategy
# * bollinger band

# %% [markdown]
# setting parameters

# %%
fund = 100
money = 100
feeRate = 0.002
length = 20
times_of_std = 2
df['ma'] = df['close'].rolling(window = length, center = False).mean()
df['std'] = df['close'].rolling(window = length, center = False).std()

for i in range(len(df)):
    df['upper_bound'] = df['ma'] + times_of_std * df['std']
    df['lower_bound'] = df['ma'] - times_of_std * df['std']
# %%
plt.figure(figsize = (12, 8))
plt.plot(df[['ma', 'upper_bound', 'lower_bound']], linestyle =  '--', color = 'gray')
plt.plot(df['close'], color = 'skyblue')
plt.grid(True)

# %%
B_or_S = None
buy = []
sell = []
short = []
buytocover = []
profit_list = [0]
profit_fee_list = [0]

for i in range(len(df)):

    if i == len(df) - 1:
        break

    if B_or_S == None:
        profit_list.append(0)
        profit_fee_list.append(0)

        if df['close'][i-1] > df['lower_bound'][i-1] and df['close'][i] < df['lower_bound'][i]:
            executeSize = money / df['open'][i+1]
            B_or_S = 'S'
            t = i + 1
            short.append(t)
        
        if df['close'][i-1] < df['upper_bound'][i-1] and df['close'][i] > df['upper_bound'][i]:
            executeSize = money / df['open'][i+1]
            B_or_S = 'B'
            t = i + 1
            buy.append(t)

    elif B_or_S == 'B':
        profit = executeSize * (df['open'][i+1] - df['open'][i])
        profit_list.append(profit)

        if (df['close'][i-1] >= df['lower_bound'][i-1] and df['close'][i] < df['lower_bound'][i]) or (i == len(df) - 2):
            pl_round = executeSize * (df['open'][i+1] - df['open'][t])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
            profit_fee_list.append(profit_fee)
            sell.append(i+1)
            B_or_S = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

    elif B_or_S == 'S':
        profit = executeSize * (df['open'][i] - df['open'][i+1])
        profit_list.append(profit)

        if (df['close'][i-1] <= df['upper_bound'][i-1] and df['close'][i] > df['upper_bound'][i]) or (i == len(df) - 2):
            pl_round = executeSize * (df['open'][t] - df['open'][i+1])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
            profit_fee_list.append(profit_fee)
            buytocover.append(i+1)
            B_or_S = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profitfee': np.cumsum(profit_fee_list)}, index = df.index)
print(equity)
equity.plot(grid = True, figsize = (12, 8))
     
# %%
print(buy)
print(short)
print(sell)
print(buytocover)
# %%
equity['drawdown_percent'] = (equity['profitfee'] / equity['profitfee'].cummax()) - 1
equity['drawdown'] = equity['profitfee'] - equity['profitfee'].cummax()
# %%
fig, ax = plt.subplots(figsize = (16, 6))
high_index = equity[equity['profitfee'].cummax() == equity['profitfee']].index
equity['profitfee'].plot(label = 'Total Profit', ax = ax, color = 'red', grid = True)
plt.fill_between(equity['drawdown'].index, equity['drawdown'], 0, facecolor = 'red', label = 'Drawdown', alpha = 0.5)
plt.scatter(high_index, equity['profitfee'].loc[high_index], color = '#02ff0f', label = 'High')

plt.legend()
plt.ylabel('Accumulated profit')
plt.xlabel('Time')
plt.title('Profit & Drawdown', fontsize = 15)


# %%
fig, ax = plt.subplots(figsize = (16, 6))
df['close'].plot(label = 'Close Price', ax = ax, color = 'gray', grid = True, alpha = 0.8)
plt.plot(df[['ma', 'upper_bound', 'lower_bound']], linestyle = '--')
plt.grid(True)
plt.scatter(df['close'].iloc[buy].index, df['close'].iloc[buy], color = 'red', label = 'Buy', marker = '^', s = 60)
plt.scatter(df['close'].iloc[sell].index, df['close'].iloc[sell], color = 'red', label = 'Sell', marker = 'v', s = 60)
plt.scatter(df['close'].iloc[buytocover].index, df['close'].iloc[buytocover], color = 'limegreen', label = 'Buy to Cover', marker = '^', s = 60)
plt.scatter(df['close'].iloc[short].index, df['close'].iloc[short], color = 'limegreen', label = 'Short', marker = 'v', s = 60)

plt.legend()
plt.ylabel('USD')
plt.xlabel('Time')
plt.title('Price Movement & Trade signal', fontsize = 15)

# %%
profit = equity['profitfee'].iloc[-1]
returns = (equity['profitfee'][-1] - equity['profitfee'][41])
mdd = abs(equity['drawdown']).max()
calmarRatio = returns / mdd
tradeTimes = len(buy) + len(short)
winRate = len([i for i in profit_fee_list if i > 0]) / len(profit_fee_list)
profitFactor = sum([i for i in profit_fee_list if i > 0]) / abs(sum([i for i in profit_fee_list if i < 0]))
WLRatio = np.mean([i for i in profit_fee_list if i > 0]) / abs(np.mean([i for i in profit_fee_list if i < 0]))

print(f'profit: ${np.round(profit,2)}')
print(f'returns: {np.round(returns,4)}%')
print(f'mdd: {np.round(mdd,4)}%')
print(f'calmarRatio: {np.round(calmarRatio,2)}')
print(f'tradeTimes: {tradeTimes}')
print(f'winRate: {np.round(winRate,4)*100}%')
print(f'profitFactor: {np.round(profitFactor,2)}')
print(f'winLossRatio: {np.round(WLRatio,2)}')
# %% [markdown]
# ## Strategy optimize

# %%
df_insample = df.loc[:'2022-2']
df_outofsample = df.loc['2022-3':]

rule = '1D'
df_open = df['open']

# %% [markdown]
# ## In-sample estimate

# %%
optimizationList = []
fund = 100
money = 100
feeRate = 0.02

rule = '1D'

for length in range(10, 110, 10):
    for times_of_std in np.arange(0.5, 3, 0.5):

        times_of_std = np.round(times_of_std, 1)

        print('----------')
        print(f'length: {length}')
        print(f'times_of_std: {times_of_std}')

        d1 = df_insample.resample(rule=rule, closed='right', label='right').first()[['open']]
        d2 = df_insample.resample(rule=rule, closed='right', label='right').max()[['high']]
        d3 = df_insample.resample(rule=rule, closed='right', label='right').min()[['low']]
        d4 = df_insample.resample(rule=rule, closed='right', label='right').last()[['close']]
        d5 = df_insample.resample(rule=rule, closed='right', label='right').sum()[['volume']]
        df = pd.concat([d1,d2,d3,d4,d5], axis=1)

        df['ma'] = df['close'].rolling(window = length, center = False).mean()
        df['std'] = df['close'].rolling(window = length, center = False).std()
        #for i in range(len(df)):
         #   df['upper_bound'][i] = df['ma'][i] + times_of_std * df['std'][i]
          #  df['lower_bound'][i] = df['ma'][i] - times_of_std * df['std'][i]

        B_or_S = None
        buy = []
        sell = []
        short = []
        buytocover = []
        profit_list = [0]
        profit_fee_list = [0]

        for i in range(len(df)):

            if i == len(df) - 1:
                break

            if B_or_S == None:
                profit_list.append(0)
                profit_fee_list.append(0)

                if df['close'][i-1] > (df['ma'][i-1] - times_of_std * df['std'][i-1]) and df['close'][i] < (df['ma'][i] - times_of_std * df['std'][i]):
                    executeSize = money / df['open'][i+1]
                    B_or_S = 'S'
                    t = i + 1
                    short.append(t)
        
                if df['close'][i-1] < (df['ma'][i-1] + times_of_std * df['std'][i-1]) and df['close'][i] > (df['ma'][i] + times_of_std * df['std'][i]):
                    executeSize = money / df['open'][i+1]
                    B_or_S = 'B'
                    t = i + 1
                    buy.append(t)

            elif B_or_S == 'B':
                profit = executeSize * (df['open'][i+1] - df['open'][i])
                profit_list.append(profit)

                if (df['close'][i-1] >= df['ma'][i-1] - times_of_std * df['std'][i-1] and df['close'][i] < df['ma'][i] - times_of_std * df['std'][i]) or (i == len(df) - 2):
                    pl_round = executeSize * (df['open'][i+1] - df['open'][t])
                    profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
                    profit_fee_list.append(profit_fee)
                    sell.append(i+1)
                    B_or_S = None
        
                else:
                    profit_fee = profit
                    profit_fee_list.append(profit_fee)

            elif B_or_S == 'S':
                profit = executeSize * (df['open'][i] - df['open'][i+1])
                profit_list.append(profit)

                if (df['close'][i-1] <= df['ma'][i-1] + times_of_std * df['std'][i-1] and df['close'][i] > df['ma'][i] + times_of_std * df['std'][i]) or (i == len(df) - 2):
                    pl_round = executeSize * (df['open'][t] - df['open'][i+1])
                    profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
                    profit_fee_list.append(profit_fee)
                    buytocover.append(i+1)
                    B_or_S = None
        
                else:
                    profit_fee = profit
                    profit_fee_list.append(profit_fee)

        equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profitfee': np.cumsum(profit_fee_list)}, index = df.index)
        equity['drawdown_percent'] = (equity['profitfee'] / equity['profitfee'].cummax()) - 1
        equity['drawdown'] = equity['profitfee'] - equity['profitfee'].cummax()
        returns = (equity['profitfee'][-1] - equity['profitfee'][41])
        mdd = abs(equity['drawdown']).max()
        calmarRatio = returns / mdd

        optimizationList.append([length, times_of_std, returns, calmarRatio])
# %%
optResult = pd.DataFrame(optimizationList, columns = ['length', 'times_of_std', 'returns', 'calmarRatio'])
optResult
# %%
pic = optResult.pivot('length', 'times_of_std', 'returns')
sns.heatmap(data = pic).set(title='Return')
# %%
pic = optResult.pivot('length', 'times_of_std', 'calmarRatio')
sns.heatmap(data = pic).set(title='Calmar Ratio')

# %% [markdown]
# ## Out of sample

# %%
fund = 100
money = 100
feeRate = 0.02
length = 40
times_of_std = 1
rule = '1D'

d1 = df_outofsample.resample(rule=rule, closed='right', label='right').first()[['open']]
d2 = df_outofsample.resample(rule=rule, closed='right', label='right').max()[['high']]
d3 = df_outofsample.resample(rule=rule, closed='right', label='right').min()[['low']]
d4 = df_outofsample.resample(rule=rule, closed='right', label='right').last()[['close']]
d5 = df_outofsample.resample(rule=rule, closed='right', label='right').sum()[['volume']]
df = pd.concat([d1,d2,d3,d4,d5], axis=1)

df['ma'] = df['close'].rolling(window = length, center = False).mean()
df['std'] = df['close'].rolling(window = length, center = False).std()
##for i in range(len(df)):
    ##df['upper_bound'][i] = df['ma'][i] + times_of_std * df['std'][i]
    ##df['lower_bound'][i] = df['ma'][i] - times_of_std * df['std'][i]
#%%
B_or_S = None
buy = []
sell = []
short = []
buytocover = []
profit_list = [0]
profit_fee_list = [0]

for i in range(len(df)):

    if i == len(df) - 1:
        break

    if B_or_S == None:
        profit_list.append(0)
        profit_fee_list.append(0)

        if df['close'][i-1] > (df['ma'][i-1] - times_of_std * df['std'][i-1]) and df['close'][i] < (df['ma'][i] - times_of_std * df['std'][i]):
            executeSize = money / df['open'][i+1]
            B_or_S = 'S'
            t = i + 1
            short.append(t)

        if df['close'][i-1] < (df['ma'][i-1] + times_of_std * df['std'][i-1]) and df['close'][i] > (df['ma'][i] + times_of_std * df['std'][i]):
            executeSize = money / df['open'][i+1]
            B_or_S = 'B'
            t = i + 1
            buy.append(t)

    elif B_or_S == 'B':
        profit = executeSize * (df['open'][i+1] - df['open'][i])
        profit_list.append(profit)

        if (df['close'][i-1] >= df['ma'][i-1] - times_of_std * df['std'][i-1] and df['close'][i] < df['ma'][i] - times_of_std * df['std'][i]) or (i == len(df) - 2):
            pl_round = executeSize * (df['open'][i+1] - df['open'][t])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
            profit_fee_list.append(profit_fee)
            sell.append(i+1)
            B_or_S = None

        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

    elif B_or_S == 'S':
        profit = executeSize * (df['open'][i] - df['open'][i+1])
        profit_list.append(profit)

        if (df['close'][i-1] <= df['ma'][i-1] + times_of_std * df['std'][i-1] and df['close'][i] > df['ma'][i] + times_of_std * df['std'][i]) or (i == len(df) - 2):
            pl_round = executeSize * (df['open'][t] - df['open'][i+1])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
            profit_fee_list.append(profit_fee)
            buytocover.append(i+1)
            B_or_S = None

        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=df.index)
equity['drawdown_percent'] = (equity['profitfee'] / equity['profitfee'].cummax()) - 1
equity['drawdown'] = equity['profitfee'] - equity['profitfee'].cummax()

profit = equity['profitfee'].iloc[-1]
returns = (equity['profitfee'][-1] - equity['profitfee'][41])
mdd = abs(equity['drawdown']).max()
calmarRatio = returns / mdd
tradeTimes = len(buy) + len(short)
winRate = len([i for i in profit_fee_list if i > 0]) / len(profit_fee_list)
profitFactor = sum([i for i in profit_fee_list if i > 0]) / abs(sum([i for i in profit_fee_list if i < 0]))
WLRatio = np.mean([i for i in profit_fee_list if i > 0]) / abs(np.mean([i for i in profit_fee_list if i < 0]))

print(f'profit: ${np.round(profit,2)}')
print(f'returns: {np.round(returns,4)}%')
print(f'mdd: {np.round(mdd,4)}%')
print(f'calmarRatio: {np.round(calmarRatio,2)}')
print(f'tradeTimes: {tradeTimes}')
print(f'winRate: {np.round(winRate,4)*100}%')
print(f'profitFactor: {np.round(profitFactor,2)}')
print(f'winLossRatio: {np.round(WLRatio,2)}')

# %%
fig, ax = plt.subplots(figsize = (16,6))

high_index = equity[equity['profitfee'].cummax() == equity['profitfee']].index
equity['profitfee'].plot(label = 'Total Profit', ax = ax, color = 'red', grid=True)
plt.fill_between(equity['drawdown'].index, equity['drawdown'], 0, facecolor  = 'r', label = 'Drawdown', alpha=0.5)
plt.scatter(high_index, equity['profitfee'].loc[high_index],c = '#02ff0f', label = 'High')

plt.legend()
plt.ylabel('Accumulated Profit')
plt.xlabel('Time')
plt.title('Profit & Drawdown',fontsize  = 16)

# %%
