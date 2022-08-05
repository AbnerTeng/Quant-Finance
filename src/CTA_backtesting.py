# %% [markdown]
# ## CTA backtesting framework with BTCUSD and bollingerband strategy

# %% [markdown]
# ## import packages

# %%
from ast import parse
from http.client import responses
import os
from re import S
from sqlite3 import PrepareProtocol
from tkinter import Grid
from tkinter.messagebox import NO
from tkinter.ttk import Style
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
symbol = 'BTC-PERP'

startTime = '2021-1-1'
endTime = '2022-7-31'
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
# ## Tidy data

# %%
data = data[['open', 'high', 'low', 'close', 'volume']]
data.to_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTC_PERP.csv')
pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTC_PERP.csv', parse_dates = True, index_col = 'startTime')
# %% [markdown]
# ## Fee rate

# %%
symbol = 'BTC-PERP'
startTime = '2021-01-01'
endTime = '2022-07-31'
startTimeStamp = dt.datetime.strptime(startTime, '%Y-%m-%d').timestamp()
endTimeStamp = dt.datetime.strptime(endTime, '%Y-%m-%d').timestamp()
data = []

while True:
    if startTimeStamp < endTimeStamp:
        t1 = startTimeStamp
        t2 = startTimeStamp + 86400

        url = 'https://ftx.com/api/funding_rates?start_time={}&end_time={}&future={}'.format(t1, t2, symbol)
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()['result'][:-1][::-1]
            data += result
        else:
            print('error: {}, {} ~ {}'.format(symbol, t1, t2))
        startTimeStamp += 86400
    else:
        break

data = pd.DataFrame(data)
data.index = pd.to_datetime(data['time'])
data = data[['rate']]
data.to_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTC_funding.csv'.format(symbol.split('-')[0]))

# %%
pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTC_funding.csv')
pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTC_funding.csv', parse_dates=True, index_col='time')

# %%
data = pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTC_PERP.csv', parse_dates=True, index_col='startTime')
funding = pd.read_csv('/Users/abnerteng/GitHub/TMBA-projects/data/BTC_funding.csv', parse_dates=True, index_col='time')
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
feeRate = 0.0015
length = 20
times_of_std = 2
df['ma'] = df['close'].rolling(window = length, center = False).mean()
df['std'] = df['close'].rolling(window = length, center = False).std()
df['shortSMA'] = df['close'].rolling(window = 13, center = False).mean()
df['longSMA'] = df['close'].rolling(window = 21, center = False).mean()

# %%
upper_bound_list = [0]
lower_bound_list = [0]
BBW_list = [0]
threshold_list = [0]
spread_list = [0]

for i in range(len(df)):
    upper_bound = df['ma'][i] + times_of_std * df['std'][i]
    upper_bound_list.append(upper_bound)

for i in  range(len(df)):
    lower_bound = df['ma'][i] - times_of_std * df['std'][i]
    lower_bound_list.append(lower_bound)

for i in range(len(df)):
    BBW = (upper_bound_list[i] - lower_bound_list[i]) / df['ma'][i]
    BBW_list.append(BBW)

for i in range(len(df)):
    threshold = 0.25
    threshold_list.append(threshold)

for i in range(len(df)):
    spread = df['shortSMA'][i] - df['shortSMA'][i-1]
    spread_list.append(spread)

BBdata = pd.DataFrame([upper_bound_list, lower_bound_list, BBW_list, threshold_list, spread_list])
BBdata = pd.DataFrame.transpose(BBdata)
BBdata.columns = ['upper_bound', 'lower_bound', 'BBW', 'threshold', 'spread']

BBdata = BBdata.drop(0)
BBdata = BBdata.set_index(df.index)
# %%
import mplfinance as mpf

candle_data = df[['open', 'high', 'low', 'close', 'volume']]
BBand = [ mpf.make_addplot(BBdata['lower_bound'], color = 'blue'), 
          mpf.make_addplot(BBdata['upper_bound'], color = 'blue'),
          mpf.make_addplot(df['ma'], color = 'gray'),
          mpf.make_addplot(BBdata['BBW'], panel = 1, ylabel = 'BBW'),
          mpf.make_addplot(BBdata['threshold'], panel = 1, secondary_y = False), 
          mpf.make_addplot(df['shortSMA'], panel = 2, ylabel = 'SMA'),
          mpf.make_addplot(df['longSMA'], panel = 2, secondary_y = False) ]
mpf.plot(candle_data, type = 'candle', style = 'binance', addplot = BBand, figratio = (18, 10), title = 'Band width')


# %%
# Include funding rate
rule = '1H'
df_hour = data.resample(rule = rule, closed = 'right', label = 'right').first()['open']
df_funding = pd.concat([df_hour, funding], axis = 1)
df_funding = df_funding.fillna(method = 'bfill', limit = 2).fillna(0)
df_funding

# %%
def fundingPayment(df_funding, side, unit, t1, t2):
    if len(df_funding.loc[t1:t2]) == 0:
        fee = 0
    else:
        funding_rate = np.array(df_funding.loc[t1:t2])
        fee = unit * np.dot(funding_rate[:,0], funding_rate[:,1])
    
    if side == 'long':
        return -fee
    elif side == 'short':
        return fee

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

        if BBdata['BBW'][i-1] < BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i] and df['shortSMA'][i] < df['longSMA'][i]:
            executeSize = money / df['open'][i+1]
            B_or_S = 'S'
            t = i + 1
            short.append(t)
            t1 = df.index[i+1]
        
        if BBdata['BBW'][i-1] < BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i] and df['shortSMA'][i] > df['longSMA'][i]:
            executeSize = money / df['open'][i+1]
            B_or_S = 'B'
            t = i + 1
            buy.append(t)
            t1 = df.index[i+1]

    elif B_or_S == 'B':
        profit = executeSize * (df['open'][i+1] - df['open'][i])
        profit_list.append(profit)
        t2 = df.index[i+1]
        fundingFee = fundingPayment(df_funding, 'long', executeSize, df.index[t], t2)

        if (BBdata['BBW'][i-1] >= BBdata['threshold'][i-1] and BBdata['BBW'][i] < BBdata['threshold'][i]) and df['shortSMA'][i] < df['longSMA'][i] or (i == len(df) - 2):
            pl_round = executeSize * (df['open'][i+1] - df['open'][t])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate + fundingFee
            profit_fee_list.append(profit_fee)
            sell.append(i+1)
            B_or_S = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)
            t1 = df.index[i+1]

    elif B_or_S == 'S':
        profit = executeSize * (df['open'][i] - df['open'][i+1])
        profit_list.append(profit)
        t2 = df.index[i+1]
        fundingFee = fundingPayment(df_funding, 'short', executeSize, df.index[t], t2)

        if (BBdata['BBW'][i-1] >= BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i]) and df['shortSMA'][i] > df['longSMA'][i] or (i == len(df) - 2):
            pl_round = executeSize * (df['open'][t] - df['open'][i+1])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate + fundingFee
            profit_fee_list.append(profit_fee)
            buytocover.append(i+1)
            B_or_S = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)
            t1 = df.index[i+1]

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
fig, ax = plt.subplots(figsize = (16,6))

df['close'].plot(label = 'Close Price', ax = ax, c = 'gray', grid=True, alpha=0.8)
plt.scatter(df['close'].iloc[buy].index, df['close'].iloc[buy],c = 'orangered', label = 'Buy', marker='^', s=60)
plt.scatter(df['close'].iloc[sell].index, df['close'].iloc[sell],c = 'orangered', label = 'Sell', marker='v', s=60)
plt.scatter(df['close'].iloc[short].index, df['close'].iloc[short],c = 'limegreen', label = 'Sellshort', marker='v', s=60)
plt.scatter(df['close'].iloc[buytocover].index, df['close'].iloc[buytocover],c = 'limegreen', label = 'Buytocover', marker='^', s=60)

plt.legend()
plt.ylabel('USD')
plt.xlabel('Time')
plt.title('Price Movement',fontsize  = 16)

# %%
profit = equity['profitfee'].iloc[-1]
returns = (equity['profitfee'][-1] - equity['profitfee'][37])
mdd = abs(equity['drawdown']).max()
calmarRatio = returns / mdd
tradeTimes = len(buy) + len(short)
winRate = len([i for i in profit_fee_list if i > 0]) / len(profit_fee_list)
profitFactor = sum([i for i in profit_fee_list if i > 0]) / abs(sum([i for i in profit_fee_list if i < 0]))
WLRatio = np.mean([i for i in profit_fee_list if i > 0]) / abs(np.mean([i for i in profit_fee_list if i < 0]))

# %%
winRate2 = len([i for i in profit_fee_list if i > 0]) / (len(profit_fee_list) - len([i for i in profit_fee_list if i == 0]))
print(winRate2)
# %%
print(f'profit: ${np.round(profit,2)}')
print(f'returns: {np.round(returns,4)}%')
print(f'mdd: {np.round(mdd,4)}%')
print(f'calmarRatio: {np.round(calmarRatio,2)}')
print(f'tradeTimes: {tradeTimes}')
print(f'winRate: {np.round(winRate,4)*100}%')
print(f'profitFactor: {np.round(profitFactor,2)}')
print(f'winLossRatio: {np.round(WLRatio,2)}')
# %%
var_list = [0]
for i in range(len(equity)):
    var = (equity['profitfee'][i]/100 - np.mean(equity['profitfee'])/100) ** 2
    var_list.append(var)
annual_vol = ((sum(var_list) ** 0.5)*len([i for i in profit_fee_list if i != 0])) ** 0.5

# %%
equity_value = equity['profitfee']*100 + 10000
returns = (equity['profitfee'][-1] - equity['profitfee'][37])
annual_returns = ((equity_value[-1] / 10000) ** (1 / (len(df) / 365))) -1
annual_vol = ((sum(var_list) ** 0.5)*len([i for i in profit_fee_list if i != 0])) ** 0.5
mdd = abs(equity['drawdown']).max()
sharpe_ratio = annual_returns * 100 / annual_vol
calmarRatio = returns / mdd
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
feeRate = 0.0015

rule = '1D'

# %%

for Short in [5, 8, 13]:
    for Long in [21, 34, 55]:
        for threshold in np.arange(0.01, 0.1, 0.01):

            print('----------')
            print(f'short: {Short}')
            print(f'Long: {Long}')
            print(f'threshold: {threshold}')

            d1 = df_insample.resample(rule=rule, closed='right', label='right').first()[['open']]
            d2 = df_insample.resample(rule=rule, closed='right', label='right').max()[['high']]
            d3 = df_insample.resample(rule=rule, closed='right', label='right').min()[['low']]
            d4 = df_insample.resample(rule=rule, closed='right', label='right').last()[['close']]
            d5 = df_insample.resample(rule=rule, closed='right', label='right').sum()[['volume']]
            df = pd.concat([d1,d2,d3,d4,d5], axis=1)

            df['ma'] = df['close'].rolling(window = 20, center = False).mean()
            df['std'] = df['close'].rolling(window = 20, center = False).std()
            df['shortSMA'] = df['close'].rolling(window = Short, center = False).mean()
            df['longSMA'] = df['close'].rolling(window = Long, center = False).mean()
            
            upper_bound_list = [0]
            lower_bound_list = [0]
            BBW_list = [0]
            threshold_list = [0]
            spread_list = [0]

            for i in range(len(df)):
                upper_bound = df['ma'][i] + times_of_std * df['std'][i]
                upper_bound_list.append(upper_bound)

            for i in  range(len(df)):
                lower_bound = df['ma'][i] - times_of_std * df['std'][i]
                lower_bound_list.append(lower_bound)

            for i in range(len(df)):
                BBW = (upper_bound_list[i] - lower_bound_list[i]) / df['ma'][i]
                BBW_list.append(BBW)

            for i in range(len(df)):
                threshold = threshold
                threshold_list.append(threshold)

            BBdata = pd.DataFrame([upper_bound_list, lower_bound_list, BBW_list, threshold_list])
            BBdata = pd.DataFrame.transpose(BBdata)
            BBdata.columns = ['upper_bound', 'lower_bound', 'BBW', 'threshold']

            BBdata = BBdata.drop(0)
            BBdata = BBdata.set_index(df.index)

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

                    if BBdata['BBW'][i-1] < BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i] and df['shortSMA'][i] < df['longSMA'][i]:
                        executeSize = money / df['open'][i+1]
                        B_or_S = 'S'
                        t = i + 1
                        short.append(t)
                        t1 = df.index[i+1]
            
                    if BBdata['BBW'][i-1] < BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i] and df['shortSMA'][i] > df['longSMA'][i]:
                        executeSize = money / df['open'][i+1]
                        B_or_S = 'B'
                        t = i + 1
                        buy.append(t)
                        t1 = df.index[i+1]

                elif B_or_S == 'B':
                    profit = executeSize * (df['open'][i+1] - df['open'][i])
                    profit_list.append(profit)
                    t2 = df.index[i+1]
                    fundingFee = fundingPayment(df_funding, 'long', executeSize, df.index[t], t2)

                    if (BBdata['BBW'][i-1] >= BBdata['threshold'][i-1] and BBdata['BBW'][i] < BBdata['threshold'][i]) and df['shortSMA'][i] < df['longSMA'][i] or (i == len(df) - 2):
                        pl_round = executeSize * (df['open'][i+1] - df['open'][t])
                        profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
                        profit_fee_list.append(profit_fee)
                        sell.append(i+1)
                        B_or_S = None
            
                    else:
                        profit_fee = profit
                        profit_fee_list.append(profit_fee)
                        t1 = df.index[i+1]

                elif B_or_S == 'S':
                    profit = executeSize * (df['open'][i] - df['open'][i+1])
                    profit_list.append(profit)
                    t2 = df.index[i+1]
                    fundingFee = fundingPayment(df_funding, 'short', executeSize, df.index[t], t2)

                    if (BBdata['BBW'][i-1] >= BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i]) and df['shortSMA'][i] > df['longSMA'][i] or (i == len(df) - 2):
                        pl_round = executeSize * (df['open'][t] - df['open'][i+1])
                        profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
                        profit_fee_list.append(profit_fee)
                        buytocover.append(i+1)
                        B_or_S = None
            
                    else:
                        profit_fee = profit
                        profit_fee_list.append(profit_fee)
                        t1 = df.index[i+1]

            equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profitfee': np.cumsum(profit_fee_list)}, index = df.index)
            equity['drawdown_percent'] = (equity['profitfee'] / equity['profitfee'].cummax()) - 1
            equity['drawdown'] = equity['profitfee'] - equity['profitfee'].cummax()
            returns = (equity['profitfee'][-1] - equity['profitfee'][41])
            mdd = abs(equity['drawdown']).max()
            calmarRatio = returns / mdd

            optimizationList.append([Short, Long, threshold, returns, calmarRatio])
# %%
optResult = pd.DataFrame(optimizationList, columns = ['Short', 'Long', 'threshold', 'returns', 'calmarRatio'])
optResult
# %%
pic = optResult.pivot('Short', 'Long', 'threshold', 'returns')
sns.heatmap(data = pic).set(title='Return')
# %%
pic = optResult.pivot('Short', 'threshold', 'calmarRatio')
sns.heatmap(data = pic).set(title='Calmar Ratio')

# %% [markdown]
# ## Out of sample

# %%
fund = 100
money = 100
feeRate = 0.02
length = 20
times_of_std = 2
rule = '1D'

d1 = df_outofsample.resample(rule=rule, closed='right', label='right').first()[['open']]
d2 = df_outofsample.resample(rule=rule, closed='right', label='right').max()[['high']]
d3 = df_outofsample.resample(rule=rule, closed='right', label='right').min()[['low']]
d4 = df_outofsample.resample(rule=rule, closed='right', label='right').last()[['close']]
d5 = df_outofsample.resample(rule=rule, closed='right', label='right').sum()[['volume']]
df = pd.concat([d1,d2,d3,d4,d5], axis=1)

df['ma'] = df['close'].rolling(window = length, center = False).mean()
df['std'] = df['close'].rolling(window = length, center = False).std()
df['shortSMA'] = df['close'].rolling(window = 5, center = False).mean()
df['longSMA'] = df['close'].rolling(window = 21, center = False).mean()
for i in range(len(df)):
    df['upper_bound'] = df['ma'] + times_of_std * df['std']
    df['lower_bound'] = df['ma'] - times_of_std * df['std']
    BBdata['BBW'] = (df['upper_bound'] - df['lower_bound']) / df['ma']
    BBdata['threshold'] = 0.25
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

        if BBdata['BBW'][i-1] < BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i] and df['shortSMA'][i] < df['longSMA'][i]:
            executeSize = money / df['open'][i+1]
            B_or_S = 'S'
            t = i + 1
            short.append(t)
        
        if BBdata['BBW'][i-1] < BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i] and df['shortSMA'][i] > df['longSMA'][i]:
            executeSize = money / df['open'][i+1]
            B_or_S = 'B'
            t = i + 1
            buy.append(t)

    elif B_or_S == 'B':
        profit = executeSize * (df['open'][i+1] - df['open'][i])
        profit_list.append(profit)

        if (BBdata['BBW'][i-1] >= BBdata['threshold'][i-1] and BBdata['BBW'][i] < BBdata['threshold'][i]) and df['shortSMA'][i] < df['longSMA'][i] or (i == len(df) - 2):
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

        if (BBdata['BBW'][i-1] >= BBdata['threshold'][i-1] and BBdata['BBW'][i] > BBdata['threshold'][i]) and df['shortSMA'][i] > df['longSMA'][i] or (i == len(df) - 2):
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
returns = (equity['profitfee'][-1] - equity['profitfee'][0])
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
