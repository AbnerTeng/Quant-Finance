# %% [markdown]
# ## Project 1
# SMA strategy backtesting return analysis by using pyfolio

# %% [markdown]
# import data

# %%
import numpy as np
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt
import yfinance as yf
from pandas import Series, DataFrame

TSMC_price_data = yf.download('2330.TW', start = '2018-01-01', end = '2022-07-18')
TSMC_price_data['Date'] = TSMC_price_data.index
print(TSMC_price_data)
TSMC_price_data['Date'] = pd.to_datetime(TSMC_price_data['Date'], format = '%Y-%m-%d')

# %% [markdown]
# Setting SMA5, SMA20 line, and plot them

# %%
TSMC_price_data['Close'] = pd.to_numeric(TSMC_price_data['Close'])
TSMC_price_data['SMA5'] = TSMC_price_data['Close'].rolling(5).mean()
TSMC_price_data['SMA20'] = TSMC_price_data['Close'].rolling(20).mean()

plt.style.use('seaborn-darkgrid')
plt.figure(figsize = (12, 6))
plt.plot(TSMC_price_data['Close'])
plt.plot(TSMC_price_data['SMA5'])
plt.plot(TSMC_price_data['SMA20'])
plt.legend(['Close', 'SMA5', 'SMA20'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Basic status of 2330.TW')
plt.show()

# %% [markdown]
# My strategy: SMA5 > SMA20 is the first condition, and SMA5 < SMA20 is the second condition \
# I use an empty array to store my signal

# %%
condition = []

for i in range(len(TSMC_price_data)):

    if TSMC_price_data['SMA5'][i] >= TSMC_price_data['SMA20'][i]:
        condition_list = condition.append('+')

    elif TSMC_price_data['SMA5'][i] < TSMC_price_data['SMA20'][i]:
        condition_list = condition.append('-')

    else:
        condition_list = condition.append('None')

signal = []

for i in range(len(TSMC_price_data)):
    if condition[i] == '+':
        stock = 1
        signal.append('Buy')

    elif condition[i] == '-' and stock == 1:
        stock -= 1
        signal.append('Sell')
    else:
        signal.append(0)

TSMC_price_data['SMA_signal'] = pd.Series(index = TSMC_price_data.index, data = signal)

# %% [markdown]
# Calculate the daily return

# %%
buy_n_hold_list = [0]

for i in range(1, len(TSMC_price_data), 1):
    buy_n_hold= ((TSMC_price_data['Close'][i] - TSMC_price_data['Close'][i-1]) / TSMC_price_data['Close'][i-1])
    buy_n_hold_list.append(buy_n_hold)
    
strat_return_list = [0]

for i in range(1, len(TSMC_price_data), 1):
    if TSMC_price_data['SMA_signal'][i] == 'Buy':
        strat_return = ((TSMC_price_data['Open'][i+1] - TSMC_price_data['Close'][i]) / TSMC_price_data['Close'][i]) * 1
        strat_return_list.append(strat_return)

    elif TSMC_price_data['SMA_signal'][i] == 'Sell' or TSMC_price_data['SMA_signal'][i] == 0:
        strat_return = 0
        strat_return_list.append(strat_return)

TSMC_price_data['benchmark'] = buy_n_hold_list
TSMC_price_data['strat_return'] = strat_return_list

# %%
pf.create_returns_tear_sheet(returns = TSMC_price_data['strat_return'], benchmark_rets = TSMC_price_data['benchmark'])


# %% [markdown]
# export dataFrame to .csv file

# %%
TSMC_price_data = TSMC_price_data.drop(columns = ['High', 'Low', 'Adj Close', 'return'])
from pathlib import Path
filepath = Path('/Users/abnerteng/GitHub/TMBA-projects/data/TSMC_price_data.csv')
filepath.parent.mkdir(parents = True, exist_ok = True)
TSMC_price_data.to_csv(filepath)


# %%
