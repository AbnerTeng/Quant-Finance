# %% [markdown]
# ## Project 1

# %%
from tracemalloc import start
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

# %%
TSMC_price_data['Close'] = pd.to_numeric(TSMC_price_data['Close'])
TSMC_price_data['SMA5'] = TSMC_price_data['Close'].rolling(5).mean()
TSMC_price_data['SMA20'] = TSMC_price_data['Close'].rolling(20).mean()

plt.figure(figsize = (12, 6))
plt.plot(TSMC_price_data['Close'])
plt.plot(TSMC_price_data['SMA5'])
plt.plot(TSMC_price_data['SMA20'])
plt.legend(['Close', 'SMA5', 'SMA20'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Basic status of 2330.TW')
plt.show()

# %%
condition1 = TSMC_price_data['SMA5'] > TSMC_price_data['SMA20']
condition2 = TSMC_price_data['SMA5'] < TSMC_price_data['SMA20']
signal = []
## stock = 0

for i in range(len(TSMC_price_data)):
    if  condition1[i]:
        stock = 1
        signal.append(1)
    elif condition2[i] and stock == 1:
        stock -= 1
        signal.append(-1)
    else:
        signal.append(0)

TSMC_price_data['SMA_signal'] = pd.Series(index = TSMC_price_data.index, data = signal)


# %%
''';
zero_data = np.zeros(len(TSMC_price_data))
strat_return = pd.Series(zero_data)
stock = 0
stock_his = []
buy_price = 0
sell_price = 0
for i in range(len(TSMC_price_data) - 1):
    stock_his.append(stock)
    if TSMC_price_data['SMA_signal'][i] == 1:
        buy_price = TSMC_price_data['Open'][i+1]
        stock += 1
    elif TSMC_price_data['SMA_signal'][i] == -1:
        sell_price = TSMC_price_data['Open'][i+1]
        stock -= 1
        strat_return.append((sell_price - buy_price)/buy_price)
        buy_price = 0
        sell_price = 0

if stock == 1 and buy_price != 0 and sell_price == 0:
    sell_price = TSMC_price_data['Open'][-1]
    strat_return.append((sell_price - buy_price)/buy_price)
    stock -= 1
'''
# %%
'''
stock = 0
condition1 = (TSMC_price_data['SMA5'] > TSMC_price_data['SMA20'])
condition2 = (TSMC_price_data['SMA5'] < TSMC_price_data['SMA20'])

for i in range(len(TSMC_price_data)):
    if condition1[i] and stock == 0:
        TSMC_price_data['SMA_signal'][i] = 1
        stock += 1
    elif condition2[i] and stock == 1:
        TSMC_price_data['SMA_signal'][i] = -1
        stock -=  1
    else:
        TSMC_price_data['SMA_signal'][i] = 0
'''

# %%
TSMC_price_data['return'] = TSMC_price_data['Close'].pct_change()
strat_return = np.zeros(len(TSMC_price_data))

for i in range(len(TSMC_price_data)):
    if TSMC_price_data['SMA_signal'][i] == 1:
        strat_return[i] = TSMC_price_data['return'][i]*TSMC_price_data['SMA_signal'][i]
    elif TSMC_price_data['SMA_signal'][i] == -1:
        strat_return[i] = TSMC_price_data['return'][i]*TSMC_price_data['SMA_signal'][i]

TSMC_price_data['strat_return'] = strat_return
# %%
pf.create_returns_tear_sheet(TSMC_price_data['strat_return'])

