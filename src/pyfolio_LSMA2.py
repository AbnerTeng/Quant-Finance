# %% [markdown]
# ## Another strategy for project 1

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
SMA_difference = TSMC_price_data['SMA5'] - TSMC_price_data['SMA20']
stock = 0
signal = []
for i in range(len(TSMC_price_data)):
    if SMA_difference[i-1] < 0 and SMA_difference[i] > 0 and stock == 0:
        signal.append(1)
        stock += 1 
    elif SMA_difference[i-1] > 0 and SMA_difference[i] < 0 and stock == 1:
        signal.append(-1)
        stock -= 1
    else:
        signal.append(0)

TSMC_price_data['SMA_signal'] = pd.DataFrame(data = signal, index = TSMC_price_data.index)

# %%
Returns = []
stock = 0
stock_stat = []
buy_price = 0
sell_price = 0

for i in range(len(TSMC_price_data)-1):
    stock_stat.append(stock)
    if TSMC_price_data['SMA_signal'][i] == 1:
        buy_price = TSMC_price_data['Open'][i+1]
        stock += 1
    elif TSMC_price_data['SMA_signal'][i] == -1:
        sell_price = TSMC_price_data['Open'][i+1]
        stock -= 1
        Returns.append((sell_price - buy_price) / buy_price)
        buy_price = 0
        sell_price = 0

# %%
benchmark_return = TSMC_price_data['Close'].pct_change()
# %%
