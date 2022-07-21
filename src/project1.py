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
condition1 = TSMC_price_data['SMA5'] >= TSMC_price_data['SMA20']
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

# %% [markdown]
# Calculate the daily return

# %%
TSMC_price_data['return'] = pd.Series(np.zeros(len(TSMC_price_data)))
i = 2
for i in range(len(TSMC_price_data)):
    TSMC_price_data['return'][i] = ((TSMC_price_data['Close'][i] - TSMC_price_data['Close'][i-1]) / TSMC_price_data['Close'][i-1])
    
strat_return = np.zeros(len(TSMC_price_data))

for i in range(len(TSMC_price_data)):
    if TSMC_price_data['SMA_signal'][i] == 1:
        strat_return[i] = TSMC_price_data['return'][i+1]*TSMC_price_data['SMA_signal'][i]
    elif TSMC_price_data['SMA_signal'][i] == -1:
        strat_return[i] = TSMC_price_data['return'][i+1]*TSMC_price_data['SMA_signal'][i]

TSMC_price_data['strat_return'] = strat_return

# %%
pf.create_returns_tear_sheet(returns = TSMC_price_data['strat_return'], benchmark_rets = TSMC_price_data['return'])


# %% [markdown]
# export dataFrame to .csv file

# %%
TSMC_price_data = TSMC_price_data.drop(columns = ['High', 'Low', 'Adj Close', 'return'])
from pathlib import Path
filepath = Path('/Users/abnerteng/GitHub/TMBA-projects/data/TSMC_price_data.csv')
filepath.parent.mkdir(parents = True, exist_ok = True)
TSMC_price_data.to_csv(filepath)


# %%
