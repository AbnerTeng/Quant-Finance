# %%
from turtle import position
import requests
import os
from io import StringIO
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = "whitegrid")
import mplfinance as mpf
import numpy as np

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


def find_cointegrated_pairs(df):
    n = df.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = df.keys()
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            S1 = df[keys[i]]
            S2 = df[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

fwd_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
num = ['1101', '1102', '1103', '1104', '1108', '1109', '1110', '1201', '1203', '1210']
df = pd.DataFrame()

for i in num:
    file_path = fwd_path + "/stock_data/stock_data"+ i +".TW.csv"
    file = pd.read_csv(file_path, index_col = 'Date')
    file = file.drop(columns = 'Adj Close')
    file = pd.DataFrame(file['Close'])
    df = pd.concat([df, file], axis = 1)

df.columns = [['1101', '1102', '1103', '1104', '1108', '1109', '1110', '1201', '1203', '1210']]
print(df)


scores, pvalues, pairs = find_cointegrated_pairs(df)
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(pvalues, xticklabels = num, yticklabels = num, cmap = 'RdYlGn_r', mask = (pvalues >= 0.05), annot = True)
plt.show()
print(pairs)


stock1 = df['1101']
stock2 = df['1210']
score, pvalue, _ = coint(stock1, stock2)
print(pvalue)

stock1 = sm.add_constant(stock1)
model = sm.OLS(stock2, stock1).fit()
stock1 = stock1[('1101',)]
b = model.params[('1101',)]

stock2 = stock2[('1210',)]
spread = stock2 - b * stock1
mean = spread.mean()
vol = spread.std()
up_b = mean + 1.5 * vol
low_b = mean - 1.5 * vol

spread.plot(figsize = (12, 6))
stock1.plot()
stock2.plot()
plt.show()

spread.plot(figsize = (12, 6))
plt.axhline(spread.mean(), color = 'black')
plt.axhline(up_b, color = 'red')
plt.axhline(low_b, color = 'green')
plt.legend(['Spread', 'mean', 'up', 'low'])
plt.show()

# %%
split_ratio = 0.8
split_num = len(spread) * split_ratio
print(split_num)
train = spread[:323]
train_mean = train.mean()
train_vol = train.std()
train_up_b = train_mean + 1.5 * train_vol
train_low_b = train_mean - 1.5 * train_vol
test = spread[323:]
test_mean = test.mean()
test_vol = test.std()
test_up_b = test_mean + 1.5 * test_vol
test_low_b = test_mean - 1.5 * test_vol

train.plot(figsize = (12, 6))
plt.axhline(train.mean(), color = 'black')
plt.axhline(train_up_b, color = 'red')
plt.axhline(train_low_b, color = 'green')
plt.legend(['Spread', 'mean', 'up', 'low'])
plt.show()


# %%
fund = 10000
money = 10000
fee = 0.001425

position = None
buy1 = []
buy2 = []
sell1 = []
sell2 = []
short1 = []
short2 = []
buytocover1 = []
buytocover2 = []
profit_list = [0]
profit_fee_list = [0]

for i in range(len(spread)):
    if i == len(spread)-1:
        break

    if position == None:
        profit_list.append(0)
        profit_fee_list.append(0)

        if spread[i] < up_b and spread[i+1] > up_b:
            position = 'short1_buy2'
            t = i + 1
            short1.append(t)
            buy2.append(t)
            executesize = money / stock1[i+1] + stock2[i+1]
          
        if spread[i] > low_b and spread[i+1] < low_b:
            position = 'buy1_short2'
            t = i + 1
            buy1.append(t)
            short2.append(t)
    
    elif position == 'short1_buy2':
        profit = (stock1[i] - stock1[i+1]) * executesize - (stock2[i+1] - stock2[i]) * executesize
        profit_list.append(profit)

        if spread[i] > mean and spread[i+1] < mean or i == len(spread)-2:
            pl_round = executesize * (stock1[i] - stock1[i+1]) - (stock2[i+1] - stock2[i]) * executesize
            profit_fee = profit - money * fee - (money+pl_round) * fee
            profit_fee_list.append(profit_fee)
            buytocover1.append(i+1)
            sell2.append(i+1)
            position = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

    elif position == 'buy1_short2':
        profit = (stock1[i+1] - stock1[i]) * executesize - (stock2[i] - stock2[i+1]) * executesize
        profit_list.append(profit)

        if spread[i] < mean and spread[i+1] > mean or i == len(spread)-2:
            pl_round = executesize * (stock1[i+1] - stock1[i]) - (stock2[i] - stock2[i+1]) * executesize
            profit_fee = profit - money * fee - (money+pl_round) * fee
            profit_fee_list.append(profit_fee)
            sell1.append(i+1)
            buytocover2.append(i+1)
            position = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profit_fee': np.cumsum(profit_fee_list)}, index = spread.index)
print(equity)
equity.plot(figsize = (12, 6), grid = True)
plt.show()

# %%
print(buy1, short2)
print(buy2, short1)
print(sell1, buytocover2)
print(sell2, buytocover1)

# %%
