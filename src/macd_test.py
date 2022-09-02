import pandas as pd #引入pandas讀取股價歷史資料CSV檔
import sys
import yfinance as yf
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def get_data(data):
    start = '2020-01-01'
    end = '2022-01-01'
    df = yf.download(data, start = start, end = end)
    df = df.drop(columns = ['Adj Close'])
    return df

df = get_data('2330.TW')

df['200EMA'] = df['Close'].ewm(span = 200).mean()
df['fastEMA'] = df['Close'].ewm(span = 12).mean()
df['slowEMA'] = df['Close'].ewm(span = 26).mean()
df['MACD'] = df['fastEMA'] - df['slowEMA']
df['signal'] = df['MACD'].ewm(span = 9).mean()
df['hist'] = df['MACD'] - df['signal']
df['green_hist'] = np.where(df['hist'] > 0, df['hist'], 0)
df['red_hist'] = np.where(df['hist'] < 0, df['hist'], 0)


print(df.tail())

candle_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
macd = [mpf.make_addplot(df['200EMA'], panel = 0, secondary_y = False, color = 'red'),
        mpf.make_addplot(df['MACD'], panel = 1, secondary_y = True, color = 'blue'),
        mpf.make_addplot(df['signal'], panel = 1, secondary_y = False, color = 'orange'),
        mpf.make_addplot(df['green_hist'], panel = 1, secondary_y = False, type = 'bar', color = 'green'),
        mpf.make_addplot(df['red_hist'], panel = 1, secondary_y = False, type = 'bar', color = 'red')
]

mpf.plot(candle_data, type = 'candle', style = 'binance', addplot = macd, figratio = (18, 10), title = 'MACD')



