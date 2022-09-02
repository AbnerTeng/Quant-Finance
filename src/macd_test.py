import pandas as pd #引入pandas讀取股價歷史資料CSV檔
import sys, os
sys.path.insert(1, '../')

import yfinance as yf
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

from data.common_tools import DataIO

## def get_data(stockNO, start, end ):
##     df = yf.download(stockNO, start = start, end = end)
##     df = df.drop(columns = ['Adj Close'])
##     df.to_csv()
##     return df


## df['200EMA'] = df['Close'].ewm(span = 200).mean()
## df['fastEMA'] = df['Close'].ewm(span = 12).mean()
## df['slowEMA'] = df['Close'].ewm(span = 26).mean()
## df['MACD'] = df['fastEMA'] - df['slowEMA']
## df['signal'] = df['MACD'].ewm(span = 9).mean()
## df['hist'] = df['MACD'] - df['signal']
## df['green_hist'] = np.where(df['hist'] > 0, df['hist'], 0)
## df['red_hist'] = np.where(df['hist'] < 0, df['hist'], 0)
## 
## 
## print(df.tail())
## 
## candle_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
## macd = [mpf.make_addplot(df['200EMA'], panel = 0, secondary_y = False, color = 'red'),
##         mpf.make_addplot(df['MACD'], panel = 1, secondary_y = True, color = 'blue'),
##         mpf.make_addplot(df['signal'], panel = 1, secondary_y = False, color = 'orange'),
##         mpf.make_addplot(df['green_hist'], panel = 1, secondary_y = False, type = 'bar', color = 'green'),
##         mpf.make_addplot(df['red_hist'], panel = 1, secondary_y = False, type = 'bar', color = 'red')
## ]
## 
## mpf.plot(candle_data, type = 'candle', style = 'binance', addplot = macd, figratio = (18, 10), title = 'MACD')



class MACD():
    def __init__(self, df:DataFrame, _fast:int=12, _slow:int=26, _MACD:int=9):
        self.df = df
        self.df['200EMA'] = self.df['Close'].ewm(span = 200).mean()
        self.df['fastEMA'] = self.df['Close'].ewm(span = _fast).mean()
        self.df['slowEMA'] = self.df['Close'].ewm(span = _slow).mean()
        self.df['MACD'] = self.df['fastEMA'] - self.df['slowEMA']
        self.df['signal'] = self.df['MACD'].ewm(span = _MACD).mean()
        self.df['hist'] = self.df['MACD'] - self.df['signal']
        self.df['green_hist'] = np.where(self.df['hist'] > 0, self.df['hist'], 0)
        self.df['red_hist'] = np.where(self.df['hist'] < 0, self.df['hist'], 0)

    def drawPicture(self):
        candleData = self.df[['Open', 'High', 'Low', 'Close', 'Volume']]
        macd = [mpf.make_addplot(self.df['200EMA'], panel = 0, secondary_y = False, color = 'red'),
                mpf.make_addplot(self.df['MACD'], panel = 1, secondary_y = True, color = 'blue'),
                mpf.make_addplot(self.df['signal'], panel = 1, secondary_y = False, color = 'orange'),
                mpf.make_addplot(self.df['green_hist'], panel = 1, secondary_y = False, type = 'bar', color = 'green'),
                mpf.make_addplot(self.df['red_hist'], panel = 1, secondary_y = False, type = 'bar', color = 'red')
        ]

        mpf.plot(candleData, type = 'candle', style = 'binance', addplot = macd, figratio = (18, 10), title = 'MACD')

    def signalLine(self) -> Series:
        return self.df['signal']

    def MACDLine(self) -> Series:
        return self.df['MACD']

    def histogram(self) -> Series:
        return self.df['hist']


if (__name__ == "__main__"):
    df = DataIO().readCSV("data\\2330.TW.csv")
    macd = MACD(df)
    #print(list(macd.signalLine()))
    macd.drawPicture()
