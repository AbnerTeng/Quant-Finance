# %%
import pandas as pd #引入pandas讀取股價歷史資料CSV檔
import sys, os
sys.path.insert(1, '../')

import yfinance as yf
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

from data.common_tools import DataIO


class MACD():
    def __init__(self, df:DataFrame, _len:int=200, _fast:int=12, _slow:int=26, _MACD:int=9):
        self.df = df
        self.df['200EMA'] = self.df['Close'].ewm(span = _len).mean()
        ## print(type(self.df['200EMA']))
        self.df['fastEMA'] = self.df['Close'].ewm(span = _fast).mean()
        self.df['slowEMA'] = self.df['Close'].ewm(span = _slow).mean()
        self.df['MACD'] = self.df['fastEMA'] - self.df['slowEMA']
        ## print(type(self.df['MACD']))
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

class strategy(MACD):
    def __init__(self, df):
        super().__init__(df)

        self.money = 100
        self.fee = 0.001425
        self.position = None
        self.buy = []
        self.short = []
        self.sell = []
        self.buy_to_cover = []
        self.profit_list = [0]
        self.profit_fee_list = [0]

    def MACD_strat(self):
        for i in range(len(self.df)):

            if i == len(self.df) - 1:
                break

            if self.position == None:
                self.profit_list.append(0)
                self.profit_fee_list.append(0)

                if (self.df['MACD'][i-1] < self.df['signal'][i-1] and self.df['MACD'][i] > self.df['signal'][i]) and (self.df['MACD'][i] < 0) and (self.df['Close'][i] > self.df['200EMA'][i]):
                    self.executeSize = self.money / self.df['Open'][i+1]
                    self.position = 'B'
                    self.t = i + 1
                    self.buy.append(self.t)

                if (self.df['MACD'][i-1] > self.df['signal'][i-1] and self.df['MACD'][i] < self.df['signal'][i]) and (self.df['MACD'][i] > 0) and (self.df['Close'][i] < self.df['200EMA'][i]):
                    self.executeSize = self.money / self.df['Open'][i+1]
                    self.position = 'S'
                    self.t = i + 1
                    self.short.append(self.t)

            elif self.position == 'B':
                self.profit = self.executeSize * (self.df['Open'][i+1] - self.df['Open'][i])
                self.profit_list.append(self.profit)

                if (self.df['MACD'][i-1] > self.df['signal'][i-1] and self.df['MACD'][i] < self.df['signal'][i]) and (self.df['MACD'][i] > 0) and (self.df['Close'][i] < self.df['200EMA'][i]) or i == len(self.df) - 2:
                    self.pnl = self.executeSize * (self.df['Open'][i+1] - self.df['Open'][self.t])
                    self.profit_fee = self.profit - self.money * self.fee - (self.money + self.pnl) * self.fee
                    self.profit_fee_list.append(self.profit_fee)
                    self.sell.append(i+1)
                    self.position = None

                else:
                    self.profit_fee = self.profit
                    self.profit_fee_list.append(self.profit_fee)

            elif self.position == 'S':
                self.profit = self.executeSize * (self.df['Open'][i] - self.df['Open'][i+1])
                self.profit_list.append(self.profit)

                if (self.df['MACD'][i-1] < self.df['signal'][i-1] and self.df['MACD'][i] > self.df['signal'][i]) and (self.df['MACD'][i] < 0) and (self.df['Close'][i] > self.df['200EMA'][i]) or i == len(self.df) - 2:
                    self.pnl = self.executeSize * (self.df['Open'][self.t] - self.df['Open'][i+1])
                    self.profit_fee = self.profit - self.money * self.fee - (self.money + self.pnl) * self.fee
                    self.profit_fee_list.append(self.profit_fee)
                    self.buy_to_cover.append(i+1)
                    self.position = None

                else:
                    self.profit_fee = self.profit
                    self.profit_fee_list.append(self.profit_fee)

        self.equity = pd.DataFrame({'profit': np.cumsum(self.profit_list), 'profitfee': np.cumsum(self.profit_fee_list)}, index = self.df.index)
        print(self.equity)
        self.equity.plot(grid = True, figsize = (12, 8))
        plt.show()




if (__name__ == "__main__"):
    df = DataIO().readCSV("../data/2330.TW.csv")
    macd = MACD(df)
    #print(list(macd.signalLine()))
    macd.drawPicture()
    strat = strategy(df)
    strat.MACD_strat()



# %%
