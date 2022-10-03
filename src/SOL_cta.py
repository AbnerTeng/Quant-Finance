# %%
import symbol
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import datetime as dt
import plotly.graph_objs as go
import plotly.express as px

class strategy():
    def get_data(self):
        self.above_path = os.path.dirname(os.getcwd())
        self.data = pd.read_csv(self.above_path + '/data/SOL.csv', parse_dates = True, index_col = 'startTime')
        return self.data

    def different_time(self, scale):
        self.rule = scale
        self.d1 = self.data.resample(rule = self.rule, closed = 'right', label = 'right').first()[['open']]
        self.d2 = self.data.resample(rule = self.rule, closed = 'right', label = 'right').max()[['high']]
        self.d3 = self.data.resample(rule = self.rule, closed = 'right', label = 'right').min()[['low']]
        self.d4 = self.data.resample(rule = self.rule, closed = 'right', label = 'right').last()[['close']]
        self.d5 = self.data.resample(rule = self.rule, closed = 'right', label = 'right').sum()[['volume']]
        self.df = pd.concat([self.d1, self.d2, self.d3, self.d4, self.d5], axis = 1)
        self.df.fillna(0, inplace = True)
        ##print(type(self.df['open']))
        return self.df

    def check(self):
        plt.figure(figsize = (16, 8))
        plt.plot(self.df['close'])
        plt.show()

    def define_var(self, length, time_std, short, long, percentage):
        self.BBC = []
        self.BBU = []
        self.BBL = []
        self.BBW = []
        self.shortMA = []
        self.longMA = []
        self.threshold = []

        self.BBC = self.df['close'].rolling(window = length, center = False).mean()
        self.BBU = self.BBC + time_std * self.df['close'].rolling(window = length, center = False).std()
        self.BBL = self.BBC - time_std * self.df['close'].rolling(window = length, center = False).std()
        self.BBW = (self.BBU - self.BBL) / self.BBC
        self.shortSMA = self.df['close'].rolling(window = short).mean()
        self.longSMA = self.df['close'].rolling(window = long).mean()

        for i in np.arange(0, len(self.BBW), 1):
            self.thres = self.BBW.quantile(percentage)
            self.threshold.append(self.thres)
        self.threshold = pd.DataFrame(self.threshold, index = self.BBW.index)

        self.strat_df = pd.concat([self.BBC, self.BBU, self.BBL, self.BBW, self.threshold, self.shortSMA, self.longSMA], axis = 1)
        self.strat_df.columns = [['BBC', 'BBU', 'BBL', 'BBW', 'threshold', 'shortSMA', 'longSMA']]
        self.strat_df.fillna(0, inplace = True)


    def OHLC_plot(self):
        self.figure = go.Figure(data = [go.Candlestick(x = self.df.index,
            open = self.df['open'],
            high = self.df['high'],
            low = self.df['low'],
            close = self.df['close'])])
        self.figure.update_layout(
            title = 'SOL OHLC',
            yaxis_title = 'Price',
            xaxis_title = 'Time')
        ##self.figure.show()

    def strat_plot(self):
        plt.figure(figsize = (16, 8))
        plt.grid(True)
        plt.plot(self.strat_df['BBC'], label = 'BBC')
        plt.plot(self.strat_df['BBU'], label = 'BBU')
        plt.plot(self.strat_df['BBL'], label = 'BBL')
        plt.plot(self.strat_df['shortSMA'], label = 'shortSMA')
        plt.plot(self.strat_df['longSMA'], label = 'longSMA')
        plt.legend()
        ##plt.show()
    
    def BBW_plot(self):
        plt.figure(figsize = (16, 8))
        plt.grid(True)
        plt.plot(self.strat_df['BBW'], label = 'BBW')
        plt.plot(self.strat_df['threshold'], label = 'threshold')
        plt.legend()
        ##plt.show()

    def to_series(self):
        self.strat_series_BBW = self.strat_df['BBW'].squeeze()
        self.strat_series_threshold = self.strat_df['threshold'].squeeze()
        self.strat_series_sma = self.strat_df['shortSMA'].squeeze()
        self.strat_series_lma = self.strat_df['longSMA'].squeeze()

    def BB_strat(self):
        self.money = 10000
        self.fee = 0.0002
        self.B_or_S = None
        self.buy = []
        self.sell = []
        self.short = []
        self.buytocover = []
        self.profit_list = [0]
        self.profit_fee_list = [0]

        for i in range(len(self.df)):
            if i == len(self.df) - 1:
                break

            if self.B_or_S == None:
                self.profit_list.append(0)
                self.profit_fee_list.append(0)

                if self.strat_series_BBW[i-1] < self.strat_series_threshold[i-1] and self.strat_series_BBW[i] > self.strat_series_threshold[i] and self.strat_series_sma[i] > self.strat_series_lma[i]:
                    self.executesize = self.money / self.df['open'][i+1]
                    self.B_or_S = 'B'
                    self.t = i + 1
                    self.buy.append(self.t)

                if self.strat_series_BBW[i-1] < self.strat_series_threshold[i-1] and self.strat_series_BBW[i] > self.strat_series_threshold[i] and self.strat_series_sma[i] < self.strat_series_lma[i]:
                    self.executesize = self.money / self.df['open'][i+1]
                    self.B_or_S = 'S'
                    self.t = i + 1
                    self.short.append(self.t)
                
            elif self.B_or_S == 'B':
                self.profit = self.executesize * (self.df['open'][i+1] - self.df['open'][i])
                self.profit_list.append(self.profit)

                if (self.strat_series_sma[i-1] > self.strat_series_lma[i-1] and self.strat_series_sma[i] > self.strat_series_lma[i]):
                    self.pnl = self.executesize * (self.df['open'][i+1] - self.df['open'][i])
                    self.profit_fee = self.profit - self.money*self.fee - (self.money + self.pnl)*self.fee
                    self.profit_fee_list.append(self.profit_fee)
                    self.sell.append(i+1)
                    self.B_or_S = None
                
                else:
                    self.profit_fee = self.profit
                    self.profit_fee_list.append(self.profit_fee)

            elif self.B_or_S == 'S':
                self.profit = self.executesize * (self.df['open'][i] - self.df['open'][i+1])
                self.profit_list.append(self.profit)

                if (self.strat_series_sma[i-1] < self.strat_series_lma[i-1] and self.strat_series_sma[i] < self.strat_series_lma[i]):
                    self.pnl = self.executesize * (self.df['open'][i] - self.df['open'][i+1])
                    self.profit_fee = self.profit - self.money*self.fee - (self.money + self.pnl)*self.fee
                    self.profit_fee_list.append(self.profit_fee)
                    self.buytocover.append(i+1)
                    self.B_or_S = None
                
                else:
                    self.profit_fee = self.profit
                    self.profit_fee_list.append(self.profit_fee)
        
        self.equity = pd.DataFrame({'profit': np.cumsum(self.profit_list), 'profit fee': np.cumsum(self.profit_fee_list)}, index = self.df.index)
        print(self.equity)
        self.equity.plot(figsize = (16, 8))
        plt.show()

                

                    

        
    




if __name__ == '__main__':
    strat = strategy()
    strat.get_data()  
    strat.different_time(scale = '1D')
    ##strat.check()
    strat.define_var(length = 20, time_std = 2, short = 5, long = 21, percentage = 0.1)
    strat.OHLC_plot()
    strat.strat_plot()
    strat.BBW_plot()
    strat.to_series()
    strat.BB_strat()



# %%
