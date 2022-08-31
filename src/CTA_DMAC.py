from distutils.command.build_py import build_py
from backtesting import Backtest, Strategy #引入回測和交易策略功能

from backtesting.lib import crossover #從lib子模組引入判斷均線交會功能
from backtesting.test import SMA #從test子模組引入繪製均線功能
import pandas as pd #引入pandas讀取股價歷史資料CSV檔
import sys
import yfinance as yf
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

##class SmaCross(Strategy):
##
##    def init(self):
##        self.fast_line = self.I(SMA, self.data.Close, 13)
##        self.slow_line = self.I(SMA, self.data.Close, 48)
##
##    def next(self):
##        if crossover(self.fast_line, self.slow_line):
##            print(
##                f"{self.data.index[-1]} Buy: Price: {self.data.Close[-1]}, Slow: {self.slow_line[-5:]}, Fast: {self.fast_line[-5:]} \n"
##            )
##            self.buy()
##        elif crossover(self.slow_line, self.fast_line):
##            print(
##                f"{self.data.index[-1]} Sell: Price: {self.data.Close[-1]}, Slow: {self.slow_line[-5:]}, Fast: {self.fast_line[-5:]} \n"
##            )
##
##            self.sell()


##stock = '2330.TW'
##start ='2018-01-01'
##end = '2022-01-01'
##stock_data = yf.download(stock, start = start, end = end)
##stock_data = stock_data.drop('Adj Close', axis = 1)
##
##
##stock_data.to_csv(f'{stock}.csv')
##df = pd.read_csv(f'{stock}.csv') 


def crawl_data():
    stock = 'AAPL'
    start = '2018-01-01'
    end = '2022-01-01'
    stock_data = yf.download(stock, start = start, end = end)
    stock_data = stock_data.drop('Adj Close', axis = 1)
    stock_data.to_csv(f'{stock}.csv')

crawl_data()

df = pd.read_csv('AAPL.csv')
print(df.tail())
close = df['Close']
open = df['Open']

def sma(length):
    simple_ma = close.rolling(window = length, center = False).mean()
    ##simple_ma = simple_ma.fillna(0)
    return simple_ma


## stock_no = GOOG   
## if(sys.argv[1] != None):
##    stock_no = sys.argv[1]

#df = pd.read_csv(f"{stock_no}.csv", index_col=0) #pandas讀取資料，並將第1欄作為索引欄
#df = df.interpolate() #CSV檔案中若有缺漏，會使用內插法自動補值，不一定需要的功能
#df.index = pd.to_datetime(df.index) #將索引欄資料轉換成pandas的時間格式，backtesting才有辦法排序


## build DMAC strategy and calculate equity

fund = 10000
money = 10000
feeRate = 0.0015
B_or_S = None
buy = []
sell = []
short = []
buytocover = []
profit_list = [0]
profit_fee_list = [0]

for i in range(len(df)):

    if i == len(df)-1:
        break

    if B_or_S == None:
        profit_list.append(0)
        profit_fee_list.append(0)

        if sma(8)[i] > sma(21)[i]:
            executeSize = money / open[i+1]
            B_or_S = 'B'
            t = i + 1
            buy.append(t)
        
        if sma(8)[i] < sma(21)[i]:
            executeSize = money / open[i+1]
            B_or_S = 'S'
            t = i + 1
            short.append(t)

    elif B_or_S == 'B':

        profit = executeSize * (open[i+1] - open[i])
        profit_list.append(profit)

        if sma(8)[i] <= sma(21)[i] or i == len(df)-2:
            pl_round = executeSize * (open[i+1] - open[t])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
            profit_fee_list.append(profit_fee)
            sell.append(i+1)
            B_or_S = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

    elif B_or_S == 'S':
        profit = executeSize * (open[i] - open[i+1])
        profit_list.append(profit)

        if sma(8)[i] >= sma(21)[i] or i == len(df)-2:
            pl_round = executeSize * (open[t] - open[i+1])
            profit_fee = profit - money * feeRate - (money + pl_round) * feeRate
            profit_fee_list.append(profit_fee)
            buytocover.append(i+1)
            B_or_S = None
        
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)

equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profitfee': np.cumsum(profit_fee_list)}, index = df.index)
print(equity)
equity.plot(grid = True, figsize = (12, 8))
plt.show()
     
print(buy)
print(short)
print(sell)
print(buytocover)

equity['equity_value'] = equity['profitfee'] + fund
equity['drawdown_percent'] = (equity['equity_value'] / equity['equity_value'].cummax()) - 1
equity['drawdown'] = equity['equity_value'] - equity['equity_value'].cummax()

## build Profit and Drawdown plot


fig, ax = plt.subplots(figsize = (16, 6))
high_index = equity[equity['profitfee'].cummax() == equity['profitfee']].index
equity['profitfee'].plot(label = 'Total Profit', ax = ax, color = 'red', grid = True)
plt.fill_between(equity['drawdown'].index, equity['drawdown'], 0, facecolor = 'red', label = 'Drawdown', alpha = 0.5)
plt.scatter(high_index, equity['profitfee'].loc[high_index], color = '#02ff0f', label = 'High')

plt.legend()
plt.ylabel('Accumulated Profit')
plt.xlabel('Time')
plt.title('Profit & Drawdown', fontsize  = 16)

plt.show()


## Build price and trade point plot
fig, ax = plt.subplots(figsize = (16,6))

close.plot(label = 'Close Price', ax = ax, c = 'gray', grid=True, alpha=0.8)
plt.scatter(close.iloc[buy].index, close.iloc[buy],c = 'orangered', label = 'Buy', marker='^', s=60)
plt.scatter(close.iloc[sell].index, close.iloc[sell],c = 'orangered', label = 'Sell', marker='v', s=60)
plt.scatter(close.iloc[short].index, close.iloc[short],c = 'limegreen', label = 'Sellshort', marker='v', s=60)
plt.scatter(close.iloc[buytocover].index, close.iloc[buytocover],c = 'limegreen', label = 'Buytocover', marker='^', s=60)

plt.legend()
plt.ylabel('NTD')
plt.xlabel('Time')
plt.title('Price Movement',fontsize  = 16)

plt.show()


## calculate technical indicators

def tech_indicators():
    profit = equity['profitfee'].iloc[-1]
    returns = (equity['equity_value'][-1] / equity['equity_value'][0]) - 1

# %%



##test = Backtest(
##        df,
##        SmaCross,
##        cash=1000000,
##        commission=0.004,
##        exclusive_orders=True,
##        trade_on_close=True,
##    )
# 指定回測程式為test，在Backtest函數中依序放入(資料來源、策略、現金、手續費)
#result = test.run()
#執行回測程式並存到result中

#print(result) # 直接print文字結果
#test.plot(filename="res{stock_no}.html") #將線圖網頁依照指定檔名保存
