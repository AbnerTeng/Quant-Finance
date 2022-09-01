from fileinput import close
import talib
import yfinance as yf
import backtesting
from backtesting import Strategy, Backtest
import pandas as pd
import matplotlib.pyplot as plt

def get_data(data):
    start = '2018-01-01'
    end = '2022-01-01'
    df = yf.download(data, start = start, end = end)
    df = df.drop(columns = ['Adj Close'])
    return df
df = get_data('2330.TW')
print(df.tail())
close = df['Close']
print(close.tail())

def strat(fast, slow, signal):
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    print(df)

strat(12, 26, 9)    


df['macd'].plot()
df['macdsignal'].plot()
df['macdhist'].plot()
plt.show()




 
