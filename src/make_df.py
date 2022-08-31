import pandas as pd
from pandas import DataFrame, Series
import yfinance as yf
import numpy as np

def load_data(stock, start, end):
    stock_data = yf.download(stock, start = start, end = end)
    stock_data = stock_data.drop('Adj Close', axis = 1)
    print(stock_data)

load_data('GOOG', '2018-01-01', '2022-01-01')
