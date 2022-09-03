import pandas as pd
import os
import yfinance as yf
from common_tools import DataIO

def getData(stockNo, start, end):
    df = yf.download(stockNo, start=start, end=end)
    df = df.drop(["Adj Close"], axis=1)
    df.to_csv(os.path.join(stockNo + '.csv'))
    return df


if(__name__ == "__main__"):
    startDate = '2022-01-01'
    endDate = '2022-08-29'
    stockNo = '2330.TW'

    getData(stockNo, startDate, endDate)

    # ## 直接呼叫回測腳本
    # ##TODO: Dont use fixed path
    ## os.system(f"python myBacktesting.py {stockNo}df")