# %%
import requests
import pandas as pd
from io import StringIO
import datetime
import os
import matplotlib.pyplot as plt

df = pd.read_csv("nasdaq_tech.csv")
stock_id = df['Symbol']
print(stock_id)
print(type(stock_id))


def input(stock_id, time_start, time_end):
    seconds = 24 * 60 * 60
    initial = datetime.datetime.strptime("1970-01-01", "%Y-%m-%d")
    start = datetime.datetime.strptime(str(time_start), "%Y-%m-%d")
    end = datetime.datetime.strptime(str(time_end), "%Y-%m-%d")
    period1 = start - initial
    period2 = end - initial
    second1 = period1.days * seconds
    second2 = period2.days * seconds
    print("initial :" + str(initial))
    print("start :" + str(start))
    print("end :" + str(end))

    url = "https://query1.finance.yahoo.com/v7/finance/download/"+ stock_id +"?period1="+ str(second1) +"&period2="+ str(second2) +"&interval=1d&events=history&includeAdjustedClose=true"
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    } ## 加這一行是因為yahoo finance 反爬蟲機制
    response = requests.get(url, headers = headers)
    df = pd.read_csv(StringIO(response.text), index_col = "Date", parse_dates = ['Date'])
    print(df.head())

    address = r"/Users/abnerteng/GitHub/Quant-Finance/pairs/UStech_stock_data/" + stock_id + ".csv"
    df.to_csv(address, encoding = 'utf-8')

time_start = '2021-01-01'
time_end = '2022-09-01'

for i in stock_id:
    try:
        input(i, time_start, time_end)
        print(i + 'success')
    except:
        print(i + 'fail')