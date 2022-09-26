# %%
##from tkinter.tix import DisplayStyle
import requests
import pandas as pd
from io import StringIO
import datetime
import os
import matplotlib.pyplot as plt


url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=1&industry_code=&Page=1&chklike=Y"
response = requests.get(url)
listed = pd.read_html(response.text)[0]
listed.columns = listed.iloc[0,:]
listed = listed[["有價證券代號", "有價證券名稱", "市場別", "產業別", "公開發行/上市(櫃)/發行日"]]
listed = listed.iloc[1:]
print(listed)

stock_id = listed["有價證券代號"]
stock_num = stock_id.apply(lambda x: str(x)+ ".TW")
stock_num ## pandas series
# %%

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

    address = r"/Users/abnerteng/Desktop/stock_data/" + stock_id + ".csv"
    df.to_csv(address, encoding = 'utf-8')


time_start = "2021-01-01"
time_end = "2022-09-01"
for i in stock_num:
    try:
        input(i, time_start, time_end)
        print(i + 'success')
    except:
        print(i + 'fail')

