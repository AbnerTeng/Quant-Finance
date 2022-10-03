import pandas as pd
import os
import datetime as dt
import requests

path = os.getcwd()
above_path = os.path.dirname(path)

symbol= 'SOL/USDT'
startTime = '2021-01-01'  #time format = 20XX-XX-XX
endTime = '2022-09-30'   #time format = 20XX-XX-XX
resolution = 60

startTimeStamp = dt.datetime.strptime(startTime, '%Y-%m-%d').timestamp()
endTimeStamp = dt.datetime.strptime(endTime, '%Y-%m-%d').timestamp()
data = []

import tqdm
for i in tqdm.tqdm(range(100)):
    while True:
        if startTimeStamp < endTimeStamp:
            t1 = startTimeStamp
            t2 = startTimeStamp + resolution * 1440

            url = f'https://ftx.com/api//markets/{symbol}/candles?resolution={resolution}&limit=1440&start_time={t1}&end_time={t2}'
            response = requests.get(url)
            if response.status_code == 200:
                result = response.json()['result'][:-1]
                data += result
            else:
                print('error: {}, {} ~ {}'.format(symbol, t1, t2))

            startTimeStamp += resolution * 1440
        else:
            break

data = pd.DataFrame(data)
data.index = pd.to_datetime(data['startTime'])
data = data[['open', 'high', 'low', 'close', 'volume']]
data.to_csv(above_path + '/data/{symbol}.csv')