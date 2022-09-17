from this import d
import requests
import os
from io import StringIO
import pandas as pd

import matplotlib.pyplot as plt
import mplfinance as mpf

fwd_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
file_path = fwd_path + "/stock_data/stock_data1101.TW.csv"
df = pd.read_csv(file_path, index_col = 'Date')
df = df.drop(columns = 'Adj Close')
print(df.head())


