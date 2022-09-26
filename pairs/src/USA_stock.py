# %%
import requests
import pandas as pd
from io import StringIO
import datetime
import os
import matplotlib.pyplot as plt

df = pd.read_csv("nasdaq_tech.csv")

stock_id = df['Symbol']
stock_id = pd.DataFrame(stock_id)
print(stock_id)

# %%
