# %%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from tqdm import tqdm
import statsmodels
import statsmodels.api as sm

# %%
path = '/Users/abnerteng/GitHub/TMBA-projects/data/1h_crypto_price/'
csv_files = glob.glob(path + '/*.csv')
crypto_list = [pd.read_csv(file) for file in csv_files]

# %%
for i in range(len(crypto_list)):
    crypto_list[i] = crypto_list[i][['startTime', 'close']]
    print(crypto_list[i])
# %%
