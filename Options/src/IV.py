import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objs as go

# Newton Method for finding IV
n = norm.pdf
N = norm.cdf

def call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1- sigma*np.sqrt(T)
    call = S*N(d1)-(K*(1+r)**-T)*N(d2)
    return call

def put_price(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1- sigma*np.sqrt(T)
    put = (K*(1+r)**-T)*N(-d2)-S*N(-d1)
    return put

def call_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    c_vega = S*N(d1)*np.sqrt(T)
    return c_vega

def put_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    p_vega = S*N(d1)*np.sqrt(T)
    return p_vega

def Newton_call_iv(S, K, T, r, c, iv):
    Max_iter = 2000
    for i in range(0, Max_iter):
        iv -= (call_price(S, K, T, r, iv)-c)/call_vega(S, K, T, r, iv)*0.001
    return iv

def Newton_put_iv(S, K, T, r, p, iv):
    Max_iter = 2000
    for i in range(0, Max_iter):
        iv -= (put_price(S, K, T, r, iv)-p)/put_vega(S, K, T, r, iv)*0.001
    return iv

S = 13801.43
c_week_price = [283.5, 245.5, 216.5, 180.5, 145.5, 119, 95.5, 73.5, 55.5]
p_week_price = [69, 80.5, 94.5, 110.5, 132, 155.5, 181, 206, 243]
c_month_price = [325.5, 258.5, 200.5, 149.5, 110.5]
p_month_price = [151, 186, 229, 283, 343]
K_week = [13600, 13650, 13700, 13750, 13800, 13850, 13900, 13950, 14000]
K_month = [13700, 13800, 13900, 14000, 14100]
T1 = 7/365
T2 = 25/365
r = 0.01
sigma_init = 0.5
iv_call_week = []
iv_call_month = []
iv_put_week = []
iv_put_month = []

for i in range(0, len(c_week_price)):
    iv_call_week.append(Newton_call_iv(S, K_week[i], T1, r, c_week_price[i], sigma_init))
iv_call_week_df = pd.merge(pd.DataFrame(K_week), pd.DataFrame(iv_call_week), left_index=True, right_index=True)
iv_call_week_df.columns = ['Strike', 'IV']
iv_call_week_df.index = iv_call_week_df['Strike']
iv_call_week_df = iv_call_week_df.drop('Strike', axis=1)
print(f"call_implied_vol:", iv_call_week_df)

for i in range(0, len(p_week_price)):
    iv_put_week.append(Newton_put_iv(S, K_week[i], T1, r, p_week_price[i], sigma_init))
iv_put_week_df = pd.merge(pd.DataFrame(K_week), pd.DataFrame(iv_put_week), left_index=True, right_index=True)
iv_put_week_df.columns = ['Strike', 'IV']
iv_put_week_df.index = iv_put_week_df['Strike']
iv_put_week_df = iv_put_week_df.drop('Strike', axis=1)
print(f"put_implied_vol:", iv_put_week_df)

for i in range(0, len(c_month_price)):
    iv_call_month.append(Newton_call_iv(S, K_month[i], T2, r, c_month_price[i], sigma_init))
iv_call_month_df = pd.merge(pd.DataFrame(K_month), pd.DataFrame(iv_call_month), left_index=True, right_index=True)
iv_call_month_df.columns = ['Strike', 'IV']
iv_call_month_df.index = iv_call_month_df['Strike']
iv_call_month_df = iv_call_month_df.drop('Strike', axis=1)
print(f"call_implied_vol:", iv_call_month_df)

for i in range(0, len(p_month_price)):
    iv_put_month.append(Newton_put_iv(S, K_month[i], T2, r, p_month_price[i], sigma_init))
iv_put_month_df = pd.merge(pd.DataFrame(K_month), pd.DataFrame(iv_put_month), left_index=True, right_index=True)
iv_put_month_df.columns = ['Strike', 'IV']
iv_put_month_df.index = iv_put_month_df['Strike']
iv_put_month_df = iv_put_month_df.drop('Strike', axis=1)
print(f"put_implied_vol:", iv_put_month_df)



##for i in range(0, len(p_price)):
##    iv_put.append(Newton_put_iv(S, K[i], T, r, p_price[i], sigma_init))
##iv_put_df = pd.merge(pd.DataFrame(K), pd.DataFrame(iv_put), left_index=True, right_index=True)
##iv_put_df.columns = ['Strike', 'IV']
##iv_put_df.index = iv_put_df['Strike']
##iv_put_df = iv_put_df.drop('Strike', axis=1)
##print(f"put_implied_vol:", iv_put)

## S+P = C+Ke^(-rT) => P = C+Ke^(-rT)-S


    

def graph_week():
    plt.figure(figsize = (20, 10))
    plt.grid(True)
    plt.plot(iv_call_week_df, label = 'week call')
    plt.plot(iv_put_week_df, label = 'week put')
    plt.legend()
    plt.title('Call & Put Implied Volatility-Week', size = 14)
    plt.xlabel('Strike')
    plt.ylabel('IV')
    plt.show()

def graph_month():
    plt.figure(figsize = (20, 10))
    plt.grid(True)
    plt.plot(iv_call_month_df, label = 'month call')
    plt.plot(iv_put_month_df, label = 'month put')
    plt.legend()
    plt.title('Call & Put Implied Volatility-Month', size = 14)
    plt.xlabel('Strike')
    plt.ylabel('IV')
    plt.show()

graph_week()
graph_month()
# %%
