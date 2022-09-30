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
    Max_iter = 100
    for i in range(0, Max_iter):
        iv -= (call_price(S, K, T, r, iv)-c)/call_vega(S, K, T, r, iv)
    return iv

def Newton_put_iv(S, K, T, r, p, iv):
    Max_iter = 100
    for i in range(0, Max_iter):
        iv -= (put_price(S, K, T, r, iv)-p)/put_vega(S, K, T, r, iv)
    return iv

c_price = [563, 442.5, 400.5, 351.5, 261, 197, 144.5, 98.5, 64, 37.5, 18.75]
p_price = [93, 123.5, 163, 213.5, 279.5]
S = 13503
K = [13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700, 13800, 13900, 14000]
T1 = 15/252
T2 = 30/252
T3 = 60/252
r = 0.01
sigma_init = 0.5
iv_call_t1 = []
iv_call_t2 = []
iv_call_t3 = []
iv_put = []

for i in range(0, len(c_price)):
    iv_call_t1.append(Newton_call_iv(S, K[i], T1, r, c_price[i], sigma_init))
iv_call_t1_df = pd.merge(pd.DataFrame(K), pd.DataFrame(iv_call_t1), left_index=True, right_index=True)
iv_call_t1_df.columns = ['Strike', 'IV']
iv_call_t1_df.index = iv_call_t1_df['Strike']
iv_call_t1_df = iv_call_t1_df.drop('Strike', axis=1)
print(f"call_implied_vol:", iv_call_t1_df)

for i in range(0, len(c_price)):
    iv_call_t2.append(Newton_call_iv(S, K[i], T2, r, c_price[i], sigma_init))
iv_call_t2_df = pd.DataFrame(iv_call_t2)
iv_call_t2_df.columns = ["IV"]
iv_call_t2_df.index = iv_call_t1_df.index
print(f"call_implied_vol:", iv_call_t2_df)

for i in range(0, len(c_price)):
    iv_call_t3.append(Newton_call_iv(S, K[i], T3, r, c_price[i], sigma_init))
iv_call_t3_df = pd.DataFrame(iv_call_t3)
iv_call_t3_df.columns = ["IV"]
iv_call_t3_df.index = iv_call_t1_df.index
print(f"call_implied_vol:", iv_call_t3_df)

##for i in range(0, len(p_price)):
##    iv_put.append(Newton_put_iv(S, K[i], T, r, p_price[i], sigma_init))
##iv_put_df = pd.merge(pd.DataFrame(K), pd.DataFrame(iv_put), left_index=True, right_index=True)
##iv_put_df.columns = ['Strike', 'IV']
##iv_put_df.index = iv_put_df['Strike']
##iv_put_df = iv_put_df.drop('Strike', axis=1)
##print(f"put_implied_vol:", iv_put)

def graph():
    plt.figure(figsize = (20, 10))
    plt.grid(True)
    plt.plot(iv_call_t1_df, marker = "o")
    plt.plot(iv_call_t2_df, marker = "o")
    plt.plot(iv_call_t3_df, marker = "o")
    plt.legend(['15 days implied volatility curve', '30 days implied volatility curve', '60 days implied volatility curve'])
    plt.title('Call Implied Volatility', size = 14)
    plt.xlabel('Strike')
    plt.ylabel('IV')
    plt.show()

graph()

# %%
