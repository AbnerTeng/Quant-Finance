# %%
from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
# %%
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

# %%
call = pd.DataFrame()
put = pd.DataFrame()
import numpy as np
S = 12998
T = 21/365
r = 0.0126
sigma = 0.5
c_buy = np.array([1890, 1800, 1700, 1610, 1510, 1420, 1320, 1230, 1170, 1080, 1000, 915, 830, 750, 670, 610, 505, 433, 354, 381, 324, 277, 231, 190, 157, 122, 97, 78, 61, 47, 37, 28.5, 22.5, 17, 12.5, 11.5, 9.1, 3.8, 5, 3.4, 4.5])
c_sell = np.array([2070, 1970, 1870, 1780, 1680, 1590, 1490, 1400, 1280, 1190, 1110, 1020, 935, 855, 775, 700, 630, 560, 454, 392, 339, 284, 240, 191, 161, 126, 100, 79, 64, 49.5, 38, 29, 23, 17.5, 14.5, 12, 10, 8.2, 7.5, 7.3, 6.4])
p_buy = np.array([16.5, 18, 21, 27, 32, 37, 30.5, 53, 52, 74, 90, 102, 119, 129, 158, 183, 213, 242, 285, 318, 340, 373, 380, 476, 540, 645, 700, 765, 885, 930, 1060, 1110, 1200, 1290, 1390, 1490, 1590, 1680, 1750, 1880, 1980])
p_sell = np.array([35, 19.5, 23.5, 30.5, 33.5, 38.5, 49.5, 55, 64, 80, 91, 104, 122, 141, 164, 189, 218, 275, 286, 348, 371, 420, 500, 555, 645, 715, 750, 870, 950, 1040, 1090, 1220, 1310, 1400, 1500, 1600, 1700, 1790, 1890, 1990, 2090])
##c_buy = np.array([1420, 1320, 1230, 1170, 1080, 1000, 915, 830, 750, 670, 610, 505, 433, 354, 381, 324, 277, 231, 190, 157, 122, 97, 78, 61, 47, 37, 28.5, 22.5, 17, 12.5, 11.5])
##c_sell = np.array([1590, 1490, 1400, 1280, 1190, 1110, 1020, 935, 855, 775, 700, 630, 560, 454, 392, 339, 284, 240, 191, 161, 126, 100, 79, 64, 49.5, 38, 29, 23, 17.5, 14.5, 12])
##p_buy = np.array([37, 30.5, 53, 52, 74, 90, 102, 119, 129, 158, 183, 213, 242, 285, 318, 340, 373, 380, 476, 540, 645, 700, 765, 885, 930, 1060, 1110, 1200, 1290, 1390, 1490])
##p_sell = np.array([38.5, 49.5, 55, 64, 80, 91, 104, 122, 141, 164, 189, 218, 275, 286, 348, 371, 420, 500, 555, 645, 715, 750, 870, 950, 1040, 1090, 1220, 1310, 1400, 1500, 1600])
c_month_price = (c_buy + c_sell)/2
p_month_price = (p_buy + p_sell)/2
##K_month = np.arange(11500, 14600, 100)
K_month = np.arange(11000, 15100, 100)
print(K_month)
call['price'] = c_month_price
call['k'] = K_month
call['f/k'] = S/call['k']
iv = Newton_call_iv(S, K_month, T, r, c_month_price, sigma)
call['iv'] = iv

put['price'] = p_month_price
put['k'] = K_month
put['f/k'] = S/put['k']
iv = Newton_put_iv(S, K_month, T, r, p_month_price, sigma)
put['iv'] = iv

print(call)
print(put)

# %%
def interpolation(fk_new, call):
    high_fk=0
    low_fk=0
    high=0
    low=0
    for i in range (len(call)):
        if fk_new < call['f/k'][i]:
            high_fk=call['f/k'][i]
            low_fk=call['f/k'][i+1]
            high =call['iv'][i]
            low = call['iv'][i+1]
        else:
            break
            
    a = high_fk - fk_new
    b = fk_new - low_fk
    vol_new = ((b*high) + (a*low))/(a+b)
    
    return vol_new
# %%
##def interpolation_put(fk_new, put):
##    high_fk=0
##    low_fk=0
##    high=0
##    low=0
##    for i in range (len(put)):
##        if fk_new > put['f/k'][i]:
##            high_fk = put['f/k'][i]
##            low_fk=put['f/k'][i+1]
##            high =put['iv'][i]
##            low = put['iv'][i+1]
##        else:
##            break
##            
##    a = high_fk - fk_new
##    b = fk_new - low_fk
##    vol_new = ((b*high) + (a*low))/(a+b)
##    
##    return vol_new
# %%
S_new = np.arange(11000, 15100, 1)
K_new_c = 12800
K_new_p = 12800
fk_new_c = S_new/K_new_c
fk_new_p = S_new/K_new_p
maxfk = max(call['f/k'])
minfk = min(call['f/k'])
delta = (maxfk-minfk)/(2*len(S_new)-1)
fkn = np.arange(minfk, maxfk+delta, delta)
fkn = fkn[0:8201]
# %%

S_newnew_c = K_new_c * fkn
S_newnew_p = K_new_p * fkn

vol_new_c = []
for i in range (len(fkn)):
    vol_new_c.append(interpolation(fkn[i], call))
print(vol_new_c)
# %%
plt.plot(S_newnew_c, vol_new_c)
# %%
vol_new_p = []
for i in range (len(fkn)):
    vol_new_p.append(interpolation(fkn[i], put))
print(vol_new_p)
# %%
plt.plot(S_newnew_p, vol_new_p)

# %%
S_newnew_c = pd.Series(S_newnew_c)
vol_new_c = pd.Series(vol_new_c)
S_newnew_p = pd.Series(S_newnew_p)
vol_new_p = pd.Series(vol_new_p)

def call_price(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, cheat):
    d1 = (np.log(S_newnew_c/K_new_c)+(r+0.5*vol_new_c**2)*T)/(vol_new_c*np.sqrt(T))
    d2 = d1- vol_new_c*np.sqrt(T)
    call = S_newnew_c*N(d1)-(K_new_c*(1+r)**-T)*N(d2)
    return -call - fee + right_c + cheat

def put_price(S_newnew_p, K_new_p, T, r, vol_new_p, fee, right_p, cheat):
    d1 = (np.log(S_newnew_p/K_new_p)+(r+0.5*vol_new_p**2)*T)/(vol_new_p*np.sqrt(T))
    d2 = d1- vol_new_p*np.sqrt(T)
    put = (K_new_p*(1+r)**-T)*N(-d2)-S_newnew_p*N(-d1)
    return -put - fee + right_p + cheat

##def long_put_price(S_newnew_p, K_new_p, T, r, vol_new_p, fee, right_p):
##    d1 = (np.log(S_newnew_p/K_new_p)+(r+0.5*vol_new_p**2)*T)/(vol_new_p*np.sqrt(T))
##    d2 = d1- vol_new_p*np.sqrt(T)
##    put = (K_new_p*(1+r)**-T)*N(-d2)-S_newnew_p*N(-d1)
##    return put - fee - right_p

fee = 2
right_c = 404
right_p = 285.5
cheat = 100
price_c = call_price(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, cheat)
print(price_c)
price_p = put_price(S_newnew_p, K_new_p, T, r, vol_new_p, fee, right_p, cheat)
print(price_p)
plt.plot(S_newnew_c, price_c)
plt.plot(S_newnew_p, price_p)

# %%
##test = long_put_price(S_newnew_p, K_new_p, T, r, vol_new_p, fee, right_p)
##plt.plot(S_newnew_p, test)
# %%
def short_strangle(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, S_newnew_p, K_new_p, vol_new_p, right_p, cheat):
    price_c = call_price(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, cheat)
    price_p = put_price(S_newnew_p, K_new_p, T, r, vol_new_p, fee, right_p, cheat)
    p = price_c + price_p
    return p
# %%
price_c = pd.Series(price_c)
price_p = pd.Series(price_p)
dataframe = pd.DataFrame()
dataframe['S'] = S_newnew_c
dataframe['price_c'] = price_c
dataframe['price_p'] = price_p

# %%
def short_strangle_plot(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, S_newnew_p, K_new_p, vol_new_p, right_p, cheat):
    price_c = call_price(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, cheat)
    price_p = put_price(S_newnew_p, K_new_p, T, r, vol_new_p, fee, right_p, cheat)
    p = short_strangle(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, S_newnew_p, K_new_p, vol_new_p, right_p, cheat)
    plt.figure(figsize=(16, 8))
    plt.suptitle('Payoff and profit of Options', fontsize = 20, fontweight = 'bold')
    ##plt.text(0.5, 0.04, 'Stock/Underlying Price ($)', ha='center', fontsize=14, fontweight='bold')
    ##plt.text(0.08, 0.5, 'Option Payoff and Profit($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    plt.plot(S_newnew_c, p, 'green')
    plt.plot(S_newnew_c, price_c, 'r--')
    plt.plot(S_newnew_c, price_p, 'b--')
    plt.axvline(x = 13083, color = 'black')
    plt.axvline(x = 12140, color = 'black')
    plt.ylim(-1000,750)
    plt.legend(['Short Straddle Profit', 'Short Call', 'Short Put'])
    plt.title('Short Straddle')
    plt.grid('True')
    plt.show()
# %%
plt.figure(figsize=(16, 8))
plt.plot(S_newnew_p, price_p)
plt.grid('True')
# %%
plt.figure(figsize=(16, 8))
plt.plot(S_newnew_c, price_c)
plt.grid('True')
    

# %%
short_strangle_plot(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, S_newnew_p, K_new_p, vol_new_p, right_p, cheat)
# %%

price_c = call_price(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c)
price_p = put_price(S_newnew_p, K_new_p, T, r, vol_new_p, fee, right_p)
p = short_strangle(S_newnew_c, K_new_c, T, r, vol_new_c, fee, right_c, S_newnew_p, K_new_p, vol_new_p, right_p)

print(price_c[1000])
print(price_p[1000])
print(p[10])
print(price_c[1000] + price_p[1000])
# %%
print(S_newnew_c[1000])
print(S_newnew_p[1000])
# %%
for i in range(len(p)):
    if p[i] > -10 and p[i] < 10:
        print(S_newnew_c[i])
        print("\n", p[i])

# %%
