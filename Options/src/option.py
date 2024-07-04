"""
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

call = pd.DataFrame()
import numpy as np
S = 12998
T = 21/365
r = 0.0126
sigma = 0.5
c_buy = np.array([1890, 1800, 1700, 1610, 1510, 1420, 1320, 1230, 1170, 1080, 1000, 915, 830, 750, 670, 610, 505, 433, 354, 381, 324, 277, 231, 190, 157, 122, 97, 78, 61, 47, 37, 28.5, 22.5, 17, 12.5, 11.5, 9.1, 3.8, 5, 3.4, 4.5])
c_sell = np.array([2070, 1970, 1870, 1780, 1680, 1590, 1490, 1400, 1280, 1190, 1110, 1020, 935, 855, 775, 700, 630, 560, 454, 392, 339, 284, 240, 191, 161, 126, 100, 79, 64, 49.5, 38, 29, 23, 17.5, 14.5, 12, 10, 8.2, 7.5, 7.3, 6.4])
c_month_price = (c_buy + c_sell)/2
K_month = np.arange(11000, 15100, 100)
print(K_month)
call['price'] = c_month_price
call['k'] = K_month
call['f/k'] = S/call['k']
iv = Newton_call_iv(S, K_month, T, r, c_month_price, sigma)
call['iv'] = iv

print(call)

def paynpro_graph(S, K, K_lc, K_sc, P_lc, P_lp, P_sc):
    plt.subplot(2, 1, 1)
    P = bear_spread(S, K_lc, K_sc, P_lc, P_sc)
    P1 = bear_spread_payoff(S, K_lc, K_sc)
    long_c = long_call(S, K_lc, P_lc)
    short_c = short_call(S, K_sc, P_sc)
    plt.plot(S, P, 'black')
    plt.plot(S, P1, 'red')
    plt.plot(S, long_c, 'r--')
    plt.plot(S, short_c, 'b--')
    plt.legend(['Bear Spread Profit', 'Bear Spread Payoff', 'Long Call', 'Short Call'])
    plt.title('Bear Spread')

    plt.subplot(2, 1, 2)
    P = straddle(S, K, P_lc, P_lp)
    P1 = straddle_payoff(S, K)
    P_longcall = long_call(S, K, P_lc)
    P_longput = long_put(S, K, P_lp)
    plt.plot(S, P, 'black')
    plt.plot(S, P1, 'red')
    plt.plot(S, P_longcall, 'r--')
    plt.plot(S, P_longput, 'b--')
    plt.legend(['Straddle Profit', 'Straddle Payoff', 'Long Call', 'Long Put'])
    plt.title('Straddle')
    plt.show()
"""