# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp as e
from scipy.stats import norm



plt.style.use('ggplot')
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.titleweight'] = 'medium'
plt.rcParams['lines.linewidth'] = 2.5

## define Put and Call options with Price and w/o Price
##def long_call(S, K_lc, P_lc, fee):
##    P = list(map(lambda x: max(x-K_lc, 0) - P_lc - fee, S))
##    return P
##
##def long_call_payoff(S, K_lc):
##    P = list(map(lambda x: max(x-K_lc, 0), S))
##    return P
##
##def long_put(S, K_lp, P_lp):
##    P = list(map(lambda x: max(K_lp-x, 0) - P_lp, S))
##    return P
##
##def long_put_payoff(S, K_lp):
##    P = list(map(lambda x: max(K_lp-x, 0), S))
##    return P
##
##def short_call(S, K_sc, P_sc, fee):
##    P = list(map(lambda x: P_sc - max(x - K_sc, 0) - fee, S))
##    return P
##
##def short_call_payoff(S, K_sc):
##    P = long_call_payoff(S, K_sc)
##    return [-1.0*p for p in P]
##
##def short_put(S, K_sp, P_sp):
##    P = long_put(S, K_sp, P_sp)
##    return [-1.0*p for p in P]
##
##def short_put_payoff(S, K_sp):  
##   P = long_put_payoff(S, K_sp)
##   return [-1.0*p for p in P]

## Use those basic options to construct strategies with Price and w/o Price
##def bear_spread(S, K_lc, K_sc, P_lc, P_sc):
##    P1 = long_call(S, K_lc, P_lc)
##    P2 = short_call(S, K_sc, P_sc)
##    return [x+y for x,y in zip(P1, P2)]
##
##def bear_spread_payoff(S, K_lc, K_sc):
##    P1 = long_call_payoff(S, K_lc)
##    P2 = short_call_payoff(S, K_sc)
##    return [x+y for x,y in zip(P1, P2)]
##
##def straddle(S, K, P_lc, P_lp):
##    P1 = long_call(S, K, P_lc)
##    P2 = long_put(S, K, P_lp)
##    return [x+y for x,y in zip(P1, P2)]
##
##def straddle_payoff(S, K):
##    P1 = long_call_payoff(S, K)
##    P2 = long_put_payoff(S, K)
##    return [x+y for x,y in zip(P1, P2)]
##
##def strangle(S, K_lc, K_lp, P_lc, P_lp):
##    P1 = long_call(S, K_lc, P_lc)
##    P2 = long_put(S, K_lp, P_lp)
##    return [x+y for x,y in zip(P1, P2)]
##
##def strangle_payoff(S, K_lc, K_lp):
##    P1 = long_call_payoff(S, K_lc)
##    P2 = long_put_payoff(S, K_lp)
##    return [x+y for x,y in zip(P1, P2)]
##
##def short_strangle(S, K_sc, K_sp, P_sc, P_sp):
##    P1 = short_call(S, K_sc, P_sc)
##    P2 = short_put(S, K_sp, P_sp)
##    return [x+y for x,y in zip(P1, P2)]
##
##def short_strangle_payoff(S, K_sc, K_sp):
##    P1 = short_call_payoff(S, K_sc)
##    P2 = short_put_payoff(S, K_sp)
##    return [x+y for x,y in zip(P1, P2)]

##def bull_spread(S, K_lc, K_sc, P_lc, P_sc, fee):
##    P1 = short_call(S, K_sc, P_sc, fee)
##    P2 = long_call(S, K_lc, P_lc, fee)
##    return [x+y for x,y in zip(P1, P2)]
##
##def bull_spread_payoff(S, K_lc, K_sc):
##    P1 = short_call_payoff(S, K_sc)
##    P2 = long_call_payoff(S, K_lc)
##    return [x+y for x,y in zip(P1, P2)]

## define variables
Spot = 13100
low = Spot - 2000
high = Spot + 2000
step = 1
S = [x for x in range(low, high, step)]
S = pd.Series(S)
K_lc = 13200
K_lp = 12800
c_vol = 0.313723
p_vol = 0.352223
r = 0.01
T = 28/365
fee = 2
call = 356
put = 333

def c_d1_calc(S, K_lc, r, c_vol, T):
    return (np.log(S/K_lc) + (r + 0.5*c_vol**2)*T)/(c_vol*np.sqrt(T))

def p_d1_calc(S, K_lp, r, p_vol, T):
    return (np.log(S/K_lp) + (r + 0.5*p_vol**2)*T)/(p_vol*np.sqrt(T))

def BS_call(S, K_lc, r, c_vol, T, fee):
    d1 = c_d1_calc(S, K_lc, r, c_vol, T)
    d2 = d1 - c_vol*np.sqrt(T)
    return S*norm.cdf(d1)-K_lc*e(-r*T)*norm.cdf(d2)-fee-call

def BS_put(S, K_lp, r, p_vol, T, fee):
    return BS_call(S, K_lp, r, p_vol, T, fee) - S + e(-r*T)*K_lp-put

def BS_strangle(S, K_lc, K_lp, r, c_vol, p_vol, T, fee):
    p_call = BS_call(S, K_lc, r, c_vol, T, fee)
    p_put = BS_put(S, K_lp, r, p_vol, T, fee)
    sell_call_1 = -1.0*BS_call(S, K_lc + 200 , r, c_vol, T, fee) + 247
    ##sell_call_2 = -1.0*BS_call(S, K_lc + , r, c_vol, T, fee)
    return p_call + p_put + sell_call_1


def strangle_plot(S, K_lc, K_lp, r, c_vol, p_vol, T, fee):
    P = BS_strangle(S, K_lc, K_lp, r, c_vol, p_vol, T, fee)
    P_longcall = BS_call(S, K_lc, r, c_vol, T, fee)
    P_longput = BS_put(S, K_lp, r, p_vol, T, fee)
    P_shortcall = -1.0*BS_call(S, K_lc +200 , r, c_vol, T, fee)+247
    plt.figure(figsize = (16, 8))
    plt.suptitle('Payoff and profit of Options', fontsize = 20, fontweight = 'bold')
    plt.text(0.5, 0.04, 'Stock/Underlying Price ($)', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.08, 0.5, 'Option Payoff and Profit($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    plt.plot(S, P, 'red')
    plt.plot(S, P_longcall, 'r--')
    plt.plot(S, P_longput, 'b--')
    plt.plot(S, P_shortcall, 'g--')
    plt.axvline(x = 13110, color = 'black')
    plt.legend(['Strangle Profit', 'Long Call', 'Long Put'])
    plt.title('Strangle')
    plt.show()

a = BS_strangle(S, K_lc, K_lp, r, c_vol, p_vol, T, fee)
print(a[2010])

##Pp = BS_strangle(S, K_lc, K_lp, 56, 12.5, 2)
##max_loss = -min(Pp)
##max_profit = max(Pp)
##print(f"Maximum Loss: {max_loss}, Maximum Profit: {max_profit}")
##
##def BEP(K_lc, P_lc, P_sc, fee):
##    bep = K_lc + abs(P_sc - P_lc) + fee
##    print(bep)
##    return bep


## Define payoff and profit graph
##fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True, figsize = (20, 10))
##fig.suptitle('Payoff of Options', fontsize = 20, fontweight = 'bold')
##fig.text(0.5, 0.04, 'Stock/Underlying Price ($)', ha='center', fontsize=14, fontweight='bold')
##fig.text(0.08, 0.5, 'Option Payoff and Profit($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

## Enter those params
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

##paynpro_graph(S, 13800, 14000, 13600, 26, 75, 265)

def straddle_plot(S, K, P_lc, P_lp):
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

##def strangle_plot(S, K_lc, K_lp, P_lc, P_lp):
##    P = strangle(S, K_lc, K_lp, P_lc, P_lp)
##    P1 = strangle_payoff(S, K_lc, K_lp)
##    P_longcall = long_call(S, K_lc, P_lc)
##    P_longput = long_put(S, K_lp, P_lp)
##    plt.figure(figsize = (16, 8))
##    plt.suptitle('Payoff and profit of Options', fontsize = 20, fontweight = 'bold')
##    plt.text(0.5, 0.04, 'Stock/Underlying Price ($)', ha='center', fontsize=14, fontweight='bold')
##    plt.text(0.08, 0.5, 'Option Payoff and Profit($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
##    plt.plot(S, P, 'black')
##    plt.plot(S, P1, 'red')
##    plt.plot(S, P_longcall, 'r--')
##    plt.plot(S, P_longput, 'b--')
##    plt.legend(['Strangle Profit', 'Strangle Payoff', 'Long Call', 'Long Put'])
##    plt.title('Strangle')
##    plt.show()

def short_strangle_plot(S, K_sc, K_sp, P_sc, P_sp):
    P = short_strangle(S, K_sc, K_sp, P_sc, P_sp)
    P1 = short_strangle_payoff(S, K_sc, K_sp)
    P_shortcall = short_call(S, K_sc, P_sc)
    P_shortput = short_put(S, K_sp, P_sp)
    plt.figure(figsize = (16, 8))
    plt.suptitle('Payoff and profit of Options', fontsize = 20, fontweight = 'bold')
    plt.text(0.5, 0.04, 'Stock/Underlying Price ($)', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.08, 0.5, 'Option Payoff and Profit($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    plt.plot(S, P, 'black')
    plt.plot(S, P1, 'red')
    plt.plot(S, P_shortcall, 'r--')
    plt.plot(S, P_shortput, 'b--')
    plt.legend(['Short Strangle Profit', 'Short Strangle Payoff', 'Short Call', 'Short Put'])
    plt.title('Short Strangle')
    plt.show()

def bull_spread_plot(S, K_lc, K_sc, P_lc, P_sc, fee):
    P = bull_spread(S, K_lc, K_sc, P_lc, P_sc, fee)
    P1 = bull_spread_payoff(S, K_lc, K_sc)
    P_longcall = long_call(S, K_lc, P_lc, fee)
    P_shortcall = short_call(S, K_sc, P_sc, fee)
    bep = BEP(K_sc, P_sc, P_lc, fee)
    plt.figure(figsize = (16, 8))
    plt.suptitle('Payoff and profit of Options', fontsize = 20, fontweight = 'bold')
    plt.text(0.5, 0.04, 'Stock/Underlying Price ($)', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.08, 0.5, 'Option Payoff and Profit($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    plt.plot(S, P, 'black')
    plt.plot(S, P1, 'red')
    plt.plot(S, P_longcall, 'r--')
    plt.plot(S, P_shortcall, 'b--')
    plt.axvline(x = 14045.5, color = 'g')
    ##plt.axhline(y = ml, color = 'b', linestyle = '--')
    ##plt.axhline(y = mp, color = 'b', linestyle = '--')
    plt.legend(['Bull Spread Profit', 'Bull Spread Payoff', 'Long Call', 'Short Call'])
    plt.title('Bull Spread')
    plt.show()




##strangle_plot(S, 13500, 13400, 198, 140)
##short_strangle_plot(S, 14000, 13500, 56, 48)
##bull_spread_plot(S, 13900, 13700, 96, 225)
##bull_spread_plot(S, 14100, 13900, 29.5, 96)
strangle_plot(S, K_lc, K_lp, r, c_vol, p_vol, T, fee)