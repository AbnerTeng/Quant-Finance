# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.titleweight'] = 'medium'
plt.rcParams['lines.linewidth'] = 2.5

## define Put and Call options with Price and w/o Price
def long_call(S, K_lc, P_lc):
    P = list(map(lambda x: max(x-K_lc, 0) - P_lc, S))
    return P

def long_call_payoff(S, K_lc):
    P = list(map(lambda x: max(x-K_lc, 0), S))
    return P

def long_put(S, K_lp, P_lp):
    P = list(map(lambda x: max(K_lp-x, 0) - P_lp, S))
    return P

def long_put_payoff(S, K_lp):
    P = list(map(lambda x: max(K_lp-x, 0), S))
    return P

def short_call(S, K_sc, P_sc):
    P = long_call(S, K_sc, P_sc)
    return [-1.0*p for p in P]

def short_call_payoff(S, K_sc):
    P = long_call_payoff(S, K_sc)
    return [-1.0*p for p in P]

def short_put(S, K_sp, P_sp):
    P = long_put(S, K_sp, P_sp)
    return [-1.0*p for p in P]

def short_put_payoff(S, K_sp):  
    P = long_put_payoff(S, K_sp)
    return [-1.0*p for p in P]

## Use those basic options to construct strategies with Price and w/o Price
def bear_spread(S, K_lc, K_sc, P_lc, P_sc):
    P1 = long_call(S, K_lc, P_lc)
    P2 = short_call(S, K_sc, P_sc)
    return [x+y for x,y in zip(P1, P2)]

def bear_spread_payoff(S, K_lc, K_sc):
    P1 = long_call_payoff(S, K_lc)
    P2 = short_call_payoff(S, K_sc)
    return [x+y for x,y in zip(P1, P2)]

def straddle(S, K, P_lc, P_lp):
    P1 = long_call(S, K, P_lc)
    P2 = long_put(S, K, P_lp)
    return [x+y for x,y in zip(P1, P2)]

def straddle_payoff(S, K):
    P1 = long_call_payoff(S, K)
    P2 = long_put_payoff(S, K)
    return [x+y for x,y in zip(P1, P2)]

## Define stock price range
S = [x for x in range(13000, 15000, 10)]

## Define payoff and profit graph
fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True, figsize = (20, 20))
fig.suptitle('Payoff of Options', fontsize = 20, fontweight = 'bold')
fig.text(0.5, 0.04, 'Stock/Underlying Price ($)', ha='center', fontsize=14, fontweight='bold')
fig.text(0.08, 0.5, 'Option Payoff and Profit($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

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

paynpro_graph(S, 13800, 14000, 13600, 26, 75, 265)
# %%
