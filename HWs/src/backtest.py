"""
Backtest module
"""
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

CAPITAL = 1000000
FEE = 0.001425


def get_moving_average(data: pd.DataFrame, k: int) -> None:
    """
    Get k-day moving average line
    """
    if not isinstance(k, int):
        k = int(k)

    data[f"MA_{k}"] = data["Close"].rolling(window=k).mean()


def strategy(
    data: pd.DataFrame,
    k: int
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    """
    Strategy function
    """
    signal_map = {
        "buy": [],
        "sell": [],
        "buytocover": [],
        "short": []
    }
    profit_map = {
        "profit": [],
        "profitwithfee": []
    }
    curr_cond = None

    for i in range(len(data) - 1):
        if curr_cond is None:
            profit_map["profit"].append(0)
            profit_map["profitwithfee"].append(0)

            if data["Close"].iloc[i] > data[f"MA_{k}"].iloc[i]:
                execute_size = CAPITAL / data["Open"].iloc[i+1]
                curr_cond = "buy"
                t = i + 1
                signal_map["buy"].append(t)

            elif data["Close"].iloc[i] < data[f"MA_{k}"].iloc[i]:
                execute_size = CAPITAL / data["Open"].iloc[i+1]
                curr_cond = "short"
                t = i + 1
                signal_map["short"].append(t)

        elif curr_cond == "buy":
            profit = execute_size * (
                data["Open"].iloc[i+1] - data["Open"].iloc[i]
            )
            profit_map["profit"].append(profit)

            if data["Close"].iloc[i] < data[f"MA_{k}"].iloc[i]:
                pnl = execute_size * (
                    data["Open"].iloc[i+1] - data["Open"].iloc[t]
                )
                profitwithfee = profit - (2 * CAPITAL + pnl) * FEE
                profit_map["profitwithfee"].append(profitwithfee)
                curr_cond = None
                signal_map["sell"].append(i + 1)

            else:
                profitwithfee = profit
                profit_map["profitwithfee"].append(profitwithfee)

        elif curr_cond == "short":
            profit = execute_size * (
                data["Open"].iloc[i] - data["Open"].iloc[i+1]
            )
            profit_map["profit"].append(profit)

            if data["Close"].iloc[i] > data[f"MA_{k}"].iloc[i]:
                pnl = execute_size * (
                    data["Open"].iloc[t] - data["Open"].iloc[i+1]
                )
                profitwithfee = profit - (2 * CAPITAL + pnl) * FEE
                profit_map["profitwithfee"].append(profitwithfee)
                curr_cond = None
                signal_map["buytocover"].append(i + 1)

            else:
                profitwithfee = profit
                profit_map["profitwithfee"].append(profitwithfee)

    return signal_map, profit_map


def cal_equity(
    profit_map: Dict[str, List[float]],
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate equity based on profit
    """
    equity = pd.DataFrame(
        {
            "profit": np.cumsum(profit_map["profit"]),
            "profitwithfee": np.cumsum(profit_map["profitwithfee"])
        },
        index=data.index[:-1]
    )
    equity["equity_val"] = CAPITAL + equity["profitwithfee"]
    equity["drawdown"] = equity["equity_val"] - equity["equity_val"].cummax()
    equity["drawdown%"] = (
        equity["equity_val"] / equity["equity_val"].cummax()
    ) - 1
    return equity


def get_ret(equity_df: pd.DataFrame) -> List[float]:
    """
    Get daily return from profit map

    Input:
        equity_df: (pd.DataFrame) equity curve data

    Output:
        ret: (List[float]) daily return
    """
    ret = []

    for idx, val in enumerate(equity_df["equity_val"]):
        if idx == 0:
            ret.append(0)
        else:
            ret.append(
                (val - equity_df["equity_val"].iloc[idx-1])
                / equity_df["equity_val"].iloc[idx-1]
            )

    return ret
