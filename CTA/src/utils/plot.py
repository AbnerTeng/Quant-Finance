"""
All about plots
"""

from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt


def profit_graph(equity: pd.DataFrame) -> None:
    """
    Equity: profit_map dataframe
    """
    _, ax = plt.subplots(figsize=(16, 6))
    high_index = equity[
        equity["profitwithfee"].cummax() == equity["profitwithfee"]
    ].index
    equity["profitwithfee"].plot(label="Total Profit", ax=ax, color="red", grid=True)
    plt.fill_between(
        equity["dd"].index,
        equity["dd"],
        0,
        facecolor="red",
        label="Drawdown",
        alpha=0.5,
    )
    plt.scatter(
        high_index,
        equity["profitwithfee"].loc[high_index],
        color="#02ff0f",
        label="High",
    )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Accumulated Profit")
    plt.title("Profit & Drawdown", fontsize=15)
    plt.show()


def trade_position(
    df: pd.DataFrame, indicators: List[pd.Series], log: Dict[str, List[int]]
) -> None:
    """
    df: trade_log dataframe
    """
    _, ax = plt.subplots(figsize=(16, 6))
    df["close"].plot(label="Close Price", ax=ax, color="grey", grid=True, alpha=0.8)

    for indicator in indicators:
        indicator.plot(label="Indicator", ax=ax, color="blue", grid=True, alpha=0.8)

    for signal in log.keys():
        color = "orangered" if signal in ["long", "sell"] else "limegreen"
        marker = "^" if signal in ["long", "buytocover"] else "v"
        plt.scatter(
            df["close"].iloc[log[signal]].index,
            df["close"].iloc[log[signal]],
            color=color,
            label=signal,
            marker=marker,
            s=60,
        )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Trade Position", fontsize=15)
    plt.show()
