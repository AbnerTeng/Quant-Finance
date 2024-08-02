"""
Useful utilities
"""
from typing import Dict, Union, List
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RISK_FREE = 0.02 # not in %


def load_data(path: str) -> Union[pd.DataFrame, Dict] | None:
    """
    Load different types of data
    Can manually add more types of data

    path: (str) path to the data
    """
    suffix = path.split('.')[-1]

    if suffix == "csv":
        return pd.read_csv(path, encoding="utf-8")

    if suffix == "yaml":
        with open(path, 'r', encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file)

        raise ValueError(f"Unknown file type: {suffix}")


def plot_dd(equity: pd.DataFrame) -> None:
    """
    Plot drawdown
    """
    _, ax = plt.subplots(figsize=(16, 6))
    high_index = equity[
        equity['profitwithfee'].cummax() == equity['profitwithfee']
    ].index
    equity['profitwithfee'].plot(
        label="Total Profit",
        ax=ax,
        color="red",
        grid=True
    )
    plt.fill_between(
        equity["drawdown"].index,
        equity["drawdown"],
        0,
        facecolor="red",
        label="Drawdown",
        alpha=0.5
    )
    plt.scatter(
        high_index,
        equity["profitwithfee"].loc[high_index],
        color="#02ff0f",
        label="High"
    )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Accumulated Profit")
    plt.title("Profit & Drawdown", fontsize=15)
    plt.show()


def plot_trade_position(
    df: pd.DataFrame,
    signal_map: Dict[str, List[int]],
    indicator_name: str
) -> None:
    """
    df: OHLCV data
    signal_map: signal map
    """
    if "close" not in df.columns:
        df.rename(columns={"Close": "close"}, inplace=True)

    _, ax = plt.subplots(figsize=(16, 6))
    df["close"].plot(
        label="Close Price",
        ax=ax,
        color="grey",
        grid=True,
        alpha=0.8
    )
    df[indicator_name].plot(
        label=indicator_name,
        ax=ax,
        color="blue",
        grid=True,
        alpha=0.8
    )
    for signal in signal_map.keys():
        color = "orangered" if signal in ["buy", "sell"] else "limegreen"
        marker = "^" if signal in ["buy", "buytocover"] else "v"
        plt.scatter(
            df['close'].iloc[signal_map[signal]].index,
            df['close'].iloc[signal_map[signal]],
            color=color,
            label=signal,
            marker=marker,
            s=60
        )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Trade Position", fontsize=15)
    plt.show()


def sharpe(ret: List[float]) -> float:
    """
    Calculate Sharpe ratio

    Input:
        ret: (List[float]) daily return

    Output:
        sharpe: (float) Sharpe ratio
    """
    ret = np.array(ret)
    return (ret.mean() / ret.std()) * np.sqrt(252)


def metrics(
    equity: pd.DataFrame,
    ret: List[float]
) -> Dict[str, Union[float, str]]:
    """
    Calculate performance metrics

    Input:
        equity: (pd.DataFrame) equity curve data
        ret: (List[float]) daily return

    Output:
        metrics_map: (Dict[str, Union[str, float]]) performance metrics
    """
    metrics_map = {}
    metrics_map["sharpe_ratio"] = sharpe(ret)
    metrics_map["mdd"] = f"{-1 * equity['drawdown%'].min() * 100}%"
    metrics_map["annual_return"] = f"{np.mean(ret) * 252 * 100}%"
    return metrics_map
