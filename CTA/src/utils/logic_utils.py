"""
Logic utilities for trading strategies
"""

from typing import Union


import pandas as pd


def crossover(
    cross: Union[pd.Series, pd.DataFrame],
    crossed: Union[pd.Series, pd.DataFrame],
    idx: int,
) -> bool:
    """
    Up crossover, mainly used for moving average strategies

    Scenarios
    ---------
    1. Close[i] >= MA[i] & Close[i-1] < MA[i-1]
    2. FastMA[i] >= SlowMA[i] & FastMA[i-1] < SlowMA[i-1]
    """
    return (cross.iloc[idx] >= crossed.iloc[idx]) & (
        cross.iloc[idx - 1] < crossed.iloc[idx - 1]
    )


def crossdown(
    cross: Union[pd.Series, pd.DataFrame],
    crossed: Union[pd.Series, pd.DataFrame],
    idx: int,
) -> bool:
    """
    Down crossover, mainly used for moving average strategies

    Scenarios
    ---------
    1. Close[i] < MA[i] & Close[i-1] > MA[i-1]
    2. FastMA[i] < SlowMA[i] & FastMA[i-1] > SlowMA[i-1]
    """
    return (cross.iloc[idx] <= crossed.iloc[idx]) & (
        cross.iloc[idx - 1] > crossed.iloc[idx - 1]
    )


def stop_loss(ret: float, threshold: float = -0.07) -> bool:
    """
    Stop loss condition
    """
    return ret <= threshold


def stop_profit(ret: float, threshold: float = 0.05) -> bool:
    """
    Stop profit condition
    """
    return ret >= threshold
