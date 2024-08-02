"""
All about performance indicators and metrics
"""
import numpy as np
import pandas as pd


def calc_month_perf(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly return
    
    *Can add other indicators or metrics in future*
    """
    data["ret"] = [0] + list(data["close"][1:].values / data['close'][:-1].values -1)
    data["next_ret"] = list(data["ret"][1:].values) + [np.nan]
    return data

def sharpe_ratio(ret: np.ndarray) -> float:
    """
    Calculate Sharpe Ratio
    """
    return (np.mean(ret) / np.std(ret)) * np.sqrt(12)