from typing import List, Union
import re
import ast
import inspect
import numpy as np

SCALE_MAP = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12
}


def sharpe_ratio(
    returns: List[Union[int, float]],
    rf: Union[List, float],
    ret_scale: str,
    annual: bool = False
) -> float:
    """
    Calculate the sharpe ratio of a given return series
    """
    stdev = np.std(returns)
    if isinstance(rf, List):
        mean = np.mean(returns) - np.mean(rf)
    else:
        mean = np.mean(returns) - rf

    if annual:
        return (mean / stdev) * np.sqrt(SCALE_MAP[ret_scale])
    else:
        return mean / stdev

def calmar_ratio(returns: List[Union[int, float]], rf: Union[List, float]):
    """
    Calculate the calmar ratio of a given return series
    """
    raise NotImplementedError


def sortino_ratio(returns: List[Union[int, float]], rf: Union[List, float]):
    """
    Calculate the sortino ratio of a given return series
    """
    raise NotImplementedError


def parse_column(col):
    match = re.match(r'(\w+)\s*\((.*?)\)', col)
    if not match:
        return [col]

    ind, params_str = match.groups()
    params = ast.literal_eval(f"({params_str})")
    params = [params] if isinstance(params, int) else list(params)

    return [f"{ind}_{param}" for param in params]
