import importlib
from typing import Callable, List, Union, Dict, Any
import re
import ast

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf



SCALE_MAP = {"daily": 252, "weekly": 52, "monthly": 12}


def sharpe_ratio(
    returns: List[Union[int, float]],
    rf: Union[List, float],
    ret_scale: str,
    annual: bool = False,
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
        return float(mean / stdev)


def load_config(cfg_path: str) -> Union[DictConfig, ListConfig]:
    config = OmegaConf.load(cfg_path)

    return config


def get_class(class_object: Any) -> Callable:
    module_name, class_object = class_object.rsplit(".", 1)
    module = importlib.import_module(module_name)

    return getattr(module, class_object)


def generate_random_param(ind: Any, bound_args: Dict[str, Any]) -> Any:
    """
    Generate random parameters for the given indicator
    """
    if ind.tag in ["EMA", "SMA"]:
        bound_args.update(
            {
                k: np.random.randint(5, 100) if v is not None else v
                for k, v in bound_args.items()
            }
        )

    elif ind.tag == "RSI":
        bound_args.update({k: np.random.randint(5, 100) for (k,) in bound_args.keys()})

    elif ind.tag == "BB":
        bound_args.update(
            {
                "period": np.random.randint(5, 100),
                "std": np.random.uniform(1.2, 3.0),
                "reverse": np.random.choice([True, False]),
            }
        )

    vars(ind).update(bound_args)

    return ind


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
    match = re.match(r"(\w+)\s*\((.*?)\)", col)
    if not match:
        return [col]

    ind, params_str = match.groups()
    params = ast.literal_eval(f"({params_str})")
    params = [params] if isinstance(params, int) else list(params)

    return [f"{ind}_{param}" for param in params]
