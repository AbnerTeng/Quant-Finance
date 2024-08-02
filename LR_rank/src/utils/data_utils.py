"""
Useful utilities about data loading and preprocessing
"""
from typing import Any, Dict
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> Any:
    """
    Load data from path
    """
    file_type = path.split('.')[-1]
    if file_type == "csv":
        return pd.read_csv(path, encoding="utf-8")

    elif file_type == "yaml":
        with open(path, 'r', encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    else:
        raise ValueError("File type not supported")

def merge_to_data(_map: Dict[str, pd.Series], data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the rank values to data
    """
    for k, v in _map.items():
        data.loc[data.Date == k, 'label'] = v

    return data


def date_lab_encoder(series: pd.Series) -> np.ndarray:
    """
    trasform date data to groups
    """
    le = LabelEncoder()
    groups = le.fit_transform(series)
    return groups
