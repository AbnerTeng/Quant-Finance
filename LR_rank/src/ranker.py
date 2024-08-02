"""
Ranking class
"""
from typing import Dict
import pandas as pd
import numpy as np


class Ranker:
    """
    Ranking methods
    """
    def __init__(self, method: str) -> None:
        self.method = method
        self.rank_map = {
            "minmax": Ranker.minmax,
            "order": Ranker.order,
            "quantile": Ranker.quantile,
            "softmax": Ranker.softmax
        }
        self.rank_val = {}

    @staticmethod
    def minmax(data: pd.Series) -> pd.Series:
        """
        Min-Max scaler
        """
        return (data - data.min()) / (data.max() - data.min())

    @staticmethod
    def order(data: pd.Series) -> pd.Series:
        """
        Rank order
        """
        return data.rank(method='min')

    @staticmethod
    def quantile(data: pd.Series) -> pd.Series:
        """
        Quantile transformation
        """
        return pd.qcut(data.rank(method='first'), 10, labels=False) / 10 + 0.1

    @staticmethod
    def softmax(data: pd.Series) -> pd.Series:
        """
        Softmax transformation
        """
        return np.exp(data) / np.exp(data).sum()

    def rank(self, data: pd.Series) -> pd.Series:
        """
        Rank
        """
        if self.method not in self.rank_map:
            raise ValueError("Method not supported")
        return self.rank_map[self.method](data)

    def rank_across_stocks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Rank across stocks in given time
        """
        dates = df.Date.unique()
        for date in dates:
            subdata = df[df.Date == date]
            self.rank_val[date] = self.rank(subdata['next_ret'])
        return self.rank_val