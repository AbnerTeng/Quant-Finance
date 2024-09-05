"""
Moving average strategies
"""
import pandas as pd
from ..base.base_indicator import (
    BaseIndicator,
    GlobalDataManager
)


class SMA(BaseIndicator):
    """
    Simple moving average indicator
    """
    def __init__(self, k1: int, k2: int = None, column: str = 'close') -> None:
        super().__init__(column)
        self.k1 = k1
        self.k2 = k2

    def build(self) -> pd.DataFrame:
        data = GlobalDataManager.get_data()
        close = GlobalDataManager.get_column(self.column)
        data[f'SMA_{self.k1}'] = close.rolling(window=self.k1).mean()

        if isinstance(self.k2, int):
            data[f"SMA_{self.k2}"] = close.rolling(window=self.k2).mean()

        return data

    def __str__(self) -> str:
        return f"SMA ({self.k1}, {self.k2})"


class EMA(BaseIndicator):
    """
    Exponential moving average indicator
    """
    def __init__(self, k1: int, k2: int = None, column: str = 'close') -> None:
        super().__init__(column)
        self.k1 = k1
        self.k2 = k2

    def build(self) -> pd.DataFrame:
        data = GlobalDataManager.get_data()
        close = GlobalDataManager.get_column(self.column)
        data[f'EMA_{self.k1}'] = close.ewm(span=self.k1).mean()

        if isinstance(self.k2, int):
            data[f"EMA_{self.k2}"] = close.ewm(span=self.k2).mean()

        return data

    def __str__(self) -> str:
        return f"EMA ({self.k1}, {self.k2})"


class MACD(BaseIndicator):
    """
    MACD indicator
    """
    def __init__(self, k1: int, k2: int, k3: int, column: str = 'close') -> None:
        super().__init__(column)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        assert k1 <= k2, "k1 must be less than k2"

    def build(self) -> pd.DataFrame:
        data = GlobalDataManager.get_data()
        close = GlobalDataManager.get_column(self.column)
        data['MACD'] = close.ewm(span=self.k1).mean() - close.ewm(span=self.k2).mean()
        data['Signal'] = data['MACD'].ewm(span=self.k3).mean()

        return data

    def __str__(self) -> str:
        return f"MACD ({self.k1}, {self.k2}, {self.k3})"
