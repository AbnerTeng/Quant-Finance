"""
Base class for all indicators
"""
from abc import ABC, abstractmethod
import pandas as pd


class GlobalDataManager:
    """
    A global data manager to store data for all indicators
    """
    _instance = None
    _data = None

    @classmethod
    def set_data(cls, data: pd.DataFrame) -> None:
        cls._data = data

    @classmethod
    def get_data(cls) -> pd.DataFrame:
        if cls._data is None:
            raise ValueError("Data has not been set. Call set_data() first.")
        else:
            return cls._data

    @classmethod
    def get_column(cls, column: str) -> pd.Series:
        data = cls.get_data()
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        return data[column]


class BaseIndicator(ABC):
    """
    Base class for all indicators
    """
    def __init__(self, column: str = 'close'):
        self.column = column
        self.result = None

    @abstractmethod
    def build(self) -> pd.Series:
        pass

    def get_result(self) -> pd.Series:
        if self.result is None:
            raise ValueError("Indicator hasn't been calculated. Call calculate() first.")
        return self.result

    def __str__(self) -> str:
        return f"{self.__class__.__name__} (column: {self.column})"

    @property
    def name(self) -> str:
        return str(self)
