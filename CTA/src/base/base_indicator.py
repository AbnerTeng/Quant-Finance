"""
Base class for all indicators
"""
from abc import ABC, abstractmethod
import inspect
from typing import Tuple
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

    @property
    def name(self) -> str:
        return str(self)

    @property
    def tag(self) -> str:
        return str(self.__class__.__name__)

    def num_args(self) -> int:
        """
        Return the number of arguments a function has
        """
        init_signature = inspect.signature(self.__init__)
        bound_args = {
            param: getattr(self, param) for param in init_signature.parameters if param != 'self'
        }
        num_non_default_args = sum(
            1 for param_name, paramn in init_signature.parameters.items()
            if param_name != 'self' and bound_args[param_name] != paramn.default
        )
        return num_non_default_args

    def get_init_args(self) -> Tuple:
        """
        Get the arguments of the indicator
        """
        init_signature = inspect.signature(self.__init__)
        bound_args = {
            param: getattr(self, param) for param in init_signature.parameters if param != 'self'
        }
        return bound_args
