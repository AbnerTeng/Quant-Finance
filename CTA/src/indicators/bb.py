"""
Boolinger Band strategies
"""
import pandas as pd
from ..base.base_indicator import (
    GlobalDataManager,
    BaseIndicator
)


class BB(BaseIndicator):
    """
    Bolllinger band indicator class
    """
    def __init__(self, period: int, is_reverse: bool = False) -> None:
        super().__init__()
        self.period = period
        self.is_reverse = 1 if is_reverse else 0

    def build(self) -> pd.DataFrame:
        data = GlobalDataManager.get_data()
        close = GlobalDataManager.get_column('close')
        data['ma'] = close.rolling(window=self.period).mean()
        data['std'] = close.rolling(window=self.period).std()
        data['upper'] = data['ma'] + 2 * data['std']
        data['lower'] = data['ma'] - 2 * data['std']
        assert data['upper'].max() > data['lower'].max()
        assert data['upper'].min() > data['lower'].min()

        return data

    def __str__(self) -> str:
        return f"BB ({self.period}, {self.is_reverse})"
