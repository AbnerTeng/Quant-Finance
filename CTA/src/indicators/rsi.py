"""
Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
"""

import pandas as pd
from ..base.base_indicator import BaseIndicator, GlobalDataManager


class RSI(BaseIndicator):
    """
    RSI indicator class
    """

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

    def build(self) -> pd.DataFrame:
        data = GlobalDataManager.get_data()
        close = GlobalDataManager.get_column("close")
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        data[f"RSI_{self.period}"] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        assert (
            data[f"RSI_{self.period}"].max() <= 100
            and data[f"RSI_{self.period}"].min() >= 0
        )

        return data

    def __str__(self) -> str:
        return f"RSI ({self.period})"
