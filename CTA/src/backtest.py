"""
General backtest class
"""
from typing import Any
import pandas as pd
import numpy as np


class BackTest:
    """
    General backtest class
    """
    def __init__(
        self, data: Any, initial_cap: float, trans_cost: float
    ) -> None:
        """
        strategy: backtest strategy

        data: data to backtest

        initial_cap: initial capital to backtest
        """
        self.data = data
        self.cap = initial_cap
        self.fee = trans_cost
        self.trade_log = {
            "buy": [],
            "sell": [],
            "buytocover": [],
            "short": [],
        }
        self.profit_map = {
            "profit": [],
            "profitwithfee": []
        }

    def calculate_equity(self) -> pd.DataFrame:
        """
        Calculate equity
        """
        equity = pd.DataFrame(
            {
                "profit": np.cumsum(self.profit_map["profit"]),
                "profitwithfee": np.cumsum(self.profit_map["profitwithfee"])
            },
            index=self.data.index[:len(self.profit_map["profit"])]
        )
        equity["equity_val"] = self.cap + equity["profitwithfee"]
        equity["dd%"] = (equity["equity_val"] / equity["equity_val"].cummax()) - 1
        equity["dd"] = equity["equity_val"] - equity["equity_val"].cummax()
        return equity
