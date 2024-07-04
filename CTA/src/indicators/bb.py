"""
Boolinger Band strategies
"""
from typing import Any
import pandas as pd
from ..backtest import BackTest
from ..utils.logic_utils import (
    crossover,
    crossdown
)


class BBStrat(BackTest):
    """
    Backtest class for boolinger band strategies
    """
    def __init__(
        self, data: Any, strat_type: str,
        initial_cap: float, trans_cost: float,
        k1: int = 5
    ) -> None:
        """
        data: data to backtest

        strattype: type of moving average strategy

        initial_cap: initial capital to backtest

        trans_cost: transaction cost
        """
        super().__init__(data, initial_cap, trans_cost)
        self.strat_type = strat_type
        self.k1 = k1

    def build_bb_indicator(self, std_times: float) -> None:
        """
        Build boolinger band indicator
        """
        if isinstance(self.data, dict):
            self.data = pd.DataFrame(self.data)
        self.data["ma"] = self.data["close"].rolling(window=self.k1).mean()
        self.data["std"] = self.data["close"].rolling(window=self.k1).std()
        self.data["upper"] = self.data["ma"] + std_times * self.data["std"]
        self.data["lower"] = self.data["ma"] - std_times * self.data["std"]

        if self.strat_type == "bbw":
            self.data["bbw"] = (
                self.data["upper"] - self.data["lower"]
            ) / self.data["ma"]

    def vanilla_mom_bb(self) -> None:
        """
        Run vanilla boolinger band strategy
        """
        curr_cond = None
        for i in range(len(self.data) - 1):
            if curr_cond is None:
                self.profit_map["profit"].append(0)
                self.profit_map["profit_fee"].append(0)

                if crossover(
                    self.data["close"].iloc[i],
                    self.data["upper"].iloc[i]
                ):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "buy"
                    t = i + 1
                    self.trade_log["buy"].append(t)

                elif crossdown(
                    self.data["close"].iloc[i],
                    self.data["lower"].iloc[i]
                ):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "short"
                    t = i + 1
                    self.trade_log["short"].append(t)

            elif curr_cond == "buy":
                profit = execute_size(
                    self.data["open"].iloc[i+1] - self.data["open"].iloc[i]
                )
                self.profit_map["profit"].append(profit)

                if (
                    crossdown(
                        self.data["close"].iloc[i],
                        self.data["lower"].iloc[i]
                    )
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[i+1] - self.data["open"].iloc[t]
                    )
                    fee = self.cap * self.fee + execute_size * self.data["open"].iloc[i+1] * self.fee
                    self.profit_map["profit_fee"].append(pnl_round - fee)
                    curr_cond = None

                else:
                    self.profit_map["profitwithfee"].append(profit)

            elif curr_cond == "short":
                pass

    def vanilla_rev_bb(self) -> None:
        """
        Run vanilla boolinger band strategy
        """
        curr_cond = None
        for i in range(len(self.data) - 1):
            if curr_cond is None:
                self.profit_map["profit"].append(0)
                self.profit_map["profit_fee"].append(0)

                if crossover(
                    self.data["close"].iloc[i],
                    self.data["upper"].iloc[i]
                ):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "buy"
                    t = i + 1
                    self.trade_log["buy"].append(t)

                elif crossdown(
                    self.data["close"].iloc[i],
                    self.data["lower"].iloc[i]
                ):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "short"
                    t = i + 1
                    self.trade_log["short"].append(t)

            elif curr_cond == "buy":
                profit = execute_size(
                    self.data["open"].iloc[i+1] - self.data["open"].iloc[i]
                )
                self.profit_map["profit"].append(profit)

                if (
                    crossdown(
                        self.data["close"].iloc[i],
                        self.data["lower"].iloc[i]
                    )
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[i+1] - self.data["open"].iloc[t]
                    )
                    fee = self.cap * self.fee + execute_size * self.data["open"].iloc[i+1] * self.fee
                    self.profit_map["profit_fee"].append(pnl_round - fee)
                    curr_cond = None

                else:
                    self.profit_map["profitwithfee"].append(profit)

            elif curr_cond == "short":
                pass
