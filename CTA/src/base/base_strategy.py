"""
General backtest class
"""
from abc import abstractmethod, ABC
from typing import Any, Tuple
import pandas as pd
import numpy as np
from ..utils.logic_utils import (
    stop_loss,
    stop_profit
)


class BackTest(ABC):
    """
    General backtest class
    """
    def __init__(
            self,
            data: Any,
            initial_cap: float,
            trans_cost: float,
            sl_thres: float | None = None,
            sp_thres: float | None = None
    ) -> None:
        """
        strategy: backtest strategy

        data: data to backtest

        initial_cap: initial capital to backtest
        """
        self.data = data
        self.cap = initial_cap
        self.cost = trans_cost
        self.sl_thres = sl_thres if sl_thres is not None else -np.inf
        self.sp_thres = sp_thres if sp_thres is not None else np.inf
        self.trade_log = {
            "long": [],
            "sell": [],
            "buytocover": [],
            "short": [],
        }
        self.profit_map = {
            "profit": [],
            "profitwithfee": []
        }

    def daily_run(
            self,
            idx: int,
            curr_cond: str | None,
            logics: pd.DataFrame,
            execute_size: float,
            t: int,
            curr_ret: float
    ) -> Tuple[str | None, float, int, float]:
        if curr_cond is None:
            self.profit_map["profit"].append(0)
            self.profit_map["profitwithfee"].append(0)
            curr_ret = 0

            if all(logics["long"]):
                execute_size = self.cap / self.data["open"].iloc[idx+1]
                curr_cond = "long"
                t = idx + 1
                self.trade_log["long"].append(t)

            elif all(logics["short"]):
                execute_size = self.cap / self.data["open"].iloc[idx+1]
                curr_cond = "short"
                t = idx + 1
                self.trade_log["short"].append(t)

        elif curr_cond == "long":
            profit = execute_size * (self.data["open"].iloc[idx+1] - self.data["open"].iloc[idx])
            self.profit_map["profit"].append(profit)
            ret = (self.data["open"].iloc[idx+1] - self.data["open"].iloc[idx]) / self.data["open"].iloc[idx]
            curr_ret += ret

            if (
                all(logics["sell"])
                or idx == len(self.data) - 2
                or stop_loss(curr_ret, self.sl_thres)
                or stop_profit(curr_ret, self.sp_thres)
            ):
                pnl_round = execute_size * (self.data["open"].iloc[idx+1] - self.data["open"].iloc[t])
                fee = self.cap * self.cost + (self.cap + pnl_round) * self.cost
                self.profit_map["profitwithfee"].append(profit - fee)
                self.trade_log["sell"].append(idx+1)
                curr_cond = None
                # print(f"Current transaction return: {curr_ret}")

            else:
                self.profit_map["profitwithfee"].append(profit)

        elif curr_cond == "short":
            profit = execute_size * (self.data["open"].iloc[idx] - self.data["open"].iloc[idx+1])
            self.profit_map["profit"].append(profit)
            ret = (self.data["open"].iloc[idx] - self.data["open"].iloc[idx+1]) / self.data["open"].iloc[idx]
            curr_ret += ret

            if (
                all(logics["buytocover"])
                or idx == len(self.data) - 2
                or stop_loss(curr_ret, self.sl_thres)
                or stop_profit(curr_ret, self.sp_thres)
            ):
                pnl_round = execute_size * (self.data["open"].iloc[t] - self.data["open"].iloc[idx+1])
                fee = self.cap * self.cost + (self.cap + pnl_round) * self.cost
                self.profit_map["profitwithfee"].append(profit - fee)
                self.trade_log["buytocover"].append(idx+1)
                curr_cond = None
                # print(f"Current transaction return: {curr_ret}")

            else:
                self.profit_map["profitwithfee"].append(profit)

        return curr_cond, execute_size, t, curr_ret

    # @abstractmethod
    def run(
            self,
            start_idx: int | None = None,
            stop_idx: int | None = None
    ) -> None:
        """
        Run full backtest process
        """
        raise NotImplementedError

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

    def calculate_rolling_profit(
            self,
            status: str,
            start_idx: int,
            stop_idx: int
    ) -> pd.DataFrame:
        """
        Calculate equity with specific data window
        """
        if status == "train":
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

        elif status == "valid":
            profit = pd.DataFrame(
                {
                    "profit": np.cumsum(self.profit_map["profit"]),
                    "profitwithfee": np.cumsum(self.profit_map["profitwithfee"])
                },
                index=self.data.index[start_idx:stop_idx]
            )
            return profit
