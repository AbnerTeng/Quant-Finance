"""
General backtest class
"""

from typing import Any, Tuple

import pandas as pd
import numpy as np

from ..utils.logic_utils import stop_loss, stop_profit


class BackTest:
    """
    General backtest class
    """

    def __init__(
        self,
        data: Any,
        initial_cap: float,
        part: int,
        trans_cost: float,
        sl_thres: float | None = None,
        sp_thres: float | None = None,
    ) -> None:
        """
        strategy: backtest strategy

        data: data to backtest

        initial_cap: initial capital to backtest
        """
        self.data = data
        self.cap = initial_cap
        self.part = part
        self.unit_cap = self.cap / self.part
        self.cost = trans_cost
        self.sl_thres = sl_thres if sl_thres is not None else -np.inf
        self.sp_thres = sp_thres if sp_thres is not None else np.inf
        self.trade_log = {
            "long": [],
            "sell": [],
            "buytocover": [],
            "short": [],
        }
        self.trajectory = []
        self.return_log = {"date": [], "return": []}

    def daily_run(
        self,
        idx: int,
        curr_cond: str | None,
        logics: pd.DataFrame,
        t: int,
        curr_ret: float,
    ) -> Tuple[str | None, float, int, float]:
        try:
            self.return_log["date"].append(self.data["Date"].iloc[idx])
        except KeyError:
            self.return_log["date"].append(self.data.index[idx])

        if curr_cond is None:
            self.return_log["return"].append(0)
            curr_ret = 0

            if all(logics["long"]):
                curr_cond = "long"
                t = idx + 1
                self.trade_log["long"].append(t)
                self.return_log["return"][-1] = -self.cost

            elif all(logics["short"]):
                curr_cond = "short"
                t = idx + 1
                self.trade_log["short"].append(t)
                self.return_log["return"][-1] = -self.cost

        elif curr_cond == "long":
            ret = (
                self.data["open"].iloc[idx + 1] - self.data["open"].iloc[idx]
            ) / self.data["open"].iloc[idx]
            curr_ret += ret

            if (
                all(logics["sell"])
                or idx == len(self.data) - 2
                or stop_loss(curr_ret, self.sl_thres)
                or stop_profit(curr_ret, self.sp_thres)
            ):
                self.return_log["return"].append(ret - self.cost)
                self.trade_log["sell"].append(idx + 1)
                curr_cond = None
                self.trajectory.append((t, idx + 1, ret - self.cost))

            else:
                self.return_log["return"].append(0)

        elif curr_cond == "short":
            ret = (
                self.data["open"].iloc[idx] - self.data["open"].iloc[idx + 1]
            ) / self.data["open"].iloc[idx]
            curr_ret += ret

            if (
                all(logics["buytocover"])
                or idx == len(self.data) - 2
                or stop_loss(curr_ret, self.sl_thres)
                or stop_profit(curr_ret, self.sp_thres)
            ):
                self.return_log["return"].append(ret - self.cost)
                self.trade_log["buytocover"].append(idx + 1)
                curr_cond = None
                self.trajectory.append((t, idx + 1, ret - self.cost))

            else:
                self.return_log["return"].append(0)

        return curr_cond, t, curr_ret

    def run(self, start_idx: int | None = None, stop_idx: int | None = None) -> None:
        """
        Run full backtest process
        """
        raise NotImplementedError

    def calculate_return(self) -> float:
        """
        calculate the return from `self.return_log`
        """
        return np.cumsum(self.return_log["return"])[-1]
