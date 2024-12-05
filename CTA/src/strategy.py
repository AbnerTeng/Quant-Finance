from typing import Dict, List, SupportsIndex, Tuple, Any, Union, Optional, overload
from typing_extensions import override

import numpy as np
import pandas as pd
from sqlalchemy import over

from .base.base_strategy import BackTest
from .utils.logic_utils import (
    crossover,
    crossdown,
)


class Strategy:
    """
    Combined strategy class

    Args:
        data (pd.DataFrame): pre=built training  data with indicators
        indicator_class (BaseIndicator): indicator name
    """

    def __init__(self, data: pd.DataFrame, indicators_class: Any) -> None:
        self.data = data
        self.indicator = indicators_class
        self.trade_logics = {"long": [], "short": [], "sell": [], "buytocover": []}

    def get_strategy(self, idx: int) -> Dict[str, List[bool]]:
        """
        Get the trade logic from a specifc day

        Args:
            idx (int): day index

        Returns:
            self.trade_logics (pd.DataFrame): trade logics of indicators from a specific day
        """
        if self.indicator.tag in ["SMA", "EMA"]:
            sma = sorted([col for col in self.data.columns if "SMA" in col])
            ema = sorted([col for col in self.data.columns if "EMA" in col])

            if len(sma) == 1 or len(ema) == 1:
                self.trade_logics["long"].append(
                    crossover(
                        self.data["close"],
                        self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                        idx,
                    )
                )
                self.trade_logics["short"].append(
                    crossdown(
                        self.data["close"],
                        self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                        idx,
                    )
                )
                self.trade_logics["sell"] = self.trade_logics["short"]
                self.trade_logics["buytocover"] = self.trade_logics["long"]

            else:
                self.trade_logics["long"].append(
                    crossover(
                        self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                        self.data[sma[1]] if len(ema) == 0 else self.data[ema[1]],
                        idx,
                    )
                )
                self.trade_logics["short"].append(
                    crossdown(
                        self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                        self.data[sma[1]] if len(ema) == 0 else self.data[ema[1]],
                        idx,
                    )
                )
                self.trade_logics["sell"] = self.trade_logics["short"]
                self.trade_logics["buytocover"] = self.trade_logics["long"]

        if self.indicator.tag == "RSI":
            rsi = [col for col in self.data.columns if "RSI" in col]

            self.trade_logics["long"].append(self.data[rsi[0]].iloc[idx] < 30)
            self.trade_logics["short"].append(self.data[rsi[0]].iloc[idx] > 70)
            self.trade_logics["sell"] = self.trade_logics["short"]
            self.trade_logics["buytocover"] = self.trade_logics["long"]

        if self.indicator.tag == "MACD":
            self.trade_logics["long"].append(
                crossover(self.data["MACD"], self.data["Signal"], idx)
            )
            self.trade_logics["short"].append(
                crossdown(self.data["MACD"], self.data["Signal"], idx)
            )
            self.trade_logics["sell"] = self.trade_logics["short"]
            self.trade_logics["buytocover"] = self.trade_logics["long"]

        if self.indicator.tag == "BB":
            if self.indicator.is_reverse is False:
                self.trade_logics["long"].append(
                    crossover(self.data["close"], self.data["upper"], idx)
                )
                self.trade_logics["short"].append(
                    crossdown(self.data["close"], self.data["lower"], idx)
                )
                self.trade_logics["sell"].append(
                    crossdown(self.data["close"], self.data["ma"], idx)
                )
                self.trade_logics["buytocover"].append(
                    crossover(self.data["close"], self.data["ma"], idx)
                )

            else:
                self.trade_logics["long"].append(
                    crossover(self.data["close"], self.data["lower"], idx)
                )
                self.trade_logics["short"].append(
                    crossdown(self.data["close"], self.data["upper"], idx)
                )
                self.trade_logics["sell"].append(
                    crossover(self.data["close"], self.data["ma"], idx)
                )
                self.trade_logics["buytocover"].append(
                    crossdown(self.data["close"], self.data["ma"], idx)
                )

        return self.trade_logics


class RunStrategy(BackTest):
    """
    Run strategy class
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strat: Strategy,
        initial_cap: float = 10000,
        trans_cost: float = 0.001,
        sl_thres: Optional[float] = None,
        sp_thres: Optional[float] = None,
    ) -> None:
        super().__init__(data, initial_cap, trans_cost, sl_thres, sp_thres)
        self.data = data
        self.strat = strat

    def run(self, start_idx: SupportsIndex, stop_idx: SupportsIndex) -> float:
        """
        Run rolling strategy
        """
        curr_cond = None
        t = 0
        curr_ret = 0

        if start_idx is None:
            start_idx = 0

        if stop_idx is None:
            stop_idx = len(self.data) - 1

        for idx in range(start_idx, stop_idx, 1):
            logics = self.strat.get_strategy(idx)
            curr_cond, t, curr_ret = self.daily_run(idx, curr_cond, logics, t, curr_ret)

            for key in logics.keys():
                logics[key] = []

        ret = self.calculate_return()
        return ret


class RunRollingStrategy(BackTest):
    """
    Rolling strategy class

    Args:
        data (pd.DataFrame): training data
        strat (Strategy): strategy classs
        initial_cap (float): initial capital
        trans_cost (float): transaction cost of each trade
        sl_thres (float | None): stop loss threshold
        sp_thres (float | None): stop profit threshold

    """

    def __init__(
        self,
        data: pd.DataFrame,
        strat: Strategy,
        initial_cap: float = 10000,
        trans_cost: float = 0.001,
        sl_thres: float | None = -np.inf,
        sp_thres: float | None = np.inf,
    ) -> None:
        super().__init__(data, initial_cap, trans_cost, sl_thres, sp_thres)
        self.data = data.fillna(0) if sum(data.isna().sum()) > 0 else data
        self.strat = strat

    def day_of_year_idx(self) -> Tuple[List[int], List[int]]:
        """
        Get last day and the half day index of all years

        Returns:
            last_day: List[int] -> All last day index of years
            half_day: List[int] -> All half day index of years
        """
        last_day, half_day = [], []
        self.data["year"] = self.data.index.year
        self.data.reset_index(inplace=True)

        for year in self.data.year.unique():
            last_day.append(self.data[self.data["year"] == year].index[-1])
            half_day.append(
                self.data[self.data["year"] == year].index[
                    len(self.data[self.data["year"] == year]) // 2
                ]
            )
            half_day.append(self.data[self.data["year"] == year].index[-1])

        return last_day, half_day

    def run(
        self, start_idx: SupportsIndex, stop_idx: SupportsIndex
    ) -> Tuple[float, int]:
        """
        Run rolling strategy

        Args:
            start_idx: int | None ->
            stop_idx: int | None ->

        Returns:
            ret: float ->
            val_year_stop: int ->
        """
        curr_cond, curr_ret, t = None, 0, 0

        for idx in range(start_idx, stop_idx, 1):
            logics = self.strat.get_strategy(idx)
            curr_cond, t, curr_ret = self.daily_run(idx, curr_cond, logics, t, curr_ret)

            for key in logics.keys():
                logics[key] = []

        ret = self.calculate_return()

        if start_idx == 0:
            val_year_stop = self.data.Date.iloc[stop_idx].year
        else:
            val_year_stop = self.data.index[stop_idx].year

        return ret, val_year_stop
