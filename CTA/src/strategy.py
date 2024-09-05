from typing import List, Tuple
import numpy as np
import pandas as pd
from .base.base_strategy import BackTest
from .utils.logic_utils import (
    crossover,
    crossdown,
)
from .indicators.bb import BB
from .base.base_indicator import BaseIndicator


class CombineStrategy:
    """
    Combined strategy class
    """
    def __init__(
            self,
            data: pd.DataFrame,
            indicators_class: BaseIndicator,
            indicators_str: List[str],
    ) -> None:
        self.data = data
        self.indicators_class = indicators_class
        self.indicators_str = indicators_str
        self.trade_logics = {
            "long": [],
            "short": [],
            "sell": [],
            "buytocover": []
        }

    def get_strategy(self, idx: int) -> pd.DataFrame:
        for indicator in self.indicators_class:
            if indicator.__class__.__name__ in ["SMA", "EMA"]:
                sma = [col for col in self.data.columns if "SMA" in col]
                ema = [col for col in self.data.columns if "EMA" in col]

                if len(sma) == 1 or len(ema) == 1:
                    self.trade_logics['long'].append(
                        crossover(
                            self.data["close"],
                            self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                            idx
                        )
                    )
                    self.trade_logics['short'].append(
                        crossdown(
                            self.data["close"],
                            self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                            idx
                        )
                    )
                    self.trade_logics['sell'] = self.trade_logics['short']
                    self.trade_logics['buytocover'] = self.trade_logics['long']

                else:
                    self.trade_logics['long'].append(
                        crossover(
                            self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                            self.data[sma[1]] if len(ema) == 0 else self.data[ema[1]],
                            idx
                        )
                    )
                    self.trade_logics['short'].append(
                        crossdown(
                            self.data[sma[0]] if len(ema) == 0 else self.data[ema[0]],
                            self.data[sma[1]] if len(ema) == 0 else self.data[ema[1]],
                            idx
                        )
                    )
                    self.trade_logics['sell'] = self.trade_logics['short']
                    self.trade_logics['buytocover'] = self.trade_logics['long']

            if indicator.__class__.__name__ == "RSI":
                rsi = [col for col in self.data.columns if "RSI" in col]

                self.trade_logics['long'].append(
                    self.data[rsi[0]].iloc[idx] < 30
                )
                self.trade_logics['short'].append(
                    self.data[rsi[0]].iloc[idx] > 70
                )
                self.trade_logics['sell'] = self.trade_logics['short']
                self.trade_logics['buytocover'] = self.trade_logics['long']

            if indicator.__class__.__name__ == "MACD":
                self.trade_logics['long'].append(
                    crossover(
                        self.data["MACD"],
                        self.data["Signal"],
                        idx
                    )
                )
                self.trade_logics['short'].append(
                    crossdown(
                        self.data["MACD"],
                        self.data["Signal"],
                        idx
                    )
                )
                self.trade_logics['sell'] = self.trade_logics['short']
                self.trade_logics['buytocover'] = self.trade_logics['long']

            if indicator.__class__ == BB:
                if "BB_0" in self.indicators_str:
                    self.trade_logics['long'].append(
                        crossover(self.data['close'], self.data['upper'], idx)
                    )
                    self.trade_logics['short'].append(
                        crossdown(self.data['close'], self.data['lower'], idx)
                    )
                    self.trade_logics['sell'].append(
                        crossdown(self.data['close'], self.data['ma'], idx)
                    )
                    self.trade_logics['buytocover'].append(
                        crossover(self.data['close'], self.data['ma'], idx)
                    )

                else:
                    self.trade_logics['long'].append(
                        crossover(self.data['close'], self.data['lower'], idx)
                    )
                    self.trade_logics['short'].append(
                        crossdown(self.data['close'], self.data['upper'], idx)
                    )
                    self.trade_logics['sell'].append(
                        crossover(self.data['close'], self.data['ma'], idx)
                    )
                    self.trade_logics['buytocover'].append(
                        crossdown(self.data['close'], self.data['ma'], idx)
                    )

        return self.trade_logics


class RunStrategy(BackTest):
    """
    Run combined strategy class
    """
    def __init__(
        self,
        data: pd.DataFrame,
        combine_strat: CombineStrategy,
        initial_cap: float = 10000,
        trans_cost: float = 0.001,
        sl_thres: float | None = None,
        sp_thres: float | None = None,
    ) -> None:
        super().__init__(data, initial_cap, trans_cost, sl_thres, sp_thres)
        self.data = data
        self.combine_strat = combine_strat

    def run(
            self,
            start_idx: int | None = None,
            stop_idx: int | None = None
    ) -> pd.DataFrame:
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
            logics = self.combine_strat.get_strategy(idx)
            curr_cond, t, curr_ret = self.daily_run(
                idx, curr_cond, logics, t, curr_ret
            )

            for key in logics.keys():
                logics[key] = []

        ret = self.calculate_return()
        return ret


class RunRollingStrategy(BackTest):
    """
    Rolling strategy class
    """
    def __init__(
            self,
            data: pd.DataFrame,
            combine_strat: CombineStrategy,
            initial_cap: float = 10000,
            trans_cost: float = 0.001,
            sl_thres: float | None = -np.inf,
            sp_thres: float | None = np.inf,
    ) -> None:
        super().__init__(data, initial_cap, trans_cost)
        self.data = data
        self.combine_strat = combine_strat
        self.cap = initial_cap
        self.cost = trans_cost
        self.sl_thres = sl_thres if sl_thres is not None else -np.inf
        self.sp_thres = sp_thres if sp_thres is not None else np.inf

    def day_of_year_idx(self) -> Tuple[List[int], List[int]]:
        last_day, half_day = [], []
        self.data["year"] = self.data.index.year
        self.data.reset_index(inplace=True)

        for year in self.data.year.unique():
            last_day.append(self.data[self.data['year'] == year].index[-1])
            half_day.append(
                self.data[self.data['year'] == year].index[
                    len(self.data[self.data['year'] == year])//2
                ]
            )
            half_day.append(self.data[self.data['year'] == year].index[-1])

        return last_day, half_day

    def run_rolling(
            self,
            start_idx: int | None = None,
            stop_idx: int | None = None
    ) -> Tuple[float, int]:
        """
        Run rolling strategy
        """
        curr_cond = None
        t = 0
        curr_ret = 0
        for idx in range(start_idx, stop_idx, 1):
            logics = self.combine_strat.get_strategy(idx)
            curr_cond, t, curr_ret = self.daily_run(
                idx, curr_cond, logics, t, curr_ret
            )

            for key in logics.keys():
                logics[key] = []

        ret = self.calculate_return()

        if start_idx == 0:
            val_year_stop = self.data.Date.iloc[stop_idx].year
        else:
            val_year_stop = self.data.index[stop_idx].year

        return ret, val_year_stop
