from typing import List
import numpy as np
import pandas as pd
from .backtest import BackTest
from .utils.logic_utils import (
    crossover,
    crossdown,
    stop_loss,
    stop_profit
)
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

            if indicator.__class__.__name__ == "BB":
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

    def run(self) -> pd.DataFrame:
        curr_cond = None

        for idx in range(len(self.data) - 1):
            logics = self.combine_strat.get_strategy(idx)

            if curr_cond is None:
                self.profit_map["profit"].append(0)
                self.profit_map["profitwithfee"].append(0)
                curr_ret = 0

                if all(logics["long"]):
                    print(idx)
                    print(len(self.data))
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
                    print(f"Current transaction return: {curr_ret}")

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
                    print(f"Current transaction return: {curr_ret}")

                else:
                    self.profit_map["profitwithfee"].append(profit)

            for key in logics.keys():
                logics[key] = []

        equity_df = self.calculate_equity()
        return equity_df
