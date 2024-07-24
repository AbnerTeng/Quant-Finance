"""
Moving average strategies
"""
from typing import Union, Dict
import logging
import pandas as pd
from ..backtest import BackTest
from ..utils.logic_utils import (
    crossover,
    crossdown,
    stop_loss,
    stop_profit
)


class MAStrat(BackTest):
    """
    Backtest class for moving average strategies
    """
    def __init__(
        self,
        data: Union[pd.DataFrame, Dict],
        strat_type: str,
        initial_cap: float,
        trans_cost: float,
        k1: int,
        k2: int | str,
        sl_thres: float,
        sp_thres: float
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

        if isinstance(k2, str):
            self.k2 = None
        else:
            self.k2 = k2

        self.sl_thres = sl_thres
        self.sp_thres = sp_thres

    def build_ma_indicator(self) -> None:
        """
        Build data for indicators
        """
        if isinstance(self.data, dict):
            self.data = pd.DataFrame(self.data)

        close_price = self.data["close"]

        if self.strat_type == "macd":
            self.data[f"ma_{self.k1}"] = close_price.ewm(span=self.k1).mean()
            self.data[f"ma_{self.k2}"] = close_price.ewm(span=self.k2).mean()
            self.data["MACD"] = self.data[f"ma_{self.k1}"] - self.data[f"ma_{self.k2}"]
            self.data["signal"] = self.data["MACD"].ewm(span=9).mean()

        else:
            self.data[f"ma_{self.k1}"] = close_price.rolling(window=self.k1).mean()

            if self.k2:
                self.data[f"ma_{self.k2}"] = close_price.rolling(window=self.k2).mean()

        self.data.dropna(inplace=True)

    def singlema_strategy(self) -> None:
        """
        Run single moving average strategy
        """
        curr_cond = None

        for i in range(1, len(self.data) - 1):
            if curr_cond is None:
                self.profit_map["profit"].append(0)
                self.profit_map["profitwithfee"].append(0)
                curr_ret = 0

                if crossover(self.data["close"], self.data[f"ma_{self.k1}"], i):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "long"
                    t = i + 1
                    self.trade_log["long"].append(t)

                elif crossdown(self.data["close"], self.data[f"ma_{self.k1}"], i):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "short"
                    t = i + 1
                    self.trade_log["short"].append(t)

            elif curr_cond == "long":
                profit = execute_size * (self.data["open"].iloc[i+1] - self.data["open"].iloc[i])
                ret = (self.data["open"].iloc[i+1] - self.data["open"].iloc[i]) / self.data["open"].iloc[i]
                self.profit_map["profit"].append(profit)
                curr_ret += ret

                if (
                    crossdown(
                        self.data["close"],
                        self.data[f"ma_{self.k1}"],
                        i
                    )
                    or (
                        i == len(self.data) - 2
                    ) or (
                        stop_loss(curr_ret, self.sl_thres)
                    ) or (
                        stop_profit(curr_ret, self.sp_thres)
                    )
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[i+1] - self.data["open"].iloc[t]
                    )
                    fee = self.cap * self.fee + (self.cap + pnl_round) * self.fee
                    self.profit_map["profitwithfee"].append(profit - fee)
                    self.trade_log["sell"].append(i+1)
                    curr_cond = None
                    print(f"Current transaction return: {curr_ret}")

                else:
                    self.profit_map["profitwithfee"].append(profit)

            elif curr_cond == "short":
                profit = execute_size * (self.data["open"].iloc[i] - self.data["open"].iloc[i+1])
                ret = (self.data["open"].iloc[i] - self.data["open"].iloc[i+1]) / self.data["open"].iloc[i]
                self.profit_map["profit"].append(profit)
                curr_ret += ret

                if (
                    crossover(
                        self.data["close"],
                        self.data[f"ma_{self.k1}"],
                        i
                    ) or (
                        i == len(self.data) - 2
                    ) or (
                        stop_loss(curr_ret)
                    ) or (
                        stop_profit(curr_ret)
                    )
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[t] - self.data["open"].iloc[i+1]
                    )
                    fee = self.cap * self.fee + (self.cap + pnl_round) * self.fee
                    self.profit_map["profitwithfee"].append(profit - fee)
                    self.trade_log["buytocover"].append(i+1)
                    curr_cond = None
                    print(f"Current transaction return: {curr_ret}")

                else:
                    self.profit_map["profitwithfee"].append(profit)

    def doublema_strategy(self) -> None:
        """
        Run double moving average strategy
        """
        curr_cond = None

        if self.k2 is None:
            raise ValueError("k2 cannot be None")

        for i in range(len(self.data)):
            if curr_cond is None:
                self.profit_map["profit"].append(0)
                self.profit_map["profitwithfee"].append(0)
                curr_ret = 0

                if crossover(
                    self.data[f"ma_{self.k1}"],
                    self.data[f"ma_{self.k2}"],
                    i
                ):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "long"
                    t = i + 1
                    self.trade_log["long"].append(t)

                elif crossdown(
                    self.data[f"ma_{self.k1}"],
                    self.data[f"ma_{self.k2}"],
                    i
                ):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "short"
                    t = i + 1
                    self.trade_log["short"].append(t)

            elif curr_cond == "long":
                profit = execute_size * (self.data["open"].iloc[i+1] - self.data["open"].iloc[i])
                self.profit_map["profit"].append(profit)
                ret  = (self.data["open"].iloc[i+1] - self.data["open"].iloc[i]) / self.data["open"].iloc[i]
                curr_ret += ret

                if (
                    crossdown(
                        self.data[f"ma_{self.k1}"],
                        self.data[f"ma_{self.k2}"],
                        i
                    )
                    or (
                        i == len(self.data) - 2
                    ) or (
                        stop_loss(curr_ret, self.sl_thres)
                    ) or (
                        stop_profit(curr_ret, self.sp_thres)
                    )
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[i+1] - self.data["open"].iloc[t]
                    )
                    fee = self.cap * self.fee + (self.cap + pnl_round) * self.fee
                    self.profit_map["profitwithfee"].append(profit - fee)
                    self.trade_log["sell"].append(i+1)
                    curr_cond = None
                    print(f"Current transaction return: {curr_ret}")

                else:
                    self.profit_map["profitwithfee"].append(profit)

            elif curr_cond == "short":
                profit = execute_size * (self.data["open"].iloc[i] - self.data["open"].iloc[i+1])
                self.profit_map["profit"].append(profit)
                ret = (self.data["open"].iloc[i] - self.data["open"].iloc[i+1]) / self.data["open"].iloc[i]
                curr_ret += ret

                if (
                    crossover(
                        self.data[f"ma_{self.k1}"],
                        self.data[f"ma_{self.k2}"],
                        i
                    )
                    or (
                        i == len(self.data) - 2
                    ) or (
                        stop_loss(curr_ret, self.sl_thres)
                    ) or (
                        stop_profit(curr_ret, self.sp_thres)
                    )
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[t] - self.data["open"].iloc[i+1]
                    )
                    fee = self.cap * self.fee + (self.cap + pnl_round) * self.fee
                    self.profit_map["profitwithfee"].append(profit - fee)
                    self.trade_log["buytocover"].append(i+1)
                    curr_cond = None
                    print(f"Current transaction return: {curr_ret}")

                else:
                    self.profit_map["profitwithfee"].append(profit)

    def macd_strategy(self) -> None:
        """
        Run MACD strategy
        """
        curr_cond = None

        if self.k2 is None:
            raise ValueError("k2 cannot be None")

        for i in range(len(self.data)):
            if curr_cond is None:
                self.profit_map["profit"].append(0)
                self.profit_map["profitwithfee"].append(0)

                if crossover(self.data["MACD"], self.data["signal"], i):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "long"
                    t = i + 1
                    self.trade_log["long"].append(t)

                elif crossdown(self.data["MACD"], self.data["signal"], i):
                    execute_size = self.cap / self.data["open"].iloc[i+1]
                    curr_cond = "short"
                    t = i + 1
                    self.trade_log["short"].append(t)

            elif curr_cond == "long":
                profit = execute_size * (self.data["open"].iloc[i+1] - self.data["open"].iloc[i])
                self.profit_map["profit"].append(profit)

                if (
                    crossdown(self.data["MACD"], self.data["signal"], i)
                    or i == len(self.data) - 2
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[i+1] - self.data["open"].iloc[t]
                    )
                    fee = self.cap * self.fee + (self.cap + pnl_round) * self.fee
                    self.profit_map["profitwithfee"].append(profit - fee)
                    self.trade_log["sell"].append(i+1)
                    curr_cond = None

                else:
                    self.profit_map["profitwithfee"].append(profit)

            elif curr_cond == "short":
                profit = execute_size * (self.data["open"].iloc[i] - self.data["open"].iloc[i+1])
                self.profit_map["profit"].append(profit)

                if (
                    crossover(self.data["MACD"], self.data["signal"], i)
                    or i == len(self.data) - 2
                ):
                    pnl_round = execute_size * (
                        self.data["open"].iloc[t] - self.data["open"].iloc[i+1]
                    )
                    fee = self.cap * self.fee + (self.cap + pnl_round) * self.fee
                    self.profit_map["profitwithfee"].append(profit - fee)
                    self.trade_log["buytocover"].append(i+1)
                    curr_cond = None

                else:
                    self.profit_map["profitwithfee"].append(profit)

    def run_strategy(self) -> None:
        """
        runner
        """
        self.build_ma_indicator()

        if self.strat_type == "singlema":
            self.singlema_strategy()

        elif self.strat_type == "doublema":
            self.doublema_strategy()

        elif self.strat_type == "macd":
            self.macd_strategy()

        else:
            raise ValueError("Strategy not supported")

        equity_df = self.calculate_equity()
        return equity_df
