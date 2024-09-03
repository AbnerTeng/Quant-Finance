"""
Rolling main script to run the rolling strategy
"""
import os
from multiprocessing import Pool
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from .indicators.ma import SMA, EMA, MACD
from .indicators.rsi import RSI
from .indicators.bb import BB
from .strategy import (
    CombineStrategy,
    RunRollingStrategy
)
from .utils.general_utils import (
    parse_column,
    num_args
)
from .utils.data_utils import (
    get_self,
    get_taifex,
    transfer_colnames
)
from .get_data import DataAPI
# from .utils.plot import profit_graph, trade_position
from .base.base_indicator import BaseIndicator, GlobalDataManager


class Combination:
    """
    Combination class to combine multiple strategies
    """
    def __init__(self, *args: BaseIndicator) -> None:
        self.indicators = list(args)
        self.used_columns = [
            name.name for name in args
        ]
        self.cleaned_columns = [
            cleaned_col
            for col in self.used_columns
            for cleaned_col in parse_column(col)
        ]

    def run_combination(self) -> pd.DataFrame:
        full_df = pd.DataFrame()

        for strat in self.indicators:
            strat_data = strat.build()
            full_df = pd.concat([full_df, strat_data], axis=1)

        full_df = full_df.loc[:, ~full_df.columns.duplicated()]

        return full_df


def parse_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", "-cp", type=str, default="config/combine_test.yaml"
    )
    parser.add_argument(
        "--config_type", "-ctp", type=str, default="yahoo"
    )
    parser.add_argument(
        "--df_path", "-dp", type=str, default="data/strat_df.csv"
    )
    parser.add_argument(
        "--trials", '-t', type=int, default=50
    )
    parser.add_argument(
        '--plot', '-p', action='store_true'
    )
    return parser.parse_args()


if __name__ == "__main__":
    p_args = parse_args()
    cfg = get_self(p_args.config_path)

    if os.path.exists(p_args.df_path):
        strat_df = pd.read_csv(p_args.df_path, index_col=0)
    else:
        strat_df = pd.DataFrame()

    if p_args.config_type == "yahoo":
        fetcher = DataAPI(**cfg["DataAPIYahoo"])
        data = fetcher.fetch()

    elif p_args.config_type == "taifex":
        fetcher = DataAPI(**cfg["DataAPITAIFEX"])
        date_list = pd.date_range(
            cfg["DataAPITAIFEX"]["start"],
            cfg["DataAPITAIFEX"]["end"], freq='D'
        ).strftime("%Y/%m/%d").tolist()

        with Pool(10) as pool:
            daily_data = pool.map(get_taifex, date_list)

        data = pd.concat(daily_data, axis=0)
        data.insert(0, "Date", data.pop("Date"))

    else:
        raise ValueError("Invalid config type")

    df = transfer_colnames(data)
    GlobalDataManager.set_data(df)
    comb = Combination(*[eval(item) for item in cfg['Strat']])
    full_profit_df = pd.DataFrame()

    i = 0

    while i < 5:
        best_equity_val = -np.inf

        for _ in tqdm(range(p_args.trials)):
            for idx, ind in enumerate(comb.indicators):
                num_params = num_args(ind)

                if num_params == 1:
                    random_window = np.random.randint(5, 100)
                    indicator_class = ind.__class__
                    comb.indicators[idx] = indicator_class(
                        random_window
                    )

                elif num_params == 2:
                    random_short = np.random.randint(5, 100)
                    random_long = np.random.randint(20, 250)

                    if random_short > random_long:
                        random_short, random_long = random_long, random_short

                    indicator_class = ind.__class__
                    comb.indicators[idx] = indicator_class(
                        random_short, random_long
                    )

                elif num_params == 3:
                    random_short = np.random.randint(5, 100)
                    random_long = np.random.randint(20, 250)
                    random_signal = np.random.randint(5, 50)

                    if random_short > random_long:
                        random_short, random_long = random_long, random_short

                    indicator_class = ind.__class__
                    comb.indicators[idx] = indicator_class(
                        random_short, random_long, random_signal
                    )

            new_comb = Combination(*comb.indicators)
            train_df = new_comb.run_combination().dropna()
            used_cols = [
                col for col in new_comb.cleaned_columns if col in train_df.columns
            ]
            train_df = train_df[['open', 'high', 'low', 'close', 'volume'] + used_cols]
            combine_strat_train = CombineStrategy(
                train_df,
                comb.indicators,
                comb.cleaned_columns
            )
            runner = RunRollingStrategy(
                train_df, combine_strat_train, **cfg["Settings"]
            )  # Run the training process
            last_ids, half_ids = runner.day_of_year_idx()

            if i == 0:
                last_idx = last_ids[0]

            eqdf = runner.run_train(0, last_idx)
            now_equity = eqdf['equity_val'].iloc[-1]
            now_class = comb.indicators

            if now_equity > best_equity_val:
                best_equity_val = eqdf['equity_val'].iloc[-1]
                best_class = now_class[0]

        print(f"best class at period {i}: {str(best_class)} | Equity val: {best_equity_val}")

        val_start_idx = last_idx
        i += 1
        # valid part
        val_comb = Combination(*[best_class])
        val_df = comb.run_combination().dropna()
        combine_strat_val = CombineStrategy(
            val_df,
            [best_class],
            comb.cleaned_columns
        )
        runner = RunRollingStrategy(
            val_df, combine_strat_val, **cfg["Settings"]
        )  # Run the validation process
        prdf = runner.run_valid(val_start_idx, half_ids[i])
        print(runner.trade_log)
        last_idx = half_ids[i]
        full_profit_df = pd.concat([full_profit_df, prdf], axis=0)
