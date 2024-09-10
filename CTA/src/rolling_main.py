"""
Rolling main script to run the rolling strategy
"""
import os
import pickle
from multiprocessing import Pool
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from .indicators.ma import SMA, EMA, MACD
from .indicators.rsi import RSI
from .indicators.bb import BB
from .strategy import (
    CombineStrategy,
    RunRollingStrategy
)
from .utils.general_utils import parse_column
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

    elif p_args.config_type == "self":
        data = get_self("tick_data/sp500.csv")
        data.index = pd.to_datetime(data.index)

    else:
        raise ValueError("Invalid config type")

    df = transfer_colnames(data)
    GlobalDataManager.set_data(df)
    comb = Combination(*[eval(item) for item in cfg['Strat']])
    tags = [eval(item).tag for item in cfg['Strat']]

    i, val_year, last_year = 1, 0, df.index[-1].year

    full_trajectory, full_return_log, full_date, full_param = [], [], [], []

    while val_year < last_year:
        best_ret = -np.inf

        for _ in range(p_args.trials):
            for idx, ind in enumerate(comb.indicators):
                num_params = ind.num_args()

                if ind.__class__ in [EMA, SMA]:
                    random_short = np.random.randint(5, 100)
                    indicator_class = ind.__class__
                    if num_params == 1:
                        comb.indicators[idx] = indicator_class(
                            random_short
                        )
                    elif num_params == 2:
                        random_long = int(
                            random_short * np.random.uniform(1.2, 5.0)
                        )
                        comb.indicators[idx] = indicator_class(
                            random_short, random_long
                        )

                elif ind.__class__ == RSI:
                    random_window = np.random.randint(5, 100)
                    indicator_class = ind.__class__
                    comb.indicators[idx] = indicator_class(
                        random_window
                    )

                elif ind.__class__ == BB:
                    random_window = np.random.randint(5, 150)
                    random_mult = np.random.uniform(1.5, 3.0)
                    random_tf = np.random.choice([True, False])
                    indicator_class = ind.__class__
                    comb.indicators[idx] = indicator_class(
                        random_window, random_mult, random_tf
                    )

                elif ind.__class__ == MACD:
                    random_short = np.random.randint(5, 100)
                    random_long = random_short + np.random.randint(5, 20)
                    random_signal = np.random.randint(5, 50)

                    if random_short > random_long:
                        random_short, random_long = random_long, random_short

                    indicator_class = ind.__class__
                    comb.indicators[idx] = indicator_class(
                        random_short, random_long, random_signal
                    )

            new_comb = Combination(*comb.indicators)
            train_df = new_comb.run_combination().fillna(0)
            used_cols = [
                col for col in new_comb.cleaned_columns if col in train_df.columns
            ]
            new_classes = [ind.__class__ for ind in new_comb.indicators]

            if BB in new_classes:
                used_cols += ["upper", "lower", "ma"]

            if MACD in new_classes:
                used_cols += ["MACD", "Signal"]

            train_df = train_df[['open', 'high', 'low', 'close', 'volume'] + used_cols]
            combine_strat_train = CombineStrategy(
                train_df,
                new_comb.indicators,
                new_comb.cleaned_columns
            )
            train_runner = RunRollingStrategy(
                train_df, combine_strat_train, **cfg["Settings"]
            )  # Run the training process
            last_ids, half_ids = train_runner.day_of_year_idx()

            if i == 1:
                last_idx = last_ids[0]

            ret, val_year_stop = train_runner.run_rolling(0, last_idx)

            if ret > best_ret:
                best_ret = ret
                best_class = new_comb.indicators
                best_used_cols = used_cols
                best_new_classes = [
                    ind.__class__ for ind in best_class
                ]

        print(f"best class from {train_df.loc[0, 'Date']} to {train_df.loc[last_idx, 'Date']}: {[str(bc) for bc in best_class]} | Best Cum Ret: {best_ret}")
        val_start_idx = last_idx
        i += 1
        # valid part
        val_comb = Combination(*best_class)
        val_df = val_comb.run_combination().fillna(0)
        val_df = val_df[['open', 'high', 'low', 'close', 'volume'] + best_used_cols]

        combine_strat_val = CombineStrategy(
            val_df,
            best_class,
            val_comb.cleaned_columns
        )
        valid_runner = RunRollingStrategy(
            val_df, combine_strat_val, **cfg["Settings"]
        )  # Run the validation process
        print(f"Validation period {i - 1}: {val_df.index[val_start_idx]}, {val_df.index[half_ids[i]]}")

        if i == len(half_ids) - 2:
            ret, val_year = valid_runner.run_rolling(val_start_idx, val_df.shape[0] - 1)
        else:
            ret, val_year = valid_runner.run_rolling(val_start_idx, half_ids[i])

        last_idx = half_ids[i]
        print(f"Validation period {i - 1} | Cum Ret: {ret}")
        period_param = [
            {
            ind.tag: ind.get_init_args() for ind in val_comb.indicators
            }
        ]
        full_trajectory.extend(valid_runner.trajectory)
        full_return_log.extend(valid_runner.return_log["return"])
        full_date.extend(valid_runner.return_log["date"])
        full_param.extend(period_param)

    dic = {
        "date": full_date,
        "return": full_return_log,
        "trajectory": full_trajectory,
        "param": full_param
    }
    print(np.cumsum(dic['return'])[-1])

    pkl_filename = f"{'_'.join(tags)}.pkl"

    if pkl_filename in os.listdir("trade_log"):
        pkl_filename = f"{'_'.join(tags)}_{np.random.randint(1000)}.pkl"

    with open(f"trade_log/{pkl_filename}", 'wb') as pkl_file:
        pickle.dump(dic, pkl_file)
