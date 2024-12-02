"""
Rolling main script to run the rolling strategy
"""
import os
import pickle
from multiprocessing import Pool

from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd

from .strategy import (
    Strategy,
    RunRollingStrategy
)
from .utils.data_utils import (
    get_self,
    get_taifex,
    transfer_colnames
)
from .get_data import DataAPI
from .base.base_indicator import GlobalDataManager


def get_args() -> Namespace:
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
        "--random", "-r", action="store_true"
    )
    parser.add_argument(
        '--plot', '-p', action='store_true'
    )
    parser.add_argument(
        "--task", "-tk", type=str, default="synthetic_spsl"
    )
    return parser.parse_args()


if __name__ == "__main__":
    p_args = get_args()
    cfg = get_self(p_args.config_path)

    if p_args.task == "synthetic_spsl":
        p_args.random = False

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
    i, val_year, last_year = 1, 0, df.index[-1].year
    full_log, full_trajectory, full_return_log, full_date, full_param = [], [], [], [], []
    best_class = None
    ind = eval(cfg["Strat"])

    while val_year < last_year:
        best_ret = -np.inf

        for _ in range(p_args.trials):
            num_params = ind.num_args()
            args = ind.get_init_args()

            if p_args.random:
                if ind.tag in ["EMA", "SMA"]:
                    args.update(
                        {
                            k: np.random.randint(5, 100) if v is not None else v for k, v in args.items()
                        }
                    )

                elif ind.tag == "RSI":
                    args.update(
                        {
                            k: np.random.randint(5, 100) for k, in args.keys()
                        }
                    )

                elif ind.tag == "BB":
                    args.update(
                        {
                            "period": np.random.randint(5, 100),
                            "std": np.random.uniform(1.2, 3.0),
                            "reverse": np.random.choice([True, False])
                        }
                    )

                vars(ind).update(args)

            else:
                print("use self-defined parameters")

            train_df = ind.build()
            spec_strategy = Strategy(train_df, ind)
            train_runner = RunRollingStrategy(
                train_df, spec_strategy, **cfg["Settings"]
            )  # Run the training process
            last_ids, half_ids = train_runner.day_of_year_idx()

            if i == 1:
                last_idx = last_ids[0]

            ret, val_year_stop = train_runner.run_rolling(0, last_idx)

            if ret > best_ret:
                best_ret = ret
                best_class = spec_strategy.indicator

            GlobalDataManager.reset()

        print(f"best class from {train_df.index[0]} to {train_df.index[last_idx]}: {[str(best_class)]} | Best Cum Ret: {best_ret}")
        val_start_idx = last_idx
        i += 1
        val_df = best_class.build()
        combine_strat_val = Strategy(val_df, best_class)
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
        period_param = {ind.tag: ind.get_init_args()}
        full_log.append(valid_runner.trade_log)
        full_trajectory.extend(valid_runner.trajectory)
        full_return_log.extend(valid_runner.return_log["return"])
        full_date.extend(valid_runner.return_log["date"])
        full_param.extend(period_param)

        GlobalDataManager.reset()

    dic = {
        "date": full_date,
        "return": full_return_log,
        "trade_log": full_log,
        "trajectory": full_trajectory,
        "param": full_param
    }
    print(np.cumsum(dic['return'])[-1])

    if p_args.task == "synthetic_spsl":
        pkl_filename = f"{ind.__str__()}_{cfg['Settings']['sl_thres']}_{cfg['Settings']['sp_thres']}.pkl"
        folder_name = f"synthetic_{ind.__str__()}"

        if folder_name not in os.listdir("trade_log"):
            os.mkdir(f"trade_log/{folder_name}")

        with open(f"trade_log/{folder_name}/{pkl_filename}", "wb") as pkl_file:
            pickle.dump(dic, pkl_file)

    else:
        pkl_filename = f"{ind.tag}_new_{cfg['Settings']['sl_thres']}_{cfg['Settings']['sp_thres']}.pkl"

        if pkl_filename in os.listdir("trade_log"):
            pkl_filename = f"{ind.tag}_new_{np.random.randint(1000)}.pkl"

        with open(f"trade_log/{pkl_filename}", 'wb') as pkl_file:
            pickle.dump(dic, pkl_file)
