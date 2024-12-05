"""
Rolling main script to run the rolling strategy
"""

import os
import pickle
from typing import Union

from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd

from .strategy import Strategy, RunRollingStrategy
from .utils.general_utils import load_config, generate_random_param, get_class
from .utils.data_utils import transfer_colnames
from .get_data import DataAPI
from .base.base_indicator import GlobalDataManager
from .indicators import BB, EMA, RSI, SMA


def get_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--data_source", type=str, default="self")
    parser.add_argument("--df_path", "-dp", type=str, default="data/strat_df.csv")
    parser.add_argument("--plot", "-p", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = load_config("config/gen_strat.yaml")
    RANDOM_PARAM = False if cfg.task == "syn_spsl" else True

    if os.path.exists(args.df_path):
        strat_df = pd.read_csv(args.df_path, index_col=0)
    else:
        strat_df = pd.DataFrame()

    fetcher = DataAPI(args.data_source)
    api_map = {
        "yahoo": cfg.DataAPIYahoo,
        "self": cfg.DataLocal,
        "taifex": cfg.DataAPITAIFEX,
    }
    data = fetcher.fetch(api_map[args.data_source])
    df = transfer_colnames(data)
    GlobalDataManager.set_data(df)
    i, val_year, last_year = 1, 0, df.index[-1].year
    full_log, full_trajectory, full_return_log, full_date, full_param = (
        [],
        [],
        [],
        [],
        [],
    )
    ModelClass = get_class(cfg.Class.strat)
    ind = ModelClass(cfg.Class.params)
    last_idx, half_ids, BEST_CLASS = None, [], ind
    ind: Union[BB, EMA, RSI, SMA]

    while val_year < last_year:
        best_ret = -np.inf

        if not RANDOM_PARAM:
            cfg.trials = 1

        for _ in range(cfg.trials):
            num_params = ind.num_args()
            bound_args = ind.get_init_args()

            if RANDOM_PARAM:
                ind = generate_random_param(ind, bound_args)

            train_df = ind.build()
            spec_strategy = Strategy(train_df, ind)
            train_runner = RunRollingStrategy(train_df, spec_strategy, **cfg.Settings)
            last_ids, half_ids = train_runner.day_of_year_idx()

            if i == 1:
                last_idx = last_ids[0]

            ret, val_year_stop = train_runner.run(0, last_idx)

            if ret > best_ret:
                best_ret = ret
                BEST_CLASS = spec_strategy.indicator

            GlobalDataManager.reset()

        print(f"training period {i - 1} | Cum Ret: {best_ret}")
        val_start_idx = last_idx
        i += 1

        BEST_CLASS: Union[BB, EMA, RSI, SMA]
        val_df = BEST_CLASS.build()
        combine_strat_val = Strategy(val_df, BEST_CLASS)
        valid_runner = RunRollingStrategy(val_df, combine_strat_val, **cfg.Settings)

        if i == len(half_ids) - 2:
            ret, val_year = valid_runner.run(val_start_idx, val_df.shape[0] - 1)
        else:
            ret, val_year = valid_runner.run(val_start_idx, half_ids[i])

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
        "param": full_param,
    }
    print(np.cumsum(dic["return"])[-1])

    if cfg.task == "syn_spsl":
        pkl_filename = f"{str(ind)}_{cfg.Settings.sl_thres}_{cfg.Settings.sp_thres}.pkl"
        folder_name = f"synthetic_{str(ind)}"

        if folder_name not in os.listdir("trade_log"):
            os.mkdir(f"trade_log/{folder_name}")

        with open(f"trade_log/{folder_name}/{pkl_filename}", "wb") as pkl_file:
            pickle.dump(dic, pkl_file)

    else:
        pkl_filename = (
            f"{ind.tag}_new_{cfg.Settings.sl_thres}_{cfg.Setting.sp_thres}.pkl"
        )

        if pkl_filename in os.listdir("trade_log"):
            pkl_filename = f"{ind.tag}_new_{np.random.randint(1000)}.pkl"

        with open(f"trade_log/{pkl_filename}", "wb") as pkl_file:
            pickle.dump(dic, pkl_file)
