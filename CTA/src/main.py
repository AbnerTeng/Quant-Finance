import os
from multiprocessing import Pool
from argparse import ArgumentParser
import pandas as pd
from .indicators.ma import SMA, EMA, MACD
from .indicators.rsi import RSI
from .indicators.bb import BB
from .strategy import (
    CombineStrategy,
    RunStrategy,
)
from .utils.general_utils import parse_column
from .utils.data_utils import (
    get_self,
    get_taifex,
    transfer_colnames
)
from .get_data import DataAPI
from .utils.plot import profit_graph
from .base.base_indicator import BaseIndicator, GlobalDataManager


class Combination:
    """
    Combination class to combine multiple strategies
    """
    def __init__(self, *args: BaseIndicator) -> None:
        self.indicators = args
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
        "--df_path", "-dp", type=str, default="tick_data/strat_df.csv"
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
    big_df = comb.run_combination().dropna()
    combine_strat = CombineStrategy(
        big_df,
        comb.indicators,
        comb.cleaned_columns
    )
    runner = RunStrategy(big_df, combine_strat, **cfg["Settings"])
    ret = runner.run()
    print(ret)
    print(runner.trajectory)
    print(runner.return_log)

    full_strat = ""

    for strt in cfg["Strat"]:
        full_strat += strt + "_"

    # eqdf["strats"] = full_strat

    # if strat_df.empty:
    #     strat_df = pd.concat([strat_df, eqdf], axis=0)
    # else:
    #     if eqdf["strats"].values[0] not in strat_df["strats"].values:
    #         strat_df = pd.concat([strat_df, eqdf], axis=0)

    # strat_df.to_csv(p_args.df_path)

    # if p_args.plot:
    #     profit_graph(eqdf)
