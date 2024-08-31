from multiprocessing import Pool
from argparse import ArgumentParser
import pandas as pd
from .indicators.ma import SMA, EMA, MACD
from .indicators.rsi import RSI
from .indicators.bb import BB
from .strategy import (
    CombineStrategy,
    RunStrategy
)
from .utils.general_utils import parse_column
from .utils.data_utils import (
    get_self,
    get_taifex,
    transfer_colnames
)
from .get_data import DataAPI
from .utils.plot import profit_graph, trade_position
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
            data = strat.build()
            full_df = pd.concat([full_df, data], axis=1)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_self(args.config_path)

    if args.config_type == "yahoo":
        fetcher = DataAPI(**cfg["DataAPIYahoo"])
        data = fetcher.fetch()

    elif args.config_type == "taifex":
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
    comb = Combination(BB(20, True), SMA(5))
    big_df = comb.run_combination()
    combine_strat = CombineStrategy(
        big_df,
        comb.indicators,
        comb.cleaned_columns
    )
    runner = RunStrategy(big_df, combine_strat, cfg["Settings"])
    eqdf = runner.run()
    profit_graph(eqdf)
