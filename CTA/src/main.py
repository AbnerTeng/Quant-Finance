"""
Backtesting pipeline
"""
from multiprocessing import Pool
from argparse import ArgumentParser
import pandas as pd
from .indicators.ma import MAStrat
from .utils.data_utils import (
    get_self,
    get_taifex,
    transfer_colnames
)
from .get_data import DataAPI
from .plot import profit_graph, trade_position


def parse_args():
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config_type", "-ctp", type=str, default="yahoo"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_self("config/sma_test.yaml")

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

    data = transfer_colnames(data)
    # print(data)
    tester = MAStrat(data, **cfg["Strat"])
    eqdf = tester.run_strategy()
    profit_graph(eqdf)
    print(tester.trade_log)

    if tester.k2 is not None:
        indicators = [data[f"ma_{tester.k1}"], data[f"ma_{tester.k2}"]]
    else:
        indicators = [data[f"ma_{tester.k1}"]]

    trade_position(data, indicators, tester.trade_log)
