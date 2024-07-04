"""
Backtesting pipeline
"""
from multiprocessing import Pool
import pandas as pd
from .indicators.ma import MAStrat
from .utils.data_utils import (
    get_self,
    get_taifex,
    transfer_colnames
)
from .get_data import DataAPI
from .plot import profit_graph, trade_position


if __name__ == '__main__':
    cfg = get_self("config/sma_test.yaml")
    fetcher = DataAPI(**cfg["DataAPITAIFEX"])

    date_list = pd.date_range(
        cfg["DataAPITAIFEX"]["start"],
        cfg["DataAPITAIFEX"]["end"], freq='D'
    ).strftime("%Y/%m/%d").tolist()

    with Pool(10) as pool:
        daily_data = pool.map(get_taifex, date_list)

    data = pd.concat(daily_data, axis=0)
    data.insert(0, "Date", data.pop("Date"))
    data = transfer_colnames(data)
    tester = MAStrat(data, **cfg["Strat"])
    profit_graph(tester.profit_map)
    trade_position(tester.trade_log)
