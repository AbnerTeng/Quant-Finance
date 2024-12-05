"""
Data API
"""

from typing import Union
from multiprocessing import Pool

from omegaconf import DictConfig, ListConfig
import pandas as pd

from .utils.data_utils import get_self, get_yahoo, get_taifex


class DataAPI:
    """
    Get data from API

    Args:
        source (str): data source
    """

    def __init__(self, source: str) -> None:
        self.source = source

    def fetch(self, spec_config: Union[DictConfig, ListConfig]) -> pd.DataFrame:
        """
        get data from source
        """
        data = None

        if self.source == "self":
            data = get_self(spec_config.path)
            data.index = pd.to_datetime(data.index)

        elif self.source == "yahoo":
            data = get_yahoo(spec_config)

        elif self.source == "taifex":
            date_list = (
                pd.date_range(spec_config.start, spec_config.end, freq="D")
                .strftime("%Y/%m/%d")
                .tolist()
            )

            with Pool(10) as pool:
                daily_data = pool.map(get_taifex, date_list)

            data = pd.concat(daily_data, axis=0)
            data.insert(0, "Date", data.pop("Date"))

        elif self.source == "binance":
            raise NotImplementedError

        else:
            raise ValueError("Invalid source")

        return data
