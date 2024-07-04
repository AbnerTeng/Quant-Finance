"""
Data API
"""
from typing import List, Union, Optional
import pandas as pd
from .utils.data_utils import (
    get_self,
    get_yahoo,
    get_binance,
    get_taifex
)

class DataAPI:
    """
    Get data from API
    """
    def __init__(
        self,
        source: str,
        comp: Union[List[str], str, None],
        start: str,
        end: str,
        scale: str,
        path: Optional[str] = None
    ) -> None:
        self.source = source
        self.comp: List[str] = (
            comp if isinstance(comp, list) else [comp] if comp is not None else None
        )
        self.start = start
        self.end = end
        self.scale = scale
        self.path = path

    def fetch(self) -> pd.DataFrame:
        """
        get data from source
        """
        if self.source == "self":
            data = get_self(self.path)

        elif self.source in ("yahoo", "taifex"):
            data = (
                get_yahoo(self.comp, self.start, self.end, self.scale)
                if self.source == "yahoo"
                else get_taifex(self.start, self.end, market_code=0)
            )

        elif self.source == "binance":
            get_binance()

        else:
            raise ValueError("Invalid source")

        return data
