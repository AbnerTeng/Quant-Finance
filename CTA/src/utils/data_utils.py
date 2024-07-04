"""
Utilities for data
"""
from typing import Any, List
import warnings
import datetime
import yaml
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from bs4 import BeautifulSoup
import urllib3
from ..constants import TAIFEX_URL

warnings.filterwarnings("ignore")

def get_self(path: str) -> Any:
    """
    data loading utils with yaml, csv, and parquet
    """
    extension = path.split('.')[-1]

    if extension == "csv":
        return pd.read_csv(path, encoding='utf-8')

    elif extension == "parquet":
        return pd.read_parquet(path, engine='pyarrow')

    elif extension == "yaml":
        with open(path, 'r', encoding='utf-8') as yml:
            data = yaml.safe_load(yml)
        return data

    else:
        raise ValueError("File format not supported")

def transfer_colnames(data: Any) -> Any:
    """
    transfer column names to lower case
    """
    if isinstance(data, pd.DataFrame):
        data.columns = [col.lower() for col in data.columns]

    elif isinstance(data, dict):
        data = {key.lower(): value for key, value in data.items()}

    else:
        raise ValueError("Data type not supported")

    return data

def get_yahoo(
    stock_id: List[str],
    start: str,
    end: str,
    scale: str
) -> pd.DataFrame:
    """
    get data from yahoo finance
    
    Instrcutions:
    - stock_id: 
        - US: "AAPL", "NVDA", etc.
        - TW: "2330.TW", "2317.TW", etc.
    - start: start date in format "YYYY-MM-DD"
    - end: end date in format "YYYY-MM-DD"
    - scale: scale of data, e.g. "1d", "1h", "1m"
    
    For more informations, please refer to:
    > https://github.com/ranaroussi/yfinance/wiki/Tickers#download
    """
    if len(stock_id) == 1:
        return yf.download(
            stock_id[0],
            start=start,
            end=end,
            interval=scale
        )

    else:
        full_df = pd.DataFrame()
        false_counter = 0

        for stock in tqdm(stock_id):
            try:
                if false_counter > 5:
                    print("There may be general errors")
                    break

                else:
                    df = yf.download(
                        stock,
                        start=start,
                        end=end,
                        interval=scale,
                        progress=True
                    )
                    df['stock_id'] = stock
                    full_df = pd.concat(
                        [full_df, df],
                        axis=0
                    )

            except ValueError:
                print("Stock ID not found")
                false_counter += 1

            except TypeError:
                print("Invalid input")
                false_counter += 1

        return full_df

def get_binance() -> pd.DataFrame:
    """
    Get crypto data from binance
    """
    pass

def get_taifex(day: datetime, market_code: int = 0) -> pd.DataFrame:
    """
    Get data from Taiwan Futures Exchange

    start & end date format: YYYY-MM-DD (in datetime format)
    """
    http = urllib3.PoolManager()

    res = http.request(
        "POST",
        TAIFEX_URL,
        fields={
            "queryType": 2,
            "marketCode": market_code,
            "commodity_id": "TX",
            "queryDate": day,
            "MarketCode": market_code,
            "commodity_idt": "TX"
        }
    )
    html_doc = res.data
    soup = BeautifulSoup(html_doc, "html.parser")
    table = soup.findAll("table")[0]

    try:
        df_day = pd.DataFrame(
            pd.read_html(str(table))[0].iloc[0]
        ).T
        df_day["Date"] = day
        return df_day

    except ValueError:
        print("No data on this day")
