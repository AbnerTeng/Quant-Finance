"""
Get the list of companies in the ETF every QTR
"""
import os
import time
import warnings
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
warnings.filterwarnings("ignore")

URL = "https://mops.twse.com.tw/mops/web/ajax_t78sb04"


def get_comp(year: int, qtr: str) -> pd.DataFrame:
    """
    Get the list of companies in the ETF every QTR

    year (int): year of ROC (89, 90, 110, ...)
    qtr (str): quarter of ROC (01, 02, 03, 04)
    """
    data = {
        'encodeURIComponent': 1,
        'TYPEK': 'all',
        'step': 1,
        'run': '',
        'firstin': True,
        'FUNTYPE': 2,
        'year': year,
        'season': qtr,
        'fund_no': 0
    }
    try:
        response = requests.post(URL, data=data, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        etf_table = soup.find_all('table')[1]
        df = pd.read_html(str(etf_table))[0][:50]
        df['time'] = f"{year}Q{qtr[1]}"
        df.drop(
            columns=[
                "股票種類",
                "股票名稱",
                "產業類別",
                "持股比率.1"
            ],
            inplace=True
        )
        assert len(df) == 50, f"The number of companies is not 50 at {year}Q{qtr[1]}"
        return df

    except IndexError:
        print(f"IndexError at {year}Q{qtr[1]}")
        return pd.DataFrame()


def concat_dfs(df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    concat all components
    """
    full_df = pd.concat([df, full_df], ignore_index=True)
    return full_df


if __name__ == "__main__":
    if not os.path.exists("data/etf_components.csv"):
        output = pd.DataFrame()

    else:
        output = pd.read_csv("data/etf_components.csv", encoding="utf-8")

    yqtr = [
        int(d.split('Q')[0]) for d in output['time'].unique()
    ] if len(output) > 0 else None
    current_yqtr = max(yqtr) if yqtr else 98

    for y in tqdm(
        range(current_yqtr, current_yqtr, 1)
    ):
        for q in ['01', '02', '03', '04']:
            curr_df = get_comp(y, q)
            output = concat_dfs(curr_df, output)
            time.sleep(3)

    output.to_csv("data/etf_components.csv", index=False)
