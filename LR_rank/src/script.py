from CTA.src.get_data import DataAPI
from CTA.src.utils.data_utils import (
    get_self,
    transfer_colnames
)

# Path: ETF_entry/config/test.yaml

if __name__ == '__main__':
    cfg = get_self("LR_rank/config/fetcher.yaml")
    fetcher = DataAPI(**cfg["DataAPIYahoo"])
    data = fetcher.fetch()
    data = transfer_colnames(data)
    data.to_csv(cfg["DataDest"])
