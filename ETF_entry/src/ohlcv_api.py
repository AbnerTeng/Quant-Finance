"""
Download data
"""
from typing import List
from argparse import ArgumentParser
from rich import print as rprint
from CTA.src.get_data import DataAPI
from CTA.src.utils.data_utils import (
    get_self,
    transfer_colnames,
    filter_novol
)

def parse_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        help="config file"
    )
    return parser.parse_args()


def main(cfgs: List[str]) -> None:
    """
    Main function
    """
    for cfg_path in cfgs:
        cfg = get_self(cfg_path)
        fetcher = DataAPI(**cfg["DataAPIYahoo"])
        data = fetcher.fetch()
        data = transfer_colnames(data)
        data = filter_novol(data)
        data.to_csv(cfg["DataDest"])

        rprint(f"Config file: {cfg_path} executed successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
