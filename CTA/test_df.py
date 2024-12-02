import os
import pickle

from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd


def get_ret(path: str) -> dict:
    with open(path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    log_dict = {"log": [], "cum_ret": []}

    for logs in os.listdir(args.data_dir):
        df = get_ret(f"{args.data_dir}/{logs}")
        log_dict["log"].append(logs)
        log_dict["cum_ret"].append(np.cumsum(df["return"])[-1])

    print(pd.DataFrame(log_dict).sort_values("cum_ret", ascending=False))
