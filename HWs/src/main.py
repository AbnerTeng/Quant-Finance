"""
Main script
"""
from argparse import ArgumentParser
import yfinance as yf
from .utils import (
    load_data,
    plot_dd,
    plot_trade_position,
    metrics
)
from .backtest import (
    get_moving_average,
    strategy,
    cal_equity,
    get_ret
)

def parse_args() -> ArgumentParser:
    """
    Parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--ticker",
        "-t",
        type=str,
        help="Ticker symbol"
    )
    parser.add_argument(
        "--k",
        type=int
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ticker:
        data = yf.download(args.ticker, start="2021-01-01", end="2023-12-31")
    else:
        data = load_data("data/AAPL.csv")

    get_moving_average(data, args.k)
    signal, profit = strategy(data, args.k)
    equity_df = cal_equity(profit, data)
    daily_return = get_ret(equity_df)

    if args.plot:
        plot_dd(equity_df)
        plot_trade_position(data, signal, f"MA_{args.k}")

    print(metrics(equity_df, daily_return))
