from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Portfolio:
    def __init__(self, data: pd.DataFrame, test: pd.DataFrame):
        self.data = data
        self.test = test
        self.unique_dates = self.data['Date'].unique()

    def form_portfolio(self) -> Tuple[List, List]:
        """
        Form Long-Short Portfolio
        """
        long_rets, short_rets = [], []

        for idx, date in enumerate(self.unique_dates):
            if idx == len(self.unique_dates) - 1:
                break

            long_stocks = self.data[self.data['Date'] == date].sort_values(
                'pred',
                ascending=False
            ).index[:2].to_list()
            short_stocks = self.data[self.data['Date'] == date].sort_values(
                'pred',
                ascending=True
            ).index[:2].to_list()
            subtest = self.test[self.test['Date'] == self.unique_dates[idx + 1]]
            long_ret = subtest.loc[long_stocks, 'next_ret']
            short_ret = subtest.loc[short_stocks, 'next_ret']
            long_rets.append(long_ret.values)
            short_rets.append(short_ret.values)

        return long_rets, short_rets

    def calc_merge_ret(self) -> np.ndarray:
        """
        Calculating the merged return
        """
        long_rets, short_rets = self.form_portfolio()
        merge_ret = np.mean(np.array(long_rets), axis=1) - np.mean(np.array(short_rets), axis=1)
        return merge_ret

    def plot(self) -> None:
        """
        Plot the merged return
        """
        merge_ret = self.calc_merge_ret()
        plt.plot(merge_ret.cumsum())
        plt.show()
