"""
Option class
"""
import numpy as np
from scipy.stats import norm


class Option:
    """
    template class for option
    """
    def __init__(
        self, S: float, K: float,
        T: int, r: float, sigma: float
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n = norm.pdf
        self.N = norm.cdf

    def call(self, K: float) -> float:
        """
        call price
        """
        if K is None:
            K = self.K

        d1 = (
            np.log(self.S / K) + (self.r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.S * self.N(d1) - self.K * np.exp(-self.r * self.T) * self.N(d2)

    def put(self, K: float) -> float:
        """
        put price
        """
        if K is None:
            K = self.K

        d1 = (
            np.log(self.S / K) + (self.r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.K * np.exp(-self.r * self.T) * self.N(-d2) - self.S * self.N(-d1)
