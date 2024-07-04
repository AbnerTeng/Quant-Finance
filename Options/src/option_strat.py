from .general_option import Option


class Strat(Option):
    """
    Option strategies
    """
    def __init__(self, S: float, K: float, T: int, r: float, sigma: float) -> None:
        super().__init__(S, K, T, r, sigma)

    def long_straddle(self) -> float:
        """
        +call +put with same strike prices
        """
        return self.call(self.K) + self.put(self.K)

    def short_straddle(self) -> float:
        """
        -call -put with same strike prices
        """
        return -self.call(self.K) - self.put(self.K)

    def long_strangle(self, k2: float) -> float:
        """
        +call +put with different strike prices
        """
        if k2 > self.K:
            raise ValueError("k2 must be smaller than K")

        elif k2 == self.K:
            raise ValueError("This is a straddle strategy, use long_straddle() instead")

        else:
            return self.call(self.K) + self.put(k2)

    def short_strangle(self, k2: float) -> float:
        """
        -call -put with different strike prices
        """
        if k2 > self.K:
            raise ValueError("k2 must be smaller than K")

        elif k2 == self.K:
            raise ValueError("This is a straddle strategy, use short_straddle() instead")

        else:
            return -self.call(self.K) - self.put(k2)

    def long_call_spread(self, k2: float) -> float:
        """
        +call -call with different strike prices
        """
        return self.call(self.K) - self.call(k2)

    def short_call_spread(self, k2: float) -> float:
        """
        -call +call with different strike prices
        """
        return -self.call(self.K) + self.call(k2)

    def long_put_spread(self, k2: float) -> float:
        """
        +put -put with different strike prices
        """
        return self.put(self.K) - self.put(k2)

    def short_put_spread(self, k2: float) -> float:
        """
        -put +put with different strike prices
        """
        return -self.put(self.K) + self.put(k2)


    