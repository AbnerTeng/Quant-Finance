import numpy as np
from .general_option import Option

class Greeks(Option):
    """
    Greeks class
    """
    def vega(self):
        """
        vega
        """
        return self.S * self.N(self.d1) * np.sqrt(self.T)

    def theta(self):
        pass

    def gamma(self):
        pass

    def rho(self):
        pass

    def delta(self):
        pass

    