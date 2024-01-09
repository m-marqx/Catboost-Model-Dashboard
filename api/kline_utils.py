from math import ceil
import time
import numpy as np
from utils import interval_to_milliseconds

class KlineTimes:
    """
    Class for working with Kline times.

    Parameters
    ----------
    symbol : str
        The symbol of the asset.
    interval : str
        The interval of the Kline data.

    Attributes
    ----------
    symbol : str
        The symbol of the asset.
    interval : str
        The interval of the Kline data.

    Methods
    -------
    default_intervals()
        Returns the list of default intervals.
    calculate_max_multiplier(max_candle_limit: int = 1500)
        Calculate the maximum multiplier based on the interval.
    get_end_times(start_time=1597118400000, max_candle_limit=1500)
        Get the end times for retrieving Kline data.
    interval_max_divisor()
        Returns the maximum divisor of the interval.

    """
    def __init__(self, symbol, interval):
        """
        Initialize the KlineTimes object

        Parameters:
        -----------
        symbol : str
            The symbol of the asset.
        interval : str
            The interval of the Kline data.
        """
        self.symbol = symbol
        self.interval = interval

    @property
    def default_intervals(self):
        """
        Returns the list of default intervals.

        Returns
        -------
        list of str
            The list of default intervals.

        """
        return [
            "1s",
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]

