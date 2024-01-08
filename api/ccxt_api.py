import time
import logging
import numpy as np
import pandas as pd
import ccxt
from utils.time_utils import interval_to_milliseconds
from api.kline_utils import KlineTimes


class CcxtAPI:
    """
    A class for interacting with the CCXT library to retrieve financial
    market data.

    Parameters:
    -----------
    symbol : str
        The trading symbol for the asset pair (e.g., 'BTC/USD').
    interval : str
        The time interval for K-line data
        (e.g., '1h' for 1-hour candles).
    exchange : ccxt.Exchange
        The CCXT exchange object
        (default: ccxt.bitstamp()).
    since : int
        The Unix timestamp of the first candle
        (default: 1325296800000).
    verbose : bool
        If True, print verbose logging messages during data retrieval
        (default: False).

    Attributes:
    -----------
    symbol : str
        The trading symbol for the asset pair.
    interval : str
        The time interval for K-line data.
    since : int
        The Unix timestamp of the first candle.
    data_frame : pd.DataFrame
        DataFrame to store the K-line data.
    exchange : ccxt.Exchange
        The CCXT exchange object.
    max_interval : str
        The maximum time interval supported by the asset pair.
    utils : KlineTimes
        An instance of the KlineTimes class for time-related
        calculations.
    max_multiplier : int
        The maximum multiplier calculated based on the time interval.

    Methods:
    --------
    get_since_value_value() -> int or None:
        Search for the Unix timestamp of the first candle in the
        historical K-line data.

    get_all_klines(ignore_unsupported_exchanges=False) -> CcxtAPI:
        Fetch all K-line data for the specified symbol and interval.

    to_OHLCV() -> pd.DataFrame:
        Convert the fetched K-line data into a pandas DataFrame in
        OHLCV format.

    aggregate_klines(
        exchanges=None,
        symbols=None,
        output_format='DataFrame',
        method='mean',
        filter_by_largest_qty=True
    ) -> pd.DataFrame or dict or tuple:
        Aggregate the fetched K-line data into a pandas DataFrame.

    date_check() -> pd.DataFrame:
        Check for irregularities in the K-line data timestamps and
        return a DataFrame with discrepancies.
    """
    def __init__(
        self,
        symbol:str,
        interval:str,
        exchange:ccxt.Exchange = ccxt.bitstamp(),
        since:int = 1325296800000,
        verbose:bool = False,
    ) -> None:
        """
        Initialize the CcxtAPI object.

        Parameters:
        -----------
        symbol : str
            The trading symbol for the asset pair.
        interval : str
            The time interval for K-line data.
        exchange : ccxt.Exchange
            The CCXT exchange object.
        since : int
            The Unix timestamp of the first candle.
        verbose : bool
            If True, print verbose logging messages during data retrieval
            (default: False).
        """
        self.symbol = symbol
        self.interval = interval
        self.since = since
        self.exchange = exchange
        self.verbose = verbose
        self.max_interval = KlineTimes(symbol, interval).get_max_interval
        self.utils = KlineTimes(self.symbol, self.max_interval)
        self.max_multiplier = int(self.utils.calculate_max_multiplier()) if interval != '1w' else None
        self.data_frame = None
        self.klines_list = None
        if verbose:
            logging.basicConfig(
                format='%(levelname)s %(asctime)s: %(message)s',
                datefmt='%H:%M:%S',
                force=True,
                level=logging.INFO,
            )
        else:
            logging.basicConfig(
                force=True,
                level=logging.CRITICAL,
            )

    def _fetch_klines(self, since, limit: int=None) -> list:
        return self.exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.interval,
            since=since,
            limit=limit,
        )

