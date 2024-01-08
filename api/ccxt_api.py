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

    def get_since_value(self):
        """
        Search for the Unix timestamp of the first candle in the
        historical K-line data.

        This method iteratively fetches K-line data in reverse
        chronological order and stops when it finds the first candle.
        It can be used to determine the starting point for fetching
        historical data.

        Returns:
        --------
        int or None
            The Unix timestamp of the first candle found, or None
            if not found.
        """
        end_times = self.utils.get_end_times(
            self.since,
            self.max_multiplier
        )

        for index in range(0, len(end_times) - 1):
            klines = self._fetch_klines(
                since=int(end_times[index]),
                limit=self.max_multiplier,
            )
            if self.verbose:
                load_percentage = (index / (len(end_times) - 1)) * 100
                logging.info(
                    "Finding first candle time [%.2f%%]",
                    load_percentage
                )

            if len(klines) > 0:
                first_unix_time = klines[0][0]
                if self.verbose:
                    logging.info("Finding first candle time [100%]")
                    logging.info(
                        "First candle time found: %s\n",
                        first_unix_time
                    )
                break

        return first_unix_time

    def get_all_klines(
        self,
        until: int | None = None,
        ignore_unsupported_exchanges: bool = False
    ):
        """
        Fetch all K-line data for the specified symbol and interval
        using a for loop.

        Parameters:
        -----------
        until : None
            The end time for fetching K-line data.
        ignore_unsupported_exchanges : bool, optional
            If True, ignore exchanges that do not support the specified
            symbol.
            (default: False).

        Returns:
        --------
        CcxtAPI
            Returns the CcxtAPI object with the fetched K-line data.
        """
        if ignore_unsupported_exchanges:
            not_supported_types = None
        else:
            not_supported_types = (
                type(ccxt.bittrex()),
                type(ccxt.gemini()),
                type(ccxt.huobi()),
                type(ccxt.deribit()),
                type(ccxt.hitbtc()),
            )

        if isinstance(self.exchange, not_supported_types):
            raise ValueError(f"{self.exchange} is not supported")

        klines = []
        klines_list = []

        first_call = self._fetch_klines(self.since, self.max_multiplier)

        if first_call:
            first_unix_time = first_call[0][0]
        else:
            first_unix_time = self.get_since_value()
            first_call = self._fetch_klines(first_unix_time, self.max_multiplier)


        last_candle_interval = (
            (
                time.time() * 1000 - interval_to_milliseconds(self.interval)
                if until is None
                else until
            )
        )

        if self.verbose:
            start = time.perf_counter()
            logging.info("Starting requests")

        time_value = klines[-1][0] + 1 if klines else first_unix_time
        time_delta = first_call[-1][0] - first_call[0][0]
        step = time_delta + pd.Timedelta(self.interval).value / 1e+6
        end_times = np.arange(time_value, last_candle_interval, step)

        for current_start_time in end_times:
            klines = self._fetch_klines(
                int(current_start_time),
                self.max_multiplier
            )
            if not klines:
                break

            klines_list.extend(klines)

            if klines_list[-1][0] >= last_candle_interval:
                if self.verbose:
                    logging.info(
                        "Qty: %d - Total: 100%% complete",
                        len(klines_list)
                    )
                break

            if self.verbose:

                percentage = (
                    (np.where(end_times == current_start_time)[0][0] + 1)
                    / end_times.shape[0]
                ) * 100
                logging.info(
                    "Qty: %d - Total: %.2f%% complete",
                    len(klines_list), percentage
                )


        if self.verbose:
            logging.info(
                "Requests elapsed time: %s\n",
                time.perf_counter() - start
            )
        self.klines_list = klines_list
        return self

    def to_OHLCV(self) -> pd.DataFrame:
        """
        Convert the fetched K-line data into a pandas DataFrame in
        OHLCV format.

        Returns:
        --------
        pd.DataFrame
            Returns a pandas DataFrame containing OHLCV data.
        """
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

        self.data_frame = pd.DataFrame(
            self.klines_list,
            columns=["date"] + ohlcv_columns
        )

        self.data_frame["date"] = self.data_frame["date"].astype(
            "datetime64[ms]"
        )

        self.data_frame = self.data_frame.set_index("date")
        return self

    def update_klines(
        self,
        klines: list | pd.DataFrame
    ) -> list | pd.DataFrame:
        """
        Update historical Klines data with new data.

        This method is used to update existing historical Klines data
        with fresh data obtained from the exchange. The function
        identifies the last timestamp in the provided Klines data and
        retrieves new data starting from that point.

        Parameters:
        -----------
        klines : list or pd.DataFrame
            The historical Klines data to be updated. This can be
            either a list of lists or a pandas DataFrame.

        Returns:
        --------
        list or pd.DataFrame
            The updated Klines data with the new data appended.
        """
        last_time = (
            klines[-3][0] if isinstance(klines, list)
            else int(klines.index[-3].value / 1e+6)
        )

        capi = CcxtAPI(self.symbol, self.interval, self.exchange, last_time)
        new_data = capi.get_all_klines()

        if isinstance(klines, list):
            new_klines = new_data.klines_list
            new_data = [
                data for data in new_klines
                if data[0] not in (kline[0] for kline in klines)
            ]

            updated_klines = klines + new_data
            updated_klines.sort()

        else:
            new_df = new_data.to_OHLCV().data_frame
            updated_klines = klines.combine_first(new_df)
            updated_klines = updated_klines.sort_index()
            updated_klines = updated_klines[
                ~updated_klines.index.duplicated(keep='last')
            ]

        return updated_klines

