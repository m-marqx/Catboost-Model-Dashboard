import itertools
from ast import literal_eval
from typing import Literal

import pandas as pd
import numpy as np
import plotly.express as px

from machine_learning.ml_utils import DataHandler
from machine_learning.model_builder import model_creation
from machine_learning.model_creator import (
    adjust_predict_one_side,
)


class ModelMiner:
    """
    Class to search for the best model parameters.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the OHLC data.
    target : pd.Series
        The target series containing the adjusted values.
    max_trades : int, optional
        The maximum number of trades
        (default : 3).
    off_days : int, optional
        The number of days to consider for trade calculation
        (default : 7).
    side : int, optional
        The side of the trade to adjust
        (default : 1).

    Attributes:
    ----------
    ohlc : list
        The list of the OHLC columns.
    max_trades : int
        The maximum number of trades.
    off_days : int
        The number of days to consider for trade calculation.
    dataframe : pd.DataFrame
        The input dataframe containing the OHLC data.
    target : pd.Series
        The target series containing the adjusted values.
    ma_types : list
        The list of the moving averages types.
    ma_type_combinations : np.array
        The array of the moving averages type combinations.
    features : list
        The list of the features.
    random_features : np.array
        The array of the random features.
    adj_targets : pd.Series
        The adjusted target series.
    empty_dict : dict
        The empty dictionary.
    feat_parameters : None
        The features variables.

    Methods:
    -------
    search_model(test_index: int, pct_adj: float = 0.5, train_in_middle:
    bool = True)
        Search for the best model parameters.


    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: pd.Series,
        max_trades: int = 3,
        off_days: int = 7,
        side: int = 1,
    ):
        """
        Parameters:
        ----------
        predict : pd.Series
            The input series containing the predicted values.
        max_trades : int
            The maximum number of trades.
        target_days : int
            The number of days to consider for trade calculation.
        side : int, optional
            The side of the trade to adjust
            (default : 1).
        """
        self.ohlc = ["open", "high", "low", "close"]
        self.max_trades = max_trades
        self.off_days = off_days

        self.dataframe = dataframe.copy()
        self.target = target.copy()

        # DTW
        self.ma_types = ["sma", "ema", "dema", "tema", "rma"]

        combinations_list = []
        for r in range(1, len(self.ma_types) + 1):
            combinations_list.extend(itertools.combinations(self.ma_types, r))

        self.ma_type_combinations = np.array(
            list(combinations_list) + ["all"],
            dtype="object",
        )

        features = [
            "RSI",
            "Stoch",
            "CCI",
            "MACD",
            "SMIO",
            "TRIX",
            "DTW",
            "TSI",
            "DIDI",
            "Ichimoku"
        ]

        self.random_features = []
        for x in range(1, len(features) + 1):
            self.random_features += list(itertools.combinations(features, x))

        self.random_features = np.array(self.random_features, dtype="object")

        self.adj_targets = adjust_predict_one_side(self.target, max_trades, off_days, side)

        self.empty_dict = {
            "feat_parameters": None,
            "hyperparameters": None,
            "metrics_results": None,
            "drawdown_full_test": None,
            "drawdown_full_val": None,
            "drawdown_adj_test": None,
            "drawdown_adj_val": None,
            "expected_return_test": None,
            "expected_return_val": None,
            "precisions_test": None,
            "precisions_val": None,
            "support_diff_test": None,
            "support_diff_val": None,
            "total_operations_test": None,
            "total_operations_val": None,
            "total_operations_pct_test": None,
            "total_operations_pct_val": None,
            "r2_in_2023": None,
            "r2_val": None,
            "test_index": None,
        }

    def __generate_feat_parameters(self):
        macd_lengths = np.random.choice(range(2, 151), 2, replace=False)
        smio_lengths = np.random.choice(range(2, 151), 2, replace=False)
        tsi_lengths = np.random.choice(range(2, 151), 2, replace=False)
        didi_ma_lengths = np.random.choice(range(2, 151), 3, replace=False)
        ichimoku_lengths = np.random.choice(range(2, 151), 2, replace=False)

        fast_length = max(macd_lengths)
        slow_length = min(macd_lengths)
        signal_length = np.random.choice(range(2, 51))
        signal_length = (
            signal_length + 1 if signal_length in macd_lengths else signal_length
        )

        return {
            # General
            "random_features": list(np.random.choice(self.random_features)),
            # DTW
            "random_source_price_dtw": np.random.choice(self.ohlc),
            "random_binnings_qty_dtw": np.random.choice(range(10, 31)),
            "random_moving_averages": np.random.choice(self.ma_type_combinations),
            "random_moving_averages_length": np.random.choice(range(2, 301)),
            # RSI
            "random_source_price_rsi": np.random.choice(self.ohlc),
            "random_binnings_qty_rsi": np.random.choice(range(10, 31)),
            "random_rsi_length": np.random.choice(range(2, 151)),
            # STOCH
            "random_source_price_stoch": np.random.choice(self.ohlc),
            "random_binnings_qty_stoch": np.random.choice(range(10, 31)),
            "random_slow_stoch_length": np.random.choice(range(2, 51)),
            "random_slow_stoch_k": np.random.choice(range(1, 11)),
            "random_slow_stoch_d": np.random.choice(range(2, 11)),
            # DIDI
            "random_source_price_didi": np.random.choice(self.ohlc),
            "random_binnings_qty_didi": np.random.choice(range(10, 31)),
            "random_didi_short_length": int(np.min(didi_ma_lengths)),
            "random_didi_mid_length": int(np.median(didi_ma_lengths)),
            "random_didi_long_length": int(np.max(didi_ma_lengths)),
            "random_didi_ma_type": np.random.choice(self.ma_types),
            "random_didi_method": np.random.choice(["absolute", "ratio", "dtw"]),
            # CCI
            "random_source_price_cci": np.random.choice(self.ohlc),
            "random_binnings_qty_cci": np.random.choice(range(10, 31)),
            "random_cci_length": np.random.choice(range(2, 151)),
            "random_cci_method": np.random.choice(self.ma_types),
            # MACD
            "random_source_price_macd": np.random.choice(self.ohlc),
            "random_binnings_qty_macd": np.random.choice(range(10, 31)),
            "random_macd_fast_length": fast_length,
            "random_macd_slow_length": slow_length,
            "random_macd_signal_length": signal_length,
            "random_macd_diff_method": np.random.choice(["absolute", "dtw"]),
            "random_macd_ma_method": np.random.choice(self.ma_types),
            "random_macd_signal_method": np.random.choice(self.ma_types),
            "random_macd_column": np.random.choice(["macd", "signal", "histogram"]),
            # TRIX
            "random_source_price_trix": np.random.choice(self.ohlc),
            "random_binnings_qty_trix": np.random.choice(range(10, 31)),
            "random_trix_length": np.random.choice(range(2, 51)),
            "random_trix_signal_length": np.random.choice(range(2, 51)),
            "random_trix_ma_method": np.random.choice(self.ma_types),
            # SMIO
            "random_source_price_smio": np.random.choice(self.ohlc),
            "random_binnings_qty_smio": np.random.choice(range(10, 31)),
            "random_smio_short_length": max(smio_lengths),
            "random_smio_long_length": min(smio_lengths),
            "random_smio_signal_length": np.random.choice(range(2, 51)),
            "random_smio_ma_method": np.random.choice(self.ma_types),
            # TSI
            "random_source_price_tsi": np.random.choice(self.ohlc),
            "random_binnings_qty_tsi": np.random.choice(range(10, 31)),
            "random_tsi_short_length": max(tsi_lengths),
            "random_tsi_long_length": min(tsi_lengths),
            "random_tsi_ma_method": np.random.choice(self.ma_types),
            # Ichimoku
            "random_binnings_qty_ichimoku": np.random.choice(range(10, 31)),
            "random_ichimoku_conversion_periods": ichimoku_lengths[0],
            "random_ichimoku_base_periods": ichimoku_lengths[1],
            "random_ichimoku_lagging_span_2_periods": np.random.choice(range(2, 31)),
            "random_ichimoku_displacement": np.random.choice(range(2, 31)),
            "random_ichimoku_based_on": np.random.choice(['lead_line', 'lagging_span']),
            "random_ichimoku_method": np.random.choice(["absolute", "ratio", "dtw"]),
        }

