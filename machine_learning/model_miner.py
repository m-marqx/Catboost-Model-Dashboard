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
