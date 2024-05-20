from typing import Literal

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import klib

from machine_learning.model_handler import ModelHandler
from machine_learning.model_features import ModelFeatures

from machine_learning.model_creator import (
    adjust_predict_one_side,
)

def max_trade_adj(data_set, off_days, max_trades, pct_adj):
    """
    Adjusts the input dataset based on the maximum number of trades,
    off days, and percentage adjustment.

    Parameters
    ----------
    data_set : pd.DataFrame
        The input dataset containing the predictions and results.
    off_days : int
        The number of days to consider for trade calculation.
    max_trades : int
        The maximum number of trades.
    pct_adj : float
        The percentage adjustment.

    Returns
    -------
    pd.DataFrame
        - Returns adjusted dataset containing the following columns:
            - Predict (pandas.Series): The adjusted predictions.
            - y_pred_probs (pandas.Series): The adjusted predicted
            probabilities.
            - Position (pandas.Series): The adjusted positions.
            - Liquid_Result (pandas.Series): The adjusted liquid results.
            - Liquid_Result_pct_adj (pandas.Series): The adjusted liquid
            results with percentage adjustment.
            - Liquid_Return (pandas.Series): The cumulative liquid returns.
            - Liquid_Return_simple (pandas.Series): The cumulative simple
            liquid returns.
            - Liquid_Return_pct_adj (pandas.Series): The cumulative liquid
            returns with percentage adjustment.
            - Drawdown (pandas.Series): The drawdowns.
            - Drawdown_pct_adj (pandas.Series): The drawdowns with
            percentage adjustment.
    """
    original_dataset = data_set.copy()

    data_set["Predict"] = adjust_predict_one_side(
        data_set["Predict"], max_trades, off_days, 1
    )

    data_set["y_pred_probs"] = np.where(
        data_set["Predict"] == 0, 0, data_set["y_pred_probs"]
    )

    data_set["Position"] = data_set["Predict"].shift().fillna(0)

    data_set = data_set.iloc[:, :6]

    data_set["Liquid_Result"] = (
        np.where(data_set["Predict"] == 0, 0, original_dataset["Liquid_Result"])
        / max_trades
        + 1
    )

    data_set["Liquid_Result_pct_adj"] = (
        np.where(data_set["Predict"] == 0, 0, original_dataset["Liquid_Result"])
        / max_trades
        * pct_adj
        + 1
    )

    data_set["Liquid_Return"] = data_set["Liquid_Result"].cumprod().ffill()

    data_set["Liquid_Return_simple"] = (
        (data_set["Liquid_Result"] - 1)
        .cumsum()
        .ffill()
    )

    data_set["Liquid_Return_pct_adj"] = (
        data_set["Liquid_Result_pct_adj"].cumprod().ffill()
    )

    data_set["Drawdown"] = (
        1 - data_set["Liquid_Return"] / data_set["Liquid_Return"].cummax()
    )

    data_set["Drawdown_pct_adj"] = (
        1 - (
            data_set["Liquid_Return_pct_adj"]
            / data_set["Liquid_Return_pct_adj"].cummax()
        )
    )
    return data_set
