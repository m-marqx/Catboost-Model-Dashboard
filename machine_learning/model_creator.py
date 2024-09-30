import itertools
from typing import Literal

import pandas as pd
import numpy as np
import klib
from catboost import CatBoostClassifier, Pool

from machine_learning.model_features import ModelFeatures
from machine_learning.model_handler import ModelHandler
from machine_learning.ml_utils import DataHandler

def adjust_predict_one_side(
    predict: pd.Series,
    max_trades: int,
    target_days: int,
    side: int = 1,
) -> pd.Series:
    """
    Adjusts the maximum trades on one side of the data set.

    Parameters:
    ----------
    predict : pd.Series
        The input series containing the predicted values.
    max_trades : int
        The maximum number of trades.
    target_days : int
        The number of days to consider for trade calculation.
    side : int, optional
        The side of the trade to adjust (1 for long and -1 for short).
        (default: 1).

    Returns:
    -------
    pd.Series
        The adjusted series with maximum trades on one side.
    """
    predict_numpy = predict.to_numpy()
    target = np.where(predict_numpy == side, predict_numpy, 0)

    if side not in (-1, 1):
        raise ValueError("side must be 1 or -1")

    for idx in range(max_trades, len(predict_numpy)):
        if predict_numpy[idx] != 0:
            open_trades = np.sum(target[idx-(target_days):idx + 1])

            if side > 0 and open_trades > max_trades:
                target[idx] = 0
            elif side < 0 and open_trades < -max_trades:
                target[idx] = 0

    return pd.Series(target, index=predict.index, name=predict.name)

def adjust_predict_both_side(
    data_set: pd.DataFrame,
    off_days: int,
    max_trades: int,
):
    """
    Adjusts the maximum trades on both sides of the data set.

    Parameters:
    ----------
    data_set : pd.Series
        The input data set.
    off_days : int
        The number of off days.
    max_trades : int
        The maximum number of trades.

    Returns:
    -------
    pd.Series
        The adjusted data set.
    """
    for idx, row in data_set.iloc[max_trades:].iterrows():
        if row["Predict"] != 0:
            three_lag_days = data_set.loc[:idx].iloc[-(max_trades + 1) : -1]
            three_lag_days_trades = three_lag_days["Predict"].abs().sum()
            if three_lag_days_trades >= max_trades:
                data_set.loc[idx:, "Predict"].iloc[0:off_days] = 0
    return data_set

def adjust_max_trades(
    dataframe: pd.DataFrame,
    off_days: int,
    max_trades: int,
    pct_adj: float,
    side: Literal["both"] | int = "both",
) -> pd.DataFrame:
    """
    Adjust the dataframe based on maximum trades.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input dataframe.
    off_days : int
        Number of days to set the predictions to zero after reaching
        maximum trades.
    max_trades : int
        Maximum number of trades allowed.
    pct_adj : float
        Percentage adjustment to apply to the liquid result.

    Returns:
    --------
    pd.DataFrame
        The adjusted dataframe.
    """
    original_dataframe = dataframe.copy()
    data_frame: pd.DataFrame = dataframe.copy()

    if side == "both":
        data_frame = adjust_predict_both_side(dataframe, off_days, max_trades)
    else:
        if not isinstance(side, int):
            raise ValueError("Invalid side parameter")

        data_frame["Predict"] = adjust_predict_one_side(
            predict=data_frame["Predict"],
            max_trades=max_trades,
            target_days=off_days,
            side=side,
        )

    data_frame["y_pred_probs"] = np.where(
        data_frame["Predict"] == 0, 0, data_frame["y_pred_probs"]
    )
    data_frame["Position"] = data_frame["Predict"].shift().fillna(0)

    data_frame = data_frame.iloc[:, :6]

    data_frame["Liquid_Result"] = np.where(
        data_frame["Predict"] == 0, 0, original_dataframe["Liquid_Result"]
    ) / max_trades + 1

    data_frame["Liquid_Result_pct_adj"] = np.where(
        data_frame["Predict"] == 0, 0, original_dataframe["Liquid_Result"]
    ) / max_trades * pct_adj + 1

    data_frame["Liquid_Return"] = data_frame["Liquid_Result"].cumprod().ffill()

    data_frame["Liquid_Return_simple"] = (
        (data_frame["Liquid_Result"] - 1)
        .cumsum()
        .ffill()
    )

    data_frame["Liquid_Return_pct_adj"] = (
        data_frame["Liquid_Result_pct_adj"].cumprod().ffill()
    )

    data_frame["Drawdown"] = (
        1 - data_frame["Liquid_Return"] / data_frame["Liquid_Return"].cummax()
    )

    data_frame["Drawdown_pct_adj"] = (
        1 - data_frame["Liquid_Return_pct_adj"]
        / data_frame["Liquid_Return_pct_adj"].cummax()
    )

    return data_frame
