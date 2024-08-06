import itertools
from typing import Literal

import pandas as pd
import numpy as np
import klib
from catboost import CatBoostClassifier, Pool

from machine_learning.model_features import ModelFeatures
from machine_learning.model_handler import ModelHandler
from machine_learning.ml_utils import DataHandler

def create_catboost_model(
    dataset: pd.DataFrame,
    feats: list,
    test_index: int,
    target_series: pd.Series,
    plot: bool = False,
    output: Literal["All", "Return", "Model", "Dataset"] = "All",
    long_only: bool = False,
    short_only: bool = False,
    train_in_middle: bool = False,
    **hyperparams,
) -> tuple | CatBoostClassifier:
    """
    Create the machine learning model using the CatBoost algorithm.

    Parameters:
    -----------
    dataset : pd.DataFrame
        The input dataset.
    feats : list
        List of features to be used for training the model.
    test_index : int, optional
        Index to split the dataset into training and testing sets.
        (default: 1000)
    target_series : pd.Series
        The target variable series.
    plot : bool, optional
        Whether to plot the evaluation set during model training.
        (default: False)
    output : Literal["All", "Return", "Model", "Dataset"], optional
        Output parameter to specify the desired return values.
        (default: "All")
    long_only : bool, optional
        Whether to consider long positions only.
        (default: False)
    short_only : bool, optional
        Whether to consider short positions only.
        (default: False)
    train_in_middle : bool, optional
        Whether to train the model using the test set in the middle.
        (default: False)
    **hyperparams : dict, optional
        Additional keyword arguments to be passed to the
        CatBoostClassifier.

    Returns:
    --------
    pd.DataFrame or tuple
        Depending on the value of the `output` parameter, the function
        returns different values.
        If `output` is "All", it returns a tuple containing various
        model-related objects.
        If `output` is "Return", it returns a tuple containing the model
        returns and index splits.
        If `output` is "Model", it returns the best model.
        If `output` is "Dataset", it returns the training and testing
        datasets.

    Raises:
    -------
    ValueError
        If the `output` parameter is invalid.
    """
    data_frame = klib.convert_datatypes(dataset)

    train_set = data_frame.iloc[:test_index]
    test_set = data_frame.iloc[test_index : test_index * 2]
    validation_set = data_frame.iloc[test_index * 2 :]

    features = list(feats)
    target = ["Target_bin"]

    X_train = train_set[features]
    y_train = train_set[target]

    X_test = test_set[features]
    y_test = test_set[target]

    X_validation = validation_set[features]
    y_validation = validation_set[target]

    all_x = pd.concat([X_train, X_test, X_validation])
    all_y = pd.concat([y_train, y_test, y_validation])

    if not hyperparams:
        hyperparams = {
            "iterations": 500,
            "learning_rate": 0.1,
            "eval_metric": "Logloss",
            "random_seed": 69,
            "logging_level": "Silent",
            "use_best_model": True,
        }

    if train_in_middle:
        train_pool = Pool(X_test, y_test)
        test_pool = Pool(X_train, y_train)
    else:
        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)

    best_model = CatBoostClassifier(**hyperparams)

    best_model.fit(train_pool, eval_set=test_pool, plot=plot)

    predict = best_model.predict_proba(train_set[features])
    cutoff = np.median(predict)

    X_test = test_set[features]
    y_test = test_set[target]

    y_pred_train = np.where(best_model.predict_proba(X_train) > cutoff, 1, 0)

    y_pred_test = np.where(best_model.predict_proba(X_test) > cutoff, 1, 0)

    y_pred_train = pd.DataFrame(y_pred_train, index=y_train.index)[1]
    y_pred_test = pd.DataFrame(y_pred_test, index=y_test.index)[1]

    dataset_params = dict(X_test=all_x, y_test=all_y)

    target_index = all_y.index
    mh2 = ModelHandler(best_model, **dataset_params).model_returns(
        target_series=target_series.reindex(target_index),
        fee=0,
        cutoff=cutoff,
        long_only=long_only,
        short_only=short_only,
    )

    mh2["cuttoff"] = cutoff
    index_splits = {
        "train": pd.Interval(train_set.index[0], train_set.index[-1]),
        "test": pd.Interval(test_set.index[0], test_set.index[-1]),
        "validation": pd.Interval(
            validation_set.index[0],
            validation_set.index[-1],
        ),
    }

    match output:
        case "All":
            return (
                mh2,
                best_model,
                X_train,
                X_test,
                y_train,
                y_test,
                y_pred_train,
                y_pred_test,
                all_x,
                all_y,
                index_splits,
            )

        case "Return":
            return mh2, index_splits

        case "Model":
            return best_model

        case "Dataset":
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                y_pred_train,
                y_pred_test,
                all_x,
                all_y,
            )

        case _:
            raise ValueError("Invalid output parameter")

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
        The side of the trade to adjust
        (default: 1).

    Returns:
    -------
    pd.Series
        The adjusted series with maximum trades on one side.
    """
    predict_numpy = predict.to_numpy()
    target = np.where(predict_numpy == side, predict_numpy, 0)

    for idx in range(max_trades, len(predict_numpy)):
        if predict_numpy[idx] != 0:
            three_lag_days_trades = np.sum(target[idx-(target_days):idx + 1])

            if three_lag_days_trades > max_trades:
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
