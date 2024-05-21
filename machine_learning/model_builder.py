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

def calculate_model(
    dataset: pd.DataFrame,
    feats: list,
    test_index: int,
    plot: bool = False,
    output: Literal["All", "Return", "Model", "Dataset"] = "All",
    long_only: bool = False,
    short_only: bool = False,
    train_in_middle: bool = False,
    **hyperparams,
) -> pd.DataFrame:
    """
    Calculates the model based on the input dataset, features, test
    index,and hyperparameters.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataset.
    feats : list
        The list of features.
    test_index : int
        The test index.
    plot : bool, optional
        Whether to plot the model, by default False.
    output : Literal["All", "Return", "Model", "Dataset"], optional
        The output type, by default "All".
    long_only : bool, optional
        Whether to consider long only, by default False.
    short_only : bool, optional
        Whether to consider short only, by default False.
    train_in_middle : bool, optional
        Whether to train in the middle, by default False.
    **hyperparams
        The hyperparameters.

    Returns
    -------
    pd.DataFrame
        The calculated model.
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

    best_model = CatBoostClassifier(**hyperparams, allow_writing_files=False)

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
        target_series=data_frame["Target"].reindex(target_index),
        fee=0,
        cutoff=cutoff,
        long_only=long_only,
        short_only=short_only,
    )

    mh2["cuttoff"] = cutoff
    index_splits = {
        "train": pd.Interval(train_set.index[0], train_set.index[-1]),
        "test": pd.Interval(test_set.index[0], test_set.index[-1]),
        "validation": pd.Interval(validation_set.index[0], validation_set.index[-1]),
    }
    if output == "All":
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
    elif output == "Return":
        return mh2, index_splits
    elif output == "Model":
        return best_model
    elif output == "Dataset":
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
    else:
        raise ValueError("Invalid output parameter")

