import itertools
from typing import Literal

import pandas as pd
import numpy as np
import klib
from catboost import CatBoostClassifier, Pool
import sklearn.metrics as metrics
from utils.utils import model_metrics

from machine_learning.model_features import ModelFeatures
from machine_learning.model_handler import ModelHandler

def create_catboost_model(
    dataset: pd.DataFrame,
    feats: list,
    test_index: int = 1000,
    plot: bool = False,
    output: Literal["All", "Return", "Model", "Dataset"] = "All",
    **kwargs,
):
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
    plot : bool, optional
        Whether to plot the evaluation set during model training.
        (default: False)
    output : Literal["All", "Return", "Model", "Dataset"], optional
        Output parameter to specify the desired return values.
        (default: "All")
    **kwargs : dict, optional
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

    x_test = test_set[features]
    y_test = test_set[target]

    X_validation = validation_set[features]
    y_validation = validation_set[target]

    all_x = pd.concat([X_train, x_test, X_validation])
    all_y = pd.concat([y_train, y_test, y_validation])

    if kwargs:
        params = kwargs
    else:
        params = {
            "iterations": 500,
            "learning_rate": 0.1,
            "eval_metric": "Logloss",
            "random_seed": 69,
            "logging_level": "Silent",
            "use_best_model": True,
        }

    train_pool = Pool(X_train, y_train)
    test_pool = Pool(x_test, y_test)

    best_model = CatBoostClassifier(**params)

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
        data_frame["Target"].reindex(target_index),
        0,
        cutoff=cutoff,
        long_only=False,
    )
    mh2["cuttoff"] = cutoff
    index_splits = {
        "train": pd.Interval(train_set.index[0], train_set.index[-1]),
        "test": pd.Interval(test_set.index[0], test_set.index[-1]),
        "validation": pd.Interval(
            validation_set.index[0], validation_set.index[-1]
        ),
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

    if output == "Return":

        return mh2, index_splits

    if output == "Model":
        return best_model

    if output == "Dataset":
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

    raise ValueError("Invalid output parameter")

def adjust_max_trades(
    dataframe: pd.DataFrame,
    off_days: int,
    max_trades: int,
    pct_adj: float,
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
    original_datafrane = dataframe.copy()
    data_frame = dataframe.copy()
    for idx, row in data_frame.iloc[max_trades:].iterrows():
        if row["Predict"] != 0:
            three_lag_days = data_frame.loc[:idx].iloc[-(max_trades + 1) : -1]
            three_lag_days_trades = three_lag_days["Predict"].abs().sum()
            if three_lag_days_trades >= max_trades:
                data_frame.loc[idx:, "Predict"].iloc[0:off_days] = 0

    data_frame["y_pred_probs"] = np.where(
        data_frame["Predict"] == 0, 0, data_frame["y_pred_probs"]
    )
    data_frame["Position"] = data_frame["Predict"].shift().fillna(0)

    data_frame = data_frame.iloc[:, :6]

    data_frame["Liquid_Result"] = np.where(
        data_frame["Predict"] == 0, 0, original_datafrane["Liquid_Result"]
    ) / max_trades + 1

    data_frame["Liquid_Result_pct_adj"] = np.where(
        data_frame["Predict"] == 0, 0, original_datafrane["Liquid_Result"]
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

