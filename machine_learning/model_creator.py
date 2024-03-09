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
    test_index: int,
    target_series: pd.Series,
    plot: bool = False,
    output: Literal["All", "Return", "Model", "Dataset"] = "All",
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

    x_test = test_set[features]
    y_test = test_set[target]

    X_validation = validation_set[features]
    y_validation = validation_set[target]

    all_x = pd.concat([X_train, x_test, X_validation])
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

    train_pool = Pool(X_train, y_train)
    test_pool = Pool(x_test, y_test)

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
        The side of the trade to adjust (default is 1).

    Returns:
    -------
    pd.Series
        The adjusted series with maximum trades on one side.
    """
    target = np.where(predict == side, predict, 0)
    for idx in range(max_trades, len(predict)):
        if predict[idx] != 0:
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
        if ~isinstance(side, int):
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

def perform_general_random_search(
    dataframe: pd.DataFrame,
    test_index: int,
    max_trades:int = 3,
    off_days:int = 4,
    pct_adj:float = 0.5,
) -> dict:
    """
    Perform a general random search for model creation.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the data for model creation.
    test_index : int
        The index of the test data.
    max_trades : int, optional
        The maximum number of trades, by default 3.
    off_days : int, optional
        The number of off days, by default 4.
    pct_adj : float, optional
        The percentage adjustment, by default 0.5.

    Returns:
    -------
    dict
        A dictionary containing the model creation results.
    """
    # DTW
    ma_types = ["sma", "ema", "dema", "tema", "rma"]

    combinations_list = []
    for r in range(1, len(ma_types) + 1):
        combinations_list.extend(itertools.combinations(ma_types, r))

    ma_type_combinations = np.array(
        list(combinations_list) + ["all"], dtype="object"
    )
    random_variables = {
        # DTW
        "random_source_price_dtw": np.random.choice(["open", "high", "low"]),
        "random_binnings_qty_dtw": np.random.choice(range(10, 30)),
        "random_moving_averages": np.random.choice(ma_type_combinations),
        "random_moving_averages_length": np.random.choice(range(2, 300)),
        # RSI
        "random_source_price_rsi": np.random.choice(["open", "high", "low"]),
        "random_binnings_qty_rsi": np.random.choice(range(10, 30)),
        "random_rsi_length": np.random.choice(range(2, 150)),
        # STOCH
        "random_source_price_stoch": np.random.choice(["open", "high", "low"]),
        "random_binnings_qty_stoch": np.random.choice(range(10, 30)),
        "random_slow_stoch_length": np.random.choice(range(2, 50)),
        "random_slow_stoch_k": np.random.choice(range(1, 10)),
        "random_slow_stoch_d": np.random.choice(range(2, 10)),
        "random_features": np.random.choice(
            np.array([["RSI", "Stoch"], ["RSI"], ["Stoch"]], dtype="object")
        ),
    }

    hyperparams = {
        "iterations": 1000,
        "learning_rate": np.random.choice(np.arange(0.01, 1.01, 0.01)),
        "depth": np.random.choice(range(1, 17, 1)),
        "min_child_samples": np.random.choice(range(1, 21, 1)),
        "colsample_bylevel": np.random.choice(np.arange(0.1, 1.01, 0.01)),
        "subsample": np.random.choice(np.arange(0.1, 1.01, 0.01)),
        "reg_lambda": np.random.choice(range(1, 206, 1)),
        "use_best_model": True,
        "eval_metric": np.random.choice(
            ["Logloss", "AUC", "F1", "Precision", "Recall", "PRAUC"]
        ),
        "random_seed": np.random.choice(range(1, 50_001, 1)),
        "silent": True,
    }

    kraken_df2 = dataframe.copy()
    kraken_df2["slow_stoch_source"] = kraken_df2[
        random_variables["random_source_price_stoch"]
    ].pct_change(1)

    dtw_source = (
        kraken_df2[random_variables["random_source_price_dtw"]]
        .pct_change(1)
        .iloc[1:]
    )

    rsi_source = (
        kraken_df2[random_variables["random_source_price_rsi"]]
        .pct_change(1)
        .iloc[1:]
    )

    kraken_df2 = ModelFeatures(
        kraken_df2, test_index, random_variables["random_binnings_qty_dtw"]
    ).create_dtw_distance_feature(
        dtw_source,
        random_variables["random_moving_averages"],
        random_variables["random_moving_averages_length"],
    )

    if "RSI" in random_variables["random_features"]:
        kraken_df2 = ModelFeatures(
            kraken_df2, test_index, random_variables["random_binnings_qty_rsi"]
        ).create_rsi_feature(rsi_source, random_variables["random_rsi_length"])

    if "Stoch" in random_variables["random_features"]:
        kraken_df2 = ModelFeatures(
            kraken_df2, test_index, random_variables["random_binnings_qty_stoch"]
        ).create_slow_stoch_feature(
            random_variables["random_source_price_stoch"],
            random_variables["random_slow_stoch_length"],
            random_variables["random_slow_stoch_k"],
            random_variables["random_slow_stoch_d"],
        )

    dtw_features = [
        feature
        for feature in kraken_df2.columns[9:]
        if feature.endswith("feat") and "DTW" in feature
    ]

    other_features = [
        feature
        for feature in kraken_df2.columns[9:]
        if feature.endswith("feat") and "DTW" not in feature
    ]

    other_features_list = []
    for r in range(1, len(other_features) + 1):
        other_features_list.extend(itertools.combinations(other_features, r))

    other_features_array = np.array(list(other_features_list), dtype="object")

    if len(other_features_array) == 1:
        features = np.random.choice(other_features_array[0])
        features = list((features,) + tuple(dtw_features))
    else:
        features = np.random.choice(other_features_array)
        features = list(features + tuple(dtw_features))

    (
        mh2,
        _,_,_,
        y_train, y_test,
        y_pred_train, y_pred_test,
        _, all_y,
        index_splits,
    ) = create_catboost_model(
        kraken_df2,
        features,
        plot=False,
        output="All",
        **hyperparams
    )

    data_set = mh2.copy()

    mta = adjust_max_trades(data_set, off_days, max_trades, pct_adj)
    drawdowns = mta[["Drawdown", "Drawdown_pct_adj"]].quantile(0.95).to_list()

    y_train = all_y.loc[
        index_splits["train"].left : index_splits["train"].right
    ]

    y_test = all_y.loc[
        index_splits["test"].left : index_splits["test"].right
    ]

    y_val = all_y.loc[
        index_splits["validation"].left : index_splits["validation"].right
    ][:-7]

    y_pred_train = (
        mta[["Predict"]]
        .loc[index_splits["train"].left : index_splits["train"].right]
        .query("Predict != 0")
        .where(mta["Predict"] == 1, 0)
    )
    y_pred_test = (
        mta[["Predict"]]
        .loc[index_splits["test"].left : index_splits["test"].right]
        .query("Predict != 0")
        .where(mta["Predict"] == 1, 0)
    )
    y_pred_val = (
        mta[["Predict"]]
        .loc[
            index_splits["validation"].left : index_splits["validation"].right
        ][:-7]
        .query("Predict != 0")
        .where(mta["Predict"] == 1, 0)
    )

    y_train_adj = y_train.reindex(y_pred_train.index)
    y_test_adj = y_test.reindex(y_pred_test.index)
    y_val_adj = y_val.reindex(y_pred_val.index)

    report_train = metrics.classification_report(
        y_train_adj, y_pred_train, output_dict=True, zero_division=0
    )
    report_test = metrics.classification_report(
        y_test_adj, y_pred_test, output_dict=True, zero_division=0
    )

    report_val = metrics.classification_report(
        y_val_adj, y_pred_val, output_dict=True, zero_division=0
    )

    model_metrics_test = model_metrics(
        y_pred_test.iloc[:, -1],
        y_test_adj.iloc[:, -1],
    )

    model_metrics_val = model_metrics(
        y_pred_val.iloc[:, -1],
        y_val_adj.iloc[:, -1],
    )

    support_diff_test = model_metrics_test["support_diff"][-1]
    support_diff_val = model_metrics_val["support_diff"][-1]
    support_diffs = support_diff_test, support_diff_val

    accuracys = report_test["accuracy"], report_val["accuracy"]

    precisions_test = (
        report_test["0.0"]["precision"], report_test["1.0"]["precision"]
    )

    precisions_val = (
        report_val["0.0"]["precision"], report_val["1.0"]["precision"]
    )

    precisions = precisions_test + precisions_val

    metrics_results = {
        "accuracy_test": report_test["accuracy"],
        "accuracy_val": report_val["accuracy"],
        "precisions_test": precisions_test,
        "precisions_val": precisions_val,
        "precisions": precisions,
        "support_diff_test": support_diff_test,
        "support_diff_val": support_diff_val,
    }

    reports = {
        "report_train": report_train,
        "report_test": report_test,
        "report_val": report_val,
    }

    metrics_reports = {
        "metrics_reports_test" : model_metrics_test.iloc[-1].to_dict(),
        "metrics_reports_val" : model_metrics_val.iloc[-1].to_dict(),
    }

    has_any_low_precision = min(precisions) <= 0.5

    has_any_low_accuracy = min(accuracys) <= 0.5

    has_big_drawdown = max(drawdowns) >= 0.5

    has_high_support_diff = (
        max(support_diffs) >= 0.35
        and min(support_diffs) <= -0.35
    )

    if (
        has_any_low_precision
        or has_any_low_accuracy
        or has_big_drawdown
        or has_high_support_diff
    ):
        return {
            "features_selected": None,
            "feat_parameters": None,
            "hyperparameters": None,
            "metrics_results": None,
            "drawdown_full": None,
            "drawdown_adj": None,
            "reports": None,
        }

    return {
        "features_selected": [features],
        "feat_parameters": [random_variables],
        "hyperparameters": [hyperparams],
        "metrics_results": [metrics_results],
        "drawdown_full": drawdowns[0],
        "drawdown_adj": drawdowns[1],
        "reports": [reports],
        "model_reports": [metrics_reports],
    }
