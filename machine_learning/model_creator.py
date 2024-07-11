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

def perform_general_random_search(
    dataframe: pd.DataFrame,
    test_index: int,
    side: Literal["both"] | int = 1,
    max_trades: int = 3,
    off_days: int = 7,
    pct_adj: float = 0.5,
    train_in_middle: bool = True,
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

    empty_dict = {
        "features_selected": None,
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
    }

    data_frame = dataframe.copy()
    data_frame["slow_stoch_source"] = data_frame[
        random_variables["random_source_price_stoch"]
    ].pct_change(1)

    dtw_source = (
        data_frame[random_variables["random_source_price_dtw"]]
        .pct_change(1).iloc[1:]
    )

    rsi_source = (
        data_frame[random_variables["random_source_price_rsi"]]
        .pct_change(1).iloc[1:]
    )

    data_frame = ModelFeatures(
        data_frame, test_index, random_variables["random_binnings_qty_dtw"]
    ).create_dtw_distance_feature(
        dtw_source,
        random_variables["random_moving_averages"],
        random_variables["random_moving_averages_length"],
    )

    if "RSI" in random_variables["random_features"]:
        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_rsi"]
        ).create_rsi_feature(rsi_source, random_variables["random_rsi_length"])

    if "Stoch" in random_variables["random_features"]:
        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_stoch"]
        ).create_slow_stoch_feature(
            random_variables["random_source_price_stoch"],
            random_variables["random_slow_stoch_length"],
            random_variables["random_slow_stoch_k"],
            random_variables["random_slow_stoch_d"],
        )

    dtw_features = [
        feature
        for feature in data_frame.columns[9:]
        if feature.endswith("feat") and "DTW" in feature
    ]

    other_features = [
        feature
        for feature in data_frame.columns[9:]
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
        _,
        _,
        _,
        _,
        y_test,
        _,
        y_pred_test,
        _,
        all_y,
        index_splits,
    ) = create_catboost_model(
        dataset=data_frame,
        feats=features,
        test_index=test_index,
        target_series=data_frame["Target_bin"],
        plot=False,
        output="All",
        long_only=False,
        train_in_middle=train_in_middle,
        **hyperparams,
    )

    mh2['Liquid_Result'] = np.where(
        mh2['Predict'] == -1, 0, mh2['Liquid_Result']
    )

    mta = adjust_max_trades(mh2, off_days, max_trades, pct_adj, side)

    val_periods = (
        index_splits['validation'].left,
        index_splits['validation'].right,
    )

    if train_in_middle:
        test_periods = (index_splits['train'].left,index_splits['train'].right)
        mta_test = mta.loc[test_periods[0]:test_periods[1]]
    else:
        test_periods = (index_splits['test'].left,index_splits['test'].right)
        mta_test = mta.loc[test_periods[0]:test_periods[1]]

    mta_val = mta.loc[val_periods[0]:val_periods[1]]

    test_buys = mta_test['Predict'].value_counts()
    val_buys = mta_val['Predict'].value_counts()

    if (
        len(test_buys) <= 1 or len(val_buys) <= 1
        or len(mta_test["Liquid_Result"].value_counts()) <= 1
        or len(mta_val["Liquid_Result"].value_counts()) <= 1
    ):
        return empty_dict

    total_operations = (
        test_buys[1],
        val_buys[1],
    )

    total_operations_test_pct = test_buys[1] / (1041 // 3) * 100
    total_operations_val_pct = val_buys[1] / (1041 // 3) * 100

    total_operations_pct = (
        total_operations_test_pct,
        total_operations_val_pct,
    )

    drawdown_full_test = (
        (mta_test['Liquid_Return'].cummax() - mta_test['Liquid_Return'])
        / mta_test['Liquid_Return'].cummax()
    )

    drawdown_full_val = (
        (mta_val['Liquid_Return'].cummax() - mta_val['Liquid_Return'])
        / mta_val['Liquid_Return'].cummax()
    )

    drawdown_adj_test = (
        (
            mta_test['Liquid_Return_pct_adj'].cummax()
            - mta_test['Liquid_Return_pct_adj']
        )
        / mta_test['Liquid_Return_pct_adj'].cummax()
    )

    drawdown_adj_val = (
        (
            mta_val['Liquid_Return_pct_adj'].cummax()
            - mta_val['Liquid_Return_pct_adj']
        )
        / mta_val['Liquid_Return_pct_adj'].cummax()
    )

    drawdowns = (
        drawdown_full_test.quantile(0.95),
        drawdown_full_val.quantile(0.95),
        drawdown_adj_test.quantile(0.95),
        drawdown_adj_val.quantile(0.95),
    )

    y_test = all_y.loc[test_periods[0]:test_periods[1]]

    y_val = all_y.loc[val_periods[0]:val_periods[1]][:-7]

    y_pred_test = (
        mta[["Predict"]]
        .loc[test_periods[0] : test_periods[1]]
        .query("Predict != 0")
        .where(mta["Predict"] == 1, 0)
    )
    y_pred_val = (
        mta[["Predict"]]
        .loc[val_periods[0]:val_periods[1]][:-7]
        .query("Predict != 0")
        .where(mta["Predict"] == 1, 0)
    )

    y_test_adj = y_test.reindex(y_pred_test.index)
    y_val_adj = y_val.reindex(y_pred_val.index)

    if side != "both":
        adj_targets = adjust_predict_one_side(
            data_frame['Target_bin'],
            3,
            7,
            side,
        )
    else:
        adj_targets = adjust_predict_both_side(data_frame, 7, 3)

    test_predict_adj = adj_targets.loc[test_periods[0]:test_periods[1]]
    val_predict_adj = adj_targets.loc[val_periods[0]:val_periods[1]]

    support_diff_test = (
        test_predict_adj.value_counts(normalize=True)
        - mta_test['Predict'].value_counts(normalize=True)
    )[1]

    support_diff_val = (
        val_predict_adj.value_counts(normalize=True)
        - mta_val['Predict'].value_counts(normalize=True)
    )[1]

    support_diffs = (support_diff_test, support_diff_val)

    result_metrics_test = (
        DataHandler(mta.loc[y_test_adj.index])
        .result_metrics('Liquid_Result', is_percentage_data=True)
    )

    result_metrics_val = (
        DataHandler(mta.loc[y_val_adj.index])
        .result_metrics('Liquid_Result', is_percentage_data=True)
    )

    precisions = (
        result_metrics_test.loc['Win_Rate'][0],
        result_metrics_val.loc['Win_Rate'][0],
    )

    expected_return = (
        result_metrics_test.loc['Expected_Return'][0],
        result_metrics_val.loc['Expected_Return'][0],
    )

    metrics_results = {
        "expected_return_test": expected_return[0],
        "expected_return_val": expected_return[1],
        "precisions_test": precisions[0],
        "precisions_val": precisions[1],
        "precisions": precisions,
    }

    has_any_low_precision = min(precisions) <= 0.5
    if (
        has_any_low_precision
    ):
        return empty_dict

    return {
        "features_selected": [features],
        "feat_parameters": [random_variables],
        "hyperparameters": [hyperparams],
        "metrics_results": [metrics_results],
        "drawdown_full_test": drawdowns[0],
        "drawdown_full_val": drawdowns[1],
        "drawdown_adj_test": drawdowns[2],
        "drawdown_adj_val": drawdowns[3],
        "expected_return_test": expected_return[0],
        "expected_return_val": expected_return[1],
        "precisions_test": precisions[0],
        "precisions_val": precisions[1],
        "support_diff_test": support_diffs[0],
        "support_diff_val": support_diffs[1],
        "total_operations_test": total_operations[0],
        "total_operations_val": total_operations[1],
        "total_operations_pct_test": total_operations_pct[0],
        "total_operations_pct_val": total_operations_pct[1],
    }
