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
    cutoff_point: float | None = None,
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

    if cutoff_point:
        if cutoff_point >= 100:
            raise ValueError("Cutoff point must be less than 100")
        if cutoff_point <= 0:
            raise ValueError("Cutoff point must be greater than 0")

        predict_mask = predict > cutoff

        cutoff = np.percentile(predict[predict_mask], cutoff_point)

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

def model_creation(
    feat_parameters: dict,
    hyperparams: dict,
    test_index: int,
    dataframe: pd.DataFrame,
    max_trades: int = 3,
    off_days: int = 7,
    pct_adj: float = 0.5,
    train_in_middle: bool = True,
    cutoff_point: float | None = None,
) -> tuple[pd.DataFrame, dict, pd.Series]:
    """
    Calculate and create the model based on the input parameters.

    Parameters
    ----------
    feat_parameters : dict
        Dictionary containing the parameters for the random features.
    hyperparams : dict
        Dictionary containing the hyperparameters for the model.
    test_index : int
        Index to split the dataset into train and test sets.
    dataframe : pd.DataFrame
        Input dataset.
    max_trades : int, optional
        Maximum number of trades to consider before waiting for off days
        (default: 3)
    off_days : int, optional
        When max_trades is reached, the number of days to wait before
        opening a new trade.
        (default: 7)
    pct_adj : float, optional
        Percentage adjustment to apply to the liquid results.
        (default: 0.5)
    train_in_middle : bool, optional
        Whether to train the model in the middle of the dataset
        (default: True)

    Returns
    -------
    tuple : pd.DataFrame, dict, pd.Series
        A tuple containing the adjusted model, index splits, and target
        values.
    """
    data_frame = dataframe.copy()

    if "DTW" in feat_parameters["random_features"]:
        dtw_source = (
            data_frame[feat_parameters["random_source_price_dtw"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_dtw"]
        ).create_dtw_distance_feature(
            dtw_source,
            feat_parameters["random_moving_averages"],
            feat_parameters["random_moving_averages_length"],
        )

    if "DTW_opt" in feat_parameters["random_features"]:
        dtw_source = (
            data_frame[feat_parameters["random_source_price_dtw"]]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_dtw"]
        ).create_dtw_distance_opt_feature(
            dtw_source,
            feat_parameters["random_moving_averages"],
            feat_parameters["random_moving_averages_length"],
        )

    if "RSI" in feat_parameters["random_features"]:
        rsi_source = (
            data_frame[feat_parameters["random_source_price_rsi"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_rsi"]
        ).create_rsi_feature(rsi_source, feat_parameters["random_rsi_length"])

    if "RSI_opt" in feat_parameters["random_features"]:
        rsi_source = (
            data_frame[feat_parameters["random_source_price_rsi"]]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_rsi"]
        ).create_rsi_opt_feature(rsi_source, feat_parameters["random_rsi_length"], feat_parameters["random_rsi_ma_method"])

    if "Stoch" in feat_parameters["random_features"]:
        data_frame["slow_stoch_source"] = (
            data_frame[feat_parameters["random_source_price_stoch"]]
            .pct_change(1)
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_stoch"]
        ).create_slow_stoch_feature(
            feat_parameters["random_source_price_stoch"],
            feat_parameters["random_slow_stoch_length"],
            feat_parameters["random_slow_stoch_k"],
            feat_parameters["random_slow_stoch_d"],
        )

    if "Stoch_opt" in feat_parameters["random_features"]:
        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_stoch"]
        ).create_slow_stoch_opt_feature(
            feat_parameters["random_source_price_stoch"],
            feat_parameters["random_slow_stoch_length"],
            feat_parameters["random_slow_stoch_k"],
            feat_parameters["random_slow_stoch_d"],
            feat_parameters["random_slow_stoch_ma_method"],
        )

    if "CCI" in feat_parameters["random_features"]:
        cci_source = (
            data_frame[feat_parameters["random_source_price_cci"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_cci"]
        ).create_cci_feature(
            cci_source,
            feat_parameters["random_cci_length"],
            feat_parameters["random_cci_method"],
        )

    if "CCI_clean" in feat_parameters["random_features"]:
        cci_source = (
            data_frame[feat_parameters["random_source_price_cci"]]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_cci"]
        ).create_cci_feature(
            cci_source,
            feat_parameters["random_cci_length"],
            feat_parameters["random_cci_method"],
        )

    if "MACD" in feat_parameters["random_features"]:
        macd_source = (
            data_frame[feat_parameters["random_source_price_macd"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_macd"]
        ).create_macd_feature(
            macd_source,
            feat_parameters["random_macd_fast_length"],
            feat_parameters["random_macd_slow_length"],
            feat_parameters["random_macd_signal_length"],
            feat_parameters["random_macd_diff_method"],
            feat_parameters["random_macd_ma_method"],
            feat_parameters["random_macd_signal_method"],
            feat_parameters["random_macd_column"],
        )


    if "MACD_std" in feat_parameters["random_features"]:
        macd_source = (
            data_frame[feat_parameters["random_source_price_macd"]]
            .rolling(2)
            .std()
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_macd"]
        ).create_macd_feature(
            macd_source,
            feat_parameters["random_macd_fast_length"],
            feat_parameters["random_macd_slow_length"],
            feat_parameters["random_macd_signal_length"],
            feat_parameters["random_macd_diff_method"],
            feat_parameters["random_macd_ma_method"],
            feat_parameters["random_macd_signal_method"],
            feat_parameters["random_macd_column"],
        )

    if "MACD_clean" in feat_parameters["random_features"]:
        macd_source = (
            data_frame[feat_parameters["random_source_price_macd"]]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_macd"]
        ).create_macd_feature(
            macd_source,
            feat_parameters["random_macd_fast_length"],
            feat_parameters["random_macd_slow_length"],
            feat_parameters["random_macd_signal_length"],
            feat_parameters["random_macd_diff_method"],
            feat_parameters["random_macd_ma_method"],
            feat_parameters["random_macd_signal_method"],
            feat_parameters["random_macd_column"],
        )

    if "TRIX" in feat_parameters["random_features"]:
        trix_source = data_frame[feat_parameters["random_source_price_trix"]]

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_trix"]
        ).create_trix_feature(
            trix_source,
            feat_parameters["random_trix_length"],
            feat_parameters["random_trix_signal_length"],
            feat_parameters["random_trix_ma_method"],
        )

    if "TRIX_opt" in feat_parameters["random_features"]:
        trix_source = data_frame[feat_parameters["random_source_price_trix"]]

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_trix"]
        ).create_trix_opt_feature(
            trix_source,
            feat_parameters["random_trix_length"],
            feat_parameters["random_trix_signal_length"],
            feat_parameters["random_trix_ma_method"],
        )

    if "SMIO" in feat_parameters["random_features"]:
        smio_source = data_frame[feat_parameters["random_source_price_smio"]]

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_smio"]
        ).create_smio_feature(
            smio_source,
            feat_parameters["random_smio_short_length"],
            feat_parameters["random_smio_long_length"],
            feat_parameters["random_smio_signal_length"],
            feat_parameters["random_smio_ma_method"],
        )

    if "DIDI" in feat_parameters["random_features"]:
        didi_source = (
            data_frame[feat_parameters["random_source_price_didi"]]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_didi"]
        ).create_didi_index_feature(
            source=didi_source,
            short_length=feat_parameters["random_didi_short_length"],
            medium_length=feat_parameters["random_didi_mid_length"],
            long_length=feat_parameters["random_didi_long_length"],
            ma_type=feat_parameters["random_didi_ma_type"],
            method=feat_parameters["random_didi_method"],
        )

    if "TSI" in feat_parameters["random_features"]:
        tsi_source = data_frame[feat_parameters["random_source_price_tsi"]]

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_tsi"]
        ).create_tsi_feature(
            tsi_source,
            feat_parameters["random_tsi_short_length"],
            feat_parameters["random_tsi_long_length"],
            feat_parameters["random_tsi_ma_method"],
        )

    if "TSI_opt" in feat_parameters["random_features"]:
        tsi_source = data_frame[feat_parameters["random_source_price_tsi"]]

        data_frame = ModelFeatures(
            data_frame, test_index, feat_parameters["random_binnings_qty_tsi"]
        ).create_tsi_opt_feature(
            tsi_source,
            feat_parameters["random_tsi_short_length"],
            feat_parameters["random_tsi_long_length"],
            feat_parameters["random_tsi_ma_method"],
        )

    if "Ichimoku" in feat_parameters["random_features"]:
        data_frame = ModelFeatures(
            data_frame,
            test_index,
            feat_parameters["random_binnings_qty_ichimoku"],
        ).create_ichimoku_feature(
            feat_parameters["random_ichimoku_conversion_periods"],
            feat_parameters["random_ichimoku_base_periods"],
            feat_parameters["random_ichimoku_lagging_span_2_periods"],
            feat_parameters["random_ichimoku_displacement"],
            feat_parameters["random_ichimoku_based_on"],
            feat_parameters["random_ichimoku_method"],
        )

    if "Ichimoku Price Distance" in feat_parameters["random_features"]:
        ichimoku_source = data_frame[feat_parameters["random_source_ichimoku_price_distance"]]

        data_frame = ModelFeatures(
            data_frame,
            test_index,
            feat_parameters["random_binnings_qty_ichimoku_price_distance"],
        ).create_ichimoku_price_distance_feature(
            ichimoku_source,
            feat_parameters["random_ichimoku_price_distance_conversion_periods"],
            feat_parameters["random_ichimoku_price_distance_base_periods"],
            feat_parameters["random_ichimoku_price_distance_lagging_span_2_periods"],
            feat_parameters["random_ichimoku_price_distance_displacement"],
            feat_parameters["random_ichimoku_price_distance_based_on"],
            feat_parameters["random_ichimoku_price_distance_method"],
            feat_parameters["random_ichimoku_price_distance_use_pct"],
        )

    if "BBTrend" in feat_parameters["random_features"]:
        bb_source = data_frame[feat_parameters["random_source_bb_trend"]]

        data_frame = ModelFeatures(
            data_frame,
            test_index,
            feat_parameters["random_binnings_qty_bb_trend"],
        ).create_bb_trend_feature(
            bb_source,
            feat_parameters["random_bb_trend_short_length"],
            feat_parameters["random_bb_trend_long_length"],
            feat_parameters["random_bb_trend_stdev"],
            feat_parameters["random_bb_trend_ma_method"],
            feat_parameters["random_bb_trend_stdev_method"],
            feat_parameters["random_bb_trend_diff_method"],
            feat_parameters["random_bb_trend_based_on"],
        )

    df_columns = data_frame.columns.tolist()
    features = [x for x in df_columns if "feat" in x]

    mh2, _, _, _, _, _, _, _, _, all_y, index_splits = calculate_model(
        dataset=data_frame,
        feats=features,
        test_index=test_index,
        plot=False,
        output="All",
        long_only=False,
        train_in_middle=train_in_middle,
        cutoff_point=cutoff_point,
        **hyperparams,
    )

    mh2["Liquid_Result"] = np.where(mh2["Predict"] == -1, 0, mh2["Liquid_Result"])

    return (
        max_trade_adj(mh2, off_days, max_trades, pct_adj),
        index_splits,
        all_y
    )
