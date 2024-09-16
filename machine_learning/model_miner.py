from typing import Any
import itertools
import time

import pandas as pd
import numpy as np

from machine_learning.ml_utils import DataHandler
from machine_learning.model_builder import model_creation
from machine_learning.model_creator import (
    adjust_predict_one_side,
)
from machine_learning.ols_data import calculate_r2, calculate_coef

from return_statistics import Statistics

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
        self.ohlc: list[str] = ["open", "high", "low", "close"]
        self.max_trades: int = max_trades
        self.off_days: int = off_days
        self.side: int = side

        self.dataframe = dataframe.copy()
        self.target = target.copy()

        self.ma_types: list[str] = ["sma", "ema", "rma"]

        combinations_list = []
        for r in range(1, len(self.ma_types) + 1):
            combinations_list.extend(itertools.combinations(self.ma_types, r))

        self.ma_type_combinations = np.array(
            list(combinations_list) + ["all"],
            dtype="object",
        )

        features: list[str] = [
            "RSI_opt",
            "Stoch_opt",
            "CCI",
            "MACD_opt",
            "SMIO_opt",
            # "TRIX_opt",
            # "DTW_opt",
            "TSI_opt",
            "DIDI_opt",
            # "Ichimoku",
            # "Ichimoku Price Distance",
            "BBTrend_opt"
        ]

        self.random_features = []
        for x in range(1, len(features) + 1):
            self.random_features += list(itertools.combinations(features, x))

        self.random_features = np.array(self.random_features, dtype="object")

        self.adj_targets = adjust_predict_one_side(self.target, max_trades, off_days, side)

        self.empty_dict: dict[str, None] = {
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
            "ols_coef_2022": None,
            "ols_coef_val": None,
            "test_index": None,
            "total_time": None,
            "return_ratios": None,
            "side": None,
            "max_trades": None,
            "off_days": None,
        }

    def generate_feat_parameters(self):
        """
        Generate a dictionary of random feature parameters for the model.
        Returns
        -------
        dict
            A dictionary containing the randomly generated feature
            parameters for the model.
        Notes
        -----
        This method generates random values for various feature
        parameters used in the model. The generated parameters include:
        - General parameters:
            - random_features: A list of randomly chosen features.
        - DTW parameters:
            - random_source_price_dtw: A randomly chosen source price
            for DTW.
            - random_binnings_qty_dtw: A randomly chosen number of
            binnings for DTW.
            - random_moving_averages: A randomly chosen moving average
            type for DTW.
            - random_moving_averages_length: A randomly chosen length
            for moving averages in DTW.
        - RSI parameters:
            - random_source_price_rsi: A randomly chosen source price
            for RSI.
            - random_binnings_qty_rsi: A randomly chosen number of
            binnings for RSI.
            - random_rsi_length: A randomly chosen length for RSI.
            - random_rsi_ma_method: A randomly chosen moving average
            method for RSI.
        - STOCH parameters:
            - random_source_price_stoch: A randomly chosen source price
            for STOCH.
            - random_binnings_qty_stoch: A randomly chosen number of
            binnings for STOCH.
            - random_slow_stoch_length: A randomly chosen length for
            slow STOCH.
            - random_slow_stoch_k: A randomly chosen value for K in
            slow STOCH.
            - random_slow_stoch_d: A randomly chosen value for D in
            slow STOCH.
            - random_slow_stoch_ma_method: A randomly chosen moving
            average method for slow STOCH.
        - DIDI parameters:
            - random_source_price_didi: A randomly chosen source price
            for DIDI.
            - random_binnings_qty_didi: A randomly chosen number of
            binnings for DIDI.
            - random_didi_short_length: The minimum length among
            randomly chosen lengths for DIDI.
            - random_didi_mid_length: The median length among randomly
            chosen lengths for DIDI.
            - random_didi_long_length: The maximum length among randomly
            chosen lengths for DIDI.
            - random_didi_ma_type: A randomly chosen moving average type
            for DIDI.
            - random_didi_method: A randomly chosen distance method for
            DIDI.
        - CCI parameters:
            - random_source_price_cci: A randomly chosen source price
            for CCI.
            - random_binnings_qty_cci: A randomly chosen number of
            binnings for CCI.
            - random_cci_length: A randomly chosen length for CCI.
            - random_cci_method: A randomly chosen moving average method
            for CCI.
        - MACD parameters:
            - random_source_price_macd: A randomly chosen source price
            for MACD.
            - random_binnings_qty_macd: A randomly chosen number of
            binnings for MACD.
            - random_macd_fast_length: The maximum length among randomly
            chosen lengths for MACD.
            - random_macd_slow_length: The minimum length among randomly
            chosen lengths for MACD.
            - random_macd_signal_length: A randomly chosen length for
            MACD signal.
            - random_macd_ma_method: A randomly chosen moving average
            method for MACD.
            - random_macd_signal_method: A randomly chosen moving
            average method for MACD signal.
            - random_macd_column: A randomly chosen column for MACD.
        - TRIX parameters:
            - random_source_price_trix: A randomly chosen source price
            for TRIX.
            - random_binnings_qty_trix: A randomly chosen number of
            binnings for TRIX.
            - random_trix_length: A randomly chosen length for TRIX.
            - random_trix_signal_length: A randomly chosen length for
            TRIX signal.
            - random_trix_ma_method: A randomly chosen moving average
            method for TRIX.
        - SMIO parameters:
            - random_source_price_smio: A randomly chosen source price
            for SMIO.
            - random_binnings_qty_smio: A randomly chosen number of
            binnings for SMIO.
            - random_smio_short_length: The maximum length among
            randomly chosen lengths for SMIO.
            - random_smio_long_length: The minimum length among randomly
            chosen lengths for SMIO.
            - random_smio_signal_length: A randomly chosen length for
            SMIO signal.
            - random_smio_ma_method: A randomly chosen moving average
            method for SMIO.
        - TSI parameters:
            - random_source_price_tsi: A randomly chosen source price
            for TSI.
            - random_binnings_qty_tsi: A randomly chosen number of
            binnings for TSI.
            - random_tsi_short_length: The maximum length among randomly
            chosen lengths for TSI.
            - random_tsi_long_length: The minimum length among randomly
            chosen lengths for TSI.
            - random_tsi_ma_method: A randomly chosen moving average
            method for TSI.
        - BB Trend parameters:
            - random_source_bb_trend: A randomly chosen source price for
            BB Trend.
            - random_binnings_qty_bb_trend: A randomly chosen number of
            binnings for BB Trend.
            - random_bb_trend_short_length: A randomly chosen short
            length for BB Trend.
            - random_bb_trend_long_length: A randomly chosen long length
            for BB Trend.
            - random_bb_trend_stdev: A randomly chosen standard
            deviation for BB Trend.
            - random_bb_trend_ma_method: A randomly chosen moving
            average method for BB Trend.
            - random_bb_trend_stdev_method: A randomly chosen standard
            deviation method for BB Trend.
            - random_bb_trend_diff_method: A randomly chosen difference
            method for BB Trend.
            - random_bb_trend_based_on: A randomly chosen base method
            for BB Trend.
        """
        macd_lengths = np.random.choice(range(2, 151), 2, replace=False)
        smio_lengths = np.random.choice(range(2, 151), 2, replace=False)
        tsi_lengths = np.random.choice(range(2, 151), 2, replace=False)
        didi_ma_lengths = np.random.choice(range(2, 151), 3, replace=False)
        ichimoku_lengths = np.random.choice(range(2, 151), 2, replace=False)
        ichimoku_price_distance_lengths = np.random.choice(range(2, 151), 2, replace=False)
        bb_trend_short_length = np.random.choice(range(10, 20))
        bb_trend_long_length = (
            np.random.choice(range(1, 3))
            + bb_trend_short_length
        )

        fast_length = max(macd_lengths)
        slow_length = min(macd_lengths)
        signal_length = np.random.choice(range(2, 51))
        signal_length = (
            signal_length + 1 if signal_length in macd_lengths else signal_length
        )
        distance_types: list[str] = [
            "absolute",
            "ratio",
            # "dtw",
        ]

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
            "random_rsi_ma_method": np.random.choice(self.ma_types),
            # STOCH
            "random_source_price_stoch": np.random.choice(['open', 'close']),
            "random_binnings_qty_stoch": np.random.choice(range(10, 31)),
            "random_slow_stoch_length": np.random.choice(range(2, 51)),
            "random_slow_stoch_k": np.random.choice(range(1, 11)),
            "random_slow_stoch_d": np.random.choice(range(2, 11)),
            "random_slow_stoch_ma_method": np.random.choice(self.ma_types),
            # DIDI
            "random_source_price_didi": np.random.choice(self.ohlc),
            "random_binnings_qty_didi": np.random.choice(range(10, 31)),
            "random_didi_short_length": int(np.min(didi_ma_lengths)),
            "random_didi_mid_length": int(np.median(didi_ma_lengths)),
            "random_didi_long_length": int(np.max(didi_ma_lengths)),
            "random_didi_ma_type": np.random.choice(self.ma_types),
            "random_didi_method": np.random.choice(distance_types),
            # CCI
            "random_source_price_cci": np.random.choice(self.ohlc),
            "random_binnings_qty_cci": np.random.choice(range(10, 31)),
            "random_cci_length": np.random.choice(range(10, 151)),
            "random_cci_method": np.random.choice(self.ma_types),
            # MACD
            "random_source_price_macd": np.random.choice(self.ohlc),
            "random_binnings_qty_macd": np.random.choice(range(10, 31)),
            "random_macd_fast_length": fast_length,
            "random_macd_slow_length": slow_length,
            "random_macd_signal_length": signal_length,
            "random_macd_ma_method": np.random.choice(self.ma_types),
            "random_macd_signal_method": np.random.choice(self.ma_types),
            "random_macd_column": np.random.choice(["macd", "signal", "histogram"]),
            # TRIX
            "random_source_price_trix": np.random.choice(self.ohlc),
            "random_binnings_qty_trix": np.random.choice(range(10, 31)),
            "random_trix_length": np.random.choice(range(2, 11)),
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
            "random_tsi_ma_method": np.random.choice(["sma", "ema"]),
            # Ichimoku #Removed
            "random_binnings_qty_ichimoku": np.random.choice(range(10, 31)),
            "random_ichimoku_conversion_periods": ichimoku_lengths[0],
            "random_ichimoku_base_periods": ichimoku_lengths[1],
            "random_ichimoku_lagging_span_2_periods": np.random.choice(range(2, 31)),
            "random_ichimoku_displacement": np.random.choice(range(2, 31)),
            "random_ichimoku_based_on": np.random.choice(['lead_line', 'leading_span']),
            "random_ichimoku_method": np.random.choice(distance_types),
            # Ichimoku Price Distance #Removed
            "random_source_ichimoku_price_distance": np.random.choice(self.ohlc),
            "random_binnings_qty_ichimoku_price_distance": np.random.choice(range(10, 31)),
            "random_ichimoku_price_distance_conversion_periods": ichimoku_price_distance_lengths[0],
            "random_ichimoku_price_distance_base_periods": ichimoku_price_distance_lengths[1],
            "random_ichimoku_price_distance_lagging_span_2_periods": np.random.choice(range(2, 31)),
            "random_ichimoku_price_distance_displacement": np.random.choice(range(2, 31)),
            "random_ichimoku_price_distance_based_on": np.random.choice(['lead_line', 'leading_span']),
            "random_ichimoku_price_distance_method": np.random.choice(distance_types),
            "random_ichimoku_price_distance_use_pct": True,
            # BB Trend
            "random_source_bb_trend": np.random.choice(self.ohlc),
            "random_binnings_qty_bb_trend": np.random.choice(range(10, 31)),
            "random_bb_trend_short_length": bb_trend_short_length,
            "random_bb_trend_long_length": bb_trend_long_length,
            "random_bb_trend_stdev": np.random.choice(np.arange(1, 3.1, 0.1)),
            "random_bb_trend_ma_method": np.random.choice(['sma', 'ema']),
            "random_bb_trend_stdev_method": np.random.choice(['absolute']),
            "random_bb_trend_diff_method": np.random.choice(['normal']),
            "random_bb_trend_based_on": np.random.choice(['long_length']),
        }

    def generate_hyperparameters(self):
        """
        Generate a dictionary of hyperparameters for the model.

        Returns
        -------
        dict
            A dictionary containing the following hyperparameters:
            - iterations : int
                The number of iterations for the model.
            - learning_rate : float
                The learning rate for the model.
            - depth : int
                The depth of the model.
            - min_child_samples : int
                The minimum number of samples required to create a new
                node in the model.
            - colsample_bylevel : float
                The fraction of columns to be randomly selected for
                each level in the model.
            - subsample : float
                The fraction of samples to be randomly selected for
                each tree in the model.
            - reg_lambda : int
                The regularization lambda value for the model.
            - use_best_model : bool
                Whether to use the best model found during training.
            - eval_metric : str
                The evaluation metric to be used during training.
            - random_seed : int
                The random seed value for the model.
            - silent : bool
                Whether to print messages during training.

        """
        return {
            "iterations": 1000,
            "learning_rate": np.random.choice(np.arange(0.01, 1.01, 0.01)),
            "depth": np.random.choice(range(1, 12, 1)),
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

    def create_and_calculate_metrics(
        self,
        feat_parameters: dict,
        hyperparams: dict,
        test_index: int,
        pct_adj: float = 0.5,
        train_in_middle: bool = True,
        cutoff_point: float | None = None,
    ) -> dict[str, None] | dict[str, Any]:
        """
        create catboost catboost model and return the metrics.

        Parameters:
        ----------
        feat_parameters : dict
            The feature parameters for model creation.
        hyperparams : dict
            The hyperparameters for model creation.
        test_index : int
            The test index.
        pct_adj : float, optional
            The percentage to adjust the target
            (default: 0.5).
        train_in_middle : bool, optional
            Whether to train in the middle
            (default: True).
        cutoff_point : float or None, optional
            The cutoff point for the model
            (default: None).

        Returns:
        -------
        dict
            The dictionary containing the best model parameters.
        """
        start: float = time.perf_counter()

        try:
            mta, index_splits, all_y, _ = model_creation(
                feat_parameters,
                hyperparams,
                test_index,
                self.dataframe,
                self.max_trades,
                self.off_days,
                pct_adj,
                train_in_middle,
                cutoff_point,
                self.side,
                dev=True,
            )

            val_periods = (
                index_splits["validation"].left,
                index_splits["validation"].right,
            )

            if train_in_middle:
                test_periods = (index_splits["train"].left, index_splits["train"].right)
                mta_test = mta.loc[test_periods[0] : test_periods[1]]
            else:
                test_periods = (index_splits["test"].left, index_splits["test"].right)
                mta_test = mta.loc[test_periods[0] : test_periods[1]]

            mta_val = mta.loc[val_periods[0] : val_periods[1]]

            test_buys = mta_test["Predict"].value_counts()
            val_buys = mta_val["Predict"].value_counts()

            if (
                len(test_buys) <= 1
                or len(val_buys) <= 1
                or len(mta_test["Liquid_Result"].value_counts()) <= 1
                or len(mta_val["Liquid_Result"].value_counts()) <= 1
            ):
                return self.empty_dict

            # Criação do modelo
            y_test = all_y.loc[test_periods[0] : test_periods[1]]
            y_val = all_y.loc[val_periods[0] : val_periods[1]][:-7]

            y_pred_test = (
                mta[["Predict"]]
                .loc[test_periods[0] : test_periods[1]]
                .query("Predict != 0")
                .where(mta["Predict"] == 1, 0)
            )
            y_pred_val = (
                mta[["Predict"]]
                .loc[val_periods[0] : val_periods[1]][:-7]
                .query("Predict != 0")
                .where(mta["Predict"] == 1, 0)
            )

            y_test_adj = y_test.reindex(y_pred_test.index)
            y_val_adj = y_val.reindex(y_pred_val.index)

            result_metrics_test = DataHandler(mta.reindex(y_test_adj.index)).result_metrics(
                "Liquid_Result", is_percentage_data=True, output_format="Series"
            )

            result_metrics_val = DataHandler(mta.reindex(y_val_adj.index)).result_metrics(
                "Liquid_Result", is_percentage_data=True, output_format="Series"
            )

            precisions = (
                result_metrics_test["Win_Rate"],
                result_metrics_val["Win_Rate"],
            )

            expected_return = (
                result_metrics_test["Expected_Return"],
                result_metrics_val["Expected_Return"],
            )

            metrics_results = {
                "expected_return_test": expected_return[0],
                "expected_return_val": expected_return[1],
                "precisions_test": precisions[0],
                "precisions_val": precisions[1],
                "precisions": precisions,
            }

            if min(precisions) < 0.52:
                return self.empty_dict

            #Filter the results that doesn't have profit in bear market
            bearmarket_2022 = (
                mta.loc["2021-08-11":"2023-01-01", "Liquid_Result"]
                .cumprod()
            )

            try:
                r2_2022 = calculate_r2(bearmarket_2022)
                ols_coef_2022 = calculate_coef(bearmarket_2022)

                r2_val = calculate_r2(mta_val["Liquid_Result"].cumprod())
                ols_coef_val = calculate_coef(mta_val["Liquid_Result"].cumprod())

            except Exception:
                r2_2022 = -404
                r2_val = -404
                ols_coef_2022 = -404
                ols_coef_val = -404

            total_operations = (
                test_buys.loc[self.side],
                val_buys.loc[self.side],
            )

            total_operations_test_pct: float = total_operations[0] / (test_index // self.off_days)
            total_operations_val_pct: float = total_operations[1] / (test_index // self.off_days)

            total_operations_pct: tuple[float, float] = (
                total_operations_test_pct,
                total_operations_val_pct,
            )

            liquid_return_test = mta_test["Liquid_Result"].cumprod()
            liquid_return_val = mta_val["Liquid_Result"].cumprod()
            liquid_return_adj_test = mta_test["Liquid_Result_pct_adj"].cumprod()
            liquid_return_adj_val = mta_val["Liquid_Result_pct_adj"].cumprod()

            drawdown_full_test = (
                liquid_return_test.cummax() - liquid_return_test
            ) / liquid_return_test.cummax()

            drawdown_full_val = (
                liquid_return_val.cummax() - liquid_return_val
            ) / liquid_return_val.cummax()

            drawdown_adj_test = (
                liquid_return_adj_test.cummax() - liquid_return_adj_test
            ) / liquid_return_adj_test.cummax()

            drawdown_adj_val = (
                liquid_return_adj_val.cummax() - liquid_return_adj_val
            ) / liquid_return_adj_val.cummax()

            drawdowns = (
                drawdown_full_test.quantile(0.95),
                drawdown_full_val.quantile(0.95),
                drawdown_adj_test.quantile(0.95),
                drawdown_adj_val.quantile(0.95),
            )

            _, sharpe_test, sortino_test =  Statistics(
                dataframe = (mta_test["Liquid_Result"] - 1).drop_duplicates(),
                time_span = "Y",
                risk_free_rate = (1.12) ** (1/365.25) - 1,
                is_percent = True,
            ).calculate_all_statistics().mean()

            _, sharpe_val, sortino_val =  Statistics(
                dataframe = (mta_val["Liquid_Result"] - 1).drop_duplicates(),
                time_span = "Y",
                risk_free_rate = (1.12) ** (1/365.25) - 1,
                is_percent = True,
            ).calculate_all_statistics().mean()

            return_ratios = {
                "sharpe_test" : sharpe_test,
                "sharpe_val" : sharpe_val,
                "sortino_test" : sortino_test,
                "sortino_val" : sortino_val,
            }

            test_predict_adj = self.adj_targets.loc[test_periods[0] : test_periods[1]]
            val_predict_adj = self.adj_targets.loc[val_periods[0] : val_periods[1]]

            support_diff_test = (
                test_predict_adj.value_counts(normalize=True)
                - mta_test["Predict"].value_counts(normalize=True)
            )[self.side]

            support_diff_val = (
                val_predict_adj.value_counts(normalize=True)
                - mta_val["Predict"].value_counts(normalize=True)
            )[self.side]

            support_diffs = (support_diff_test, support_diff_val)

            return {
                "feat_parameters": [feat_parameters],
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
                "r2_in_2023": r2_2022,
                "r2_val": r2_val,
                "ols_coef_2022": ols_coef_2022,
                "ols_coef_val": ols_coef_val,
                "test_index": test_index,
                "train_in_middle": train_in_middle,
                "total_time": time.perf_counter() - start,
                "return_ratios": return_ratios,
                "side": self.side,
                "max_trades": self.max_trades,
                "off_days": self.off_days,
            }
        except Exception as e:
            raise ValueError(f"Error: {e} \n {feat_parameters} \n\n  {hyperparams}") from e


    def search_model(
        self,
        test_index: int,
        pct_adj: float = 0.5,
        train_in_middle: bool = True,
        cutoff_point: float | None = None,
    ) -> dict[str, None] | dict[str, Any]:
        """
        Search for the best model parameters.

        Parameters:
        ----------
        test_index : int
            The test index.
        pct_adj : float, optional
            The percentage to adjust the target
            (default : 0.5).
        train_in_middle : bool, optional
            The train in the middle parameter
            (default : True).

        Returns:
        -------
        dict
            The dictionary containing the best model parameters.
        """
        feat_parameters = self.generate_feat_parameters()
        hyperparams = self.generate_hyperparameters()

        selected_features = feat_parameters["random_features"]
        features = []

        for r in range(1, len(selected_features) + 1):
            features += list(itertools.combinations(selected_features, r))

        features = np.array(features, dtype="object")
        if len(features) > 1:
            feat_parameters["random_features"] = np.random.choice(
                features
            )
        else:
            feat_parameters["random_features"] = list(features[0])

        return self.create_and_calculate_metrics(
            feat_parameters,
            hyperparams,
            test_index,
            pct_adj,
            train_in_middle,
            cutoff_point
        )
