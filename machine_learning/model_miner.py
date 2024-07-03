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

        self.ma_types = ["sma", "ema", "dema", "tema", "rma"]
        self.train_in_middle = None

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
            "Ichimoku",
            "Ichimoku Price Distance",
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
            "ols_coef_2023": None,
            "ols_coef_val": None,
            "test_index": None,
        }

    def __generate_feat_parameters(self):
        macd_lengths = np.random.choice(range(2, 151), 2, replace=False)
        smio_lengths = np.random.choice(range(2, 151), 2, replace=False)
        tsi_lengths = np.random.choice(range(2, 151), 2, replace=False)
        didi_ma_lengths = np.random.choice(range(2, 151), 3, replace=False)
        ichimoku_lengths = np.random.choice(range(2, 151), 2, replace=False)
        ichimoku_price_distance_lengths = np.random.choice(range(2, 151), 2, replace=False)

        fast_length = max(macd_lengths)
        slow_length = min(macd_lengths)
        signal_length = np.random.choice(range(2, 51))
        signal_length = (
            signal_length + 1 if signal_length in macd_lengths else signal_length
        )
        distance_types = [
            "absolute",
            "ratio",
            "dtw",
        ]

        return {
            # General
            "random_features": list(np.random.choice(self.random_features)),
            "train_in_middle": self.train_in_middle,
            "random_source_price_dtw": np.random.choice(self.ohlc),
            "random_binnings_qty_dtw": np.random.choice(range(10, 31)),
            "random_moving_averages": np.random.choice(self.ma_type_combinations),
            "random_moving_averages_length": np.random.choice(range(2, 301)),
            # RSI
            "random_source_price_rsi": np.random.choice(self.ohlc),
            "random_binnings_qty_rsi": np.random.choice(range(10, 31)),
            "random_rsi_length": np.random.choice(range(2, 151)),
            "random_source_price_stoch": np.random.choice(['open', 'close']),
            "random_binnings_qty_stoch": np.random.choice(range(10, 31)),
            "random_slow_stoch_length": np.random.choice(range(2, 51)),
            "random_slow_stoch_k": np.random.choice(range(1, 11)),
            "random_slow_stoch_d": np.random.choice(range(2, 11)),
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
            "random_cci_length": np.random.choice(range(2, 151)),
            "random_cci_method": np.random.choice(self.ma_types),
            # MACD
            "random_source_price_macd": np.random.choice(self.ohlc),
            "random_binnings_qty_macd": np.random.choice(range(10, 31)),
            "random_macd_fast_length": fast_length,
            "random_macd_slow_length": slow_length,
            "random_macd_signal_length": signal_length,
            "random_macd_diff_method": np.random.choice(distance_types),
            "random_macd_ma_method": np.random.choice(self.ma_types),
            "random_macd_signal_method": np.random.choice(self.ma_types),
            "random_macd_column": np.random.choice(["macd", "signal", "histogram"]),
            # TRIX
            "random_source_price_trix": np.random.choice(self.ohlc),
            "random_binnings_qty_trix": np.random.choice(range(10, 31)),
            "random_trix_length": np.random.choice(range(2, 51)),
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
            "random_tsi_ma_method": np.random.choice(self.ma_types),
            # Ichimoku
            "random_binnings_qty_ichimoku": np.random.choice(range(10, 31)),
            "random_ichimoku_conversion_periods": ichimoku_lengths[0],
            "random_ichimoku_base_periods": ichimoku_lengths[1],
            "random_ichimoku_lagging_span_2_periods": np.random.choice(range(2, 31)),
            "random_ichimoku_displacement": np.random.choice(range(2, 31)),
            "random_ichimoku_based_on": np.random.choice(['lead_line', 'lagging_span']),
            "random_ichimoku_method": np.random.choice(distance_types),

            # Ichimoku Price Distance
            "random_source_ichimoku_price_distance": np.random.choice(self.ohlc),
            "random_binnings_qty_ichimoku_price_distance": np.random.choice(range(10, 31)),
            "random_ichimoku_price_distance_conversion_periods": ichimoku_price_distance_lengths[0],
            "random_ichimoku_price_distance_base_periods": ichimoku_price_distance_lengths[1],
            "random_ichimoku_price_distance_lagging_span_2_periods": np.random.choice(range(2, 31)),
            "random_ichimoku_price_distance_displacement": np.random.choice(range(2, 31)),
            "random_ichimoku_price_distance_based_on": np.random.choice(['lead_line', 'leading_span']),
            "random_ichimoku_price_distance_method": np.random.choice(distance_types),
            "random_ichimoku_price_distance_use_pct": True,
        }

    def __generate_hyperparameters(self):
        return {
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

    def search_model(
        self,
        test_index: int,
        pct_adj: float = 0.5,
        train_in_middle: bool = True,
    ):
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
        feat_parameters = self.__generate_feat_parameters()
        hyperparams = self.__generate_hyperparameters()
        self.train_in_middle = train_in_middle

        try:
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

            mta, index_splits, all_y = model_creation(
                feat_parameters,
                hyperparams,
                test_index,
                self.dataframe,
                self.max_trades,
                self.off_days,
                pct_adj,
                train_in_middle,
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

            results = mta.loc["2023", "Liquid_Result"].cumprod()
            if results.iloc[-1] < 1:
                return self.empty_dict
            if (results == 1).all():
                return self.empty_dict

            try:
                r2_2023 = literal_eval(
                    px.scatter(results, trendline="ols")
                    .data[1]["hovertemplate"]
                    .split(">=")[1]
                    .split("<br>")[0]
                )

                ols_coef_2023 = literal_eval(
                    px.scatter(results, trendline="ols")
                    .data[1]['hovertemplate']
                    .split('<br>')[1]
                    .split('*')[0]
                    .split(' ')[-2]
                )

                r2_val = literal_eval(
                    px.scatter(mta_val["Liquid_Result"].cumprod(), trendline="ols")
                    .data[1]["hovertemplate"]
                    .split(">=")[1]
                    .split("<br>")[0]
                )

                ols_coef_val = literal_eval(
                    px.scatter(mta_val["Liquid_Result"].cumprod(), trendline="ols")
                    .data[1]['hovertemplate']
                    .split('<br>')[1]
                    .split('*')[0]
                    .split(' ')[-2]
                )

            except Exception:
                r2_2023 = -404
                r2_val = -404

            total_operations = (
                test_buys[1],
                val_buys[1],
            )

            total_operations_test_pct = test_buys[1] / (test_index // 3) * 100
            total_operations_val_pct = val_buys[1] / (test_index // 3) * 100

            total_operations_pct = (
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

            test_predict_adj = self.adj_targets.loc[test_periods[0] : test_periods[1]]
            val_predict_adj = self.adj_targets.loc[val_periods[0] : val_periods[1]]

            support_diff_test = (
                test_predict_adj.value_counts(normalize=True)
                - mta_test["Predict"].value_counts(normalize=True)
            )[1]

            support_diff_val = (
                val_predict_adj.value_counts(normalize=True)
                - mta_val["Predict"].value_counts(normalize=True)
            )[1]

            support_diffs = (support_diff_test, support_diff_val)

            result_metrics_test = DataHandler(mta.loc[y_test_adj.index]).result_metrics(
                "Liquid_Result", is_percentage_data=True
            )

            result_metrics_val = DataHandler(mta.loc[y_val_adj.index]).result_metrics(
                "Liquid_Result", is_percentage_data=True
            )

            precisions = (
                result_metrics_test.loc["Win_Rate"][0],
                result_metrics_val.loc["Win_Rate"][0],
            )

            expected_return = (
                result_metrics_test.loc["Expected_Return"][0],
                result_metrics_val.loc["Expected_Return"][0],
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
                "r2_in_2023": r2_2023,
                "r2_val": r2_val,
                "ols_coef_2023": ols_coef_2023,
                "ols_coef_val": ols_coef_val,
                "test_index": test_index,
            }
        except Exception as e:
            raise ValueError(f"Error: {e} \n {feat_parameters} \n\n  {hyperparams}") from e

    def test_preset_model(
        self,
        feat_parameters: dict,
        hyperparams: dict,
        test_index: int,
        pct_adj: float = 0.5,
        train_in_middle: bool = True,
    ):
        """
        Check the preset model parameters and return the results.

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

        mta, index_splits, all_y = model_creation(
            feat_parameters,
            hyperparams,
            test_index,
            self.dataframe,
            self.max_trades,
            self.off_days,
            pct_adj,
            train_in_middle,
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

        results = mta.loc["2023", "Liquid_Result"].cumprod()
        if results.iloc[-1] < 1:
            return self.empty_dict
        if (results == 1).all():
            return self.empty_dict

        try:
            r2_2023 = literal_eval(
                px.scatter(results, trendline="ols")
                .data[1]["hovertemplate"]
                .split(">=")[1]
                .split("<br>")[0]
            )

            r2_val = literal_eval(
                px.scatter(mta_val["Liquid_Result"].cumprod(), trendline="ols")
                .data[1]["hovertemplate"]
                .split(">=")[1]
                .split("<br>")[0]
            )

        except Exception:
            r2_2023 = -404
            r2_val = -404

        test_buy_keys = test_buys.keys()[-1]
        val_buy_keys = val_buys.keys()[-1]

        total_operations = (
            test_buys[test_buy_keys],
            val_buys[val_buy_keys],
        )

        total_operations_test_pct = test_buys[1] / (test_index // 3) * 100
        total_operations_val_pct = val_buys[1] / (test_index // 3) * 100

        total_operations_pct = (
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

        test_predict_adj = self.adj_targets.loc[test_periods[0] : test_periods[1]]
        val_predict_adj = self.adj_targets.loc[val_periods[0] : val_periods[1]]

        support_diff_test = (
            test_predict_adj.value_counts(normalize=True)
            - mta_test["Predict"].value_counts(normalize=True)
        )[1]

        support_diff_val = (
            val_predict_adj.value_counts(normalize=True)
            - mta_val["Predict"].value_counts(normalize=True)
        )[1]

        support_diffs = (support_diff_test, support_diff_val)

        result_metrics_test = DataHandler(mta.loc[y_test_adj.index]).result_metrics(
            "Liquid_Result", is_percentage_data=True
        )

        result_metrics_val = DataHandler(mta.loc[y_val_adj.index]).result_metrics(
            "Liquid_Result", is_percentage_data=True
        )

        precisions = (
            result_metrics_test.loc["Win_Rate"][0],
            result_metrics_val.loc["Win_Rate"][0],
        )

        expected_return = (
            result_metrics_test.loc["Expected_Return"][0],
            result_metrics_val.loc["Expected_Return"][0],
        )

        metrics_results = {
            "expected_return_test": expected_return[0],
            "expected_return_val": expected_return[1],
            "precisions_test": precisions[0],
            "precisions_val": precisions[1],
            "precisions": precisions,
        }

        # if min(precisions) <= 0.52:
        #     return self.empty_dict

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
            "r2_in_2023": r2_2023,
            "r2_val": r2_val,
            "test_index": test_index,
        }
