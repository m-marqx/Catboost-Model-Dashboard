import unittest

import pandas as pd
import numpy as np

from machine_learning.model_miner import ModelMiner

class ModelMinerTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(120)
        self.dataframe = pd.read_parquet(r"data\assets\btc.parquet").loc[:"2023"]
        self.dataframe['Return'] = self.dataframe["close"].pct_change(7) + 1
        self.dataframe["Target"] = self.dataframe["Return"].shift(-7)
        self.dataframe["Target_bin"] = np.where(
            self.dataframe["Target"] > 1,
            1, -1)

        self.dataframe["Target_bin"] = np.where(
            self.dataframe['Target'].isna(),
            np.nan, self.dataframe['Target_bin']
        )
        self.target = self.dataframe["Target_bin"].copy()
        self.model_miner = ModelMiner(self.dataframe, self.target)

    def test_init(self):
        self.assertEqual(self.model_miner.ohlc, ["open", "high", "low", "close"])
        self.assertEqual(self.model_miner.max_trades, 3)
        self.assertEqual(self.model_miner.off_days, 7)
        self.assertEqual(self.model_miner.side, 1)
        pd.testing.assert_frame_equal(self.model_miner.dataframe, self.dataframe)
        pd.testing.assert_series_equal(self.model_miner.target, self.target)
        self.assertEqual(self.model_miner.ma_types, ["sma", "ema", "rma"])

    def test_generate_feat_parameters(self):
        feat_parameters = self.model_miner.generate_feat_parameters()

        self.assertIsInstance(feat_parameters, dict)

        # Check if feat_parameters contains the expected keys
        expected_feat_parameters = {
            'random_features': ['MACD_opt', 'SMIO_opt'],
            'random_source_price_dtw': 'high',
            'random_binnings_qty_dtw': 13,
            'random_moving_averages': ('rma',),
            'random_moving_averages_length': 49,
            'random_source_price_rsi': 'open',
            'random_binnings_qty_rsi': 29,
            'random_rsi_length': 129,
            'random_rsi_ma_method': 'ema',
            'random_source_price_stoch': 'open',
            'random_binnings_qty_stoch': 20,
            'random_slow_stoch_length': 27,
            'random_slow_stoch_k': 1,
            'random_slow_stoch_d': 3,
            'random_slow_stoch_ma_method': 'rma',
            'random_source_price_didi': 'close',
            'random_binnings_qty_didi': 28,
            'random_didi_short_length': 50,
            'random_didi_mid_length': 104,
            'random_didi_long_length': 137,
            'random_didi_ma_type': 'rma',
            'random_didi_method': 'ratio',
            'random_source_price_cci': 'open',
            'random_binnings_qty_cci': 30,
            'random_cci_length': 117,
            'random_cci_method': 'rma',
            'random_source_price_macd': 'low',
            'random_binnings_qty_macd': 27,
            'random_macd_fast_length': 113,
            'random_macd_slow_length': 68,
            'random_macd_signal_length': 14,
            'random_macd_ma_method': 'ema',
            'random_macd_signal_method': 'rma',
            'random_macd_column': 'histogram',
            'random_source_price_trix': 'low',
            'random_binnings_qty_trix': 15,
            'random_trix_length': 7,
            'random_trix_signal_length': 2,
            'random_trix_ma_method': 'sma',
            'random_source_price_smio': 'high',
            'random_binnings_qty_smio': 14,
            'random_smio_short_length': 45,
            'random_smio_long_length': 3,
            'random_smio_signal_length': 41,
            'random_smio_ma_method': 'sma',
            'random_source_price_tsi': 'open',
            'random_binnings_qty_tsi': 30,
            'random_tsi_short_length': 54,
            'random_tsi_long_length': 4,
            'random_tsi_ma_method': 'ema',
            'random_binnings_qty_ichimoku': 18,
            'random_ichimoku_conversion_periods': 146,
            'random_ichimoku_base_periods': 23,
            'random_ichimoku_lagging_span_2_periods': 17,
            'random_ichimoku_displacement': 16,
            'random_ichimoku_based_on': 'lead_line',
            'random_ichimoku_method': 'absolute',
            'random_source_ichimoku_price_distance': 'low',
            'random_binnings_qty_ichimoku_price_distance': 23,
            'random_ichimoku_price_distance_conversion_periods': 142,
            'random_ichimoku_price_distance_base_periods': 49,
            'random_ichimoku_price_distance_lagging_span_2_periods': 13,
            'random_ichimoku_price_distance_displacement': 24,
            'random_ichimoku_price_distance_based_on': 'lead_line',
            'random_ichimoku_price_distance_method': 'absolute',
            'random_ichimoku_price_distance_use_pct': True,
            'random_source_bb_trend': 'open',
            'random_binnings_qty_bb_trend': 21,
            'random_bb_trend_short_length': 11,
            'random_bb_trend_long_length': 12,
            'random_bb_trend_stdev': 2.7000000000000015,
            'random_bb_trend_ma_method': 'ema',
            'random_bb_trend_stdev_method': 'absolute',
            'random_bb_trend_diff_method': 'normal',
            'random_bb_trend_based_on': 'long_length',
        }
        pd.testing.assert_series_equal(pd.Series(feat_parameters), pd.Series(expected_feat_parameters))

    def test_generate_hyperparameters(self):
        hyperparams = self.model_miner.generate_hyperparameters()

        # Check if hyperparams is a dictionary
        self.assertIsInstance(hyperparams, dict)

        expected_hyperparams = {
            'iterations': 1000,
            'learning_rate': 0.4,
            'depth': 1,
            'min_child_samples': 11,
            'colsample_bylevel': 0.5699999999999997,
            'subsample': 0.6599999999999997,
            'reg_lambda': 85,
            'use_best_model': True,
            'eval_metric': 'AUC',
            'random_seed': 38552,
            'silent': True
        }

        self.assertDictEqual(hyperparams, expected_hyperparams)

    def test_search_model(self):
        expected_results = {
            'feat_parameters': [{'random_features': ('Stoch_opt', 'CCI', 'MACD_opt'),
               'random_source_price_dtw': 'close',
               'random_binnings_qty_dtw': 15,
               'random_moving_averages': 'all',
               'random_moving_averages_length': 219,
               'random_source_price_rsi': 'high',
               'random_binnings_qty_rsi': 16,
               'random_rsi_length': 103,
               'random_rsi_ma_method': 'ema',
               'random_source_price_stoch': 'close',
               'random_binnings_qty_stoch': 26,
               'random_slow_stoch_length': 50,
               'random_slow_stoch_k': 6,
               'random_slow_stoch_d': 8,
               'random_slow_stoch_ma_method': 'rma',
               'random_source_price_didi': 'high',
               'random_binnings_qty_didi': 21,
               'random_didi_short_length': 122,
               'random_didi_mid_length': 140,
               'random_didi_long_length': 143,
               'random_didi_ma_type': 'ema',
               'random_didi_method': 'ratio',
               'random_source_price_cci': 'close',
               'random_binnings_qty_cci': 26,
               'random_cci_length': 106,
               'random_cci_method': 'sma',
               'random_source_price_macd': 'low',
               'random_binnings_qty_macd': 12,
               'random_macd_fast_length': 135,
               'random_macd_slow_length': 111,
               'random_macd_signal_length': 10,
               'random_macd_ma_method': 'sma',
               'random_macd_signal_method': 'rma',
               'random_macd_column': 'signal',
               'random_source_price_trix': 'low',
               'random_binnings_qty_trix': 16,
               'random_trix_length': 6,
               'random_trix_signal_length': 3,
               'random_trix_ma_method': 'ema',
               'random_source_price_smio': 'close',
               'random_binnings_qty_smio': 14,
               'random_smio_short_length': 142,
               'random_smio_long_length': 44,
               'random_smio_signal_length': 7,
               'random_smio_ma_method': 'ema',
               'random_source_price_tsi': 'low',
               'random_binnings_qty_tsi': 13,
               'random_tsi_short_length': 104,
               'random_tsi_long_length': 62,
               'random_tsi_ma_method': 'sma',
               'random_binnings_qty_ichimoku': 27,
               'random_ichimoku_conversion_periods': 16,
               'random_ichimoku_base_periods': 122,
               'random_ichimoku_lagging_span_2_periods': 17,
               'random_ichimoku_displacement': 22,
               'random_ichimoku_based_on': 'lead_line',
               'random_ichimoku_method': 'ratio',
               'random_source_ichimoku_price_distance': 'close',
               'random_binnings_qty_ichimoku_price_distance': 14,
               'random_ichimoku_price_distance_conversion_periods': 142,
               'random_ichimoku_price_distance_base_periods': 114,
               'random_ichimoku_price_distance_lagging_span_2_periods': 19,
               'random_ichimoku_price_distance_displacement': 26,
               'random_ichimoku_price_distance_based_on': 'lead_line',
               'random_ichimoku_price_distance_method': 'absolute',
               'random_ichimoku_price_distance_use_pct': True,
               'random_source_bb_trend': 'close',
               'random_binnings_qty_bb_trend': 13,
               'random_bb_trend_short_length': 11,
               'random_bb_trend_long_length': 13,
               'random_bb_trend_stdev': 2.300000000000001,
               'random_bb_trend_ma_method': 'ema',
               'random_bb_trend_stdev_method': 'absolute',
               'random_bb_trend_diff_method': 'normal',
               'random_bb_trend_based_on': 'long_length'}],
            'hyperparameters': [{'iterations': 1000,
               'learning_rate': 0.14,
               'depth': 5,
               'min_child_samples': 18,
               'colsample_bylevel': 0.3599999999999999,
               'subsample': 0.7999999999999996,
               'reg_lambda': 176,
               'use_best_model': True,
               'eval_metric': 'F1',
               'random_seed': 12181,
               'silent': True}],
            'metrics_results': [{'expected_return_test': 1.3878393091497603,
               'expected_return_val': 0.4090202698016433,
               'precisions_test': 0.5861182519280206,
               'precisions_val': 0.5236447520184544,
               'precisions': (0.5861182519280206, 0.5236447520184544)}],
            'drawdown_full_test': 0.5084558694714835,
            'drawdown_full_val': 0.6843396662346691,
            'drawdown_adj_test': 0.2868381518242906,
            'drawdown_adj_val': 0.4245488255748184,
            'expected_return_test': 1.3878393091497603,
            'expected_return_val': 0.4090202698016433,
            'precisions_test': 0.5861182519280206,
            'precisions_val': 0.5236447520184544,
            'support_diff_test': -0.10776699029126213,
            'support_diff_val': -0.11929371231696817,
            'total_operations_test': 389,
            'total_operations_val': 870,
            'total_operations_pct_test': 2.6462585034013606,
            'total_operations_pct_val': 5.918367346938775,
            'r2_in_2023': 0.857435,
            'r2_val': 0.558247,
            'ols_coef_2022': -0.020184,
            'ols_coef_val': 0.100191,
            'test_index': 1030,
            'train_in_middle': True,
            'return_ratios': {
                'sharpe_test': 0.17999999999999997,
                'sharpe_val': 0.15285714285714286,
                'sortino_test': 0.30666666666666664,
                'sortino_val': 0.35428571428571426
            },
            'side': 1,
            'max_trades': 3,
            'off_days': 7
        }

        np.random.seed(0)
        results = ModelMiner(self.dataframe, self.target).search_model(1030)

        pd.testing.assert_series_equal(
            pd.Series(results).drop('total_time'),
            pd.Series(expected_results)
        )

    def test_create_model(self):
        np.random.seed(120)
        feat_params = self.model_miner.generate_feat_parameters()
        hyperaparams = self.model_miner.generate_hyperparameters()
        results = self.model_miner.create_and_calculate_metrics(feat_params, hyperaparams, 1030)

        expected_results = pd.Series({
            'feat_parameters': [{'random_features': ['MACD_opt', 'SMIO_opt'],
            'random_source_price_dtw': 'high',
            'random_binnings_qty_dtw': 13,
            'random_moving_averages': ('rma',),
            'random_moving_averages_length': 49,
            'random_source_price_rsi': 'open',
            'random_binnings_qty_rsi': 29,
            'random_rsi_length': 129,
            'random_rsi_ma_method': 'ema',
            'random_source_price_stoch': 'open',
            'random_binnings_qty_stoch': 20,
            'random_slow_stoch_length': 27,
            'random_slow_stoch_k': 1,
            'random_slow_stoch_d': 3,
            'random_slow_stoch_ma_method': 'rma',
            'random_source_price_didi': 'close',
            'random_binnings_qty_didi': 28,
            'random_didi_short_length': 50,
            'random_didi_mid_length': 104,
            'random_didi_long_length': 137,
            'random_didi_ma_type': 'rma',
            'random_didi_method': 'ratio',
            'random_source_price_cci': 'open',
            'random_binnings_qty_cci': 30,
            'random_cci_length': 117,
            'random_cci_method': 'rma',
            'random_source_price_macd': 'low',
            'random_binnings_qty_macd': 27,
            'random_macd_fast_length': 113,
            'random_macd_slow_length': 68,
            'random_macd_signal_length': 14,
            'random_macd_ma_method': 'ema',
            'random_macd_signal_method': 'rma',
            'random_macd_column': 'histogram',
            'random_source_price_trix': 'low',
            'random_binnings_qty_trix': 15,
            'random_trix_length': 7,
            'random_trix_signal_length': 2,
            'random_trix_ma_method': 'sma',
            'random_source_price_smio': 'high',
            'random_binnings_qty_smio': 14,
            'random_smio_short_length': 45,
            'random_smio_long_length': 3,
            'random_smio_signal_length': 41,
            'random_smio_ma_method': 'sma',
            'random_source_price_tsi': 'open',
            'random_binnings_qty_tsi': 30,
            'random_tsi_short_length': 54,
            'random_tsi_long_length': 4,
            'random_tsi_ma_method': 'ema',
            'random_binnings_qty_ichimoku': 18,
            'random_ichimoku_conversion_periods': 146,
            'random_ichimoku_base_periods': 23,
            'random_ichimoku_lagging_span_2_periods': 17,
            'random_ichimoku_displacement': 16,
            'random_ichimoku_based_on': 'lead_line',
            'random_ichimoku_method': 'absolute',
            'random_source_ichimoku_price_distance': 'low',
            'random_binnings_qty_ichimoku_price_distance': 23,
            'random_ichimoku_price_distance_conversion_periods': 142,
            'random_ichimoku_price_distance_base_periods': 49,
            'random_ichimoku_price_distance_lagging_span_2_periods': 13,
            'random_ichimoku_price_distance_displacement': 24,
            'random_ichimoku_price_distance_based_on': 'lead_line',
            'random_ichimoku_price_distance_method': 'absolute',
            'random_ichimoku_price_distance_use_pct': True,
            'random_source_bb_trend': 'open',
            'random_binnings_qty_bb_trend': 21,
            'random_bb_trend_short_length': 11,
            'random_bb_trend_long_length': 12,
            'random_bb_trend_stdev': 2.7000000000000015,
            'random_bb_trend_ma_method': 'ema',
            'random_bb_trend_stdev_method': 'absolute',
            'random_bb_trend_diff_method': 'normal',
            'random_bb_trend_based_on': 'long_length'}],
            'hyperparameters': [{'iterations': 1000,
            'learning_rate': 0.9,
            'depth': 4,
            'min_child_samples': 16,
            'colsample_bylevel': 0.34999999999999987,
            'subsample': 0.7899999999999996,
            'reg_lambda': 153,
            'use_best_model': True,
            'eval_metric': 'Precision',
            'random_seed': 15832,
            'silent': True}],
            'metrics_results': [{'expected_return_test': 1.2814119524074585,
            'expected_return_val': 0.3521897367000961,
            'precisions_test': 0.5596816976127321,
            'precisions_val': 0.5298329355608592,
            'precisions': (0.5596816976127321, 0.5298329355608592)}],
            'drawdown_full_test': 0.5876390190455973,
            'drawdown_full_val': 0.6362151655285802,
            'drawdown_adj_test': 0.349342237338609,
            'drawdown_adj_val': 0.38657173365819597,
            'expected_return_test': 1.2814119524074585,
            'expected_return_val': 0.3521897367000961,
            'precisions_test': 0.5596816976127321,
            'precisions_val': 0.5298329355608592,
            'support_diff_test': -0.0961165048543689,
            'support_diff_val': -0.10680447889750216,
            'total_operations_test': 377,
            'total_operations_val': 841,
            'total_operations_pct_test': 2.564625850340136,
            'total_operations_pct_val': 5.72108843537415,
            'r2_in_2023': 0.832332,
            'r2_val': 0.596389,
            'ols_coef_2022': -0.0166337,
            'ols_coef_val': 0.0450816,
            'test_index': 1030,
            'train_in_middle': True,
            'return_ratios': {
                'sharpe_test': 0.1333333333333333,
                'sharpe_val': 0.14142857142857143,
                'sortino_test': 0.2133333333333333,
                'sortino_val': 0.2557142857142857
            },
            'side': 1,
            'max_trades': 3,
            'off_days': 7
        })
        pd.testing.assert_series_equal(pd.Series(results).drop('total_time'), expected_results)

if __name__ == '__main__':
    unittest.main()
