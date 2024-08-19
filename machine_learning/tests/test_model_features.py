import unittest

import pandas as pd
from pandas import Timestamp, Interval
import numpy as np

from machine_learning.model_features import feature_binning, ModelFeatures


class ModelFeaturesTests(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe = btc_data.copy().loc[:"2023"]
        self.dataframe["Return"] = self.dataframe["close"].pct_change(7) + 1
        self.dataframe["Target"] = self.dataframe["Return"].shift(-7)
        self.dataframe["Target_bin"] = np.where(
            self.dataframe["Target"] > 1, 1, -1
        )

        self.dataframe["Target_bin"] = np.where(
            self.dataframe["Target"].isna(),
            np.nan,
            self.dataframe["Target_bin"],
        )
        self.test_index = 1030
        self.bins = 9

        self.target = self.dataframe["Target_bin"].copy()

        self.model_features = ModelFeatures(
            self.dataframe, self.test_index, self.bins, False
        )

    def test_feature_binning(self):
        results = feature_binning(
            self.dataframe["Target"].rename(None),
            self.test_index,
            10,
        )

        results_count = results.value_counts()
        expected_results_count = pd.Series(
            {
                4: 541,
                3: 532,
                2: 499,
                5: 497,
                6: 458,
                8: 420,
                1: 410,
                0: 385,
                7: 382,
                9: 251,
            },
            name="count",
        )

        dates = pd.Index(
            [
                pd.Timestamp("2012-01-02 00:00:00"),
                pd.Timestamp("2012-01-03 00:00:00"),
                pd.Timestamp("2012-01-04 00:00:00"),
                pd.Timestamp("2012-01-05 00:00:00"),
                pd.Timestamp("2012-01-06 00:00:00"),
                pd.Timestamp("2012-01-07 00:00:00"),
                pd.Timestamp("2012-01-08 00:00:00"),
                pd.Timestamp("2012-01-09 00:00:00"),
                pd.Timestamp("2012-01-10 00:00:00"),
                pd.Timestamp("2012-01-11 00:00:00"),
                pd.Timestamp("2012-01-12 00:00:00"),
                pd.Timestamp("2012-01-13 00:00:00"),
                pd.Timestamp("2012-01-14 00:00:00"),
                pd.Timestamp("2012-01-15 00:00:00"),
                pd.Timestamp("2012-01-16 00:00:00"),
                pd.Timestamp("2012-01-17 00:00:00"),
                pd.Timestamp("2012-01-18 00:00:00"),
                pd.Timestamp("2012-01-19 00:00:00"),
                pd.Timestamp("2012-01-20 00:00:00"),
                pd.Timestamp("2012-01-21 00:00:00"),
            ],
            name="date",
            dtype="datetime64[ms]",
        )

        values = [
            9,
            9,
            9,
            3,
            7,
            1,
            5,
            8,
            0,
            4,
            2,
            6,
            7,
            2,
            1,
            7,
            0,
            0,
            0,
            0,
        ]

        expected_results = pd.Series(values, index=dates)

        pd.testing.assert_series_equal(results_count, expected_results_count)
        pd.testing.assert_series_equal(results.iloc[:20], expected_results)

    def test_create_rsi_feature(self):
        source = self.dataframe["close"]
        length = 14
        ma_method = "ema"

        test_df = self.model_features.create_rsi_feature(
            source, length, ma_method
        ).dropna()[["RSI", "RSI_feat"]]

        rsi_values = [
            65.53254437869822,
            48.525446579036064,
            60.38736258255023,
            49.79513979849982,
            57.012399646774654,
            54.77221193135472,
            54.99890776717075,
            51.015778613199274,
            51.46640660824145,
            47.43806049658012,
        ]

        rsi_feat_values = [
            5.0,
            3.0,
            4.0,
            3.0,
            4.0,
            8.0,
            8.0,
            3.0,
            8.0,
            3.0,
        ]

        dates = pd.Index(
            [
                pd.Timestamp("2012-01-16 00:00:00"),
                pd.Timestamp("2012-01-17 00:00:00"),
                pd.Timestamp("2012-01-18 00:00:00"),
                pd.Timestamp("2012-01-19 00:00:00"),
                pd.Timestamp("2012-01-20 00:00:00"),
                pd.Timestamp("2012-01-21 00:00:00"),
                pd.Timestamp("2012-01-22 00:00:00"),
                pd.Timestamp("2012-01-23 00:00:00"),
                pd.Timestamp("2012-01-24 00:00:00"),
                pd.Timestamp("2012-01-25 00:00:00"),
            ],
            dtype="datetime64[ms]",
            name="date",
        )

        expected_df = pd.DataFrame(
            {"RSI": rsi_values, "RSI_feat": rsi_feat_values}, index=dates
        )

        rsi_count = (
            test_df["RSI"].rename(None).value_counts(bins=10)
        )
        rsi_feat_count = (
            test_df["RSI_feat"].rename(None).value_counts(bins=10)
        )

        expected_rsi_count = pd.Series(
            {
                pd.Interval(42.538, 51.888, closed="right"): 918,
                pd.Interval(51.888, 61.237, closed="right"): 751,
                pd.Interval(33.189, 42.538, closed="right"): 700,
                pd.Interval(61.237, 70.586, closed="right"): 576,
                pd.Interval(70.586, 79.936, closed="right"): 433,
                pd.Interval(23.84, 33.189, closed="right"): 345,
                pd.Interval(79.936, 89.285, closed="right"): 322,
                pd.Interval(89.285, 98.634, closed="right"): 147,
                pd.Interval(14.49, 23.84, closed="right"): 126,
                pd.Interval(5.045999999999999, 14.49, closed="right"): 43,
            },
            name="count",
        )

        expected_rsi_feat_count = pd.Series(
            {
                pd.Interval(0.8, 1.6, closed="right"): 637,
                pd.Interval(3.2, 4.0, closed="right"): 559,
                pd.Interval(4.8, 5.6, closed="right"): 550,
                pd.Interval(-0.009000000000000001, 0.8, closed="right"): 507,
                pd.Interval(7.2, 8.0, closed="right"): 474,
                pd.Interval(2.4, 3.2, closed="right"): 456,
                pd.Interval(1.6, 2.4, closed="right"): 455,
                pd.Interval(5.6, 6.4, closed="right"): 450,
                pd.Interval(6.4, 7.2, closed="right"): 273,
                pd.Interval(4.0, 4.8, closed="right"): 0,
            },
            name="count",
        )

        pd.testing.assert_frame_equal(test_df.head(10), expected_df)
        pd.testing.assert_series_equal(rsi_count, expected_rsi_count)
        pd.testing.assert_series_equal(rsi_feat_count, expected_rsi_feat_count)

    def test_create_rsi_opt_feature(self):
        source = self.dataframe["close"]
        length = 14
        ma_method = "ema"

        test_df = self.model_features.create_rsi_opt_feature(
            source, length, ma_method
        ).dropna()

        test_values: dict[str, list] = {}

        test_values["RSI"] = [
            -3.638192938493816,
            -0.8978086855093963,
            -2.386459178140159,
            -3.5193214555962395,
            -1.4237537619312812,
            2.6561994723420774,
            -2.4978555240279023,
            2.529828741435018,
            4.9874553777196,
            -2.784056738965517,
        ]

        test_values["RSI_feat"] = [
            1.0,
            3.0,
            2.0,
            1.0,
            3.0,
            8.0,
            2.0,
            8.0,
            6.0,
            2.0,
        ]

        dates = pd.Index(
            [
                Timestamp("2012-01-18 00:00:00"),
                Timestamp("2012-01-19 00:00:00"),
                Timestamp("2012-01-20 00:00:00"),
                Timestamp("2012-01-21 00:00:00"),
                Timestamp("2012-01-22 00:00:00"),
                Timestamp("2012-01-23 00:00:00"),
                Timestamp("2012-01-24 00:00:00"),
                Timestamp("2012-01-25 00:00:00"),
                Timestamp("2012-01-26 00:00:00"),
                Timestamp("2012-01-27 00:00:00"),
            ],
            dtype="datetime64[ms]",
            name="date",
        )

        expected_df = pd.DataFrame(test_values, index=dates)

        pd.testing.assert_frame_equal(test_df.iloc[:10, -2:], expected_df)

        rsi_count = (
            test_df["RSI"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )
        rsi_feat_count = (
            test_df["RSI_feat"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )

        expected_count = {}
        expected_count["rsi_count"] = pd.Series(
            {
                Interval(-1.051, 5.313, closed="right"): 2019,
                Interval(-7.414, -1.051, closed="right"): 1392,
                Interval(5.313, 11.677, closed="right"): 491,
                Interval(-13.778, -7.414, closed="right"): 281,
                Interval(11.677, 18.04, closed="right"): 79,
                Interval(-20.141, -13.778, closed="right"): 56,
                Interval(18.04, 24.404, closed="right"): 24,
                Interval(-26.563000000000002, -20.141, closed="right"): 10,
                Interval(24.404, 30.768, closed="right"): 7,
            },
            name="count",
        )

        expected_count["rsi_feat_count"] = pd.Series(
            {
                Interval(6.222, 7.111, closed="right"): 561,
                Interval(-0.009000000000000001, 0.889, closed="right"): 541,
                Interval(0.889, 1.778, closed="right"): 524,
                Interval(5.333, 6.222, closed="right"): 510,
                Interval(2.667, 3.556, closed="right"): 483,
                Interval(7.111, 8.0, closed="right"): 483,
                Interval(4.444, 5.333, closed="right"): 438,
                Interval(1.778, 2.667, closed="right"): 414,
                Interval(3.556, 4.444, closed="right"): 405,
            },
            name="count",
        )

        pd.testing.assert_series_equal(rsi_count, expected_count["rsi_count"])
        pd.testing.assert_series_equal(
            rsi_feat_count, expected_count["rsi_feat_count"]
        )

    def test_create_slow_stoch_feature(self):
        test_df: pd.DataFrame = self.model_features.create_slow_stoch_feature(
            "close"
        ).dropna()

        test_values: dict[str, list] = {}

        test_values["stoch_k"] = [
            43.67346938775511,
            83.13253012048192,
            37.634408602150536,
            76.88172043010755,
            67.2043010752688,
            68.27956989247313,
            53.76344086021504,
            55.37634408602151,
            42.47311827956989,
            0.0,
        ]

        test_values["stoch_k_feat"] = [
            2.0,
            6.0,
            2.0,
            5.0,
            4.0,
            4.0,
            3.0,
            3.0,
            2.0,
            0.0,
        ]

        test_values["stoch_d"] = [
            74.28571428571429,
            71.79247602655519,
            54.813469370129184,
            65.88288638424666,
            60.57347670250896,
            70.78853046594982,
            63.08243727598566,
            59.13978494623657,
            50.537634408602145,
            32.61648745519713,
        ]

        test_values["stoch_d_feat"] = [
            5.0,
            5.0,
            3.0,
            4.0,
            4.0,
            4.0,
            4.0,
            3.0,
            3.0,
            1.0,
        ]

        dates = pd.Index(
            [
                Timestamp("2012-01-17 00:00:00"),
                Timestamp("2012-01-18 00:00:00"),
                Timestamp("2012-01-19 00:00:00"),
                Timestamp("2012-01-20 00:00:00"),
                Timestamp("2012-01-21 00:00:00"),
                Timestamp("2012-01-22 00:00:00"),
                Timestamp("2012-01-23 00:00:00"),
                Timestamp("2012-01-24 00:00:00"),
                Timestamp("2012-01-25 00:00:00"),
                Timestamp("2012-01-26 00:00:00"),
            ],
            dtype="datetime64[ms]",
            name="date",
        )

        expected_df = pd.DataFrame(test_values, index=dates)

        pd.testing.assert_frame_equal(test_df.iloc[:10, -4:], expected_df)

        stoch_k_count = (
            test_df["stoch_k"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )
        stoch_d_count = (
            test_df["stoch_d"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )

        stoch_k_feat_count = (
            test_df["stoch_k_feat"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )
        stoch_d_feat_count = (
            test_df["stoch_d_feat"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )

        expected_count = {}
        expected_count["stoch_k_count"] = pd.Series(
            {
                Interval(88.889, 100.0, closed="right"): 702,
                Interval(77.778, 88.889, closed="right"): 635,
                Interval(66.667, 77.778, closed="right"): 507,
                Interval(55.556, 66.667, closed="right"): 486,
                Interval(44.444, 55.556, closed="right"): 465,
                Interval(33.333, 44.444, closed="right"): 459,
                Interval(22.222, 33.333, closed="right"): 455,
                Interval(11.111, 22.222, closed="right"): 428,
                Interval(-0.101, 11.111, closed="right"): 223,
            },
            name="count",
        )

        expected_count["stoch_d_count"] = pd.Series(
            {
                Interval(77.398, 88.092, closed="right"): 700,
                Interval(88.092, 98.785, closed="right"): 631,
                Interval(66.704, 77.398, closed="right"): 524,
                Interval(13.236, 23.929, closed="right"): 500,
                Interval(45.317, 56.011, closed="right"): 476,
                Interval(23.929, 34.623, closed="right"): 459,
                Interval(34.623, 45.317, closed="right"): 455,
                Interval(56.011, 66.704, closed="right"): 453,
                Interval(2.4450000000000003, 13.236, closed="right"): 162,
            },
            name="count",
        )

        expected_count["stoch_k_feat_count"] = pd.Series(
            {
                Interval(-0.009000000000000001, 0.889, closed="right"): 680,
                Interval(0.889, 1.778, closed="right"): 575,
                Interval(5.333, 6.222, closed="right"): 495,
                Interval(1.778, 2.667, closed="right"): 492,
                Interval(2.667, 3.556, closed="right"): 487,
                Interval(3.556, 4.444, closed="right"): 465,
                Interval(6.222, 7.111, closed="right"): 462,
                Interval(4.444, 5.333, closed="right"): 366,
                Interval(7.111, 8.0, closed="right"): 338,
            },
            name="count",
        )

        expected_count["stoch_d_feat_count"] = pd.Series(
            {
                Interval(-0.009000000000000001, 0.889, closed="right"): 682,
                Interval(0.889, 1.778, closed="right"): 573,
                Interval(3.556, 4.444, closed="right"): 502,
                Interval(5.333, 6.222, closed="right"): 483,
                Interval(1.778, 2.667, closed="right"): 478,
                Interval(2.667, 3.556, closed="right"): 477,
                Interval(7.111, 8.0, closed="right"): 477,
                Interval(4.444, 5.333, closed="right"): 408,
                Interval(6.222, 7.111, closed="right"): 280,
            },
            name="count",
        )

        pd.testing.assert_series_equal(
            stoch_k_count, expected_count["stoch_k_count"]
        )
        pd.testing.assert_series_equal(
            stoch_d_count, expected_count["stoch_d_count"]
        )

        pd.testing.assert_series_equal(
            stoch_k_feat_count, expected_count["stoch_k_feat_count"]
        )
        pd.testing.assert_series_equal(
            stoch_d_feat_count, expected_count["stoch_d_feat_count"]
        )

    def test_create_slow_stoch_opt_feature(self):
        test_df = self.model_features.create_slow_stoch_opt_feature(
            "close"
        ).dropna()

        test_values: dict[str, list] = {}

        test_values["stoch_k"] = [
            4.2702608334987495,
            -4.419989919970305,
            -20.909071486698963,
            -6.0826389779489825,
            9.504123403045133,
            -9.123958466923778,
            7.983463658558309,
            20.909071486699013,
            13.870913475273927,
            -23.42950827215129,
        ]

        test_values["stoch_k_feat"] = [
            8.0,
            2.0,
            0.0,
            1.0,
            7.0,
            0.0,
            6.0,
            7.0,
            7.0,
            0.0,
        ]

        test_values["stoch_d"] = [
            10.242985064405225,
            -4.178710910106136,
            -4.072940244409965,
            3.468814196259942,
            -1.7741030352350586,
            -2.6611545528526257,
            3.2947627797222516,
            6.589525559444504,
            -11.089846201276814,
            0.6188555577068462,
        ]

        test_values["stoch_d_feat"] = [
            7.0,
            0.0,
            0.0,
            8.0,
            2.0,
            1.0,
            8.0,
            7.0,
            0.0,
            5.0,
        ]

        dates = pd.Index(
            [
                Timestamp("2012-01-19 00:00:00"),
                Timestamp("2012-01-20 00:00:00"),
                Timestamp("2012-01-21 00:00:00"),
                Timestamp("2012-01-22 00:00:00"),
                Timestamp("2012-01-23 00:00:00"),
                Timestamp("2012-01-24 00:00:00"),
                Timestamp("2012-01-25 00:00:00"),
                Timestamp("2012-01-26 00:00:00"),
                Timestamp("2012-01-27 00:00:00"),
                Timestamp("2012-01-28 00:00:00"),
            ],
            dtype="datetime64[ms]",
            name="date",
        )

        expected_df = pd.DataFrame(test_values, index=dates)

        pd.testing.assert_frame_equal(test_df.iloc[:10, -4:], expected_df)

        stoch_k_count = (
            test_df["stoch_k"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )
        stoch_d_count = (
            test_df["stoch_d"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )

        stoch_k_feat_count = (
            test_df["stoch_k_feat"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )
        stoch_d_feat_count = (
            test_df["stoch_d_feat"]
            .rename(None)
            .value_counts(bins=self.model_features.bins)
        )

        expected_count = {}
        expected_count["stoch_k_count"] = pd.Series(
            {
                Interval(-2.929, 9.003, closed="right"): 2320,
                Interval(-14.862, -2.929, closed="right"): 1194,
                Interval(9.003, 20.935, closed="right"): 500,
                Interval(-26.794, -14.862, closed="right"): 197,
                Interval(20.935, 32.867, closed="right"): 82,
                Interval(-38.726, -26.794, closed="right"): 38,
                Interval(32.867, 44.8, closed="right"): 17,
                Interval(-50.766999999999996, -38.726, closed="right"): 7,
                Interval(44.8, 56.732, closed="right"): 3,
            },
            name="count",
        )

        expected_count["stoch_d_count"] = pd.Series(
            {
                Interval(-2.932, 1.22, closed="right"): 2093,
                Interval(1.22, 5.372, closed="right"): 1139,
                Interval(-7.085, -2.932, closed="right"): 631,
                Interval(5.372, 9.525, closed="right"): 264,
                Interval(-11.237, -7.085, closed="right"): 141,
                Interval(9.525, 13.677, closed="right"): 55,
                Interval(-15.39, -11.237, closed="right"): 23,
                Interval(13.677, 17.83, closed="right"): 9,
                Interval(-19.581, -15.39, closed="right"): 3,
            },
            name="count",
        )

        expected_count["stoch_k_feat_count"] = pd.Series(
            {
                Interval(-0.009000000000000001, 0.889, closed="right"): 588,
                Interval(6.222, 7.111, closed="right"): 575,
                Interval(0.889, 1.778, closed="right"): 511,
                Interval(5.333, 6.222, closed="right"): 508,
                Interval(1.778, 2.667, closed="right"): 459,
                Interval(7.111, 8.0, closed="right"): 455,
                Interval(2.667, 3.556, closed="right"): 431,
                Interval(3.556, 4.444, closed="right"): 420,
                Interval(4.444, 5.333, closed="right"): 411,
            },
            name="count",
        )

        expected_count["stoch_d_feat_count"] = pd.Series(
            {
                Interval(6.222, 7.111, closed="right"): 582,
                Interval(-0.009000000000000001, 0.889, closed="right"): 525,
                Interval(0.889, 1.778, closed="right"): 516,
                Interval(1.778, 2.667, closed="right"): 505,
                Interval(7.111, 8.0, closed="right"): 474,
                Interval(2.667, 3.556, closed="right"): 473,
                Interval(3.556, 4.444, closed="right"): 437,
                Interval(5.333, 6.222, closed="right"): 428,
                Interval(4.444, 5.333, closed="right"): 418,
            },
            name="count",
        )

        pd.testing.assert_series_equal(
            stoch_k_count, expected_count["stoch_k_count"]
        )
        pd.testing.assert_series_equal(
            stoch_d_count, expected_count["stoch_d_count"]
        )

        pd.testing.assert_series_equal(
            stoch_k_feat_count, expected_count["stoch_k_feat_count"]
        )
        pd.testing.assert_series_equal(
            stoch_d_feat_count, expected_count["stoch_d_feat_count"]
        )


if __name__ == "__main__":
    unittest.main()
