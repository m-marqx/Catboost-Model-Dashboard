import unittest

import warnings

import pandas as pd
from pandas import Timestamp, Interval
import numpy as np

from machine_learning.model_features import feature_binning, ModelFeatures


def assert_count_series(
    test_count: dict[str, dict[Interval, int]],
    expected_count: dict[str, dict[Interval, int]],
):
    """
    Compare two dictionaries of count values and assert that their
    corresponding series are equal.

    Parameters
    ----------
    - test_count (dict[str, dict[Interval, int]]): The dictionary
    containing the test count values.
    - expected_count (dict[str, dict[Interval, int]]): The dictionary
    containing the expected count values.
    """
    test_count_series: dict[str, pd.Series] = {}
    expected_count_series: dict[str, pd.Series] = {}

    for key, value in zip(test_count.keys(), test_count.values()):
        test_count_series[key] = pd.Series(value)

    for key, value in zip(test_count.keys(), expected_count.values()):
        expected_count_series[key] = pd.Series(value)

    test_count_concat: pd.Series = pd.concat(test_count_series)
    expected_count_concat: pd.Series = pd.concat(expected_count_series)

    pd.testing.assert_series_equal(test_count_concat, expected_count_concat)


class TestAssertCount(unittest.TestCase):
    def setUp(self) -> None:
        self.test_count = {
            "count": {
                Interval(42.538, 51.888, closed="right"): 918,
                Interval(51.888, 61.237, closed="right"): 751,
                Interval(33.189, 42.538, closed="right"): 700,
                Interval(61.237, 70.586, closed="right"): 576,
                Interval(70.586, 79.936, closed="right"): 433,
                Interval(23.84, 33.189, closed="right"): 345,
                Interval(79.936, 89.285, closed="right"): 322,
                Interval(89.285, 98.634, closed="right"): 147,
                Interval(14.49, 23.84, closed="right"): 126,
                Interval(5.045999999999999, 14.49, closed="right"): 43,
            },
            "feat_count": {
                Interval(0.8, 1.6, closed="right"): 637,
                Interval(3.2, 4.0, closed="right"): 559,
                Interval(4.8, 5.6, closed="right"): 550,
                Interval(-0.009000000000000001, 0.8, closed="right"): 507,
                Interval(7.2, 8.0, closed="right"): 474,
                Interval(2.4, 3.2, closed="right"): 456,
                Interval(1.6, 2.4, closed="right"): 455,
                Interval(5.6, 6.4, closed="right"): 450,
                Interval(6.4, 7.2, closed="right"): 273,
                Interval(4.0, 4.8, closed="right"): 0,
            },
        }

    def test_assert_count_series(self):
        expected_count = {
            "count": {
                Interval(42.538, 51.888, closed="right"): 918,
                Interval(51.888, 61.237, closed="right"): 751,
                Interval(33.189, 42.538, closed="right"): 700,
                Interval(61.237, 70.586, closed="right"): 576,
                Interval(70.586, 79.936, closed="right"): 433,
                Interval(23.84, 33.189, closed="right"): 345,
                Interval(79.936, 89.285, closed="right"): 322,
                Interval(89.285, 98.634, closed="right"): 147,
                Interval(14.49, 23.84, closed="right"): 126,
                Interval(5.045999999999999, 14.49, closed="right"): 43,
            },
            "feat_count": {
                Interval(0.8, 1.6, closed="right"): 637,
                Interval(3.2, 4.0, closed="right"): 559,
                Interval(4.8, 5.6, closed="right"): 550,
                Interval(-0.009000000000000001, 0.8, closed="right"): 507,
                Interval(7.2, 8.0, closed="right"): 474,
                Interval(2.4, 3.2, closed="right"): 456,
                Interval(1.6, 2.4, closed="right"): 455,
                Interval(5.6, 6.4, closed="right"): 450,
                Interval(6.4, 7.2, closed="right"): 273,
                Interval(4.0, 4.8, closed="right"): 0,
            },
        }

        assert_count_series(self.test_count, expected_count)

    def test_assert_count_invalid(self):
        expected_count = {
            "count": {
                Interval(42.538, 51.888, closed="right"): 0,
                Interval(51.888, 61.237, closed="right"): 0,
                Interval(33.189, 42.538, closed="right"): 0,
                Interval(61.237, 70.586, closed="right"): 0,
                Interval(70.586, 79.936, closed="right"): 0,
                Interval(23.84, 33.189, closed="right"): 0,
                Interval(79.936, 89.285, closed="right"): 0,
                Interval(89.285, 98.634, closed="right"): 0,
                Interval(14.49, 23.84, closed="right"): 0,
                Interval(5.045999999999999, 14.49, closed="right"): 0,
            },
            "feat_count": {
                Interval(0.8, 1.6, closed="right"): 0,
                Interval(3.2, 4.0, closed="right"): 0,
                Interval(4.8, 5.6, closed="right"): 0,
                Interval(-0.009000000000000001, 0.8, closed="right"): 0,
                Interval(7.2, 8.0, closed="right"): 0,
                Interval(2.4, 3.2, closed="right"): 0,
                Interval(1.6, 2.4, closed="right"): 0,
                Interval(5.6, 6.4, closed="right"): 0,
                Interval(6.4, 7.2, closed="right"): 0,
                Interval(4.0, 4.8, closed="right"): 0,
            },
        }

        self.assertRaises(
            AssertionError,
            assert_count_series,
            self.test_count,
            expected_count,
        )


class TestFeatureBinning(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
        self.test_index = 1030
        self.bins = 9

        self.model_features = ModelFeatures(
            self.dataframe, self.test_index, self.bins, False
        )

        source = self.dataframe["close"].copy()

        self.test_df = feature_binning(
            source,
            self.test_index,
            10,
        )

    def test_feature_binning_name(self):
        self.assertEqual(self.test_df.name, "close")

    def test_feature_binning_source_has_inf(self):
        source = pd.Series([1, 2, 3, np.inf, 5, 6, 7, 8, 9, 10])

        self.assertRaises(
            ValueError,
            feature_binning,
            source,
            3,
            0,
        )

    def test_feature_binning_values(self):
        expected_df = pd.Series(
            {
                Timestamp("2012-01-02 00:00:00"): 0,
                Timestamp("2012-01-03 00:00:00"): 1,
                Timestamp("2012-01-04 00:00:00"): 1,
                Timestamp("2012-01-05 00:00:00"): 1,
                Timestamp("2012-01-06 00:00:00"): 1,
                Timestamp("2012-01-07 00:00:00"): 1,
                Timestamp("2012-01-08 00:00:00"): 1,
                Timestamp("2012-01-09 00:00:00"): 1,
                Timestamp("2012-01-10 00:00:00"): 1,
                Timestamp("2012-01-11 00:00:00"): 1,
                Timestamp("2012-01-12 00:00:00"): 1,
                Timestamp("2012-01-13 00:00:00"): 1,
                Timestamp("2012-01-14 00:00:00"): 1,
                Timestamp("2012-01-15 00:00:00"): 1,
                Timestamp("2012-01-16 00:00:00"): 1,
                Timestamp("2012-01-17 00:00:00"): 1,
                Timestamp("2012-01-18 00:00:00"): 1,
                Timestamp("2012-01-19 00:00:00"): 1,
                Timestamp("2012-01-20 00:00:00"): 1,
                Timestamp("2012-01-21 00:00:00"): 1,
            },
            name="close",
        )

        expected_df.index.name = "date"

        pd.testing.assert_series_equal(expected_df, self.test_df.iloc[:20])

    def test_feature_binning_counts(self):
        test_count = {
            "close": self.test_df.value_counts(bins=self.bins).to_dict()
        }

        expected_count = {
            "close": {
                Interval(8.0, 9.0, closed="right"): 2778,
                Interval(5.0, 6.0, closed="right"): 550,
                Interval(6.0, 7.0, closed="right"): 240,
                Interval(-0.009999999999999998, 1.0, closed="right"): 206,
                Interval(7.0, 8.0, closed="right"): 196,
                Interval(1.0, 2.0, closed="right"): 103,
                Interval(2.0, 3.0, closed="right"): 103,
                Interval(3.0, 4.0, closed="right"): 103,
                Interval(4.0, 5.0, closed="right"): 103,
            }
        }

        assert_count_series(test_count, expected_count)


class TestRSI(unittest.TestCase):
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

        source = self.dataframe["close"]
        length = 14
        ma_method = "ema"

        self.test_df = self.model_features.create_rsi_feature(
            source, length, ma_method
        ).dropna()[["RSI", "RSI_feat"]]

    def test_create_rsi_feature_columns(self):
        expected_columns = pd.Index(["RSI", "RSI_feat"])

        pd.testing.assert_index_equal(self.test_df.columns, expected_columns)

    def test_create_rsi_feature_values(self):
        expected_df = pd.DataFrame(
            {
                "RSI": {
                    Timestamp("2012-01-16 00:00:00"): 65.53254437869822,
                    Timestamp("2012-01-17 00:00:00"): 48.525446579036064,
                    Timestamp("2012-01-18 00:00:00"): 60.38736258255023,
                    Timestamp("2012-01-19 00:00:00"): 49.79513979849982,
                    Timestamp("2012-01-20 00:00:00"): 57.012399646774654,
                    Timestamp("2012-01-21 00:00:00"): 54.77221193135472,
                    Timestamp("2012-01-22 00:00:00"): 54.99890776717075,
                    Timestamp("2012-01-23 00:00:00"): 51.015778613199274,
                    Timestamp("2012-01-24 00:00:00"): 51.46640660824145,
                    Timestamp("2012-01-25 00:00:00"): 47.43806049658012,
                },
                "RSI_feat": {
                    Timestamp("2012-01-16 00:00:00"): 5.0,
                    Timestamp("2012-01-17 00:00:00"): 3.0,
                    Timestamp("2012-01-18 00:00:00"): 4.0,
                    Timestamp("2012-01-19 00:00:00"): 3.0,
                    Timestamp("2012-01-20 00:00:00"): 4.0,
                    Timestamp("2012-01-21 00:00:00"): 8.0,
                    Timestamp("2012-01-22 00:00:00"): 8.0,
                    Timestamp("2012-01-23 00:00:00"): 3.0,
                    Timestamp("2012-01-24 00:00:00"): 8.0,
                    Timestamp("2012-01-25 00:00:00"): 3.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(self.test_df.head(10), expected_df)

    def test_create_rsi_feature_counts(self):
        feat_columns = self.test_df.columns
        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        test_count = {}
        test_count["RSI"] = self.test_df["RSI"].value_counts(bins=10).to_dict()
        test_count["RSI_feat"] = (
            self.test_df["RSI_feat"].value_counts(bins=10).to_dict()
        )

        expected_count = {
            "RSI": {
                Interval(42.538, 51.888, closed="right"): 918,
                Interval(51.888, 61.237, closed="right"): 751,
                Interval(33.189, 42.538, closed="right"): 700,
                Interval(61.237, 70.586, closed="right"): 576,
                Interval(70.586, 79.936, closed="right"): 433,
                Interval(23.84, 33.189, closed="right"): 345,
                Interval(79.936, 89.285, closed="right"): 322,
                Interval(89.285, 98.634, closed="right"): 147,
                Interval(14.49, 23.84, closed="right"): 126,
                Interval(5.045999999999999, 14.49, closed="right"): 43,
            },
            "RSI_feat": {
                Interval(0.8, 1.6, closed="right"): 637,
                Interval(3.2, 4.0, closed="right"): 559,
                Interval(4.8, 5.6, closed="right"): 550,
                Interval(-0.009000000000000001, 0.8, closed="right"): 507,
                Interval(7.2, 8.0, closed="right"): 474,
                Interval(2.4, 3.2, closed="right"): 456,
                Interval(1.6, 2.4, closed="right"): 455,
                Interval(5.6, 6.4, closed="right"): 450,
                Interval(6.4, 7.2, closed="right"): 273,
                Interval(4.0, 4.8, closed="right"): 0,
            },
        }

        assert_count_series(test_count, expected_count)


class TestRSIOpt(unittest.TestCase):
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

        self.test_df = self.model_features.create_rsi_opt_feature(
            self.dataframe["close"].copy(), 14, "ema"
        ).dropna()

    def test_create_rsi_opt_feature_columns(self):
        expected_columns = pd.Index(["RSI", "RSI_feat"])

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_rsi_opt_feature_values(self):
        expected_df = pd.DataFrame(
            {
                "RSI": {
                    Timestamp("2012-01-18 00:00:00"): -3.638192938493816,
                    Timestamp("2012-01-19 00:00:00"): -0.8978086855093963,
                    Timestamp("2012-01-20 00:00:00"): -2.386459178140159,
                    Timestamp("2012-01-21 00:00:00"): -3.5193214555962395,
                    Timestamp("2012-01-22 00:00:00"): -1.4237537619312812,
                    Timestamp("2012-01-23 00:00:00"): 2.6561994723420774,
                    Timestamp("2012-01-24 00:00:00"): -2.4978555240279023,
                    Timestamp("2012-01-25 00:00:00"): 2.529828741435018,
                    Timestamp("2012-01-26 00:00:00"): 4.9874553777196,
                    Timestamp("2012-01-27 00:00:00"): -2.784056738965517,
                },
                "RSI_feat": {
                    Timestamp("2012-01-18 00:00:00"): 1.0,
                    Timestamp("2012-01-19 00:00:00"): 3.0,
                    Timestamp("2012-01-20 00:00:00"): 2.0,
                    Timestamp("2012-01-21 00:00:00"): 1.0,
                    Timestamp("2012-01-22 00:00:00"): 3.0,
                    Timestamp("2012-01-23 00:00:00"): 8.0,
                    Timestamp("2012-01-24 00:00:00"): 2.0,
                    Timestamp("2012-01-25 00:00:00"): 8.0,
                    Timestamp("2012-01-26 00:00:00"): 6.0,
                    Timestamp("2012-01-27 00:00:00"): 2.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_rsi_opt_feature_counts(self):
        feat_columns = self.test_df.columns[8:]
        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "RSI": {
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
            "RSI_feat": {
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
        }

        assert_count_series(test_count, expected_count)


class TestSlowStoch(unittest.TestCase):
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

        self.test_df: pd.DataFrame = (
            self.model_features.create_slow_stoch_feature("close").dropna()
        )

    def test_create_slow_stoch_feature_columns(self):
        expected_columns = pd.Index(
            ["stoch_k", "stoch_k_feat", "stoch_d", "stoch_d_feat"]
        )
        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_slow_stoch_feature_values(self):
        expected_df = pd.DataFrame(
            {
                "stoch_k": {
                    Timestamp("2012-01-17 00:00:00"): 43.67346938775511,
                    Timestamp("2012-01-18 00:00:00"): 83.13253012048192,
                    Timestamp("2012-01-19 00:00:00"): 37.634408602150536,
                    Timestamp("2012-01-20 00:00:00"): 76.88172043010755,
                    Timestamp("2012-01-21 00:00:00"): 67.2043010752688,
                    Timestamp("2012-01-22 00:00:00"): 68.27956989247313,
                    Timestamp("2012-01-23 00:00:00"): 53.76344086021504,
                    Timestamp("2012-01-24 00:00:00"): 55.37634408602151,
                    Timestamp("2012-01-25 00:00:00"): 42.47311827956989,
                    Timestamp("2012-01-26 00:00:00"): 0.0,
                },
                "stoch_k_feat": {
                    Timestamp("2012-01-17 00:00:00"): 2.0,
                    Timestamp("2012-01-18 00:00:00"): 6.0,
                    Timestamp("2012-01-19 00:00:00"): 2.0,
                    Timestamp("2012-01-20 00:00:00"): 5.0,
                    Timestamp("2012-01-21 00:00:00"): 4.0,
                    Timestamp("2012-01-22 00:00:00"): 4.0,
                    Timestamp("2012-01-23 00:00:00"): 3.0,
                    Timestamp("2012-01-24 00:00:00"): 3.0,
                    Timestamp("2012-01-25 00:00:00"): 2.0,
                    Timestamp("2012-01-26 00:00:00"): 0.0,
                },
                "stoch_d": {
                    Timestamp("2012-01-17 00:00:00"): 74.28571428571429,
                    Timestamp("2012-01-18 00:00:00"): 71.79247602655519,
                    Timestamp("2012-01-19 00:00:00"): 54.813469370129184,
                    Timestamp("2012-01-20 00:00:00"): 65.88288638424666,
                    Timestamp("2012-01-21 00:00:00"): 60.57347670250896,
                    Timestamp("2012-01-22 00:00:00"): 70.78853046594982,
                    Timestamp("2012-01-23 00:00:00"): 63.08243727598566,
                    Timestamp("2012-01-24 00:00:00"): 59.13978494623657,
                    Timestamp("2012-01-25 00:00:00"): 50.537634408602145,
                    Timestamp("2012-01-26 00:00:00"): 32.61648745519713,
                },
                "stoch_d_feat": {
                    Timestamp("2012-01-17 00:00:00"): 5.0,
                    Timestamp("2012-01-18 00:00:00"): 5.0,
                    Timestamp("2012-01-19 00:00:00"): 3.0,
                    Timestamp("2012-01-20 00:00:00"): 4.0,
                    Timestamp("2012-01-21 00:00:00"): 4.0,
                    Timestamp("2012-01-22 00:00:00"): 4.0,
                    Timestamp("2012-01-23 00:00:00"): 4.0,
                    Timestamp("2012-01-24 00:00:00"): 3.0,
                    Timestamp("2012-01-25 00:00:00"): 3.0,
                    Timestamp("2012-01-26 00:00:00"): 1.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(self.test_df.iloc[:10, 8:], expected_df)

    def test_create_slow_stoch_feature_count(self):
        test_df: pd.DataFrame = self.model_features.create_slow_stoch_feature(
            "close"
        ).dropna()

        test_count = {}

        for column in ["stoch_k", "stoch_k_feat", "stoch_d", "stoch_d_feat"]:
            test_count[column] = (
                test_df[column].value_counts(bins=10).to_dict()
            )

        expected_count = {}

        expected_count["stoch_k"] = {
            Interval(90.0, 100.0, closed="right"): 627,
            Interval(80.0, 90.0, closed="right"): 601,
            Interval(70.0, 80.0, closed="right"): 490,
            Interval(60.0, 70.0, closed="right"): 439,
            Interval(40.0, 50.0, closed="right"): 418,
            Interval(30.0, 40.0, closed="right"): 414,
            Interval(20.0, 30.0, closed="right"): 413,
            Interval(50.0, 60.0, closed="right"): 406,
            Interval(10.0, 20.0, closed="right"): 367,
            Interval(-0.101, 10.0, closed="right"): 185,
        }

        expected_count["stoch_k_feat"] = {
            Interval(-0.009000000000000001, 0.8, closed="right"): 680,
            Interval(0.8, 1.6, closed="right"): 575,
            Interval(5.6, 6.4, closed="right"): 495,
            Interval(1.6, 2.4, closed="right"): 492,
            Interval(2.4, 3.2, closed="right"): 487,
            Interval(3.2, 4.0, closed="right"): 465,
            Interval(6.4, 7.2, closed="right"): 462,
            Interval(4.8, 5.6, closed="right"): 366,
            Interval(7.2, 8.0, closed="right"): 338,
            Interval(4.0, 4.8, closed="right"): 0,
        }

        expected_count["stoch_d"] = {
            Interval(79.537, 89.161, closed="right"): 680,
            Interval(89.161, 98.785, closed="right"): 531,
            Interval(69.912, 79.537, closed="right"): 500,
            Interval(12.166, 21.791, closed="right"): 437,
            Interval(21.791, 31.415, closed="right"): 435,
            Interval(60.288, 69.912, closed="right"): 428,
            Interval(41.039, 50.664, closed="right"): 419,
            Interval(50.664, 60.288, closed="right"): 406,
            Interval(31.415, 41.039, closed="right"): 401,
            Interval(2.4450000000000003, 12.166, closed="right"): 123,
        }

        expected_count["stoch_d_feat"] = {
            Interval(-0.009000000000000001, 0.8, closed="right"): 682,
            Interval(0.8, 1.6, closed="right"): 573,
            Interval(3.2, 4.0, closed="right"): 502,
            Interval(5.6, 6.4, closed="right"): 483,
            Interval(1.6, 2.4, closed="right"): 478,
            Interval(2.4, 3.2, closed="right"): 477,
            Interval(7.2, 8.0, closed="right"): 477,
            Interval(4.8, 5.6, closed="right"): 408,
            Interval(6.4, 7.2, closed="right"): 280,
            Interval(4.0, 4.8, closed="right"): 0,
        }

        assert_count_series(test_count, expected_count)


class TestSlowStochOpt(unittest.TestCase):
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

        self.test_df = self.model_features.create_slow_stoch_opt_feature(
            "close"
        ).dropna()

    def test_create_slow_stoch_opt_feature_columns(self):
        expected_columns = pd.Index(
            ["stoch_k", "stoch_k_feat", "stoch_d", "stoch_d_feat"]
        )
        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_slow_stoch_opt_feature_values(self):
        expected_df = pd.DataFrame(
            {
                "stoch_k": {
                    Timestamp("2012-01-19 00:00:00"): 4.2702608334987495,
                    Timestamp("2012-01-20 00:00:00"): -4.419989919970305,
                    Timestamp("2012-01-21 00:00:00"): -20.909071486698963,
                    Timestamp("2012-01-22 00:00:00"): -6.0826389779489825,
                    Timestamp("2012-01-23 00:00:00"): 9.504123403045133,
                    Timestamp("2012-01-24 00:00:00"): -9.123958466923778,
                    Timestamp("2012-01-25 00:00:00"): 7.983463658558309,
                    Timestamp("2012-01-26 00:00:00"): 20.909071486699013,
                    Timestamp("2012-01-27 00:00:00"): 13.870913475273927,
                    Timestamp("2012-01-28 00:00:00"): -23.42950827215129,
                },
                "stoch_k_feat": {
                    Timestamp("2012-01-19 00:00:00"): 8.0,
                    Timestamp("2012-01-20 00:00:00"): 2.0,
                    Timestamp("2012-01-21 00:00:00"): 0.0,
                    Timestamp("2012-01-22 00:00:00"): 1.0,
                    Timestamp("2012-01-23 00:00:00"): 7.0,
                    Timestamp("2012-01-24 00:00:00"): 0.0,
                    Timestamp("2012-01-25 00:00:00"): 6.0,
                    Timestamp("2012-01-26 00:00:00"): 7.0,
                    Timestamp("2012-01-27 00:00:00"): 7.0,
                    Timestamp("2012-01-28 00:00:00"): 0.0,
                },
                "stoch_d": {
                    Timestamp("2012-01-19 00:00:00"): 10.242985064405225,
                    Timestamp("2012-01-20 00:00:00"): -4.178710910106136,
                    Timestamp("2012-01-21 00:00:00"): -4.072940244409965,
                    Timestamp("2012-01-22 00:00:00"): 3.468814196259942,
                    Timestamp("2012-01-23 00:00:00"): -1.7741030352350586,
                    Timestamp("2012-01-24 00:00:00"): -2.6611545528526257,
                    Timestamp("2012-01-25 00:00:00"): 3.2947627797222516,
                    Timestamp("2012-01-26 00:00:00"): 6.589525559444504,
                    Timestamp("2012-01-27 00:00:00"): -11.089846201276814,
                    Timestamp("2012-01-28 00:00:00"): 0.6188555577068462,
                },
                "stoch_d_feat": {
                    Timestamp("2012-01-19 00:00:00"): 7.0,
                    Timestamp("2012-01-20 00:00:00"): 0.0,
                    Timestamp("2012-01-21 00:00:00"): 0.0,
                    Timestamp("2012-01-22 00:00:00"): 8.0,
                    Timestamp("2012-01-23 00:00:00"): 2.0,
                    Timestamp("2012-01-24 00:00:00"): 1.0,
                    Timestamp("2012-01-25 00:00:00"): 8.0,
                    Timestamp("2012-01-26 00:00:00"): 7.0,
                    Timestamp("2012-01-27 00:00:00"): 0.0,
                    Timestamp("2012-01-28 00:00:00"): 5.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(self.test_df.iloc[:10, 8:], expected_df)

    def test_create_slow_stoch_opt_feature_count(self):
        test_count = {}

        for column in ["stoch_k", "stoch_k_feat", "stoch_d", "stoch_d_feat"]:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "stoch_k": {
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
            "stoch_k_feat": {
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
            "stoch_d": {
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
            "stoch_d_feat": {
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
        }

        assert_count_series(test_count, expected_count)


class TestDTW(unittest.TestCase):
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

        source = self.dataframe["close"]

        self.test_df_all = self.model_features.create_dtw_distance_feature(
            source, "all", 14
        ).dropna()

        self.test_df_mas = self.model_features.create_dtw_distance_feature(
            source, ["sma", "ema", "rma", "dema", "tema"], 14
        ).dropna()

    def test_create_dtw_distance_feature_columns(self):
        expected_columns = pd.Index(
            [
                "SMA_DTW",
                "SMA_DTW_feat",
                "EMA_DTW",
                "EMA_DTW_feat",
                "RMA_DTW",
                "RMA_DTW_feat",
                "DEMA_DTW",
                "DEMA_DTW_feat",
                "TEMA_DTW",
                "TEMA_DTW_feat",
            ]
        )
        pd.testing.assert_index_equal(
            self.test_df_all.columns[8:], expected_columns
        )

        pd.testing.assert_index_equal(
            self.test_df_mas.columns[8:], expected_columns
        )

    def test_create_dtw_distance_feature_values(self):
        expected_df = pd.DataFrame(
            {
                "SMA_DTW": {
                    Timestamp("2012-02-10 00:00:00"): -0.4299999999999997,
                    Timestamp("2012-02-11 00:00:00"): 0.10785714285714221,
                    Timestamp("2012-02-12 00:00:00"): 0.49571428571428555,
                    Timestamp("2012-02-13 00:00:00"): 0.5007142857142854,
                    Timestamp("2012-02-14 00:00:00"): 0.140714285714286,
                    Timestamp("2012-02-15 00:00:00"): -0.19928571428571473,
                    Timestamp("2012-02-16 00:00:00"): -0.17000000000000082,
                    Timestamp("2012-02-17 00:00:00"): -0.08000000000000007,
                    Timestamp("2012-02-18 00:00:00"): 0.09999999999999964,
                    Timestamp("2012-02-19 00:00:00"): 0.11999999999999922,
                },
                "SMA_DTW_feat": {
                    Timestamp("2012-02-10 00:00:00"): 2.0,
                    Timestamp("2012-02-11 00:00:00"): 8.0,
                    Timestamp("2012-02-12 00:00:00"): 6.0,
                    Timestamp("2012-02-13 00:00:00"): 6.0,
                    Timestamp("2012-02-14 00:00:00"): 8.0,
                    Timestamp("2012-02-15 00:00:00"): 2.0,
                    Timestamp("2012-02-16 00:00:00"): 2.0,
                    Timestamp("2012-02-17 00:00:00"): 3.0,
                    Timestamp("2012-02-18 00:00:00"): 8.0,
                    Timestamp("2012-02-19 00:00:00"): 8.0,
                },
                "EMA_DTW": {
                    Timestamp("2012-02-10 00:00:00"): 0.009498492218143362,
                    Timestamp("2012-02-11 00:00:00"): -0.9111013067442757,
                    Timestamp("2012-02-12 00:00:00"): -0.3009544658450398,
                    Timestamp("2012-02-13 00:00:00"): -0.22095446584503975,
                    Timestamp("2012-02-14 00:00:00"): -0.50095446584504,
                    Timestamp("2012-02-15 00:00:00"): 0.07904553415496007,
                    Timestamp("2012-02-16 00:00:00"): 0.47904553415496043,
                    Timestamp("2012-02-17 00:00:00"): 0.49904553415496,
                    Timestamp("2012-02-18 00:00:00"): 0.14050612960096576,
                    Timestamp("2012-02-19 00:00:00"): -0.0782280210124977,
                },
                "EMA_DTW_feat": {
                    Timestamp("2012-02-10 00:00:00"): 4.0,
                    Timestamp("2012-02-11 00:00:00"): 1.0,
                    Timestamp("2012-02-12 00:00:00"): 2.0,
                    Timestamp("2012-02-13 00:00:00"): 2.0,
                    Timestamp("2012-02-14 00:00:00"): 2.0,
                    Timestamp("2012-02-15 00:00:00"): 5.0,
                    Timestamp("2012-02-16 00:00:00"): 7.0,
                    Timestamp("2012-02-17 00:00:00"): 7.0,
                    Timestamp("2012-02-18 00:00:00"): 6.0,
                    Timestamp("2012-02-19 00:00:00"): 3.0,
                },
                "RMA_DTW": {
                    Timestamp("2012-02-10 00:00:00"): -0.49103493962253353,
                    Timestamp("2012-02-11 00:00:00"): -0.37596101536378157,
                    Timestamp("2012-02-12 00:00:00"): -0.6355352285520839,
                    Timestamp("2012-02-13 00:00:00"): -0.04513985508407803,
                    Timestamp("2012-02-14 00:00:00"): 0.3548601449159223,
                    Timestamp("2012-02-15 00:00:00"): 0.3748601449159219,
                    Timestamp("2012-02-16 00:00:00"): 0.08879870599335682,
                    Timestamp("2012-02-17 00:00:00"): -0.16825834443474097,
                    Timestamp("2012-02-18 00:00:00"): -0.07838274840368875,
                    Timestamp("2012-02-19 00:00:00"): 0.011617251596311995,
                },
                "RMA_DTW_feat": {
                    Timestamp("2012-02-10 00:00:00"): 1.0,
                    Timestamp("2012-02-11 00:00:00"): 1.0,
                    Timestamp("2012-02-12 00:00:00"): 1.0,
                    Timestamp("2012-02-13 00:00:00"): 8.0,
                    Timestamp("2012-02-14 00:00:00"): 5.0,
                    Timestamp("2012-02-15 00:00:00"): 5.0,
                    Timestamp("2012-02-16 00:00:00"): 4.0,
                    Timestamp("2012-02-17 00:00:00"): 2.0,
                    Timestamp("2012-02-18 00:00:00"): 8.0,
                    Timestamp("2012-02-19 00:00:00"): 3.0,
                },
                "DEMA_DTW": {
                    Timestamp("2012-02-10 00:00:00"): -0.05404751200735447,
                    Timestamp("2012-02-11 00:00:00"): -0.03098801797227768,
                    Timestamp("2012-02-12 00:00:00"): -0.008316877688645974,
                    Timestamp("2012-02-13 00:00:00"): 0.011526189949968568,
                    Timestamp("2012-02-14 00:00:00"): 0.044892295154972395,
                    Timestamp("2012-02-15 00:00:00"): 0.091378084483976,
                    Timestamp("2012-02-16 00:00:00"): -0.18990819969973316,
                    Timestamp("2012-02-17 00:00:00"): -0.12589819626972254,
                    Timestamp("2012-02-18 00:00:00"): -0.035898196269721794,
                    Timestamp("2012-02-19 00:00:00"): 0.14410180373027792,
                },
                "DEMA_DTW_feat": {
                    Timestamp("2012-02-10 00:00:00"): 3.0,
                    Timestamp("2012-02-11 00:00:00"): 3.0,
                    Timestamp("2012-02-12 00:00:00"): 4.0,
                    Timestamp("2012-02-13 00:00:00"): 4.0,
                    Timestamp("2012-02-14 00:00:00"): 5.0,
                    Timestamp("2012-02-15 00:00:00"): 5.0,
                    Timestamp("2012-02-16 00:00:00"): 2.0,
                    Timestamp("2012-02-17 00:00:00"): 2.0,
                    Timestamp("2012-02-18 00:00:00"): 3.0,
                    Timestamp("2012-02-19 00:00:00"): 5.0,
                },
                "TEMA_DTW": {
                    Timestamp("2012-02-10 00:00:00"): 0.06396370376827143,
                    Timestamp("2012-02-11 00:00:00"): -0.20825600309942427,
                    Timestamp("2012-02-12 00:00:00"): -0.017882333373306913,
                    Timestamp("2012-02-13 00:00:00"): -0.046163648770589205,
                    Timestamp("2012-02-14 00:00:00"): 0.36526835215781794,
                    Timestamp("2012-02-15 00:00:00"): -0.2486646362476943,
                    Timestamp("2012-02-16 00:00:00"): 0.023810507382901136,
                    Timestamp("2012-02-17 00:00:00"): -0.034884050180921555,
                    Timestamp("2012-02-18 00:00:00"): 0.03594588005311827,
                    Timestamp("2012-02-19 00:00:00"): 0.19594588005311842,
                },
                "TEMA_DTW_feat": {
                    Timestamp("2012-02-10 00:00:00"): 5.0,
                    Timestamp("2012-02-11 00:00:00"): 2.0,
                    Timestamp("2012-02-12 00:00:00"): 4.0,
                    Timestamp("2012-02-13 00:00:00"): 3.0,
                    Timestamp("2012-02-14 00:00:00"): 6.0,
                    Timestamp("2012-02-15 00:00:00"): 2.0,
                    Timestamp("2012-02-16 00:00:00"): 5.0,
                    Timestamp("2012-02-17 00:00:00"): 3.0,
                    Timestamp("2012-02-18 00:00:00"): 5.0,
                    Timestamp("2012-02-19 00:00:00"): 6.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(
            self.test_df_all.iloc[:10, 8:], expected_df
        )
        pd.testing.assert_frame_equal(
            self.test_df_mas.iloc[:10, 8:], expected_df
        )

    def test_create_dtw_distance_feature_count(self):
        test_count = {}

        for column in self.test_df_all.columns[8:]:
            test_count[column] = (
                self.test_df_all[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count: dict[str, dict] = {
            "SMA_DTW": {
                Interval(-32.987, 1122.509, closed="right"): 3596,
                Interval(-1188.482, -32.987, closed="right"): 695,
                Interval(-2343.977, -1188.482, closed="right"): 15,
                Interval(1122.509, 2278.004, closed="right"): 13,
                Interval(-3509.873, -2343.977, closed="right"): 6,
                Interval(2278.004, 3433.5, closed="right"): 5,
                Interval(3433.5, 4588.995, closed="right"): 3,
                Interval(4588.995, 5744.49, closed="right"): 2,
                Interval(5744.49, 6899.986, closed="right"): 1,
            },
            "SMA_DTW_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1141,
                Interval(6.222, 7.111, closed="right"): 1107,
                Interval(7.111, 8.0, closed="right"): 503,
                Interval(0.889, 1.778, closed="right"): 409,
                Interval(5.333, 6.222, closed="right"): 388,
                Interval(4.444, 5.333, closed="right"): 347,
                Interval(1.778, 2.667, closed="right"): 191,
                Interval(2.667, 3.556, closed="right"): 128,
                Interval(3.556, 4.444, closed="right"): 122,
            },
            "EMA_DTW": {
                Interval(-1243.089, 165.411, closed="right"): 3944,
                Interval(165.411, 1573.911, closed="right"): 336,
                Interval(1573.911, 2982.411, closed="right"): 18,
                Interval(-2651.589, -1243.089, closed="right"): 17,
                Interval(-4060.089, -2651.589, closed="right"): 7,
                Interval(2982.411, 4390.911, closed="right"): 6,
                Interval(4390.911, 5799.411, closed="right"): 5,
                Interval(-5481.2660000000005, -4060.089, closed="right"): 2,
                Interval(5799.411, 7207.91, closed="right"): 1,
            },
            "EMA_DTW_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1469,
                Interval(7.111, 8.0, closed="right"): 1462,
                Interval(6.222, 7.111, closed="right"): 363,
                Interval(0.889, 1.778, closed="right"): 321,
                Interval(1.778, 2.667, closed="right"): 200,
                Interval(5.333, 6.222, closed="right"): 152,
                Interval(4.444, 5.333, closed="right"): 133,
                Interval(2.667, 3.556, closed="right"): 121,
                Interval(3.556, 4.444, closed="right"): 115,
            },
            "RMA_DTW": {
                Interval(-388.063, 448.552, closed="right"): 4101,
                Interval(-1224.678, -388.063, closed="right"): 117,
                Interval(448.552, 1285.167, closed="right"): 78,
                Interval(-2061.293, -1224.678, closed="right"): 17,
                Interval(1285.167, 2121.782, closed="right"): 12,
                Interval(2121.782, 2958.397, closed="right"): 4,
                Interval(-2905.438, -2061.293, closed="right"): 3,
                Interval(3795.012, 4631.627, closed="right"): 3,
                Interval(2958.397, 3795.012, closed="right"): 1,
            },
            "RMA_DTW_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1288,
                Interval(6.222, 7.111, closed="right"): 1242,
                Interval(1.778, 2.667, closed="right"): 412,
                Interval(5.333, 6.222, closed="right"): 374,
                Interval(7.111, 8.0, closed="right"): 310,
                Interval(0.889, 1.778, closed="right"): 298,
                Interval(4.444, 5.333, closed="right"): 160,
                Interval(3.556, 4.444, closed="right"): 135,
                Interval(2.667, 3.556, closed="right"): 117,
            },
            "DEMA_DTW": {
                Interval(-930.132, 116.792, closed="right"): 3843,
                Interval(116.792, 1163.716, closed="right"): 424,
                Interval(-1977.056, -930.132, closed="right"): 31,
                Interval(1163.716, 2210.641, closed="right"): 19,
                Interval(2210.641, 3257.565, closed="right"): 7,
                Interval(-3033.404, -1977.056, closed="right"): 4,
                Interval(3257.565, 4304.489, closed="right"): 4,
                Interval(4304.489, 5351.414, closed="right"): 2,
                Interval(5351.414, 6398.338, closed="right"): 2,
            },
            "DEMA_DTW_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1474,
                Interval(6.222, 7.111, closed="right"): 1155,
                Interval(7.111, 8.0, closed="right"): 665,
                Interval(0.889, 1.778, closed="right"): 324,
                Interval(5.333, 6.222, closed="right"): 184,
                Interval(1.778, 2.667, closed="right"): 158,
                Interval(4.444, 5.333, closed="right"): 129,
                Interval(2.667, 3.556, closed="right"): 126,
                Interval(3.556, 4.444, closed="right"): 121,
            },
            "TEMA_DTW": {
                Interval(-481.77, 363.106, closed="right"): 4056,
                Interval(363.106, 1207.983, closed="right"): 136,
                Interval(-1326.646, -481.77, closed="right"): 84,
                Interval(1207.983, 2052.859, closed="right"): 22,
                Interval(-2171.523, -1326.646, closed="right"): 20,
                Interval(-3016.399, -2171.523, closed="right"): 11,
                Interval(2052.859, 2897.735, closed="right"): 4,
                Interval(-3868.88, -3016.399, closed="right"): 2,
                Interval(2897.735, 3742.612, closed="right"): 1,
            },
            "TEMA_DTW_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1425,
                Interval(7.111, 8.0, closed="right"): 1340,
                Interval(6.222, 7.111, closed="right"): 443,
                Interval(0.889, 1.778, closed="right"): 390,
                Interval(1.778, 2.667, closed="right"): 190,
                Interval(5.333, 6.222, closed="right"): 180,
                Interval(2.667, 3.556, closed="right"): 128,
                Interval(4.444, 5.333, closed="right"): 124,
                Interval(3.556, 4.444, closed="right"): 116,
            },
        }

        assert_count_series(test_count, expected_count)


class TestDTWNegativeCase(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc["2023"]
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

        self.test_index = 100
        self.bins = 9

        self.target = self.dataframe["Target_bin"].copy()

        self.model_features = ModelFeatures(
            self.dataframe, self.test_index, self.bins, False
        )

        self.model_features = ModelFeatures(
            self.dataframe, self.test_index, self.bins, False
        )

    def test_create_dtw_distance_feature_empty_mas(self):
        expected_df = self.dataframe.copy()
        source = self.dataframe["close"]

        test_df = self.model_features.create_dtw_distance_feature(
            source, "", 14
        )
        pd.testing.assert_frame_equal(test_df, expected_df)

    def test_create_dtw_distance_feature_invalid_mas(self):
        self.assertRaises(
            AttributeError,
            self.model_features.create_dtw_distance_feature,
            self.dataframe["close"],
            "invalid",
            14,
        )

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

        expected_values: dict[str, list] = {}

        expected_values["RSI"] = [
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

        expected_values["RSI_feat"] = [
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

        expected_df = pd.DataFrame(expected_values, index=dates)

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

        expected_values: dict[str, list] = {}

        expected_values["stoch_k"] = [
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

        expected_values["stoch_k_feat"] = [
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

        expected_values["stoch_d"] = [
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

        expected_values["stoch_d_feat"] = [
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

        expected_df = pd.DataFrame(expected_values, index=dates)

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

        expected_values: dict[str, list] = {}

        expected_values["stoch_k"] = [
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

        expected_values["stoch_k_feat"] = [
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

        expected_values["stoch_d"] = [
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

        expected_values["stoch_d_feat"] = [
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

        expected_df = pd.DataFrame(expected_values, index=dates)

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

    #test all moving averages only using `all` keyword
    def test_create_dtw_distance_feature_all(self):
        source = self.dataframe["close"]

        test_df = self.model_features.create_dtw_distance_feature(
            source, "all", 14
        ).dropna()

        expected_columns = pd.Index(
            [
                "SMA_DTW",
                "SMA_DTW_feat",
                "EMA_DTW",
                "EMA_DTW_feat",
                "RMA_DTW",
                "RMA_DTW_feat",
                "DEMA_DTW",
                "DEMA_DTW_feat",
                "TEMA_DTW",
                "TEMA_DTW_feat",
            ]
        )

        pd.testing.assert_index_equal(test_df.columns[8:], expected_columns)

        expected_values: dict[str, list] = {}

        expected_values["SMA_DTW"] = [
            -0.4299999999999997,
            0.10785714285714221,
            0.49571428571428555,
            0.5007142857142854,
            0.140714285714286,
            -0.19928571428571473,
            -0.17000000000000082,
            -0.08000000000000007,
            0.09999999999999964,
            0.11999999999999922,
        ]

        expected_values["SMA_DTW_feat"] = [
            2.0,
            8.0,
            6.0,
            6.0,
            8.0,
            2.0,
            2.0,
            3.0,
            8.0,
            8.0,
        ]

        expected_values["EMA_DTW"] = [
            0.009498492218143362,
            -0.9111013067442757,
            -0.3009544658450398,
            -0.22095446584503975,
            -0.50095446584504,
            0.07904553415496007,
            0.47904553415496043,
            0.49904553415496,
            0.14050612960096576,
            -0.0782280210124977,
        ]

        expected_values["EMA_DTW_feat"] = [
            4.0,
            1.0,
            2.0,
            2.0,
            2.0,
            5.0,
            7.0,
            7.0,
            6.0,
            3.0,
        ]

        expected_values["RMA_DTW"] = [
            -0.49103493962253353,
            -0.37596101536378157,
            -0.6355352285520839,
            -0.04513985508407803,
            0.3548601449159223,
            0.3748601449159219,
            0.08879870599335682,
            -0.16825834443474097,
            -0.07838274840368875,
            0.011617251596311995,
        ]

        expected_values["RMA_DTW_feat"] = [
            1.0,
            1.0,
            1.0,
            8.0,
            5.0,
            5.0,
            4.0,
            2.0,
            8.0,
            3.0,
        ]

        expected_values["DEMA_DTW"] = [
            -0.05404751200735447,
            -0.03098801797227768,
            -0.008316877688645974,
            0.011526189949968568,
            0.044892295154972395,
            0.091378084483976,
            -0.18990819969973316,
            -0.12589819626972254,
            -0.035898196269721794,
            0.14410180373027792,
        ]

        expected_values["DEMA_DTW_feat"] = [
            3.0,
            3.0,
            4.0,
            4.0,
            5.0,
            5.0,
            2.0,
            2.0,
            3.0,
            5.0,
        ]

        expected_values["TEMA_DTW"] = [
            0.06396370376827143,
            -0.20825600309942427,
            -0.017882333373306913,
            -0.046163648770589205,
            0.36526835215781794,
            -0.2486646362476943,
            0.023810507382901136,
            -0.034884050180921555,
            0.03594588005311827,
            0.19594588005311842,
        ]

        expected_values["TEMA_DTW_feat"] = [
            5.0,
            2.0,
            4.0,
            3.0,
            6.0,
            2.0,
            5.0,
            3.0,
            5.0,
            6.0,
        ]

        dates = pd.Index(
            [
                Timestamp("2012-02-10 00:00:00"),
                Timestamp("2012-02-11 00:00:00"),
                Timestamp("2012-02-12 00:00:00"),
                Timestamp("2012-02-13 00:00:00"),
                Timestamp("2012-02-14 00:00:00"),
                Timestamp("2012-02-15 00:00:00"),
                Timestamp("2012-02-16 00:00:00"),
                Timestamp("2012-02-17 00:00:00"),
                Timestamp("2012-02-18 00:00:00"),
                Timestamp("2012-02-19 00:00:00"),
            ],
            dtype="datetime64[ns]",
            name="date",
        )

        expected_df = pd.DataFrame(expected_values, index=dates)

        pd.testing.assert_frame_equal(test_df.iloc[:10, 8:], expected_df)

        test_count = {}

        for col in expected_columns:
            test_count[col] = (
                test_df[col].value_counts(bins=self.bins).to_dict()
            )

        expected_count: dict[str, dict] = {}

        expected_count["SMA_DTW"] = {
            Interval(-32.987, 1122.509, closed="right"): 3596,
            Interval(-1188.482, -32.987, closed="right"): 695,
            Interval(-2343.977, -1188.482, closed="right"): 15,
            Interval(1122.509, 2278.004, closed="right"): 13,
            Interval(-3509.873, -2343.977, closed="right"): 6,
            Interval(2278.004, 3433.5, closed="right"): 5,
            Interval(3433.5, 4588.995, closed="right"): 3,
            Interval(4588.995, 5744.49, closed="right"): 2,
            Interval(5744.49, 6899.986, closed="right"): 1,
        }

        expected_count["SMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1141,
            Interval(6.222, 7.111, closed="right"): 1107,
            Interval(7.111, 8.0, closed="right"): 503,
            Interval(0.889, 1.778, closed="right"): 409,
            Interval(5.333, 6.222, closed="right"): 388,
            Interval(4.444, 5.333, closed="right"): 347,
            Interval(1.778, 2.667, closed="right"): 191,
            Interval(2.667, 3.556, closed="right"): 128,
            Interval(3.556, 4.444, closed="right"): 122,
        }

        expected_count["EMA_DTW"] = {
            Interval(-1243.089, 165.411, closed="right"): 3944,
            Interval(165.411, 1573.911, closed="right"): 336,
            Interval(1573.911, 2982.411, closed="right"): 18,
            Interval(-2651.589, -1243.089, closed="right"): 17,
            Interval(-4060.089, -2651.589, closed="right"): 7,
            Interval(2982.411, 4390.911, closed="right"): 6,
            Interval(4390.911, 5799.411, closed="right"): 5,
            Interval(-5481.2660000000005, -4060.089, closed="right"): 2,
            Interval(5799.411, 7207.91, closed="right"): 1,
        }

        expected_count["EMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1469,
            Interval(7.111, 8.0, closed="right"): 1462,
            Interval(6.222, 7.111, closed="right"): 363,
            Interval(0.889, 1.778, closed="right"): 321,
            Interval(1.778, 2.667, closed="right"): 200,
            Interval(5.333, 6.222, closed="right"): 152,
            Interval(4.444, 5.333, closed="right"): 133,
            Interval(2.667, 3.556, closed="right"): 121,
            Interval(3.556, 4.444, closed="right"): 115,
        }

        expected_count["RMA_DTW"] = {
            Interval(-388.063, 448.552, closed="right"): 4101,
            Interval(-1224.678, -388.063, closed="right"): 117,
            Interval(448.552, 1285.167, closed="right"): 78,
            Interval(-2061.293, -1224.678, closed="right"): 17,
            Interval(1285.167, 2121.782, closed="right"): 12,
            Interval(2121.782, 2958.397, closed="right"): 4,
            Interval(-2905.438, -2061.293, closed="right"): 3,
            Interval(3795.012, 4631.627, closed="right"): 3,
            Interval(2958.397, 3795.012, closed="right"): 1,
        }

        expected_count["RMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1288,
            Interval(6.222, 7.111, closed="right"): 1242,
            Interval(1.778, 2.667, closed="right"): 412,
            Interval(5.333, 6.222, closed="right"): 374,
            Interval(7.111, 8.0, closed="right"): 310,
            Interval(0.889, 1.778, closed="right"): 298,
            Interval(4.444, 5.333, closed="right"): 160,
            Interval(3.556, 4.444, closed="right"): 135,
            Interval(2.667, 3.556, closed="right"): 117,
        }

        expected_count["DEMA_DTW"] = {
            Interval(-930.132, 116.792, closed="right"): 3843,
            Interval(116.792, 1163.716, closed="right"): 424,
            Interval(-1977.056, -930.132, closed="right"): 31,
            Interval(1163.716, 2210.641, closed="right"): 19,
            Interval(2210.641, 3257.565, closed="right"): 7,
            Interval(-3033.404, -1977.056, closed="right"): 4,
            Interval(3257.565, 4304.489, closed="right"): 4,
            Interval(4304.489, 5351.414, closed="right"): 2,
            Interval(5351.414, 6398.338, closed="right"): 2,
        }

        expected_count["DEMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1474,
            Interval(6.222, 7.111, closed="right"): 1155,
            Interval(7.111, 8.0, closed="right"): 665,
            Interval(0.889, 1.778, closed="right"): 324,
            Interval(5.333, 6.222, closed="right"): 184,
            Interval(1.778, 2.667, closed="right"): 158,
            Interval(4.444, 5.333, closed="right"): 129,
            Interval(2.667, 3.556, closed="right"): 126,
            Interval(3.556, 4.444, closed="right"): 121,
        }

        expected_count["TEMA_DTW"] = {
            Interval(-481.77, 363.106, closed="right"): 4056,
            Interval(363.106, 1207.983, closed="right"): 136,
            Interval(-1326.646, -481.77, closed="right"): 84,
            Interval(1207.983, 2052.859, closed="right"): 22,
            Interval(-2171.523, -1326.646, closed="right"): 20,
            Interval(-3016.399, -2171.523, closed="right"): 11,
            Interval(2052.859, 2897.735, closed="right"): 4,
            Interval(-3868.88, -3016.399, closed="right"): 2,
            Interval(2897.735, 3742.612, closed="right"): 1,
        }

        expected_count["TEMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1425,
            Interval(7.111, 8.0, closed="right"): 1340,
            Interval(6.222, 7.111, closed="right"): 443,
            Interval(0.889, 1.778, closed="right"): 390,
            Interval(1.778, 2.667, closed="right"): 190,
            Interval(5.333, 6.222, closed="right"): 180,
            Interval(2.667, 3.556, closed="right"): 128,
            Interval(4.444, 5.333, closed="right"): 124,
            Interval(3.556, 4.444, closed="right"): 116,
        }

        test_count_series: dict[str, pd.Series] = {}
        expected_count_series: dict[str, pd.Series] = {}

        for key, value in zip(test_count.keys(), test_count.values()):
            test_count_series[key] = pd.Series(value)
        test_count_concat: pd.Series = pd.concat(test_count_series)

        for key, value in zip(test_count.keys(), expected_count.values()):
            expected_count_series[key] = pd.Series(value)
        expected_count_concat: pd.Series = pd.concat(expected_count_series)

        pd.testing.assert_series_equal(
            test_count_concat, expected_count_concat
        )

    # test all moving averages without `all` keyword
    def test_create_dtw_distance_feature_all_mas(self):
        source = self.dataframe["close"]

        test_df = self.model_features.create_dtw_distance_feature(
            source, ["sma", "ema", "rma", "dema", "tema"], 14
        ).dropna()

        expected_columns = pd.Index(
            [
                "SMA_DTW",
                "SMA_DTW_feat",
                "EMA_DTW",
                "EMA_DTW_feat",
                "RMA_DTW",
                "RMA_DTW_feat",
                "DEMA_DTW",
                "DEMA_DTW_feat",
                "TEMA_DTW",
                "TEMA_DTW_feat",
            ]
        )

        pd.testing.assert_index_equal(test_df.columns[8:], expected_columns)

        expected_values: dict[str, list] = {}

        expected_values["SMA_DTW"] = [
            -0.4299999999999997,
            0.10785714285714221,
            0.49571428571428555,
            0.5007142857142854,
            0.140714285714286,
            -0.19928571428571473,
            -0.17000000000000082,
            -0.08000000000000007,
            0.09999999999999964,
            0.11999999999999922,
        ]

        expected_values["SMA_DTW_feat"] = [
            2.0,
            8.0,
            6.0,
            6.0,
            8.0,
            2.0,
            2.0,
            3.0,
            8.0,
            8.0,
        ]

        expected_values["EMA_DTW"] = [
            0.009498492218143362,
            -0.9111013067442757,
            -0.3009544658450398,
            -0.22095446584503975,
            -0.50095446584504,
            0.07904553415496007,
            0.47904553415496043,
            0.49904553415496,
            0.14050612960096576,
            -0.0782280210124977,
        ]

        expected_values["EMA_DTW_feat"] = [
            4.0,
            1.0,
            2.0,
            2.0,
            2.0,
            5.0,
            7.0,
            7.0,
            6.0,
            3.0,
        ]

        expected_values["RMA_DTW"] = [
            -0.49103493962253353,
            -0.37596101536378157,
            -0.6355352285520839,
            -0.04513985508407803,
            0.3548601449159223,
            0.3748601449159219,
            0.08879870599335682,
            -0.16825834443474097,
            -0.07838274840368875,
            0.011617251596311995,
        ]

        expected_values["RMA_DTW_feat"] = [
            1.0,
            1.0,
            1.0,
            8.0,
            5.0,
            5.0,
            4.0,
            2.0,
            8.0,
            3.0,
        ]

        expected_values["DEMA_DTW"] = [
            -0.05404751200735447,
            -0.03098801797227768,
            -0.008316877688645974,
            0.011526189949968568,
            0.044892295154972395,
            0.091378084483976,
            -0.18990819969973316,
            -0.12589819626972254,
            -0.035898196269721794,
            0.14410180373027792,
        ]

        expected_values["DEMA_DTW_feat"] = [
            3.0,
            3.0,
            4.0,
            4.0,
            5.0,
            5.0,
            2.0,
            2.0,
            3.0,
            5.0,
        ]

        expected_values["TEMA_DTW"] = [
            0.06396370376827143,
            -0.20825600309942427,
            -0.017882333373306913,
            -0.046163648770589205,
            0.36526835215781794,
            -0.2486646362476943,
            0.023810507382901136,
            -0.034884050180921555,
            0.03594588005311827,
            0.19594588005311842,
        ]

        expected_values["TEMA_DTW_feat"] = [
            5.0,
            2.0,
            4.0,
            3.0,
            6.0,
            2.0,
            5.0,
            3.0,
            5.0,
            6.0,
        ]

        dates = pd.Index(
            [
                Timestamp("2012-02-10 00:00:00"),
                Timestamp("2012-02-11 00:00:00"),
                Timestamp("2012-02-12 00:00:00"),
                Timestamp("2012-02-13 00:00:00"),
                Timestamp("2012-02-14 00:00:00"),
                Timestamp("2012-02-15 00:00:00"),
                Timestamp("2012-02-16 00:00:00"),
                Timestamp("2012-02-17 00:00:00"),
                Timestamp("2012-02-18 00:00:00"),
                Timestamp("2012-02-19 00:00:00"),
            ],
            dtype="datetime64[ns]",
            name="date",
        )

        expected_df = pd.DataFrame(expected_values, index=dates)

        pd.testing.assert_frame_equal(test_df.iloc[:10, 8:], expected_df)

        test_count = {}

        for col in expected_columns:
            test_count[col] = (
                test_df[col].value_counts(bins=self.bins).to_dict()
            )

        expected_count: dict[str, dict] = {}

        expected_count["SMA_DTW"] = {
            Interval(-32.987, 1122.509, closed="right"): 3596,
            Interval(-1188.482, -32.987, closed="right"): 695,
            Interval(-2343.977, -1188.482, closed="right"): 15,
            Interval(1122.509, 2278.004, closed="right"): 13,
            Interval(-3509.873, -2343.977, closed="right"): 6,
            Interval(2278.004, 3433.5, closed="right"): 5,
            Interval(3433.5, 4588.995, closed="right"): 3,
            Interval(4588.995, 5744.49, closed="right"): 2,
            Interval(5744.49, 6899.986, closed="right"): 1,
        }

        expected_count["SMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1141,
            Interval(6.222, 7.111, closed="right"): 1107,
            Interval(7.111, 8.0, closed="right"): 503,
            Interval(0.889, 1.778, closed="right"): 409,
            Interval(5.333, 6.222, closed="right"): 388,
            Interval(4.444, 5.333, closed="right"): 347,
            Interval(1.778, 2.667, closed="right"): 191,
            Interval(2.667, 3.556, closed="right"): 128,
            Interval(3.556, 4.444, closed="right"): 122,
        }

        expected_count["EMA_DTW"] = {
            Interval(-1243.089, 165.411, closed="right"): 3944,
            Interval(165.411, 1573.911, closed="right"): 336,
            Interval(1573.911, 2982.411, closed="right"): 18,
            Interval(-2651.589, -1243.089, closed="right"): 17,
            Interval(-4060.089, -2651.589, closed="right"): 7,
            Interval(2982.411, 4390.911, closed="right"): 6,
            Interval(4390.911, 5799.411, closed="right"): 5,
            Interval(-5481.2660000000005, -4060.089, closed="right"): 2,
            Interval(5799.411, 7207.91, closed="right"): 1,
        }

        expected_count["EMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1469,
            Interval(7.111, 8.0, closed="right"): 1462,
            Interval(6.222, 7.111, closed="right"): 363,
            Interval(0.889, 1.778, closed="right"): 321,
            Interval(1.778, 2.667, closed="right"): 200,
            Interval(5.333, 6.222, closed="right"): 152,
            Interval(4.444, 5.333, closed="right"): 133,
            Interval(2.667, 3.556, closed="right"): 121,
            Interval(3.556, 4.444, closed="right"): 115,
        }

        expected_count["RMA_DTW"] = {
            Interval(-388.063, 448.552, closed="right"): 4101,
            Interval(-1224.678, -388.063, closed="right"): 117,
            Interval(448.552, 1285.167, closed="right"): 78,
            Interval(-2061.293, -1224.678, closed="right"): 17,
            Interval(1285.167, 2121.782, closed="right"): 12,
            Interval(2121.782, 2958.397, closed="right"): 4,
            Interval(-2905.438, -2061.293, closed="right"): 3,
            Interval(3795.012, 4631.627, closed="right"): 3,
            Interval(2958.397, 3795.012, closed="right"): 1,
        }

        expected_count["RMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1288,
            Interval(6.222, 7.111, closed="right"): 1242,
            Interval(1.778, 2.667, closed="right"): 412,
            Interval(5.333, 6.222, closed="right"): 374,
            Interval(7.111, 8.0, closed="right"): 310,
            Interval(0.889, 1.778, closed="right"): 298,
            Interval(4.444, 5.333, closed="right"): 160,
            Interval(3.556, 4.444, closed="right"): 135,
            Interval(2.667, 3.556, closed="right"): 117,
        }

        expected_count["DEMA_DTW"] = {
            Interval(-930.132, 116.792, closed="right"): 3843,
            Interval(116.792, 1163.716, closed="right"): 424,
            Interval(-1977.056, -930.132, closed="right"): 31,
            Interval(1163.716, 2210.641, closed="right"): 19,
            Interval(2210.641, 3257.565, closed="right"): 7,
            Interval(-3033.404, -1977.056, closed="right"): 4,
            Interval(3257.565, 4304.489, closed="right"): 4,
            Interval(4304.489, 5351.414, closed="right"): 2,
            Interval(5351.414, 6398.338, closed="right"): 2,
        }

        expected_count["DEMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1474,
            Interval(6.222, 7.111, closed="right"): 1155,
            Interval(7.111, 8.0, closed="right"): 665,
            Interval(0.889, 1.778, closed="right"): 324,
            Interval(5.333, 6.222, closed="right"): 184,
            Interval(1.778, 2.667, closed="right"): 158,
            Interval(4.444, 5.333, closed="right"): 129,
            Interval(2.667, 3.556, closed="right"): 126,
            Interval(3.556, 4.444, closed="right"): 121,
        }

        expected_count["TEMA_DTW"] = {
            Interval(-481.77, 363.106, closed="right"): 4056,
            Interval(363.106, 1207.983, closed="right"): 136,
            Interval(-1326.646, -481.77, closed="right"): 84,
            Interval(1207.983, 2052.859, closed="right"): 22,
            Interval(-2171.523, -1326.646, closed="right"): 20,
            Interval(-3016.399, -2171.523, closed="right"): 11,
            Interval(2052.859, 2897.735, closed="right"): 4,
            Interval(-3868.88, -3016.399, closed="right"): 2,
            Interval(2897.735, 3742.612, closed="right"): 1,
        }

        expected_count["TEMA_DTW_feat"] = {
            Interval(-0.009000000000000001, 0.889, closed="right"): 1425,
            Interval(7.111, 8.0, closed="right"): 1340,
            Interval(6.222, 7.111, closed="right"): 443,
            Interval(0.889, 1.778, closed="right"): 390,
            Interval(1.778, 2.667, closed="right"): 190,
            Interval(5.333, 6.222, closed="right"): 180,
            Interval(2.667, 3.556, closed="right"): 128,
            Interval(4.444, 5.333, closed="right"): 124,
            Interval(3.556, 4.444, closed="right"): 116,
        }

        test_count_series: dict[str, pd.Series] = {}
        expected_count_series: dict[str, pd.Series] = {}

        for key, value in zip(test_count.keys(), test_count.values()):
            test_count_series[key] = pd.Series(value)
        test_count_concat: pd.Series = pd.concat(test_count_series)

        for key, value in zip(test_count.keys(), expected_count.values()):
            expected_count_series[key] = pd.Series(value)
        expected_count_concat: pd.Series = pd.concat(expected_count_series)

        pd.testing.assert_series_equal(
            test_count_concat, expected_count_concat
        )

    def test_create_dtw_distance_feature_empty_mas(self):
        expected_df = self.dataframe.copy()

        source = self.dataframe["close"]

        model_features = ModelFeatures(
            self.dataframe.copy(), self.test_index, self.bins, False
        )

        test_df = model_features.create_dtw_distance_feature(source, "", 14)

        pd.testing.assert_frame_equal(test_df, expected_df)

    def test_create_dtw_distance_feature_invalid_mas(self):
        model_features = ModelFeatures(
            self.dataframe.copy(), self.test_index, self.bins, False
        )

        self.assertRaises(
            AttributeError,
            model_features.create_dtw_distance_feature,
            self.dataframe["close"],
            "invalid",
            14,
        )


class TestDTWDistanceOpt(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc["2023"]
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

        self.test_index = 100
        self.bins = 9

        self.target = self.dataframe["Target_bin"].copy()

        self.model_features = ModelFeatures(
            self.dataframe, self.test_index, self.bins, False
        )

        source = self.dataframe["close"]

        self.test_df_all = self.model_features.create_dtw_distance_opt_feature(
            source, "all", 14
        ).dropna()

        self.test_df_mas = self.model_features.create_dtw_distance_opt_feature(
            source, ["sma", "ema", "rma", "dema", "tema"], 14
        ).dropna()

    def test_create_dtw_distance_opt_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "SMA_DTW",
                "SMA_DTW_feat",
                "EMA_DTW",
                "EMA_DTW_feat",
                "RMA_DTW",
                "RMA_DTW_feat",
                "DEMA_DTW",
                "DEMA_DTW_feat",
                "TEMA_DTW",
                "TEMA_DTW_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df_all.columns[8:], expected_columns
        )
        pd.testing.assert_index_equal(
            self.test_df_mas.columns[8:], expected_columns
        )

    def test_create_dtw_distance_opt_feature_values(self):
        expected_df = pd.DataFrame(
            {
                "SMA_DTW": {
                    Timestamp("2023-02-13 00:00:00"): 0.5439029532924422,
                    Timestamp("2023-02-14 00:00:00"): -0.19191604467835344,
                    Timestamp("2023-02-15 00:00:00"): 0.1407371263667816,
                    Timestamp("2023-02-16 00:00:00"): -0.6763075050020219,
                    Timestamp("2023-02-17 00:00:00"): 0.16060395573023703,
                    Timestamp("2023-02-18 00:00:00"): -0.26054951933526094,
                    Timestamp("2023-02-19 00:00:00"): 0.38098169236155516,
                    Timestamp("2023-02-20 00:00:00"): -0.414070682557484,
                    Timestamp("2023-02-21 00:00:00"): -0.057995860419852774,
                    Timestamp("2023-02-22 00:00:00"): 0.04318271880306619,
                },
                "SMA_DTW_feat": {
                    Timestamp("2023-02-13 00:00:00"): 7.0,
                    Timestamp("2023-02-14 00:00:00"): 1.0,
                    Timestamp("2023-02-15 00:00:00"): 6.0,
                    Timestamp("2023-02-16 00:00:00"): 0.0,
                    Timestamp("2023-02-17 00:00:00"): 6.0,
                    Timestamp("2023-02-18 00:00:00"): 1.0,
                    Timestamp("2023-02-19 00:00:00"): 7.0,
                    Timestamp("2023-02-20 00:00:00"): 0.0,
                    Timestamp("2023-02-21 00:00:00"): 2.0,
                    Timestamp("2023-02-22 00:00:00"): 5.0,
                },
                "EMA_DTW": {
                    Timestamp("2023-02-13 00:00:00"): -0.013763175498455654,
                    Timestamp("2023-02-14 00:00:00"): 0.43481193117784284,
                    Timestamp("2023-02-15 00:00:00"): -0.48610441285798744,
                    Timestamp("2023-02-16 00:00:00"): 0.24148170295956536,
                    Timestamp("2023-02-17 00:00:00"): 0.012595718696672031,
                    Timestamp("2023-02-18 00:00:00"): 0.4000531659666247,
                    Timestamp("2023-02-19 00:00:00"): -0.12119500724482213,
                    Timestamp("2023-02-20 00:00:00"): 0.08624044130764696,
                    Timestamp("2023-02-21 00:00:00"): -0.26537695169524517,
                    Timestamp("2023-02-22 00:00:00"): -0.47532671469579235,
                },
                "EMA_DTW_feat": {
                    Timestamp("2023-02-13 00:00:00"): 3.0,
                    Timestamp("2023-02-14 00:00:00"): 7.0,
                    Timestamp("2023-02-15 00:00:00"): 0.0,
                    Timestamp("2023-02-16 00:00:00"): 6.0,
                    Timestamp("2023-02-17 00:00:00"): 4.0,
                    Timestamp("2023-02-18 00:00:00"): 7.0,
                    Timestamp("2023-02-19 00:00:00"): 2.0,
                    Timestamp("2023-02-20 00:00:00"): 5.0,
                    Timestamp("2023-02-21 00:00:00"): 1.0,
                    Timestamp("2023-02-22 00:00:00"): 1.0,
                },
                "RMA_DTW": {
                    Timestamp("2023-02-13 00:00:00"): 0.02334534036636611,
                    Timestamp("2023-02-14 00:00:00"): 0.5363960881696349,
                    Timestamp("2023-02-15 00:00:00"): -0.5768106837815005,
                    Timestamp("2023-02-16 00:00:00"): 0.2610949694575298,
                    Timestamp("2023-02-17 00:00:00"): 0.08200297350415975,
                    Timestamp("2023-02-18 00:00:00"): 0.6264230740458234,
                    Timestamp("2023-02-19 00:00:00"): -0.18883377711260907,
                    Timestamp("2023-02-20 00:00:00"): 0.12069743235336527,
                    Timestamp("2023-02-21 00:00:00"): -0.34977754155006135,
                    Timestamp("2023-02-22 00:00:00"): -0.149145503972211,
                },
                "RMA_DTW_feat": {
                    Timestamp("2023-02-13 00:00:00"): 4.0,
                    Timestamp("2023-02-14 00:00:00"): 7.0,
                    Timestamp("2023-02-15 00:00:00"): 0.0,
                    Timestamp("2023-02-16 00:00:00"): 8.0,
                    Timestamp("2023-02-17 00:00:00"): 5.0,
                    Timestamp("2023-02-18 00:00:00"): 7.0,
                    Timestamp("2023-02-19 00:00:00"): 2.0,
                    Timestamp("2023-02-20 00:00:00"): 6.0,
                    Timestamp("2023-02-21 00:00:00"): 1.0,
                    Timestamp("2023-02-22 00:00:00"): 2.0,
                },
                "DEMA_DTW": {
                    Timestamp("2023-02-13 00:00:00"): 0.011946534165291402,
                    Timestamp("2023-02-14 00:00:00"): -0.00015757916054042788,
                    Timestamp("2023-02-15 00:00:00"): 0.010186207157313991,
                    Timestamp("2023-02-16 00:00:00"): 0.048564110199953214,
                    Timestamp("2023-02-17 00:00:00"): -0.003958777015175885,
                    Timestamp("2023-02-18 00:00:00"): -0.014437253666313439,
                    Timestamp("2023-02-19 00:00:00"): -0.021645148197416995,
                    Timestamp("2023-02-20 00:00:00"): -0.007692776289891148,
                    Timestamp("2023-02-21 00:00:00"): 0.07717844427972073,
                    Timestamp("2023-02-22 00:00:00"): -0.022773107588911934,
                },
                "DEMA_DTW_feat": {
                    Timestamp("2023-02-13 00:00:00"): 5.0,
                    Timestamp("2023-02-14 00:00:00"): 4.0,
                    Timestamp("2023-02-15 00:00:00"): 4.0,
                    Timestamp("2023-02-16 00:00:00"): 5.0,
                    Timestamp("2023-02-17 00:00:00"): 4.0,
                    Timestamp("2023-02-18 00:00:00"): 3.0,
                    Timestamp("2023-02-19 00:00:00"): 3.0,
                    Timestamp("2023-02-20 00:00:00"): 4.0,
                    Timestamp("2023-02-21 00:00:00"): 5.0,
                    Timestamp("2023-02-22 00:00:00"): 3.0,
                },
                "TEMA_DTW": {
                    Timestamp("2023-02-13 00:00:00"): -0.02328105195558007,
                    Timestamp("2023-02-14 00:00:00"): 0.41253005482817184,
                    Timestamp("2023-02-15 00:00:00"): -0.22540800517494608,
                    Timestamp("2023-02-16 00:00:00"): 0.13285643797824215,
                    Timestamp("2023-02-17 00:00:00"): 0.02979338024631234,
                    Timestamp("2023-02-18 00:00:00"): -0.026675730495424155,
                    Timestamp("2023-02-19 00:00:00"): -0.14197005316776398,
                    Timestamp("2023-02-20 00:00:00"): -0.19932806806007158,
                    Timestamp("2023-02-21 00:00:00"): -0.00885990041080887,
                    Timestamp("2023-02-22 00:00:00"): 0.22584091773847068,
                },
                "TEMA_DTW_feat": {
                    Timestamp("2023-02-13 00:00:00"): 4.0,
                    Timestamp("2023-02-14 00:00:00"): 7.0,
                    Timestamp("2023-02-15 00:00:00"): 1.0,
                    Timestamp("2023-02-16 00:00:00"): 6.0,
                    Timestamp("2023-02-17 00:00:00"): 5.0,
                    Timestamp("2023-02-18 00:00:00"): 4.0,
                    Timestamp("2023-02-19 00:00:00"): 2.0,
                    Timestamp("2023-02-20 00:00:00"): 1.0,
                    Timestamp("2023-02-21 00:00:00"): 4.0,
                    Timestamp("2023-02-22 00:00:00"): 8.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(
            self.test_df_all.iloc[:10, 8:], expected_df
        )
        pd.testing.assert_frame_equal(
            self.test_df_mas.iloc[:10, 8:], expected_df
        )

    def test_create_dtw_distance_opt_feature_count(self):
        feat_columns = self.test_df_all.columns[8:]

        test_count_all = {}
        test_count_mas = {}

        for column in feat_columns:
            test_count_all[column] = (
                self.test_df_all[column].value_counts(bins=self.bins).to_dict()
            )
            test_count_mas[column] = (
                self.test_df_mas[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count: dict[str, dict] = {
            "SMA_DTW": {
                Interval(-0.306, 0.386, closed="right"): 185,
                Interval(-0.998, -0.306, closed="right"): 59,
                Interval(0.386, 1.078, closed="right"): 53,
                Interval(-1.69, -0.998, closed="right"): 8,
                Interval(1.078, 1.77, closed="right"): 4,
                Interval(-2.382, -1.69, closed="right"): 2,
                Interval(1.77, 2.462, closed="right"): 2,
                Interval(-3.081, -2.382, closed="right"): 1,
                Interval(2.462, 3.153, closed="right"): 1,
            },
            "SMA_DTW_feat": {
                Interval(6.222, 7.111, closed="right"): 58,
                Interval(-0.009000000000000001, 0.889, closed="right"): 53,
                Interval(0.889, 1.778, closed="right"): 39,
                Interval(7.111, 8.0, closed="right"): 39,
                Interval(1.778, 2.667, closed="right"): 28,
                Interval(4.444, 5.333, closed="right"): 28,
                Interval(2.667, 3.556, closed="right"): 26,
                Interval(3.556, 4.444, closed="right"): 26,
                Interval(5.333, 6.222, closed="right"): 18,
            },
            "EMA_DTW": {
                Interval(-0.09, 0.24, closed="right"): 109,
                Interval(-0.42, -0.09, closed="right"): 68,
                Interval(0.24, 0.571, closed="right"): 51,
                Interval(0.571, 0.901, closed="right"): 29,
                Interval(-0.751, -0.42, closed="right"): 25,
                Interval(-1.081, -0.751, closed="right"): 16,
                Interval(-1.4149999999999998, -1.081, closed="right"): 9,
                Interval(0.901, 1.231, closed="right"): 4,
                Interval(1.231, 1.561, closed="right"): 4,
            },
            "EMA_DTW_feat": {
                Interval(6.222, 7.111, closed="right"): 62,
                Interval(-0.009000000000000001, 0.889, closed="right"): 44,
                Interval(1.778, 2.667, closed="right"): 44,
                Interval(0.889, 1.778, closed="right"): 39,
                Interval(5.333, 6.222, closed="right"): 30,
                Interval(3.556, 4.444, closed="right"): 28,
                Interval(2.667, 3.556, closed="right"): 27,
                Interval(7.111, 8.0, closed="right"): 23,
                Interval(4.444, 5.333, closed="right"): 18,
            },
            "RMA_DTW": {
                Interval(-0.126, 0.427, closed="right"): 157,
                Interval(-0.68, -0.126, closed="right"): 69,
                Interval(0.427, 0.981, closed="right"): 51,
                Interval(-1.233, -0.68, closed="right"): 25,
                Interval(-1.787, -1.233, closed="right"): 6,
                Interval(0.981, 1.534, closed="right"): 3,
                Interval(1.534, 2.088, closed="right"): 2,
                Interval(-2.346, -1.787, closed="right"): 1,
                Interval(2.088, 2.641, closed="right"): 1,
            },
            "RMA_DTW_feat": {
                Interval(6.222, 7.111, closed="right"): 46,
                Interval(-0.009000000000000001, 0.889, closed="right"): 44,
                Interval(1.778, 2.667, closed="right"): 38,
                Interval(5.333, 6.222, closed="right"): 34,
                Interval(7.111, 8.0, closed="right"): 34,
                Interval(4.444, 5.333, closed="right"): 33,
                Interval(0.889, 1.778, closed="right"): 32,
                Interval(3.556, 4.444, closed="right"): 32,
                Interval(2.667, 3.556, closed="right"): 22,
            },
            "DEMA_DTW": {
                Interval(-0.218, 0.298, closed="right"): 148,
                Interval(-0.734, -0.218, closed="right"): 56,
                Interval(0.298, 0.815, closed="right"): 56,
                Interval(-1.251, -0.734, closed="right"): 20,
                Interval(0.815, 1.331, closed="right"): 18,
                Interval(-1.767, -1.251, closed="right"): 8,
                Interval(-2.2889999999999997, -1.767, closed="right"): 3,
                Interval(1.331, 1.848, closed="right"): 3,
                Interval(1.848, 2.364, closed="right"): 3,
            },
            "DEMA_DTW_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 56,
                Interval(6.222, 7.111, closed="right"): 44,
                Interval(4.444, 5.333, closed="right"): 39,
                Interval(1.778, 2.667, closed="right"): 34,
                Interval(3.556, 4.444, closed="right"): 34,
                Interval(5.333, 6.222, closed="right"): 31,
                Interval(7.111, 8.0, closed="right"): 31,
                Interval(0.889, 1.778, closed="right"): 25,
                Interval(2.667, 3.556, closed="right"): 21,
            },
            "TEMA_DTW": {
                Interval(-0.262, 0.0146, closed="right"): 85,
                Interval(0.0146, 0.291, closed="right"): 82,
                Interval(0.291, 0.567, closed="right"): 52,
                Interval(-0.538, -0.262, closed="right"): 41,
                Interval(-0.814, -0.538, closed="right"): 22,
                Interval(0.567, 0.844, closed="right"): 19,
                Interval(-1.091, -0.814, closed="right"): 6,
                Interval(-1.371, -1.091, closed="right"): 4,
                Interval(0.844, 1.12, closed="right"): 4,
            },
            "TEMA_DTW_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 61,
                Interval(7.111, 8.0, closed="right"): 48,
                Interval(6.222, 7.111, closed="right"): 46,
                Interval(4.444, 5.333, closed="right"): 33,
                Interval(2.667, 3.556, closed="right"): 29,
                Interval(5.333, 6.222, closed="right"): 29,
                Interval(0.889, 1.778, closed="right"): 28,
                Interval(1.778, 2.667, closed="right"): 21,
                Interval(3.556, 4.444, closed="right"): 20,
            },
        }

        assert_count_series(test_count_all, expected_count)
        assert_count_series(test_count_mas, expected_count)


class TestCCI(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_cci_feature(
            source, 14, "sma"
        ).dropna()

    def test_create_cci_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "CCI",
                "CCI_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_cci_feature_values(self):
        expected_df = pd.DataFrame(
            {
                "CCI": {
                    Timestamp("2012-01-15 00:00:00"): 94.2593285910396,
                    Timestamp("2012-01-16 00:00:00"): 83.08080808080821,
                    Timestamp("2012-01-17 00:00:00"): -96.66666666666654,
                    Timestamp("2012-01-18 00:00:00"): 77.75524002704529,
                    Timestamp("2012-01-19 00:00:00"): -76.3681592039801,
                    Timestamp("2012-01-20 00:00:00"): 42.960784313725675,
                    Timestamp("2012-01-21 00:00:00"): 8.831066430120346,
                    Timestamp("2012-01-22 00:00:00"): 16.19574119574133,
                    Timestamp("2012-01-23 00:00:00"): -43.48484848484832,
                    Timestamp("2012-01-24 00:00:00"): -29.807692307692033,
                },
                "CCI_feat": {
                    Timestamp("2012-01-15 00:00:00"): 6.0,
                    Timestamp("2012-01-16 00:00:00"): 6.0,
                    Timestamp("2012-01-17 00:00:00"): 1.0,
                    Timestamp("2012-01-18 00:00:00"): 5.0,
                    Timestamp("2012-01-19 00:00:00"): 1.0,
                    Timestamp("2012-01-20 00:00:00"): 4.0,
                    Timestamp("2012-01-21 00:00:00"): 3.0,
                    Timestamp("2012-01-22 00:00:00"): 4.0,
                    Timestamp("2012-01-23 00:00:00"): 2.0,
                    Timestamp("2012-01-24 00:00:00"): 3.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_cci_feature_count(self):
        feat_columns = self.test_df.columns[8:]

        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "CCI": {
                Interval(36.552, 131.956, closed="right"): 1367,
                Interval(-58.852, 36.552, closed="right"): 1178,
                Interval(-154.256, -58.852, closed="right"): 931,
                Interval(131.956, 227.36, closed="right"): 565,
                Interval(-249.66, -154.256, closed="right"): 195,
                Interval(227.36, 322.764, closed="right"): 79,
                Interval(-345.064, -249.66, closed="right"): 32,
                Interval(-441.327, -345.064, closed="right"): 8,
                Interval(322.764, 418.168, closed="right"): 7,
            },
            "CCI_feat": {
                Interval(2.667, 3.556, closed="right"): 568,
                Interval(0.889, 1.778, closed="right"): 540,
                Interval(5.333, 6.222, closed="right"): 494,
                Interval(-0.009000000000000001, 0.889, closed="right"): 491,
                Interval(1.778, 2.667, closed="right"): 477,
                Interval(3.556, 4.444, closed="right"): 477,
                Interval(6.222, 7.111, closed="right"): 457,
                Interval(7.111, 8.0, closed="right"): 446,
                Interval(4.444, 5.333, closed="right"): 412,
            },
        }

        assert_count_series(test_count, expected_count)


class TestDidiIndex(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df_absolute = self.model_features.create_didi_index_feature(
            source, 4, 19, 21, "sma", "absolute"
        ).dropna()

        self.test_df_dtw = self.model_features.create_didi_index_feature(
            source, 4, 19, 21, "sma", "dtw"
        ).dropna()

    def test_create_didi_index_invalid_method(self):
        self.assertRaises(
            ValueError,
            self.model_features.create_didi_index_feature,
            self.dataframe["close"],
            4,
            19,
            21,
            "sma",
            "invalid",
        )

    def test_create_didi_index_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "DIDI",
                "DIDI_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df_absolute.columns[8:], expected_columns
        )
        pd.testing.assert_index_equal(
            self.test_df_dtw.columns[8:], expected_columns
        )

    def test_create_didi_index_feature_values(self) -> None:
        expected_df_absolute = pd.DataFrame(
            {
                "DIDI": {
                    Timestamp("2012-01-22 00:00:00"): -0.19488095238095227,
                    Timestamp("2012-01-23 00:00:00"): -0.1974999999999998,
                    Timestamp("2012-01-24 00:00:00"): -0.037499999999999645,
                    Timestamp("2012-01-25 00:00:00"): 0.11273809523809497,
                    Timestamp("2012-01-26 00:00:00"): 0.3804761904761911,
                    Timestamp("2012-01-27 00:00:00"): 0.5347619047619059,
                    Timestamp("2012-01-28 00:00:00"): 0.8547619047619053,
                    Timestamp("2012-01-29 00:00:00"): 0.9858333333333329,
                    Timestamp("2012-01-30 00:00:00"): 0.9315476190476186,
                    Timestamp("2012-01-31 00:00:00"): 0.9889285714285716,
                },
                "DIDI_feat": {
                    Timestamp("2012-01-22 00:00:00"): 4.0,
                    Timestamp("2012-01-23 00:00:00"): 4.0,
                    Timestamp("2012-01-24 00:00:00"): 4.0,
                    Timestamp("2012-01-25 00:00:00"): 5.0,
                    Timestamp("2012-01-26 00:00:00"): 6.0,
                    Timestamp("2012-01-27 00:00:00"): 6.0,
                    Timestamp("2012-01-28 00:00:00"): 6.0,
                    Timestamp("2012-01-29 00:00:00"): 6.0,
                    Timestamp("2012-01-30 00:00:00"): 6.0,
                    Timestamp("2012-01-31 00:00:00"): 6.0,
                },
            }
        )

        expected_df_absolute.index.name = "date"

        expected_df_dtw = pd.DataFrame(
            {
                "DIDI": {
                    Timestamp("2012-01-22 00:00:00"): -0.27238095238095195,
                    Timestamp("2012-01-23 00:00:00"): -0.07249999999999979,
                    Timestamp("2012-01-24 00:00:00"): -0.017763157894736814,
                    Timestamp("2012-01-25 00:00:00"): -0.09120927318295813,
                    Timestamp("2012-01-26 00:00:00"): -0.05436716791979901,
                    Timestamp("2012-01-27 00:00:00"): -0.014630325814536604,
                    Timestamp("2012-01-28 00:00:00"): -0.07412907268170432,
                    Timestamp("2012-01-29 00:00:00"): -0.10721177944861982,
                    Timestamp("2012-01-30 00:00:00"): 0.12370927318295788,
                    Timestamp("2012-01-31 00:00:00"): 0.07491228070175282,
                },
                "DIDI_feat": {
                    Timestamp("2012-01-22 00:00:00"): 2.0,
                    Timestamp("2012-01-23 00:00:00"): 2.0,
                    Timestamp("2012-01-24 00:00:00"): 3.0,
                    Timestamp("2012-01-25 00:00:00"): 2.0,
                    Timestamp("2012-01-26 00:00:00"): 3.0,
                    Timestamp("2012-01-27 00:00:00"): 3.0,
                    Timestamp("2012-01-28 00:00:00"): 2.0,
                    Timestamp("2012-01-29 00:00:00"): 2.0,
                    Timestamp("2012-01-30 00:00:00"): 6.0,
                    Timestamp("2012-01-31 00:00:00"): 5.0,
                },
            }
        )

        expected_df_dtw.index.name = "date"

        pd.testing.assert_frame_equal(
            expected_df_absolute, self.test_df_absolute.iloc[:10, 8:]
        )
        pd.testing.assert_frame_equal(
            expected_df_dtw, self.test_df_dtw.iloc[:10, 8:]
        )

    def test_create_didi_index_feature_count(self):
        feat_columns = self.test_df_absolute.columns[8:]
        test_count_absolute = {}
        test_count_dtw = {}
        for column in feat_columns:
            test_count_absolute[column] = (
                self.test_df_absolute[column]
                .value_counts(bins=self.bins)
                .to_dict()
            )
            test_count_dtw[column] = (
                self.test_df_dtw[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count_absolute = {
            "DIDI": {
                Interval(-24.193, 2368.777, closed="right"): 2597,
                Interval(-2417.163, -24.193, closed="right"): 1393,
                Interval(-4810.132, -2417.163, closed="right"): 143,
                Interval(2368.777, 4761.746, closed="right"): 115,
                Interval(4761.746, 7154.716, closed="right"): 36,
                Interval(-9617.609, -7203.102, closed="right"): 33,
                Interval(-7203.102, -4810.132, closed="right"): 28,
                Interval(9547.685, 11940.655, closed="right"): 7,
                Interval(7154.716, 9547.685, closed="right"): 3,
            },
            "DIDI_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1576,
                Interval(7.111, 8.0, closed="right"): 1112,
                Interval(6.222, 7.111, closed="right"): 404,
                Interval(0.889, 1.778, closed="right"): 381,
                Interval(5.333, 6.222, closed="right"): 277,
                Interval(1.778, 2.667, closed="right"): 215,
                Interval(2.667, 3.556, closed="right"): 148,
                Interval(4.444, 5.333, closed="right"): 124,
                Interval(3.556, 4.444, closed="right"): 118,
            },
        }

        expected_count_dtw = {
            "DIDI": {
                Interval(-402.707, 26.945, closed="right"): 3525,
                Interval(26.945, 456.598, closed="right"): 750,
                Interval(-832.359, -402.707, closed="right"): 31,
                Interval(456.598, 886.25, closed="right"): 29,
                Interval(-1262.012, -832.359, closed="right"): 7,
                Interval(886.25, 1315.902, closed="right"): 6,
                Interval(-1691.664, -1262.012, closed="right"): 3,
                Interval(-2554.837, -2121.316, closed="right"): 2,
                Interval(-2121.316, -1691.664, closed="right"): 2,
            },
            "DIDI_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1333,
                Interval(6.222, 7.111, closed="right"): 1086,
                Interval(7.111, 8.0, closed="right"): 801,
                Interval(0.889, 1.778, closed="right"): 424,
                Interval(5.333, 6.222, closed="right"): 185,
                Interval(1.778, 2.667, closed="right"): 165,
                Interval(4.444, 5.333, closed="right"): 125,
                Interval(2.667, 3.556, closed="right"): 121,
                Interval(3.556, 4.444, closed="right"): 115,
            },
        }

        assert_count_series(test_count_absolute, expected_count_absolute)
        assert_count_series(test_count_dtw, expected_count_dtw)


class TestDidiIndexOPT(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_didi_index_opt_feature(
            source, 4, 19, 21, "sma"
        ).dropna()

    def test_create_didi_index_opt_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "DIDI",
                "DIDI_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_didi_index_opt_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "DIDI": {
                    Timestamp("2012-01-26 00:00:00"): -0.10532102461334121,
                    Timestamp("2012-01-27 00:00:00"): 0.17037644345509384,
                    Timestamp("2012-01-28 00:00:00"): 0.004103969594908968,
                    Timestamp("2012-01-29 00:00:00"): 0.010728917545322458,
                    Timestamp("2012-01-30 00:00:00"): -0.22363972913178015,
                    Timestamp("2012-01-31 00:00:00"): 0.09644406011981785,
                    Timestamp("2012-02-01 00:00:00"): -0.030020374606961525,
                    Timestamp("2012-02-02 00:00:00"): 0.1931907970853935,
                    Timestamp("2012-02-03 00:00:00"): -0.2304334254631607,
                    Timestamp("2012-02-04 00:00:00"): -0.04097030948498748,
                },
                "DIDI_feat": {
                    Timestamp("2012-01-26 00:00:00"): 1.0,
                    Timestamp("2012-01-27 00:00:00"): 6.0,
                    Timestamp("2012-01-28 00:00:00"): 4.0,
                    Timestamp("2012-01-29 00:00:00"): 4.0,
                    Timestamp("2012-01-30 00:00:00"): 0.0,
                    Timestamp("2012-01-31 00:00:00"): 8.0,
                    Timestamp("2012-02-01 00:00:00"): 3.0,
                    Timestamp("2012-02-02 00:00:00"): 7.0,
                    Timestamp("2012-02-03 00:00:00"): 0.0,
                    Timestamp("2012-02-04 00:00:00"): 3.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_didi_index_opt_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "DIDI": {
                Interval(-0.00581, 0.214, closed="right"): 1843,
                Interval(-0.226, -0.00581, closed="right"): 1784,
                Interval(0.214, 0.435, closed="right"): 305,
                Interval(-0.446, -0.226, closed="right"): 290,
                Interval(0.435, 0.655, closed="right"): 52,
                Interval(-0.666, -0.446, closed="right"): 51,
                Interval(0.655, 0.875, closed="right"): 17,
                Interval(-0.89, -0.666, closed="right"): 8,
                Interval(0.875, 1.095, closed="right"): 1,
            },
            "DIDI_feat": {
                Interval(6.222, 7.111, closed="right"): 509,
                Interval(4.444, 5.333, closed="right"): 501,
                Interval(1.778, 2.667, closed="right"): 498,
                Interval(5.333, 6.222, closed="right"): 495,
                Interval(-0.009000000000000001, 0.889, closed="right"): 491,
                Interval(0.889, 1.778, closed="right"): 489,
                Interval(3.556, 4.444, closed="right"): 473,
                Interval(2.667, 3.556, closed="right"): 464,
                Interval(7.111, 8.0, closed="right"): 431,
            },
        }

        assert_count_series(test_count, expected_count)


class TestMacd(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_macd_feature(source).dropna()

    def test_create_macd_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "MACD",
                "MACD_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_macd_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "MACD": {
                    Timestamp("2012-02-04 00:00:00"): 0.05552801159392079,
                    Timestamp("2012-02-05 00:00:00"): 0.030096544872609293,
                    Timestamp("2012-02-06 00:00:00"): 0.01444043466269182,
                    Timestamp("2012-02-07 00:00:00"): 0.011442724773684687,
                    Timestamp("2012-02-08 00:00:00"): 0.022253709336925026,
                    Timestamp("2012-02-09 00:00:00"): 0.030891439472990168,
                    Timestamp("2012-02-10 00:00:00"): 0.03572927399634637,
                    Timestamp("2012-02-11 00:00:00"): 0.014237830430749165,
                    Timestamp("2012-02-12 00:00:00"): 0.012437586630658304,
                    Timestamp("2012-02-13 00:00:00"): 0.01985454614973764,
                },
                "MACD_feat": {
                    Timestamp("2012-02-04 00:00:00"): 5.0,
                    Timestamp("2012-02-05 00:00:00"): 4.0,
                    Timestamp("2012-02-06 00:00:00"): 4.0,
                    Timestamp("2012-02-07 00:00:00"): 4.0,
                    Timestamp("2012-02-08 00:00:00"): 4.0,
                    Timestamp("2012-02-09 00:00:00"): 4.0,
                    Timestamp("2012-02-10 00:00:00"): 4.0,
                    Timestamp("2012-02-11 00:00:00"): 4.0,
                    Timestamp("2012-02-12 00:00:00"): 4.0,
                    Timestamp("2012-02-13 00:00:00"): 4.0,
                },
            },
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(self.test_df.iloc[:10, 8:], expected_df)

    def test_create_macd_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "MACD": {
                Interval(-43.229, 293.51, closed="right"): 3245,
                Interval(-379.968, -43.229, closed="right"): 665,
                Interval(293.51, 630.249, closed="right"): 179,
                Interval(-716.707, -379.968, closed="right"): 114,
                Interval(630.249, 966.989, closed="right"): 47,
                Interval(-1053.447, -716.707, closed="right"): 45,
                Interval(966.989, 1303.728, closed="right"): 26,
                Interval(-1390.186, -1053.447, closed="right"): 16,
                Interval(-1729.9569999999999, -1390.186, closed="right"): 5,
            },
            "MACD_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 1315,
                Interval(7.111, 8.0, closed="right"): 1169,
                Interval(6.222, 7.111, closed="right"): 565,
                Interval(5.333, 6.222, closed="right"): 366,
                Interval(0.889, 1.778, closed="right"): 334,
                Interval(1.778, 2.667, closed="right"): 220,
                Interval(4.444, 5.333, closed="right"): 132,
                Interval(2.667, 3.556, closed="right"): 126,
                Interval(3.556, 4.444, closed="right"): 115,
            },
        }

        assert_count_series(test_count, expected_count)


class TestMacdOPT(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_macd_opt_feature(
            source
        ).dropna()

    def test_create_macd_opt_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "MACD",
                "MACD_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_macd_opt_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "MACD": {
                    Timestamp("2012-02-08 00:00:00"): -0.0038759340792576067,
                    Timestamp("2012-02-09 00:00:00"): -0.005031868116942268,
                    Timestamp("2012-02-10 00:00:00"): 0.010667460962038598,
                    Timestamp("2012-02-11 00:00:00"): 0.03458006422034684,
                    Timestamp("2012-02-12 00:00:00"): 0.02223619782615388,
                    Timestamp("2012-02-13 00:00:00"): -0.03996247216414588,
                    Timestamp("2012-02-14 00:00:00"): 0.107128540130638,
                    Timestamp("2012-02-15 00:00:00"): -0.11637512459952784,
                    Timestamp("2012-02-16 00:00:00"): 0.050900294968030585,
                    Timestamp("2012-02-17 00:00:00"): -0.06494074941314602,
                },
                "MACD_feat": {
                    Timestamp("2012-02-08 00:00:00"): 4.0,
                    Timestamp("2012-02-09 00:00:00"): 3.0,
                    Timestamp("2012-02-10 00:00:00"): 6.0,
                    Timestamp("2012-02-11 00:00:00"): 8.0,
                    Timestamp("2012-02-12 00:00:00"): 8.0,
                    Timestamp("2012-02-13 00:00:00"): 0.0,
                    Timestamp("2012-02-14 00:00:00"): 7.0,
                    Timestamp("2012-02-15 00:00:00"): 0.0,
                    Timestamp("2012-02-16 00:00:00"): 7.0,
                    Timestamp("2012-02-17 00:00:00"): 0.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(self.test_df.iloc[:10, 8:], expected_df)

    def test_create_macd_opt_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "MACD": {
                Interval(-0.0342, 0.0439, closed="right"): 3400,
                Interval(-0.112, -0.0342, closed="right"): 526,
                Interval(0.0439, 0.122, closed="right"): 317,
                Interval(0.122, 0.2, closed="right"): 40,
                Interval(-0.19, -0.112, closed="right"): 35,
                Interval(0.2, 0.278, closed="right"): 10,
                Interval(-0.268, -0.19, closed="right"): 6,
                Interval(-0.348, -0.268, closed="right"): 3,
                Interval(0.278, 0.356, closed="right"): 1,
            },
            "MACD_feat": {
                Interval(7.111, 8.0, closed="right"): 527,
                Interval(0.889, 1.778, closed="right"): 526,
                Interval(6.222, 7.111, closed="right"): 508,
                Interval(-0.009000000000000001, 0.889, closed="right"): 500,
                Interval(5.333, 6.222, closed="right"): 471,
                Interval(4.444, 5.333, closed="right"): 467,
                Interval(3.556, 4.444, closed="right"): 453,
                Interval(1.778, 2.667, closed="right"): 443,
                Interval(2.667, 3.556, closed="right"): 443,
            },
        }

        assert_count_series(test_count, expected_count)

    def test_create_macd_opt_feature_invalid_ma_method(self):
        self.assertRaises(
            ValueError,
            self.model_features.create_macd_opt_feature,
            self.dataframe["close"],
            12,
            26,
            9,
            "invalid",
        )


class TestTrix(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_trix_feature(source).dropna()

    def test_create_trix_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "TRIX",
                "TRIX_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_trix_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "TRIX": {
                    Timestamp("2012-02-14 00:00:00"): -43.29498053644576,
                    Timestamp("2012-02-15 00:00:00"): -47.65132139439743,
                    Timestamp("2012-02-16 00:00:00"): -54.269902408885514,
                    Timestamp("2012-02-17 00:00:00"): -61.3327226954441,
                    Timestamp("2012-02-18 00:00:00"): -70.20263299674534,
                    Timestamp("2012-02-19 00:00:00"): -79.08430873189154,
                    Timestamp("2012-02-20 00:00:00"): -87.17203495208992,
                    Timestamp("2012-02-21 00:00:00"): -93.60375939056365,
                    Timestamp("2012-02-22 00:00:00"): -98.76772628595987,
                    Timestamp("2012-02-23 00:00:00"): -99.81212615194801,
                },
                "TRIX_feat": {
                    Timestamp("2012-02-14 00:00:00"): 1.0,
                    Timestamp("2012-02-15 00:00:00"): 1.0,
                    Timestamp("2012-02-16 00:00:00"): 1.0,
                    Timestamp("2012-02-17 00:00:00"): 1.0,
                    Timestamp("2012-02-18 00:00:00"): 0.0,
                    Timestamp("2012-02-19 00:00:00"): 0.0,
                    Timestamp("2012-02-20 00:00:00"): 0.0,
                    Timestamp("2012-02-21 00:00:00"): 0.0,
                    Timestamp("2012-02-22 00:00:00"): 0.0,
                    Timestamp("2012-02-23 00:00:00"): 0.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_trix_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "TRIX": {
                Interval(-26.111, 50.351, closed="right"): 2120,
                Interval(-102.572, -26.111, closed="right"): 882,
                Interval(50.351, 126.812, closed="right"): 857,
                Interval(126.812, 203.274, closed="right"): 211,
                Interval(-179.723, -102.572, closed="right"): 160,
                Interval(203.274, 279.735, closed="right"): 49,
                Interval(279.735, 356.197, closed="right"): 22,
                Interval(356.197, 432.658, closed="right"): 18,
                Interval(432.658, 509.12, closed="right"): 13,
            },
            "TRIX_feat": {
                Interval(1.778, 2.667, closed="right"): 597,
                Interval(7.111, 8.0, closed="right"): 596,
                Interval(2.667, 3.556, closed="right"): 577,
                Interval(0.889, 1.778, closed="right"): 560,
                Interval(5.333, 6.222, closed="right"): 504,
                Interval(4.444, 5.333, closed="right"): 503,
                Interval(3.556, 4.444, closed="right"): 416,
                Interval(-0.009000000000000001, 0.889, closed="right"): 415,
                Interval(6.222, 7.111, closed="right"): 164,
            },
        }

        assert_count_series(test_count, expected_count)


class TestTrixOPT(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)

            self.test_df = self.model_features.create_trix_opt_feature(
                source, 3, 2, "sma"
            ).dropna()

    def test_create_trix_opt_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "TRIX",
                "TRIX_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_trix_opt_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "TRIX": {
                    Timestamp("2012-01-14 00:00:00"): -196.61558376973153,
                    Timestamp("2012-01-15 00:00:00"): -84.72279286582875,
                    Timestamp("2012-01-16 00:00:00"): -94.93608635656972,
                    Timestamp("2012-01-17 00:00:00"): 1513.7844836003042,
                    Timestamp("2012-01-18 00:00:00"): 86.87825368232234,
                    Timestamp("2012-01-19 00:00:00"): -1091.8176832742556,
                    Timestamp("2012-01-20 00:00:00"): -940.643700987545,
                    Timestamp("2012-01-21 00:00:00"): 486.1776640622003,
                    Timestamp("2012-01-22 00:00:00"): 1106.6485467083794,
                    Timestamp("2012-01-23 00:00:00"): 1356.487327583774,
                },
                "TRIX_feat": {
                    Timestamp("2012-01-14 00:00:00"): 3.0,
                    Timestamp("2012-01-15 00:00:00"): 4.0,
                    Timestamp("2012-01-16 00:00:00"): 4.0,
                    Timestamp("2012-01-17 00:00:00"): 6.0,
                    Timestamp("2012-01-18 00:00:00"): 4.0,
                    Timestamp("2012-01-19 00:00:00"): 2.0,
                    Timestamp("2012-01-20 00:00:00"): 2.0,
                    Timestamp("2012-01-21 00:00:00"): 5.0,
                    Timestamp("2012-01-22 00:00:00"): 8.0,
                    Timestamp("2012-01-23 00:00:00"): 6.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_trix_opt_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}

        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "TRIX": {
                Interval(-670.083, 972.694, closed="right"): 1894,
                Interval(-2312.86, -670.083, closed="right"): 1072,
                Interval(972.694, 2615.472, closed="right"): 836,
                Interval(-3955.638, -2312.86, closed="right"): 266,
                Interval(2615.472, 4258.249, closed="right"): 210,
                Interval(-5598.415, -3955.638, closed="right"): 41,
                Interval(4258.249, 5901.026, closed="right"): 26,
                Interval(-7255.978, -5598.415, closed="right"): 6,
                Interval(5901.026, 7543.804, closed="right"): 5,
            },
            "TRIX_feat": {
                Interval(4.444, 5.333, closed="right"): 540,
                Interval(2.667, 3.556, closed="right"): 529,
                Interval(5.333, 6.222, closed="right"): 509,
                Interval(7.111, 8.0, closed="right"): 488,
                Interval(1.778, 2.667, closed="right"): 487,
                Interval(-0.009000000000000001, 0.889, closed="right"): 464,
                Interval(3.556, 4.444, closed="right"): 456,
                Interval(0.889, 1.778, closed="right"): 454,
                Interval(6.222, 7.111, closed="right"): 429,
            },
        }

        assert_count_series(test_count, expected_count)


class TestSMIO(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_smio_feature(source).dropna()

    def test_create_smio_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "SMIO",
                "SMIO_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_smio_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "SMIO": {
                    Timestamp("2012-01-30 00:00:00"): -0.045308619997770555,
                    Timestamp("2012-01-31 00:00:00"): -0.04179235749352472,
                    Timestamp("2012-02-01 00:00:00"): -0.007754126493321065,
                    Timestamp("2012-02-02 00:00:00"): 0.025680381002036637,
                    Timestamp("2012-02-03 00:00:00"): 0.037552918932248866,
                    Timestamp("2012-02-04 00:00:00"): 0.020552162563117652,
                    Timestamp("2012-02-05 00:00:00"): -0.006184165031534353,
                    Timestamp("2012-02-06 00:00:00"): -0.017058217095675593,
                    Timestamp("2012-02-07 00:00:00"): -0.015064748433394188,
                    Timestamp("2012-02-08 00:00:00"): -0.0019615921983414455,
                },
                "SMIO_feat": {
                    Timestamp("2012-01-30 00:00:00"): 1.0,
                    Timestamp("2012-01-31 00:00:00"): 2.0,
                    Timestamp("2012-02-01 00:00:00"): 3.0,
                    Timestamp("2012-02-02 00:00:00"): 5.0,
                    Timestamp("2012-02-03 00:00:00"): 6.0,
                    Timestamp("2012-02-04 00:00:00"): 5.0,
                    Timestamp("2012-02-05 00:00:00"): 3.0,
                    Timestamp("2012-02-06 00:00:00"): 3.0,
                    Timestamp("2012-02-07 00:00:00"): 3.0,
                    Timestamp("2012-02-08 00:00:00"): 4.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_smio_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "SMIO": {
                Interval(-0.0235, 0.055, closed="right"): 2061,
                Interval(-0.102, -0.0235, closed="right"): 1141,
                Interval(0.055, 0.133, closed="right"): 725,
                Interval(-0.18, -0.102, closed="right"): 261,
                Interval(0.133, 0.212, closed="right"): 102,
                Interval(-0.259, -0.18, closed="right"): 35,
                Interval(0.212, 0.29, closed="right"): 12,
                Interval(-0.337, -0.259, closed="right"): 8,
                Interval(-0.418, -0.337, closed="right"): 2,
            },
            "SMIO_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 565,
                Interval(6.222, 7.111, closed="right"): 556,
                Interval(7.111, 8.0, closed="right"): 545,
                Interval(2.667, 3.556, closed="right"): 499,
                Interval(1.778, 2.667, closed="right"): 491,
                Interval(3.556, 4.444, closed="right"): 472,
                Interval(0.889, 1.778, closed="right"): 453,
                Interval(4.444, 5.333, closed="right"): 393,
                Interval(5.333, 6.222, closed="right"): 373,
            },
        }

        assert_count_series(test_count, expected_count)


class TestSMIOOPT(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_smio_opt_feature(
            source
        ).dropna()

    def test_create_smio_opt_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "SMIO",
                "SMIO_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_smio_opt_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "SMIO": {
                    Timestamp("2012-02-01 00:00:00"): 0.02158229089865395,
                    Timestamp("2012-02-02 00:00:00"): -0.0004268969842382876,
                    Timestamp("2012-02-03 00:00:00"): -0.015246614895252315,
                    Timestamp("2012-02-04 00:00:00"): 0.0036261980335655034,
                    Timestamp("2012-02-05 00:00:00"): 0.006884088432290375,
                    Timestamp("2012-02-06 00:00:00"): -0.011216322592673623,
                    Timestamp("2012-02-07 00:00:00"): -0.006279520744347903,
                    Timestamp("2012-02-08 00:00:00"): 0.00785573541957065,
                    Timestamp("2012-02-09 00:00:00"): -0.004043591928205108,
                    Timestamp("2012-02-10 00:00:00"): -0.0036610795575005607,
                },
                "SMIO_feat": {
                    Timestamp("2012-02-01 00:00:00"): 7.0,
                    Timestamp("2012-02-02 00:00:00"): 4.0,
                    Timestamp("2012-02-03 00:00:00"): 1.0,
                    Timestamp("2012-02-04 00:00:00"): 5.0,
                    Timestamp("2012-02-05 00:00:00"): 8.0,
                    Timestamp("2012-02-06 00:00:00"): 1.0,
                    Timestamp("2012-02-07 00:00:00"): 2.0,
                    Timestamp("2012-02-08 00:00:00"): 8.0,
                    Timestamp("2012-02-09 00:00:00"): 3.0,
                    Timestamp("2012-02-10 00:00:00"): 3.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_smio_opt_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "SMIO": {
                Interval(-0.014, 0.0237, closed="right"): 3060,
                Interval(-0.0516, -0.014, closed="right"): 752,
                Interval(0.0237, 0.0613, closed="right"): 383,
                Interval(-0.0892, -0.0516, closed="right"): 74,
                Interval(0.0613, 0.0989, closed="right"): 47,
                Interval(-0.128, -0.0892, closed="right"): 14,
                Interval(0.0989, 0.137, closed="right"): 13,
                Interval(0.174, 0.212, closed="right"): 2,
                Interval(0.137, 0.174, closed="right"): 0,
            },
            "SMIO_feat": {
                Interval(6.222, 7.111, closed="right"): 617,
                Interval(-0.009000000000000001, 0.889, closed="right"): 597,
                Interval(1.778, 2.667, closed="right"): 486,
                Interval(7.111, 8.0, closed="right"): 481,
                Interval(5.333, 6.222, closed="right"): 453,
                Interval(0.889, 1.778, closed="right"): 450,
                Interval(3.556, 4.444, closed="right"): 444,
                Interval(4.444, 5.333, closed="right"): 412,
                Interval(2.667, 3.556, closed="right"): 405,
            },
        }

        assert_count_series(test_count, expected_count)


class TestTSI(unittest.TestCase):
    def setUp(self):
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_tsi_feature(source).dropna()

    def test_create_tsi_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "TSI",
                "TSI_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_tsi_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "TSI": {
                    Timestamp("2012-02-08 00:00:00"): -0.01239051622704824,
                    Timestamp("2012-02-09 00:00:00"): -0.010854895473568351,
                    Timestamp("2012-02-10 00:00:00"): -0.00979058364416123,
                    Timestamp("2012-02-11 00:00:00"): -0.018859278235780083,
                    Timestamp("2012-02-12 00:00:00"): -0.021811392247394847,
                    Timestamp("2012-02-13 00:00:00"): -0.020824541290120045,
                    Timestamp("2012-02-14 00:00:00"): -0.04557262215613051,
                    Timestamp("2012-02-15 00:00:00"): -0.07146536178889104,
                    Timestamp("2012-02-16 00:00:00"): -0.0977874283135179,
                    Timestamp("2012-02-17 00:00:00"): -0.11410801234672113,
                },
                "TSI_feat": {
                    Timestamp("2012-02-08 00:00:00"): 3.0,
                    Timestamp("2012-02-09 00:00:00"): 3.0,
                    Timestamp("2012-02-10 00:00:00"): 3.0,
                    Timestamp("2012-02-11 00:00:00"): 3.0,
                    Timestamp("2012-02-12 00:00:00"): 3.0,
                    Timestamp("2012-02-13 00:00:00"): 3.0,
                    Timestamp("2012-02-14 00:00:00"): 2.0,
                    Timestamp("2012-02-15 00:00:00"): 2.0,
                    Timestamp("2012-02-16 00:00:00"): 2.0,
                    Timestamp("2012-02-17 00:00:00"): 1.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_tsi_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "TSI": {
                Interval(-0.158, 0.0132, closed="right"): 1186,
                Interval(0.0132, 0.184, closed="right"): 996,
                Interval(0.184, 0.355, closed="right"): 678,
                Interval(-0.329, -0.158, closed="right"): 586,
                Interval(0.355, 0.526, closed="right"): 402,
                Interval(0.526, 0.697, closed="right"): 247,
                Interval(-0.5, -0.329, closed="right"): 170,
                Interval(0.697, 0.868, closed="right"): 61,
                Interval(-0.673, -0.5, closed="right"): 12,
            },
            "TSI_feat": {
                Interval(5.333, 6.222, closed="right"): 692,
                Interval(0.889, 1.778, closed="right"): 688,
                Interval(4.444, 5.333, closed="right"): 642,
                Interval(1.778, 2.667, closed="right"): 500,
                Interval(-0.009000000000000001, 0.889, closed="right"): 457,
                Interval(3.556, 4.444, closed="right"): 400,
                Interval(2.667, 3.556, closed="right"): 391,
                Interval(7.111, 8.0, closed="right"): 369,
                Interval(6.222, 7.111, closed="right"): 199,
            },
        }

        assert_count_series(test_count, expected_count)


class TestTSIOPT(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_tsi_opt_feature(
            source
        ).dropna()

    def test_create_tsi_opt_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "TSI",
                "TSI_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_tsi_opt_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "TSI": {
                    Timestamp("2012-02-10 00:00:00"): -0.000333265736245586,
                    Timestamp("2012-02-11 00:00:00"): 0.005659953330372618,
                    Timestamp("2012-02-12 00:00:00"): -0.004325075605794851,
                    Timestamp("2012-02-13 00:00:00"): -0.0013896508325392139,
                    Timestamp("2012-02-14 00:00:00"): 0.01680172679779965,
                    Timestamp("2012-02-15 00:00:00"): 0.0008093959761135902,
                    Timestamp("2012-02-16 00:00:00"): 0.00030357995658442105,
                    Timestamp("2012-02-17 00:00:00"): -0.007072116091604212,
                    Timestamp("2012-02-18 00:00:00"): 0.006578158149953115,
                    Timestamp("2012-02-19 00:00:00"): -0.007137925776676615,
                },
                "TSI_feat": {
                    Timestamp("2012-02-10 00:00:00"): 4.0,
                    Timestamp("2012-02-11 00:00:00"): 6.0,
                    Timestamp("2012-02-12 00:00:00"): 2.0,
                    Timestamp("2012-02-13 00:00:00"): 3.0,
                    Timestamp("2012-02-14 00:00:00"): 7.0,
                    Timestamp("2012-02-15 00:00:00"): 5.0,
                    Timestamp("2012-02-16 00:00:00"): 4.0,
                    Timestamp("2012-02-17 00:00:00"): 1.0,
                    Timestamp("2012-02-18 00:00:00"): 6.0,
                    Timestamp("2012-02-19 00:00:00"): 1.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_tsi_opt_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "TSI": {
                Interval(-0.00951, 0.0148, closed="right"): 3493,
                Interval(-0.0338, -0.00951, closed="right"): 513,
                Interval(0.0148, 0.0391, closed="right"): 257,
                Interval(0.0391, 0.0634, closed="right"): 34,
                Interval(-0.0581, -0.0338, closed="right"): 26,
                Interval(0.0634, 0.0878, closed="right"): 4,
                Interval(0.0878, 0.112, closed="right"): 4,
                Interval(-0.108, -0.0825, closed="right"): 3,
                Interval(-0.0825, -0.0581, closed="right"): 2,
            },
            "TSI_feat": {
                Interval(0.889, 1.778, closed="right"): 569,
                Interval(5.333, 6.222, closed="right"): 555,
                Interval(-0.009000000000000001, 0.889, closed="right"): 510,
                Interval(2.667, 3.556, closed="right"): 481,
                Interval(6.222, 7.111, closed="right"): 475,
                Interval(1.778, 2.667, closed="right"): 448,
                Interval(3.556, 4.444, closed="right"): 444,
                Interval(4.444, 5.333, closed="right"): 432,
                Interval(7.111, 8.0, closed="right"): 422,
            },
        }

        assert_count_series(test_count, expected_count)


class TestBBTrend(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_bb_trend_feature(
            source
        ).dropna()

    def test_create_bb_trend_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "bb_trend",
                "bb_trend_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_bb_trend_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "bb_trend": {
                    Timestamp("2012-01-26 00:00:00"): -1.165536797820343,
                    Timestamp("2012-01-27 00:00:00"): -1.096006796738471,
                    Timestamp("2012-01-28 00:00:00"): 4.720702576572254,
                    Timestamp("2012-01-29 00:00:00"): 4.364689034802541,
                    Timestamp("2012-01-30 00:00:00"): 4.273053422797143,
                    Timestamp("2012-01-31 00:00:00"): 0.9592663716378317,
                    Timestamp("2012-02-01 00:00:00"): 0.8247050413064548,
                    Timestamp("2012-02-02 00:00:00"): -1.781663883186952,
                    Timestamp("2012-02-03 00:00:00"): -4.285561135235777,
                    Timestamp("2012-02-04 00:00:00"): -4.141180231599691,
                },
                "bb_trend_feat": {
                    Timestamp("2012-01-26 00:00:00"): 2.0,
                    Timestamp("2012-01-27 00:00:00"): 3.0,
                    Timestamp("2012-01-28 00:00:00"): 8.0,
                    Timestamp("2012-01-29 00:00:00"): 8.0,
                    Timestamp("2012-01-30 00:00:00"): 8.0,
                    Timestamp("2012-01-31 00:00:00"): 4.0,
                    Timestamp("2012-02-01 00:00:00"): 4.0,
                    Timestamp("2012-02-02 00:00:00"): 2.0,
                    Timestamp("2012-02-03 00:00:00"): 1.0,
                    Timestamp("2012-02-04 00:00:00"): 1.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_bb_trend_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "bb_trend": {
                Interval(-6.034, 2.063, closed="right"): 2276,
                Interval(2.063, 10.16, closed="right"): 1212,
                Interval(-14.13, -6.034, closed="right"): 357,
                Interval(10.16, 18.257, closed="right"): 336,
                Interval(-22.301000000000002, -14.13, closed="right"): 76,
                Interval(18.257, 26.354, closed="right"): 67,
                Interval(26.354, 34.45, closed="right"): 15,
                Interval(34.45, 42.547, closed="right"): 6,
                Interval(42.547, 50.644, closed="right"): 6,
            },
            "bb_trend_feat": {
                Interval(5.333, 6.222, closed="right"): 570,
                Interval(3.556, 4.444, closed="right"): 567,
                Interval(2.667, 3.556, closed="right"): 554,
                Interval(4.444, 5.333, closed="right"): 530,
                Interval(7.111, 8.0, closed="right"): 503,
                Interval(0.889, 1.778, closed="right"): 483,
                Interval(-0.009000000000000001, 0.889, closed="right"): 474,
                Interval(1.778, 2.667, closed="right"): 429,
                Interval(6.222, 7.111, closed="right"): 241,
            },
        }

        assert_count_series(test_count, expected_count)


class TestBBTrendOPT(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"]

        self.test_df = self.model_features.create_bb_trend_feature_opt(
            source
        ).dropna()

    def test_create_bb_trend_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "bb_trend",
                "bb_trend_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_bb_trend_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "bb_trend": {
                    Timestamp("2012-01-30 00:00:00"): -5.145784196728778,
                    Timestamp("2012-01-31 00:00:00"): 2.012203936130575,
                    Timestamp("2012-02-01 00:00:00"): 0.9920688311179973,
                    Timestamp("2012-02-02 00:00:00"): 0.18008685970188232,
                    Timestamp("2012-02-03 00:00:00"): -2.793305214203219,
                    Timestamp("2012-02-04 00:00:00"): -0.28928190061285664,
                    Timestamp("2012-02-05 00:00:00"): 0.20314442932593613,
                    Timestamp("2012-02-06 00:00:00"): -1.9703927247428688,
                    Timestamp("2012-02-07 00:00:00"): 0.38394839823717275,
                    Timestamp("2012-02-08 00:00:00"): 1.1745460235444356,
                },
                "bb_trend_feat": {
                    Timestamp("2012-01-30 00:00:00"): 1.0,
                    Timestamp("2012-01-31 00:00:00"): 5.0,
                    Timestamp("2012-02-01 00:00:00"): 5.0,
                    Timestamp("2012-02-02 00:00:00"): 4.0,
                    Timestamp("2012-02-03 00:00:00"): 2.0,
                    Timestamp("2012-02-04 00:00:00"): 4.0,
                    Timestamp("2012-02-05 00:00:00"): 4.0,
                    Timestamp("2012-02-06 00:00:00"): 3.0,
                    Timestamp("2012-02-07 00:00:00"): 4.0,
                    Timestamp("2012-02-08 00:00:00"): 5.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_bb_trend_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "bb_trend": {
                Interval(-11.32, 8.009, closed="right"): 3583,
                Interval(8.009, 27.337, closed="right"): 400,
                Interval(-30.648, -11.32, closed="right"): 239,
                Interval(27.337, 46.666, closed="right"): 58,
                Interval(-49.977, -30.648, closed="right"): 46,
                Interval(46.666, 65.994, closed="right"): 12,
                Interval(-69.305, -49.977, closed="right"): 4,
                Interval(65.994, 85.323, closed="right"): 3,
                Interval(-88.80900000000001, -69.305, closed="right"): 2,
            },
            "bb_trend_feat": {
                Interval(0.889, 1.778, closed="right"): 536,
                Interval(7.111, 8.0, closed="right"): 534,
                Interval(5.333, 6.222, closed="right"): 519,
                Interval(3.556, 4.444, closed="right"): 501,
                Interval(1.778, 2.667, closed="right"): 473,
                Interval(-0.009000000000000001, 0.889, closed="right"): 466,
                Interval(2.667, 3.556, closed="right"): 451,
                Interval(6.222, 7.111, closed="right"): 447,
                Interval(4.444, 5.333, closed="right"): 420,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuDistanceLeadLine(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        self.test_df = self.model_features.create_ichimoku_feature(
            9,
            26,
            52,
            26,
            based_on="lead_line",
        ).dropna()

    def test_create_ichimoku_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-02-22 00:00:00"): -0.5299999999999994,
                    Timestamp("2012-02-23 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-02-24 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-02-25 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-02-26 00:00:00"): -0.5899999999999999,
                    Timestamp("2012-02-27 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-02-28 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-02-29 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-01 00:00:00"): -0.5975000000000001,
                    Timestamp("2012-03-02 00:00:00"): -0.6375000000000002,
                },
                "ichimoku_feat": {
                    Timestamp("2012-02-22 00:00:00"): 3.0,
                    Timestamp("2012-02-23 00:00:00"): 2.0,
                    Timestamp("2012-02-24 00:00:00"): 2.0,
                    Timestamp("2012-02-25 00:00:00"): 2.0,
                    Timestamp("2012-02-26 00:00:00"): 3.0,
                    Timestamp("2012-02-27 00:00:00"): 3.0,
                    Timestamp("2012-02-28 00:00:00"): 3.0,
                    Timestamp("2012-02-29 00:00:00"): 3.0,
                    Timestamp("2012-03-01 00:00:00"): 3.0,
                    Timestamp("2012-03-02 00:00:00"): 2.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-1471.706, 756.136, closed="right"): 3232,
                Interval(756.136, 2983.977, closed="right"): 525,
                Interval(-3699.547, -1471.706, closed="right"): 176,
                Interval(2983.977, 5211.818, closed="right"): 157,
                Interval(-5927.388, -3699.547, closed="right"): 99,
                Interval(5211.818, 7439.659, closed="right"): 45,
                Interval(-8155.229, -5927.388, closed="right"): 38,
                Interval(7439.659, 9667.5, closed="right"): 34,
                Interval(-10403.122, -8155.229, closed="right"): 18,
            },
            "ichimoku_feat": {
                Interval(7.111, 8.0, closed="right"): 1347,
                Interval(-0.009000000000000001, 0.889, closed="right"): 1231,
                Interval(5.333, 6.222, closed="right"): 411,
                Interval(6.222, 7.111, closed="right"): 409,
                Interval(1.778, 2.667, closed="right"): 285,
                Interval(0.889, 1.778, closed="right"): 283,
                Interval(3.556, 4.444, closed="right"): 130,
                Interval(2.667, 3.556, closed="right"): 114,
                Interval(4.444, 5.333, closed="right"): 114,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuDistanceLeadingSpan(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        self.test_df = self.model_features.create_ichimoku_feature(
            9,
            26,
            52,
            26,
            based_on="leading_span",
        ).dropna()

    def test_create_ichimoku_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-18 00:00:00"): -0.5299999999999994,
                    Timestamp("2012-03-19 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-20 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-21 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-22 00:00:00"): -0.5899999999999999,
                    Timestamp("2012-03-23 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-24 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-25 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-26 00:00:00"): -0.5975000000000001,
                    Timestamp("2012-03-27 00:00:00"): -0.6375000000000002,
                },
                "ichimoku_feat": {
                    Timestamp("2012-03-18 00:00:00"): 3.0,
                    Timestamp("2012-03-19 00:00:00"): 2.0,
                    Timestamp("2012-03-20 00:00:00"): 2.0,
                    Timestamp("2012-03-21 00:00:00"): 2.0,
                    Timestamp("2012-03-22 00:00:00"): 2.0,
                    Timestamp("2012-03-23 00:00:00"): 3.0,
                    Timestamp("2012-03-24 00:00:00"): 3.0,
                    Timestamp("2012-03-25 00:00:00"): 3.0,
                    Timestamp("2012-03-26 00:00:00"): 2.0,
                    Timestamp("2012-03-27 00:00:00"): 2.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-1471.706, 756.136, closed="right"): 3232,
                Interval(756.136, 2983.977, closed="right"): 514,
                Interval(-3699.547, -1471.706, closed="right"): 176,
                Interval(2983.977, 5211.818, closed="right"): 143,
                Interval(-5927.388, -3699.547, closed="right"): 99,
                Interval(5211.818, 7439.659, closed="right"): 45,
                Interval(-8155.229, -5927.388, closed="right"): 38,
                Interval(7439.659, 9667.5, closed="right"): 34,
                Interval(-10403.122, -8155.229, closed="right"): 18,
            },
            "ichimoku_feat": {
                Interval(1.778, 2.667, closed="right"): 1260,
                Interval(7.111, 8.0, closed="right"): 1184,
                Interval(5.333, 6.222, closed="right"): 405,
                Interval(6.222, 7.111, closed="right"): 397,
                Interval(0.889, 1.778, closed="right"): 318,
                Interval(4.444, 5.333, closed="right"): 258,
                Interval(-0.009000000000000001, 0.889, closed="right"): 235,
                Interval(2.667, 3.556, closed="right"): 128,
                Interval(3.556, 4.444, closed="right"): 114,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuDistanceAbsolute(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        self.test_df = self.model_features.create_ichimoku_feature(
            9,
            26,
            52,
            26,
            method="absolute",
        ).dropna()

    def test_create_ichimoku_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-18 00:00:00"): -0.5299999999999994,
                    Timestamp("2012-03-19 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-20 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-21 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-22 00:00:00"): -0.5899999999999999,
                    Timestamp("2012-03-23 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-24 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-25 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-26 00:00:00"): -0.5975000000000001,
                    Timestamp("2012-03-27 00:00:00"): -0.6375000000000002,
                },
                "ichimoku_feat": {
                    Timestamp("2012-03-18 00:00:00"): 3.0,
                    Timestamp("2012-03-19 00:00:00"): 2.0,
                    Timestamp("2012-03-20 00:00:00"): 2.0,
                    Timestamp("2012-03-21 00:00:00"): 2.0,
                    Timestamp("2012-03-22 00:00:00"): 2.0,
                    Timestamp("2012-03-23 00:00:00"): 3.0,
                    Timestamp("2012-03-24 00:00:00"): 3.0,
                    Timestamp("2012-03-25 00:00:00"): 3.0,
                    Timestamp("2012-03-26 00:00:00"): 2.0,
                    Timestamp("2012-03-27 00:00:00"): 2.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-1471.706, 756.136, closed="right"): 3232,
                Interval(756.136, 2983.977, closed="right"): 514,
                Interval(-3699.547, -1471.706, closed="right"): 176,
                Interval(2983.977, 5211.818, closed="right"): 143,
                Interval(-5927.388, -3699.547, closed="right"): 99,
                Interval(5211.818, 7439.659, closed="right"): 45,
                Interval(-8155.229, -5927.388, closed="right"): 38,
                Interval(7439.659, 9667.5, closed="right"): 34,
                Interval(-10403.122, -8155.229, closed="right"): 18,
            },
            "ichimoku_feat": {
                Interval(1.778, 2.667, closed="right"): 1260,
                Interval(7.111, 8.0, closed="right"): 1184,
                Interval(5.333, 6.222, closed="right"): 405,
                Interval(6.222, 7.111, closed="right"): 397,
                Interval(0.889, 1.778, closed="right"): 318,
                Interval(4.444, 5.333, closed="right"): 258,
                Interval(-0.009000000000000001, 0.889, closed="right"): 235,
                Interval(2.667, 3.556, closed="right"): 128,
                Interval(3.556, 4.444, closed="right"): 114,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuDistanceRatio(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        self.test_df = self.model_features.create_ichimoku_feature(
            9,
            26,
            52,
            26,
            method="ratio",
        ).dropna()

    def test_create_ichimoku_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-18 00:00:00"): 0.9051878354203937,
                    Timestamp("2012-03-19 00:00:00"): 0.8792486583184258,
                    Timestamp("2012-03-20 00:00:00"): 0.8792486583184258,
                    Timestamp("2012-03-21 00:00:00"): 0.8792486583184258,
                    Timestamp("2012-03-22 00:00:00"): 0.8944543828264758,
                    Timestamp("2012-03-23 00:00:00"): 0.9016100178890877,
                    Timestamp("2012-03-24 00:00:00"): 0.9016100178890877,
                    Timestamp("2012-03-25 00:00:00"): 0.9016100178890877,
                    Timestamp("2012-03-26 00:00:00"): 0.8931127012522361,
                    Timestamp("2012-03-27 00:00:00"): 0.8859570661896243,
                },
                "ichimoku_feat": {
                    Timestamp("2012-03-18 00:00:00"): 1.0,
                    Timestamp("2012-03-19 00:00:00"): 0.0,
                    Timestamp("2012-03-20 00:00:00"): 0.0,
                    Timestamp("2012-03-21 00:00:00"): 0.0,
                    Timestamp("2012-03-22 00:00:00"): 1.0,
                    Timestamp("2012-03-23 00:00:00"): 1.0,
                    Timestamp("2012-03-24 00:00:00"): 1.0,
                    Timestamp("2012-03-25 00:00:00"): 1.0,
                    Timestamp("2012-03-26 00:00:00"): 1.0,
                    Timestamp("2012-03-27 00:00:00"): 0.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(0.976, 1.045, closed="right"): 1162,
                Interval(0.907, 0.976, closed="right"): 958,
                Interval(1.045, 1.114, closed="right"): 913,
                Interval(1.114, 1.184, closed="right"): 525,
                Interval(0.838, 0.907, closed="right"): 451,
                Interval(1.184, 1.253, closed="right"): 135,
                Interval(0.768, 0.838, closed="right"): 104,
                Interval(0.698, 0.768, closed="right"): 37,
                Interval(1.253, 1.322, closed="right"): 14,
            },
            "ichimoku_feat": {
                Interval(3.556, 4.444, closed="right"): 668,
                Interval(5.333, 6.222, closed="right"): 594,
                Interval(2.667, 3.556, closed="right"): 561,
                Interval(1.778, 2.667, closed="right"): 550,
                Interval(7.111, 8.0, closed="right"): 506,
                Interval(0.889, 1.778, closed="right"): 485,
                Interval(-0.009000000000000001, 0.889, closed="right"): 424,
                Interval(4.444, 5.333, closed="right"): 286,
                Interval(6.222, 7.111, closed="right"): 225,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuDistanceDTW(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        self.test_df = self.model_features.create_ichimoku_feature(
            9,
            26,
            52,
            26,
            method="dtw",
        ).dropna()

    def test_create_ichimoku_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-18 00:00:00"): -0.5299999999999994,
                    Timestamp("2012-03-19 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-20 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-21 00:00:00"): -0.6749999999999998,
                    Timestamp("2012-03-22 00:00:00"): -0.5899999999999999,
                    Timestamp("2012-03-23 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-24 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-25 00:00:00"): -0.5499999999999998,
                    Timestamp("2012-03-26 00:00:00"): -0.5975000000000001,
                    Timestamp("2012-03-27 00:00:00"): -0.6375000000000002,
                },
                "ichimoku_feat": {
                    Timestamp("2012-03-18 00:00:00"): 1.0,
                    Timestamp("2012-03-19 00:00:00"): 1.0,
                    Timestamp("2012-03-20 00:00:00"): 1.0,
                    Timestamp("2012-03-21 00:00:00"): 1.0,
                    Timestamp("2012-03-22 00:00:00"): 1.0,
                    Timestamp("2012-03-23 00:00:00"): 1.0,
                    Timestamp("2012-03-24 00:00:00"): 1.0,
                    Timestamp("2012-03-25 00:00:00"): 1.0,
                    Timestamp("2012-03-26 00:00:00"): 1.0,
                    Timestamp("2012-03-27 00:00:00"): 1.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-163.067, 76.578, closed="right"): 3830,
                Interval(76.578, 316.222, closed="right"): 210,
                Interval(-402.711, -163.067, closed="right"): 109,
                Interval(316.222, 555.867, closed="right"): 90,
                Interval(-642.356, -402.711, closed="right"): 22,
                Interval(555.867, 795.511, closed="right"): 17,
                Interval(795.511, 1035.156, closed="right"): 10,
                Interval(-884.158, -642.356, closed="right"): 8,
                Interval(1035.156, 1274.8, closed="right"): 3,
            },
            "ichimoku_feat": {
                Interval(7.111, 8.0, closed="right"): 1638,
                Interval(-0.009000000000000001, 0.889, closed="right"): 901,
                Interval(6.222, 7.111, closed="right"): 800,
                Interval(0.889, 1.778, closed="right"): 274,
                Interval(1.778, 2.667, closed="right"): 166,
                Interval(5.333, 6.222, closed="right"): 145,
                Interval(3.556, 4.444, closed="right"): 132,
                Interval(4.444, 5.333, closed="right"): 125,
                Interval(2.667, 3.556, closed="right"): 118,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuDistanceNegativeCase(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

    def test_create_ichimoku_distance_invalid_method(self) -> None:
        self.assertRaises(
            ValueError,
            self.model_features.create_ichimoku_feature,
            9,
            26,
            52,
            26,
            method="invalid",
        )

    def test_create_ichimoku_distance_invalid_based_on(self) -> None:
        self.assertRaises(
            ValueError,
            self.model_features.create_ichimoku_feature,
            9,
            26,
            52,
            26,
            based_on="invalid",
        )


class TestIchimokuPriceDistanceLeadingSpan(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"].copy()

        self.test_df = (
            self.model_features.create_ichimoku_price_distance_feature(
                source, 9, 26, 52, 26, based_on="leading_span"
            ).dropna()
        )

    def test_create_ichimoku_price_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_distance_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-19 00:00:00"): -0.026269209405665572,
                    Timestamp("2012-03-20 00:00:00"): 0.022630340689408363,
                    Timestamp("2012-03-21 00:00:00"): 0.057617871428714856,
                    Timestamp("2012-03-22 00:00:00"): -0.015980823414409623,
                    Timestamp("2012-03-23 00:00:00"): 0.027254212182929116,
                    Timestamp("2012-03-24 00:00:00"): -0.0033187150858822456,
                    Timestamp("2012-03-25 00:00:00"): -0.0063106172731604016,
                    Timestamp("2012-03-26 00:00:00"): 0.028071254132409934,
                    Timestamp("2012-03-27 00:00:00"): 0.010620941822817809,
                    Timestamp("2012-03-28 00:00:00"): 0.037773894681839615,
                },
                "ichimoku_distance_feat": {
                    Timestamp("2012-03-19 00:00:00"): 0.0,
                    Timestamp("2012-03-20 00:00:00"): 5.0,
                    Timestamp("2012-03-21 00:00:00"): 6.0,
                    Timestamp("2012-03-22 00:00:00"): 1.0,
                    Timestamp("2012-03-23 00:00:00"): 5.0,
                    Timestamp("2012-03-24 00:00:00"): 3.0,
                    Timestamp("2012-03-25 00:00:00"): 2.0,
                    Timestamp("2012-03-26 00:00:00"): 5.0,
                    Timestamp("2012-03-27 00:00:00"): 4.0,
                    Timestamp("2012-03-28 00:00:00"): 6.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-0.0371, 0.0633, closed="right"): 3533,
                Interval(-0.138, -0.0371, closed="right"): 404,
                Interval(0.0633, 0.164, closed="right"): 304,
                Interval(0.164, 0.264, closed="right"): 28,
                Interval(-0.238, -0.138, closed="right"): 18,
                Interval(0.264, 0.365, closed="right"): 5,
                Interval(-0.338, -0.238, closed="right"): 3,
                Interval(-0.441, -0.338, closed="right"): 2,
                Interval(0.365, 0.465, closed="right"): 1,
            },
            "ichimoku_distance_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 660,
                Interval(7.111, 8.0, closed="right"): 555,
                Interval(1.778, 2.667, closed="right"): 499,
                Interval(0.889, 1.778, closed="right"): 478,
                Interval(2.667, 3.556, closed="right"): 473,
                Interval(3.556, 4.444, closed="right"): 472,
                Interval(5.333, 6.222, closed="right"): 449,
                Interval(4.444, 5.333, closed="right"): 433,
                Interval(6.222, 7.111, closed="right"): 279,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuPriceDistanceLeadLine(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"].copy()

        self.test_df = (
            self.model_features.create_ichimoku_price_distance_feature(
                source, 9, 26, 52, 26, based_on="lead_line"
            ).dropna()
        )

    def test_create_ichimoku_price_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_distance_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-02-23 00:00:00"): 0.17224624561875204,
                    Timestamp("2012-02-24 00:00:00"): 0.0020066526633000237,
                    Timestamp("2012-02-25 00:00:00"): 0.032228276101750136,
                    Timestamp("2012-02-26 00:00:00"): -0.009900178253119257,
                    Timestamp("2012-02-27 00:00:00"): -0.0028519854820351354,
                    Timestamp("2012-02-28 00:00:00"): 0.021984758848302643,
                    Timestamp("2012-02-29 00:00:00"): 0.008762017474650302,
                    Timestamp("2012-03-01 00:00:00"): -0.011774958115875284,
                    Timestamp("2012-03-02 00:00:00"): 0.01173079059433435,
                    Timestamp("2012-03-03 00:00:00"): -0.010048704051747792,
                },
                "ichimoku_distance_feat": {
                    Timestamp("2012-02-23 00:00:00"): 7.0,
                    Timestamp("2012-02-24 00:00:00"): 3.0,
                    Timestamp("2012-02-25 00:00:00"): 8.0,
                    Timestamp("2012-02-26 00:00:00"): 1.0,
                    Timestamp("2012-02-27 00:00:00"): 2.0,
                    Timestamp("2012-02-28 00:00:00"): 8.0,
                    Timestamp("2012-02-29 00:00:00"): 4.0,
                    Timestamp("2012-03-01 00:00:00"): 1.0,
                    Timestamp("2012-03-02 00:00:00"): 5.0,
                    Timestamp("2012-03-03 00:00:00"): 1.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-0.0224, 0.0672, closed="right"): 3376,
                Interval(-0.112, -0.0224, closed="right"): 660,
                Interval(0.0672, 0.157, closed="right"): 233,
                Interval(-0.202, -0.112, closed="right"): 23,
                Interval(0.157, 0.246, closed="right"): 16,
                Interval(0.246, 0.336, closed="right"): 8,
                Interval(-0.291, -0.202, closed="right"): 3,
                Interval(0.336, 0.426, closed="right"): 3,
                Interval(-0.383, -0.291, closed="right"): 1,
            },
            "ichimoku_distance_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 682,
                Interval(0.889, 1.778, closed="right"): 552,
                Interval(1.778, 2.667, closed="right"): 514,
                Interval(2.667, 3.556, closed="right"): 510,
                Interval(4.444, 5.333, closed="right"): 469,
                Interval(5.333, 6.222, closed="right"): 455,
                Interval(3.556, 4.444, closed="right"): 435,
                Interval(7.111, 8.0, closed="right"): 419,
                Interval(6.222, 7.111, closed="right"): 287,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuPriceDistanceAbsolute(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"].copy()

        self.test_df = (
            self.model_features.create_ichimoku_price_distance_feature(
                source, 9, 26, 52, 26, method="absolute"
            ).dropna()
        )

    def test_create_ichimoku_price_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_distance_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_price_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-19 00:00:00"): -0.026269209405665572,
                    Timestamp("2012-03-20 00:00:00"): 0.022630340689408363,
                    Timestamp("2012-03-21 00:00:00"): 0.057617871428714856,
                    Timestamp("2012-03-22 00:00:00"): -0.015980823414409623,
                    Timestamp("2012-03-23 00:00:00"): 0.027254212182929116,
                    Timestamp("2012-03-24 00:00:00"): -0.0033187150858822456,
                    Timestamp("2012-03-25 00:00:00"): -0.0063106172731604016,
                    Timestamp("2012-03-26 00:00:00"): 0.028071254132409934,
                    Timestamp("2012-03-27 00:00:00"): 0.010620941822817809,
                    Timestamp("2012-03-28 00:00:00"): 0.037773894681839615,
                },
                "ichimoku_distance_feat": {
                    Timestamp("2012-03-19 00:00:00"): 0.0,
                    Timestamp("2012-03-20 00:00:00"): 5.0,
                    Timestamp("2012-03-21 00:00:00"): 6.0,
                    Timestamp("2012-03-22 00:00:00"): 1.0,
                    Timestamp("2012-03-23 00:00:00"): 5.0,
                    Timestamp("2012-03-24 00:00:00"): 3.0,
                    Timestamp("2012-03-25 00:00:00"): 2.0,
                    Timestamp("2012-03-26 00:00:00"): 5.0,
                    Timestamp("2012-03-27 00:00:00"): 4.0,
                    Timestamp("2012-03-28 00:00:00"): 6.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_price_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-0.0371, 0.0633, closed="right"): 3533,
                Interval(-0.138, -0.0371, closed="right"): 404,
                Interval(0.0633, 0.164, closed="right"): 304,
                Interval(0.164, 0.264, closed="right"): 28,
                Interval(-0.238, -0.138, closed="right"): 18,
                Interval(0.264, 0.365, closed="right"): 5,
                Interval(-0.338, -0.238, closed="right"): 3,
                Interval(-0.441, -0.338, closed="right"): 2,
                Interval(0.365, 0.465, closed="right"): 1,
            },
            "ichimoku_distance_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 660,
                Interval(7.111, 8.0, closed="right"): 555,
                Interval(1.778, 2.667, closed="right"): 499,
                Interval(0.889, 1.778, closed="right"): 478,
                Interval(2.667, 3.556, closed="right"): 473,
                Interval(3.556, 4.444, closed="right"): 472,
                Interval(5.333, 6.222, closed="right"): 449,
                Interval(4.444, 5.333, closed="right"): 433,
                Interval(6.222, 7.111, closed="right"): 279,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuPriceDistanceRatio(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"].copy()

        self.test_df = (
            self.model_features.create_ichimoku_price_distance_feature(
                source, 9, 26, 52, 26, method="ratio"
            ).dropna()
        )

    def test_create_ichimoku_price_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_distance_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_price_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-19 00:00:00"): -0.026269209405665572,
                    Timestamp("2012-03-20 00:00:00"): 0.04544921709536953,
                    Timestamp("2012-03-21 00:00:00"): 0.08922139602230927,
                    Timestamp("2012-03-22 00:00:00"): 0.04365014543304413,
                    Timestamp("2012-03-23 00:00:00"): 0.027254212182929116,
                    Timestamp("2012-03-24 00:00:00"): 0.06419104164035938,
                    Timestamp("2012-03-25 00:00:00"): -0.0063106172731604016,
                    Timestamp("2012-03-26 00:00:00"): 0.09558101085865156,
                    Timestamp("2012-03-27 00:00:00"): 0.07698660454782258,
                    Timestamp("2012-03-28 00:00:00"): 0.07232583191664832,
                },
                "ichimoku_distance_feat": {
                    Timestamp("2012-03-19 00:00:00"): 1.0,
                    Timestamp("2012-03-20 00:00:00"): 4.0,
                    Timestamp("2012-03-21 00:00:00"): 6.0,
                    Timestamp("2012-03-22 00:00:00"): 4.0,
                    Timestamp("2012-03-23 00:00:00"): 8.0,
                    Timestamp("2012-03-24 00:00:00"): 5.0,
                    Timestamp("2012-03-25 00:00:00"): 2.0,
                    Timestamp("2012-03-26 00:00:00"): 6.0,
                    Timestamp("2012-03-27 00:00:00"): 6.0,
                    Timestamp("2012-03-28 00:00:00"): 5.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_price_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-0.0185, 0.0866, closed="right"): 2643,
                Interval(-0.124, -0.0185, closed="right"): 1160,
                Interval(0.0866, 0.192, closed="right"): 403,
                Interval(-0.229, -0.124, closed="right"): 46,
                Interval(0.192, 0.297, closed="right"): 33,
                Interval(-0.334, -0.229, closed="right"): 6,
                Interval(0.297, 0.402, closed="right"): 4,
                Interval(-0.441, -0.334, closed="right"): 2,
                Interval(0.402, 0.507, closed="right"): 1,
            },
            "ichimoku_distance_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 783,
                Interval(0.889, 1.778, closed="right"): 580,
                Interval(2.667, 3.556, closed="right"): 528,
                Interval(3.556, 4.444, closed="right"): 479,
                Interval(1.778, 2.667, closed="right"): 475,
                Interval(7.111, 8.0, closed="right"): 454,
                Interval(4.444, 5.333, closed="right"): 418,
                Interval(5.333, 6.222, closed="right"): 349,
                Interval(6.222, 7.111, closed="right"): 232,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuPriceDistanceDTW(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

        source = self.dataframe["close"].copy()

        self.test_df = (
            self.model_features.create_ichimoku_price_distance_feature(
                source, 9, 26, 52, 26, method="dtw"
            ).dropna()
        )

    def test_create_ichimoku_price_distance_feature_columns(self) -> None:
        expected_columns = pd.Index(
            [
                "ichimoku_distance",
                "ichimoku_distance_feat",
            ]
        )

        pd.testing.assert_index_equal(
            self.test_df.columns[8:], expected_columns
        )

    def test_create_ichimoku_price_distance_feature_values(self) -> None:
        expected_df = pd.DataFrame(
            {
                "ichimoku_distance": {
                    Timestamp("2012-03-08 00:00:00"): 0.05504420687066988,
                    Timestamp("2012-03-09 00:00:00"): 0.025689675683105345,
                    Timestamp("2012-03-13 00:00:00"): 0.12865211792248468,
                    Timestamp("2012-03-15 00:00:00"): 0.08929826790746412,
                    Timestamp("2012-03-16 00:00:00"): 0.026067536925923368,
                    Timestamp("2012-03-17 00:00:00"): 0.08060452628094641,
                    Timestamp("2012-03-18 00:00:00"): 0.0123029513856443,
                    Timestamp("2012-03-19 00:00:00"): -0.06518169229699075,
                    Timestamp("2012-03-20 00:00:00"): 0.04544921709536953,
                    Timestamp("2012-03-21 00:00:00"): 0.08922139602230927,
                },
                "ichimoku_distance_feat": {
                    Timestamp("2012-03-08 00:00:00"): 8.0,
                    Timestamp("2012-03-09 00:00:00"): 5.0,
                    Timestamp("2012-03-13 00:00:00"): 7.0,
                    Timestamp("2012-03-15 00:00:00"): 6.0,
                    Timestamp("2012-03-16 00:00:00"): 5.0,
                    Timestamp("2012-03-17 00:00:00"): 6.0,
                    Timestamp("2012-03-18 00:00:00"): 4.0,
                    Timestamp("2012-03-19 00:00:00"): 0.0,
                    Timestamp("2012-03-20 00:00:00"): 8.0,
                    Timestamp("2012-03-21 00:00:00"): 6.0,
                },
            }
        )

        expected_df.index.name = "date"

        pd.testing.assert_frame_equal(expected_df, self.test_df.iloc[:10, 8:])

    def test_create_ichimoku_price_distance_feature_count(self) -> None:
        feat_columns = self.test_df.columns[8:]
        test_count = {}
        for column in feat_columns:
            test_count[column] = (
                self.test_df[column].value_counts(bins=self.bins).to_dict()
            )

        expected_count = {
            "ichimoku_distance": {
                Interval(-0.0185, 0.0866, closed="right"): 2902,
                Interval(-0.124, -0.0185, closed="right"): 1027,
                Interval(0.0866, 0.192, closed="right"): 297,
                Interval(-0.229, -0.124, closed="right"): 46,
                Interval(0.192, 0.297, closed="right"): 21,
                Interval(-0.334, -0.229, closed="right"): 6,
                Interval(0.297, 0.402, closed="right"): 3,
                Interval(-0.441, -0.334, closed="right"): 2,
                Interval(0.402, 0.507, closed="right"): 1,
            },
            "ichimoku_distance_feat": {
                Interval(-0.009000000000000001, 0.889, closed="right"): 664,
                Interval(0.889, 1.778, closed="right"): 613,
                Interval(1.778, 2.667, closed="right"): 530,
                Interval(3.556, 4.444, closed="right"): 523,
                Interval(2.667, 3.556, closed="right"): 512,
                Interval(7.111, 8.0, closed="right"): 494,
                Interval(4.444, 5.333, closed="right"): 417,
                Interval(5.333, 6.222, closed="right"): 299,
                Interval(6.222, 7.111, closed="right"): 253,
            },
        }

        assert_count_series(test_count, expected_count)


class TestIchimokuPriceDistanceNegativeCases(unittest.TestCase):
    def setUp(self) -> None:
        btc_data = pd.read_parquet(r"data\assets\btc.parquet")
        self.dataframe: pd.DataFrame = btc_data.copy().loc[:"2023"]
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

    def test_create_ichimoku_price_distance_invalid_method(self) -> None:
        source = self.dataframe["close"].copy()

        self.assertRaises(
            ValueError,
            self.model_features.create_ichimoku_price_distance_feature,
            source,
            9,
            26,
            52,
            26,
            method="invalid",
        )

    def test_create_ichimoku_price_distance_invalid_based_on(self) -> None:
        source = self.dataframe["close"].copy()

        self.assertRaises(
            ValueError,
            self.model_features.create_ichimoku_price_distance_feature,
            source,
            9,
            26,
            52,
            26,
            based_on="invalid",
        )
