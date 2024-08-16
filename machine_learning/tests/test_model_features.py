import unittest

import pandas as pd
import numpy as np

from machine_learning.model_features import feature_binning, ModelFeatures


class ModelFeaturesTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
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


if __name__ == "__main__":
    unittest.main()
