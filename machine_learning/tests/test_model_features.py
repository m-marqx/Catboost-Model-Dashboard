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

