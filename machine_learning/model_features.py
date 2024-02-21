import pandas as pd
import tradingview_indicators as ta
from utils import DynamicTimeWarping
from typing import Literal

def feature_binning(
    feature: pd.Series,
    test_index: str | int,
    bins: int = 10,
) -> pd.Series:
    """
    Perform feature binning using quantiles.

    Parameters:
    -----------
    feature : pd.Series
        The input feature series to be binned.
    test_index : str or int
        The index or label up to which the training data is considered.
    bins : int, optional
        The number of bins to use for binning the feature.
        (default: 10)

    Returns:
    --------
    pd.Series
        The binned feature series.
    """
    train_series = (
        feature.iloc[:test_index].copy() if isinstance(test_index, int)
        else feature.loc[:test_index].copy()
    )

    intervals = pd.qcut(train_series, bins).value_counts().index.to_list()

    lows = pd.Series([interval.left for interval in intervals])
    highs = pd.Series([interval.right for interval in intervals])
    lows.iloc[0] = -999999999999
    highs.iloc[-1] = 999999999999

    intervals_range = (
        pd.concat([lows.rename("lowest"), highs.rename("highest")], axis=1)
        .sort_values("highest")
        .reset_index(drop=True)
    )

    return feature.dropna().apply(
        lambda x: intervals_range[
            (x >= intervals_range["lowest"])
            & (x <= intervals_range["highest"])
        ].index[0]
    )


class ModelFeatures:
    """
    Class for creating and manipulating features for a machine learning model.

    Parameters:
    -----------
    dataset : pd.DataFrame
        The dataset containing the features.
    test_index : int
        The index or label up to which the training data is considered.
    bins : int, optional
        The number of bins to use for binning the features.
        (default: 10)

    Methods:
    --------
    create_rsi_feature(source: pd.Series, length: int) -> pd.DataFrame:
        Create the RSI (Relative Strength Index) feature.

    create_slow_stoch_feature(
        source_column: str,
        k_length: int = 14,
        k_smoothing: int = 1,
        d_smoothing: int = 3,
    ) -> pd.DataFrame:
        Create the slow stochastic feature.

    create_dtw_distance_feature(
        source: pd.Series,
        feats: list,
        length: int,
    ) -> pd.DataFrame:
        Create the DTW distance feature.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        test_index: int,
        bins: int = 10,
    ):
        self.dataset = dataset.copy()
        self.test_index = test_index
        self.bins = bins

    def create_rsi_feature(self, source: pd.Series, length: int):
        """
        Create the RSI (Relative Strength Index) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating RSI.
        length : int
            The length of the RSI calculation.

        Returns:
        --------
        pd.DataFrame
            The dataset with the RSI feature added.
        """
        self.dataset["RSI"] = ta.RSI(source, length)
        self.dataset.loc[:, "RSI_feat"] = feature_binning(
            self.dataset["RSI"],
            self.test_index,
            self.bins,
        )

        return self.dataset

    def create_slow_stoch_feature(
        self,
        source_column: str,
        k_length: int = 14,
        k_smoothing: int = 1,
        d_smoothing: int = 3,
    ):
        """
        Create the slow stochastic feature.

        Parameters:
        -----------
        source_column : str
            The column name of the source data.
        k_length : int, optional
            The length of the %K calculation. (default: 14)
        k_smoothing : int, optional
            The smoothing factor for %K. (default: 1)
        d_smoothing : int, optional
            The smoothing factor for %D. (default: 3)

        Returns:
        --------
        pd.DataFrame
            The dataset with the slow stochastic feature added.
        """
        stoch_k, stoch_d = (
            ta.SlowStochastic(self.dataset, source_column)
            .slow_stoch(k_length, k_smoothing, d_smoothing)
        )

        self.dataset["stoch_k"] = stoch_k
        self.dataset.loc[:, "stoch_k_feat"] = feature_binning(
            self.dataset["stoch_k"],
            self.test_index,
            self.bins,
        )

        self.dataset["stoch_d"] = stoch_d
        self.dataset.loc[:, "stoch_d_feat"] = feature_binning(
            self.dataset["stoch_d"],
            self.test_index,
            self.bins,
        )

        return self.dataset

    def create_dtw_distance_feature(
        self,
        source: pd.Series,
        feats: Literal["sma", "ema", "dema", "tema", "rma", "all"] | list,
        length: int,
    ) -> pd.DataFrame:
        """
        Create the DTW distance feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the DTW distance.
        feats : Literal["sma", "ema", "dema", "tema", "rma", "all"]
        | list
            The list of features to calculate the DTW distance for.
        length : int
            The length of the moving average calculation.

        Returns:
        --------
        pd.DataFrame
            The dataset with the DTW distance features added.
        """
        all_mas = feats == "all"

        if any(feat.lower().startswith("sma") for feat in feats) or all_mas:
            sma = ta.sma(source, length).dropna()

            self.dataset["SMA_DTW"] = (
                DynamicTimeWarping(source, sma)
                .calculate_dtw_distance("absolute", True)
            )

            self.dataset.loc[:, "SMA_DTW_feat"] = feature_binning(
                self.dataset["SMA_DTW"],
                self.test_index,
                self.bins,
            )

        if any(feat.lower().startswith("ema") for feat in feats) or all_mas:
            ema = ta.ema(source, length).dropna()

            self.dataset["EMA_DTW"] = (
                DynamicTimeWarping(source, ema)
                .calculate_dtw_distance("absolute", True)
            )

            self.dataset.loc[:, "EMA_DTW_feat"] = feature_binning(
                self.dataset["EMA_DTW"],
                self.test_index,
                self.bins,
            )

        if any(feat.lower().startswith("rma") for feat in feats) or all_mas:
            rma = ta.rma(source, length).dropna()

            self.dataset["RMA_DTW"] = (
                DynamicTimeWarping(source, rma)
                .calculate_dtw_distance("absolute", True)
            )

            self.dataset.loc[:, "RMA_DTW_feat"] = feature_binning(
                self.dataset["RMA_DTW"],
                self.test_index,
                self.bins,
            )

        if any(feat.lower().startswith("dema") for feat in feats) or all_mas:
            dema = ta.sema(source, length, 2).dropna()
            self.dataset["DEMA_DTW"] = (
                DynamicTimeWarping(source, dema)
                .calculate_dtw_distance("absolute", True)
            )

            self.dataset.loc[:, "DEMA_DTW_feat"] = feature_binning(
                self.dataset["DEMA_DTW"],
                self.test_index,
                self.bins,
            )

        if any(feat.lower().startswith("tema") for feat in feats) or all_mas:
            tema = ta.sema(source, length, 3).dropna()
            self.dataset["TEMA_DTW"] = (
                DynamicTimeWarping(source, tema)
                .calculate_dtw_distance("absolute", True)
            )

            self.dataset.loc[:, "TEMA_DTW_feat"] = feature_binning(
                self.dataset["TEMA_DTW"],
                self.test_index,
                self.bins,
            )

        return self.dataset
