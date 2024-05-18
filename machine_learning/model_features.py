from typing import Literal
import logging
import time

import numpy as np
import pandas as pd
import tradingview_indicators as ta
from utils import DynamicTimeWarping

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

    Raises:
    -------
    ValueError
        If the feature contains NaN or infinite values.
    """
    has_inf = np.sum(np.isinf(feature.dropna().to_numpy())) >= 1
    has_na = np.sum(np.isnan(feature.dropna().to_numpy())) >= 1

    if has_inf or has_na:
        raise ValueError(
            "Feature contains NaN or infinite values. "
            "Please clean the data before binning."
        )

    train_series = (
        feature.iloc[:test_index].copy() if isinstance(test_index, int)
        else feature.loc[:test_index].copy()
    )

    intervals = (
        pd.qcut(train_series, bins, duplicates='drop')
        .value_counts()
        .index
        .to_list()
    )

    lows = pd.Series([interval.left for interval in intervals])
    highs = pd.Series([interval.right for interval in intervals])
    lows.iloc[0] = -np.inf
    highs.iloc[-1] = np.inf

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
        verbose: bool = True,
    ):
        self.dataset = dataset.copy()
        self.test_index = test_index
        self.bins = bins

        self.logger = logging.getLogger("Model_Features")
        formatter = logging.Formatter(
            '%(levelname)s %(asctime)s: %(message)s', datefmt='%H:%M:%S'
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(handler)
        self.logger.propagate = False

        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

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
        self.logger.info("Calculating RSI...")
        start = time.perf_counter()

        self.dataset["RSI"] = ta.RSI(source, length)
        self.dataset.loc[:, "RSI_feat"] = feature_binning(
            self.dataset["RSI"],
            self.test_index,
            self.bins,
        )
        self.logger.info(
            "RSI calculated in %.2f seconds.", time.perf_counter() - start
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

        self.logger.info("Calculating slow stochastic...")
        start = time.perf_counter()

        stoch_k, stoch_d = (
            ta.slow_stoch(
                self.dataset[source_column],
                self.dataset['high'],
                self.dataset['low'],
                k_length,
                k_smoothing,
                d_smoothing,
                'sma'
            )
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

        self.logger.info(
            "Slow stochastic calculated in %.2f seconds.",
            time.perf_counter() - start,
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
        self.logger.info("Calculating DTW distance for moving averages...\n")
        start_time_dtw = time.perf_counter()

        if any(feat.lower().startswith("sma") for feat in feats) or all_mas:
            self.logger.info("Calculating DTW distance for SMA...")
            start = time.perf_counter()

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
            self.logger.info(
                "DTW distance for SMA calculated in %.2f seconds.",
                time.perf_counter() - start,
            )

        if any(feat.lower().startswith("ema") for feat in feats) or all_mas:
            self.logger.info("Calculating DTW distance for EMA...")
            start = time.perf_counter()

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

            self.logger.info(
                "DTW distance for EMA calculated in %.2f seconds.",
                time.perf_counter() - start,
            )

        if any(feat.lower().startswith("rma") for feat in feats) or all_mas:
            self.logger.info("Calculating DTW distance for RMA...")
            start = time.perf_counter()

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

            self.logger.info(
                "DTW distance for RMA calculated in %.2f seconds.",
                time.perf_counter() - start,
            )

        if any(feat.lower().startswith("dema") for feat in feats) or all_mas:
            self.logger.info("Calculating DTW distance for DEMA...")
            start = time.perf_counter()

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

            self.logger.info(
                "DTW distance for DEMA calculated in %.2f seconds.",
                time.perf_counter() - start,
            )

        if any(feat.lower().startswith("tema") for feat in feats) or all_mas:
            self.logger.info("Calculating DTW distance for TEMA...")
            start = time.perf_counter()

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

            self.logger.info(
                "DTW distance for TEMA calculated in %.2f seconds.",
                time.perf_counter() - start,
            )

        self.logger.info(
            "\nDTW distance for moving averages calculated in %.2f seconds.\n",
            time.perf_counter() - start_time_dtw,
        )

        return self.dataset

    def create_cci_feature(
        self,
        source: pd.Series,
        length: int = 20,
        method: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'sma',
    ):
        """
        Create the CCI (Commodity Channel Index) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating CCI.
        length : int, optional
            The length of the CCI calculation.
            (default: 20)
        method : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for CCI calculation.
            (default: 'sma')

        Returns:
        --------
        pd.DataFrame
            The dataset with the CCI feature added.
        """
        self.logger.info("Calculating CCI...")
        start = time.perf_counter()

        self.dataset['CCI'] =  ta.CCI(source, length, method=method)['CCI']

        self.dataset.loc[:, "CCI_feat"] = feature_binning(
            self.dataset["CCI"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "CCI calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_didi_index_feature(
        self,
        source: pd.Series,
        short_length: int = 3,
        medium_length: int = 18,
        long_length: int = 20,
        ma_type: Literal['sma', 'ema','dema','tema', 'rma'] = 'sma',
        method: Literal['absolute', 'ratio', 'dtw'] = 'absolute',
    ):
        """
        Create the Didi Index feature.

        Parameters:
        -----------

        source : pd.Series
            The source series for calculating the DIDI index.
        short_length : int, optional
            The length of the short EMA.
            (default: 3)
        medium_length : int, optional
            The length of the medium EMA.
            (default: 18)
        long_length : int, optional
            The length of the long EMA.
            (default: 20)

        Returns:
        --------
        pd.DataFrame
            The dataset with the DIDI index feature added.
        """
        if method not in ["absolute", "ratio", "dtw"]:
            raise ValueError(
                "Invalid method provided. Use 'absolute', 'ratio', or 'dtw'."
            )

        self.logger.info("Calculating new DIDI index...")
        start = time.perf_counter()

        is_dtw_distance = False

        if method == "dtw":
            method = "absolute"
            is_dtw_distance = True

        self.dataset["DIDI"] = ta.didi_index(
            source,
            short_length,
            medium_length,
            long_length,
            ma_type,
            method,
            is_dtw_distance,
        )

        self.dataset.loc[:, "DIDI_feat"] = feature_binning(
            self.dataset["DIDI"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "DIDI index calculated in %.2f seconds.",
            time.perf_counter() - start
        )

        return self.dataset

    def create_macd_feature(
        self,
        source: pd.Series,
        fast_length: int = 12,
        slow_length: int = 26,
        signal_length: int = 9,
        diff_method: Literal['absolute', 'ratio', 'dtw'] = 'absolute',
        ma_method: Literal['sma', 'ema','dema','tema', 'rma'] = 'ema',
        signal_method: Literal['sma', 'ema','dema','tema', 'rma'] = 'ema',
        column: Literal['macd', 'signal', 'histogram'] = 'histogram',
    ):
        """
        Create the MACD index feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the MACD index.
        short_length : int, optional
            The length of the short EMA. (default: 12)
        long_length : int, optional
            The length of the long EMA. (default: 26)
        signal_length : int, optional
            The length of the signal line. (default: 9)
        diff_method : Literal['absolute', 'ratio', 'dtw'], optional
            The method to use for calculating the MACD index.
            (default: 'absolute')
        ma_method : Literal['sma', 'ema','dema','tema', 'rma'], optional
            The moving average method to use for MACD calculation.
            (default: 'ema')
        signal_method : Literal['sma', 'ema','dema','tema', 'rma'], optional
            The moving average method to use for signal line calculation.
            (default: 'ema')
        column : Literal['macd', 'signal', 'histogram'], optional
            The column to return from the MACD calculation.
            (default: 'histogram')

        Returns:
        --------
        pd.DataFrame
            The dataset with the MACD index feature added.
        """
        self.logger.info("Calculating MACD index...")
        start = time.perf_counter()

        self.dataset["MACD"] = ta.MACD(
            source=source,
            fast_length=fast_length,
            slow_length=slow_length,
            signal_length=signal_length,
            diff_method=diff_method,
            ma_method=ma_method,
            signal_method=signal_method,
        )[column]

        self.dataset.loc[:, "MACD_feat"] = feature_binning(
            self.dataset["MACD"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "MACD index calculated in %.2f seconds.",
            time.perf_counter() - start
        )

        return self.dataset

    def create_trix_feature(
        self,
        source: pd.Series,
        length: int = 15,
        signal_length: int = 1,
        method: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
    ):
        """
        Create the TRIX (Triple Exponential Moving Average) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the TRIX.
        length : int, optional
            The length of the TRIX calculation.
            (default: 15)
        signal_length : int, optional
            The length of the signal line.
            (default: 1)
        method : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for TRIX calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the TRIX feature added.
        """
        self.logger.info("Calculating TRIX...")
        start = time.perf_counter()

        self.dataset["TRIX"] = ta.TRIX(source, length, signal_length, method)

        self.dataset.loc[:, "TRIX_feat"] = feature_binning(
            self.dataset["TRIX"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "TRIX calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_smio_feature(
        self,
        source: pd.Series,
        short_length: int = 20,
        long_length: int = 5,
        signal_length: int = 5,
        ma_type: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
    ):
        """
        Create the SMIO (SMI Ergotic Oscillator) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the SMIO.
        short_length : int, optional
            The length of the faster moving average. (default: 3)
        long_length : int, optional
            The length of the slower moving average. (default: 18)
        ma_type : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for SMIO calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the SMIO feature added.
        """
        self.logger.info("Calculating SMIO...")
        start = time.perf_counter()

        self.dataset["SMIO"] = ta.SMIO(
            source=source,
            long_length=long_length,
            short_length=short_length,
            signal_length=signal_length,
            ma_method=ma_type,
        )

        self.dataset.loc[:, "SMIO_feat"] = feature_binning(
            self.dataset["SMIO"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "SMIO calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_tsi_feature(
        self,
        source: pd.Series,
        short_length: int = 13,
        long_length: int = 25,
        ma_type: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
    ):
        """
        Create the TSI (True Strength Index) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the TSI.
        short_length : int, optional
            The length of the faster MA.
            (default: 13)
        long_length : int, optional
            The length of the slower MA.
            (default: 25)
        ma_type : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for TSI calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the TSI feature added.
        """
        self.logger.info("Calculating TSI...")
        start = time.perf_counter()

        self.dataset["TSI"] = ta.tsi(
            source=source,
            short_length=short_length,
            long_length=long_length,
            ma_method=ma_type,
        )

        self.dataset.loc[:, "TSI_feat"] = feature_binning(
            self.dataset["TSI"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "TSI calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

def model_creation(
    random_variables: dict,
    hyperparams: dict,
    test_index: int,
    dataframe: pd.DataFrame,
    max_trades: int = 3,
    off_days: int = 7,
    pct_adj: float = 0.5,
    train_in_middle: bool = True,
):
    data_frame = dataframe.copy()

    if "DTW" in random_variables["random_features"]:
        dtw_source = (
            data_frame[random_variables["random_source_price_dtw"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_dtw"]
        ).create_dtw_distance_feature(
            dtw_source,
            random_variables["random_moving_averages"],
            random_variables["random_moving_averages_length"],
        )

    if "RSI" in random_variables["random_features"]:
        rsi_source = (
            data_frame[random_variables["random_source_price_rsi"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_rsi"]
        ).create_rsi_feature(rsi_source, random_variables["random_rsi_length"])

    if "Stoch" in random_variables["random_features"]:
        data_frame["slow_stoch_source"] = (
            data_frame[random_variables["random_source_price_stoch"]]
            .pct_change(1)
        )

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_stoch"]
        ).create_slow_stoch_feature(
            random_variables["random_source_price_stoch"],
            random_variables["random_slow_stoch_length"],
            random_variables["random_slow_stoch_k"],
            random_variables["random_slow_stoch_d"],
        )

    if "CCI" in random_variables["random_features"]:
        cci_source = (
            data_frame[random_variables["random_source_price_cci"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_cci"]
        ).create_cci_feature(
            cci_source,
            random_variables["random_cci_length"],
            random_variables["random_cci_method"],
        )

    if "MACD" in random_variables["random_features"]:
        macd_source = (
            data_frame[random_variables["random_source_price_macd"]]
            .pct_change(1)
            .iloc[1:]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_macd"]
        ).create_macd_feature(
            macd_source,
            random_variables["random_macd_fast_length"],
            random_variables["random_macd_slow_length"],
            random_variables["random_macd_signal_length"],
            random_variables["random_macd_diff_method"],
            random_variables["random_macd_ma_method"],
            random_variables["random_macd_signal_method"],
            random_variables["random_macd_column"],
        )

    if "TRIX" in random_variables["random_features"]:
        trix_source = data_frame[random_variables["random_source_price_trix"]]

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_trix"]
        ).create_trix_feature(
            trix_source,
            random_variables["random_trix_length"],
            random_variables["random_trix_signal_length"],
            random_variables["random_trix_ma_method"],
        )

    if "SMIO" in random_variables["random_features"]:
        smio_source = data_frame[random_variables["random_source_price_smio"]]

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_smio"]
        ).create_smio_feature(
            smio_source,
            random_variables["random_smio_short_length"],
            random_variables["random_smio_long_length"],
            random_variables["random_smio_signal_length"],
            random_variables["random_smio_ma_method"],
        )

    if "DIDI" in random_variables["random_features"]:
        didi_source = (
            data_frame[random_variables["random_source_price_didi"]]
        )

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_didi"]
        ).create_didi_index_feature(
            source=didi_source,
            short_length=random_variables["random_didi_short_length"],
            medium_length=random_variables["random_didi_mid_length"],
            long_length=random_variables["random_didi_long_length"],
            ma_type=random_variables["random_didi_ma_type"],
            method=random_variables["random_didi_method"],
        )

    if "TSI" in random_variables["random_features"]:
        tsi_source = data_frame[random_variables["random_source_price_tsi"]]

        data_frame = ModelFeatures(
            data_frame, test_index, random_variables["random_binnings_qty_tsi"]
        ).create_tsi_feature(
            tsi_source,
            random_variables["random_tsi_short_length"],
            random_variables["random_tsi_long_length"],
            random_variables["random_tsi_ma_method"],
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
        **hyperparams,
    )

    mh2["Liquid_Result"] = np.where(mh2["Predict"] == -1, 0, mh2["Liquid_Result"])

    return max_trade_adj(mh2, off_days, max_trades, pct_adj), index_splits, all_y