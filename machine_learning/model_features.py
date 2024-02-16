import pandas as pd
import tradingview_indicators as ta
from utils.dynamic_time_warping import DynamicTimeWarping

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


