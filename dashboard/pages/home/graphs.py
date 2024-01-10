from typing import Literal
import pandas as pd
import plotly.express as px
from machine_learning.utils import DataHandler

def resample_calculate_drawdown(
    series: pd.Series,
    period: str = "W",
) -> pd.DataFrame:
    """
    Calculate drawdown using resampling method.

    Parameters:
        series (pd.Series): The input time series data.
        period (str): The resampling period.
        (default: "W" (weekly))

    Returns:
        pd.DataFrame: The calculated drawdown.

    """
    result_df = series.to_frame()
    result_df = result_df + 1
    max_results = result_df.expanding(365).max()
    result_df = (max_results - result_df) / max_results
    return result_df.fillna(0).astype("float32").resample(period).mean()

def resample_calculate_payoff(
    series: pd.Series,
    method: Literal["sum", "mean"] = "sum",
    period: str = "W"
) -> pd.DataFrame:
    """
    Calculate payoff using resampling method.

    Parameters:
        series (pd.Series): The input time series data.
        method (Literal["sum", "mean"]): The method for calculating
        payoff.
        (default: "sum")
        period (str): The resampling period.
        (default: "W" (weekly))

    Returns:
        pd.DataFrame: The calculated payoff.

    """
    rt = series.to_frame().diff().fillna(0)
    rt_sum_pos = rt[rt > 0].resample(period).sum()
    rt_mean_pos = rt[rt > 0].resample(period).mean()
    rt_sum_neg = abs(rt[rt < 0]).resample(period).sum()
    rt_mean_neg = abs(rt[rt < 0]).resample(period).mean()
    rt = pd.DataFrame([])
    payoff = (
        rt_sum_pos / rt_sum_neg if method == "sum"
        else rt_mean_pos / rt_mean_neg
    )
    return payoff.dropna()

def resample_calculate_expected_return(
    series: pd.Series,
    period: str = "W"
) -> pd.DataFrame:
    """
    Calculate expected return using resampling method.

    Parameters:
        series (pd.Series): The input time series data.
        period (str): The resampling period.
        (default: "W" (weekly))

    Returns:
        pd.DataFrame: The calculated expected return.

    """
    rt = series.to_frame().diff().fillna(0)

    win_rate = rt[rt > 0].fillna(0)
    win_rate = win_rate.where(win_rate == 0, 1).astype("bool")
    win_rate = (
        win_rate.resample(period).sum()
        / win_rate.resample(period).count()
    )

    rt_mean_pos = rt[rt > 0].resample(period).mean()
    rt_mean_neg = abs(rt[rt < 0]).resample(period).mean()

    expected_return = rt_mean_pos * win_rate - rt_mean_neg * (1 - win_rate)
    expected_return = expected_return.astype("float32")

    return expected_return.dropna()

def resample_calculate_win_rate(
    series: pd.Series,
    period: str = "W"
) -> pd.DataFrame:
    """
    Calculate win rate using resampling method.

    Parameters:
        series (pd.Series): The input time series data.
        period (str): The resampling period.
        (default: "W" (weekly))

    Returns:
        pd.DataFrame: The calculated win rate.

    """
    rt = series.to_frame().diff().fillna(0)

    win_rate = rt[rt > 0].fillna(0)
    win_rate = win_rate.where(win_rate == 0, 1).astype("bool")
    win_rate = (
        win_rate.resample(period).sum()
        / win_rate.resample(period).count()
    )
    return win_rate

def rolling_calculate_drawdown(
    series: pd.Series,
    period: int = 30,
) -> pd.DataFrame:
    """
    Calculate drawdown using rolling method.

    Parameters:
        series (pd.Series): The input time series data.
        period (int): The rolling period.
        (default: 30)

    Returns:
        pd.DataFrame: The calculated drawdown.

    """
    result_df = series.to_frame()
    result_df = result_df + 1
    max_results = result_df.expanding(365).max()
    result_df = (max_results - result_df) / max_results
    return result_df.fillna(0).astype("float32").rolling(period).mean()

