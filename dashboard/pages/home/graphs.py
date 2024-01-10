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

