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

