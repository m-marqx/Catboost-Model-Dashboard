from typing import Literal
import pandas as pd
import plotly.express as px
from machine_learning.ml_utils import DataHandler

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

def rolling_calculate_payoff(
    series: pd.Series,
    method: Literal["sum", "mean"] = "sum",
    period: int = 30,
) -> pd.DataFrame:
    """
    Calculate payoff using rolling method.

    Parameters:
        series (pd.Series): The input time series data.
        method (Literal["sum", "mean"]): The method for calculating
        payoff.
        (default: "sum")
        period (int): The rolling period.
        (default: 30)

    Returns:
        pd.DataFrame: The calculated payoff.

    """
    rt = series.to_frame().diff().fillna(0)
    pos_values = rt[rt > 0]
    neg_values = abs(rt[rt < 0])

    if method == "sum":
        pos_values = pos_values.fillna(0).rolling(period).sum()
        neg_values = neg_values.fillna(0).rolling(period).sum()
    else:
        pos_values = pos_values.ffill().rolling(period).mean().fillna(0)
        neg_values = neg_values.ffill().rolling(period).mean().fillna(0)

    payoff = pos_values / neg_values
    return payoff.dropna()

def rolling_calculate_expected_return(
    series: pd.Series,
    period: int = 30,
) -> pd.DataFrame:
    """
    Calculate expected return using rolling method.

    Parameters:
        series (pd.Series): The input time series data.
        period (int): The rolling period.
        (default: 30)

    Returns:
        pd.DataFrame: The calculated expected return.

    """
    rt = series.to_frame().diff().fillna(0)

    pos_values = rt[rt > 0]
    neg_values = abs(rt[rt < 0])

    pos_count = pos_values.rolling(period).count()
    neg_count = neg_values.rolling(period).count()

    win_rate = pos_count / (pos_count + neg_count)

    pos_mean = pos_values.ffill().rolling(period).mean().fillna(0)
    neg_mean = neg_values.ffill().rolling(period).mean().fillna(0)

    expected_return = pos_mean * win_rate - neg_mean * (1 - win_rate)
    expected_return = expected_return.astype("float32")

    return expected_return.dropna()

def rolling_calculate_win_rate(
    series: pd.Series,
    period: int = 30,
) -> pd.DataFrame:
    """
    Calculate win rate using rolling method.

    Parameters:
        series (pd.Series): The input time series data.
        period (int): The rolling period.
        (default: 30)

    Returns:
        pd.DataFrame: The calculated win rate.

    """
    rt = series.to_frame().diff().fillna(0)

    pos_values = rt[rt > 0]
    neg_values = abs(rt[rt < 0])

    pos_count = pos_values.rolling(period).count()
    neg_count = neg_values.rolling(period).count()

    win_rate = pos_count / (pos_count + neg_count)
    return win_rate

def display_linechart(
    return_source: pd.DataFrame,
    validation_date: str | pd.Timestamp,
    stat: str,
    period: Literal["full", "test", "validation"] = "validation",
    time_period: int | str = 30,
    iqr_scales: None | list[float, float] = None,
    get_data: bool = False,
) -> px.line:
    """
    Display a line chart.

    Parameters:
        asset_source (pd.DataFrame): The asset source data.
        return_source (pd.DataFrame): The return source data.
        validation_date (str | pd.Timestamp): The validation date.
        stat (str): The statistic to display on the chart.
        period (Literal["full", "test", "validation"]): The period of
        data to display.
        (default: "validation")
        time_period (int | str): The time period for calculations.
        (default: 30)
        iqr_scales (list[float, float]): The IQR scales for outlier
        detection.
        (default: [1.0, 1.5])
        get_data (bool): Whether to return the calculated data.
        (default: False)

    Returns:
        px.line: The line chart.

    """
    time_period = int(time_period)
    method = "rolling"

    calculate_drawdown = (
        resample_calculate_drawdown
        if method == "resample"
        else rolling_calculate_drawdown
    )

    calculate_payoff = (
        resample_calculate_payoff
        if method == "resample"
        else rolling_calculate_payoff
    )

    calculate_expected_return = (
        resample_calculate_expected_return
        if method == "resample"
        else rolling_calculate_expected_return
    )

    calculate_win_rate = (
        resample_calculate_win_rate
        if method == "resample"
        else rolling_calculate_win_rate
    )


    match stat:
        case "drawdown":
            result_df = calculate_drawdown(return_source, time_period)
        case "expected_return":
            result_df = calculate_expected_return(return_source, time_period)
        case "payoff_sum":
            result_df = calculate_payoff(return_source, "sum", time_period)
        case "payoff_mean":
            result_df = calculate_payoff(return_source, "mean", time_period)
        case "winrate":
            result_df = calculate_win_rate(return_source, time_period)

    match period:
        case "test":
            result_df = result_df.loc[:validation_date]
        case "validation":
            result_df = result_df.loc[validation_date:]

    if get_data:
        return result_df

    if iqr_scales is None:
        iqr_scales = [1.0, 1.5]

    limits = {}

    column = result_df.columns[0]

    for scale in iqr_scales:
        limits[f"upper_limit_{scale}"], limits[f"lower_limit_{scale}"] = (
            DataHandler(result_df)
            .calculate_outlier_values(column, iqr_scale=scale)
        )
        if stat == "drawdown":
            limits[f"lower_limit_{scale}"] = 0

    fig = px.line(result_df, width=650, title=str(stat))

    line_params = dict(line_width=3, line_dash="dash", line_color="white")

    for idx, key in enumerate(limits):
        if idx < 2:
            fig.add_hline(limits[key])
        else:
            fig.add_hline(limits[key], **line_params)

    fig = (
        fig.add_vline(x=validation_date, line_dash="dash", line_color="yellow")
        .update_layout(title_x=0.5)
    )
    return fig

def add_outlier_lines(
    result_df: pd.DataFrame,
    fig,
    iqr_scales: None | list[float, float] = None,
    min_value=None,
    **line_kwargs
    ):
    limits = {}

    if iqr_scales is None:
        iqr_scales = [1.0, 1.5]

    column = result_df.columns[0]

    for scale in iqr_scales:
        limits[f"upper_limit_{scale}"], limits[f"lower_limit_{scale}"] = (
            DataHandler(result_df)
            .calculate_outlier_values(column, iqr_scale=scale)
        )
        if min_value or min_value == 0:
            limits[f"lower_limit_{scale}"] = (
                min_value
                if limits[f"lower_limit_{scale}"] <= min_value
                else limits[f"lower_limit_{scale}"]
            )

    if not line_kwargs:
        line_kwargs = dict(
            line_width=1,
            # line_dash="dashdot",
            line_color="#D3F2BB"
        )

    for idx, key in enumerate(limits):
        if idx < 2:
            fig.add_hline(limits[key], **line_kwargs)
        else:
            fig.add_hline(limits[key], **line_kwargs)

def calculate_sequencial_results(
    dataframe: pd.DataFrame,
    base_value: int = 0,
) -> pd.DataFrame:
    """
    Calculate the sequential count of profits or losses for a given dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the results.

    base_value : int, optional
        The base value used for calculating the results count.
        (default: 0)

    Returns
    -------
    pd.DataFrame
        The dataframe with the sequential count of profits or losses.
    """
    data_frame = dataframe.copy()

    liquid_results = (
        data_frame.query(f"Liquid_Result != {base_value}")
        ["Liquid_Result"] - base_value
    )

    sequencial_results = liquid_results.to_frame()

    loss_results_arrays = np.array([])
    gain_results_arrays = np.array([])

    loss_counter = 0
    gain_counter = 0

    for x in liquid_results.to_numpy():
        if x < 0:
            loss_counter += 1
            gain_counter = 0
        else:
            loss_counter = 0
            gain_counter += 1

        gain_results_arrays = np.append(gain_results_arrays, gain_counter)
        loss_results_arrays = np.append(loss_results_arrays, loss_counter)

    sequencial_results['loss_count'] = loss_results_arrays
    sequencial_results['gain_count'] = gain_results_arrays
    sequencial_results['sequential_count'] = (
        sequencial_results['gain_count'] - sequencial_results['loss_count']
    )
    data_frame['sequential_count'] = sequencial_results['sequential_count']
    return data_frame[['sequential_count']]
