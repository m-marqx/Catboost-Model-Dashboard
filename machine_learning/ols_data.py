from ast import literal_eval

import pandas as pd
import plotly.express as px


def calculate_r2(data: pd.Series) -> float:
    """
    Calculate the R-squared value from the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data used to calculate the R-squared value.

    Returns
    -------
    float
        The R-squared value.

    """
    return literal_eval(
        px.scatter(data, trendline="ols")
        .data[1]["hovertemplate"]
        .split(">=")[1]
        .split("<br>")[0]
    )


def calculate_coef(data: pd.Series) -> float:
    """
    Calculate the coefficient value from the given data.

    Parameters
    ----------
    data : pandas.Series
        The data used to calculate the coefficient value.

    Returns
    -------
    float
        The coefficient value.

    """
    return literal_eval(
        px.scatter(data, trendline="ols")
        .data[1]["hovertemplate"]
        .split("<br>")[1]
        .split("*")[0]
        .split(" ")[-2]
    )
