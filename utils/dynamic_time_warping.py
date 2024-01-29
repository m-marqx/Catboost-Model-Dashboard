"""
Dynamic Time Warping module.

This module provides a class for computing Dynamic Time Warping (DTW)
between two input sequences.

Classes:
- DynamicTimeWarping: Class for computing DTW.
"""

from typing import Literal
import pandas as pd
import numpy as np
import fastdtw
from utils.exceptions import InvalidArgumentError


class DynamicTimeWarping:
    """Class for computing Dynamic Time Warping (DTW).

    This class provides methods to calculate the DTW distance and ratio
    between two input sequences.

    Parameters
    ----------
    input_x : numpy.ndarray or pandas.Series
        The first input sequence.
    input_y : numpy.ndarray or pandas.Series
        The second input sequence.

    Attributes
    ----------
    input_x : numpy.ndarray or pandas.Series
        The first input sequence.
    input_y : numpy.ndarray or pandas.Series
        The second input sequence.

    Methods
    -------
    get_dtw_df()
        Get the DTW values between the input sequences.

    calculate_dtw_distance()
        Calculate the DTW distance between the input sequences.

    """

    def __init__(
        self,
        input_x: np.ndarray | pd.Series,
        input_y: np.ndarray | pd.Series,
    ):
        """
        Initialize the DynamicTimeWarping class with the input
        sequences.

        Parameters
        ----------
        input_x : numpy.ndarray or pandas.Series
            The first input sequence.
        input_y : numpy.ndarray or pandas.Series
            The second input sequence.

        """
        self.input_x = input_x
        self.input_y = input_y
        _, self.path = fastdtw.fastdtw(input_x, input_y)
        self.dtw = pd.DataFrame(self.path)

        self.column_x = (
            input_x.name if isinstance(input_x, pd.Series)
            else "input_x"
        )

        self.column_y = (
            input_y.name if isinstance(input_y, pd.Series)
            else "input_y"
        )
