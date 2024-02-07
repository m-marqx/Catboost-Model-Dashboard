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
    __init__(self, input_x, input_y)
        Initialize the DynamicTimeWarping class with the input sequences.

    dtw_df(self)
        Get the DTW dataframe between the input sequences.

    calculate_dtw_distance(self, method="ratio", align_sequences=False)
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

    @property
    def get_dtw_df(self) -> pd.DataFrame:
        """
        Get the DTW values between the input sequences.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the DTW values between the input
            sequences.
            The DataFrame has the following columns:
            - <column_x>_path: The path of the input_x sequence in
            the DTW alignment.
            - <column_y>_path: The path of the input_y sequence in
            the DTW alignment.
            - <column_x>: The values of the input_x sequence.
            - <column_y>: The values of the input_y sequence.
        """
        dtw_df1 = (
            self.input_x.reset_index(drop=True)
            .reindex(self.dtw[0])
            .rename_axis([self.column_x + "_path"])
            .reset_index()
        )

        dtw_df2 = (
            self.input_y.reset_index(drop=True)
            .reindex(self.dtw[1])
            .rename_axis([self.column_y + "_path"])
            .reset_index()
        )

        dtw_concat = pd.concat([dtw_df1, dtw_df2], axis=1)
        ordered_columns = [
            self.column_x + "_path",
            self.column_y + "_path",
            self.column_x,
            self.column_y,
        ]
        dtw_concat = dtw_concat[ordered_columns]
        return dtw_concat

    def calculate_dtw_distance(
        self,
        method:Literal['ratio', 'absolute'] = 'ratio',
    ) -> pd.Series:
        """
        Calculate the DTW distance between the input sequences.

        Parameters
        ----------
        method : str, optional
            The method to calculate the DTW distance.
            The options are:
            - 'ratio': The ratio between the input_x and input_y
            sequences.
            - 'absolute': The absolute difference between the input_x
            and input_y sequences.
            (default: 'ratio')

        Returns
        -------
        pd.Series
            A Series containing the DTW distance between the input
            sequences.
        """
        dtw_df = self.get_dtw_df

        match method:
            case 'ratio':
                return dtw_df[self.column_x] / dtw_df[self.column_y]
            case 'absolute':
                return dtw_df[self.column_x] - dtw_df[self.column_y]
            case _:
                raise InvalidArgumentError(
                    "method must be either 'ratio' or 'absolute'"
                )
