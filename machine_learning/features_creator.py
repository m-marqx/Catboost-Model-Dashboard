from itertools import combinations
from typing import Literal

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import catboost
import tradingview_indicators as ta
from utils.exceptions import InvalidArgumentError
from machine_learning.utils import (
    DataHandler,
    ModelHandler,
)
import pickle


from machine_learning.feature_params import FeaturesParamsComplete

from utils.math_features import MathFeature

class FeaturesCreator:
    """
    A class for creating features and evaluating machine learning
    models.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing financial data.
    target_series : pd.Series
        Series containing return values.
    source : pd.Series
        Series containing source data.
    validation_index : str | int, optional
        Index or column to split the data for training and development.
        (default: None)

    Attributes:
    -----------
    data_frame : pd.DataFrame
        Processed DataFrame using DataHandler.
    target_series : pd.Series
        Series containing return values.
    validation_index : int
        Index to split the data for training and development.
    train_development : pd.DataFrame
        Subset of the data for training and development.
    train_development_index : int
        Index used for splitting training and development data.
    source : pd.Series
        Series containing source data.
    split_params : dict
        Parameters for splitting the data.
    split_paramsH : dict
        Parameters for splitting the data with a higher threshold.
    split_paramsL : dict
        Parameters for splitting the data with a lower threshold.

    Methods:
    --------
    calculate_results(features_columns=None, model_params=None, fee=0.1) \
    -> pd.DataFrame:
        Calculate the results of the model pipeline.
    temp_indicator(value: int | list, \
    indicator: Literal['RSI', 'rolling_ratio'] = 'RSI') \
    -> pd.Series:
        Calculate a temporary indicator series.
    results_model_pipeline(value, indicator, model_params=None, \
    fee=0.1, train_end_index=None, results_column=None) -> dict:
        Calculate drawdown results for different variable combinations.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_series: pd.Series,
        source: pd.Series,
        feature_params: FeaturesParamsComplete,
        validation_index: str | int = None,
    ):
        """
        Initialize the FeaturesCreator instance.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input DataFrame containing financial data.
        target_series : pd.Series
            Series containing return values.
        source : pd.Series
            Series containing source data.
        validation_index : str | int, optional
            Index or column to split the data for training and
            development.
            (default: None)

        """
        self.data_frame = DataHandler(dataframe).calculate_targets()
        self.target_series = target_series
        self.validation_index = (
            validation_index
            or int(self.data_frame.shape[0] * 0.7)
        )
        self.temp_indicator_series = None

        self.train_development = (
            self.data_frame.loc[:self.validation_index]
            if isinstance(self.validation_index, str)
            else self.data_frame.iloc[:self.validation_index]
        )

        self.train_development_index = int(
            self.train_development.shape[0]
            * 0.5
        )

        self.source = source

        self.split_params = feature_params.split_features.dict()
        self.split_paramsH = feature_params.high_features.dict()
        self.split_paramsL = feature_params.low_features.dict()

