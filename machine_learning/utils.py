from typing import Literal
import re

import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.model_selection import (
    learning_curve,
    train_test_split,
)

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go


class DataHandler:
    """
    Class for handling data preprocessing tasks.

    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        The input DataFrame or Series to be processed.

    Attributes
    ----------
    data_frame : pd.DataFrame
        The processed DataFrame.

    Methods
    -------
    get_datasets(feature_columns, test_size=0.5, split_size=0.7)
        Splits the data into development and validation datasets.
    drop_zero_predictions(column)
        Drops rows where the specified column has all zero values.
    get_splits(target, features)
        Splits the DataFrame into training and testing sets.
    get_best_results(target_column)
        Gets the rows with the best accuracy for each unique value in
        the target column.
    result_metrics(result_column=None, is_percentage_data=False,
    output_format="DataFrame")
        Calculates result-related statistics like expected return and win rate.
    fill_outlier(column=None, iqr_scale=1.5, upper_quantile=0.75,
    down_quantile=0.25)
        Removes outliers from a specified column using the IQR method.
    quantile_split(target_input, column=None, method="ratio",
    quantiles=None, log_values=False)
        Splits data into quantiles and analyzes the relationship with a target.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | pd.Series | np.ndarray,
    ) -> None:
        """
        Initialize the DataHandler object.

        Parameters:
        -----------
        dataframe : pd.DataFrame, pd.Series, or np.ndarray
            The input data to be processed. It can be a pandas DataFrame,
            Series, or a numpy array.

        """
        self.data_frame = dataframe.copy()

        if isinstance(dataframe, np.ndarray):
            self.data_frame = pd.Series(dataframe)

    def get_datasets(
        self,
        feature_columns: list,
        test_size: float = 0.5,
        split_size: float = 0.7
    ) -> dict[dict[pd.DataFrame, pd.Series]]:
        """
        Splits the data into development and validation datasets.

        Separates the DataFrame into training and testing sets for
        development, and a separate validation set, based on the
        specified split and test sizes.

        Parameters
        ----------
        feature_columns : list
            List of column names to be used as features.
        test_size : float
            Proportion of the dataset to include in the test split.
            (default: 0.5)
        split_size : float
            Proportion of the dataset to include in the development
            split.
            (default: 0.7)

        Returns
        -------
        dict
            A dictionary containing the development and validation
            datasets, each itself a dictionary with DataFrames and
            Series for features and target values respectively.

        Raises
        ------
        ValueError
            If the provided data_frame is not a Pandas DataFrame.
        """
        if not isinstance(self.data_frame, pd.DataFrame):
            raise ValueError("The dataframe must be a Pandas DataFrame")

        split_index = int(self.data_frame.shape[0] * split_size)
        development_df = self.data_frame.iloc[:split_index].copy()
        validation_df = self.data_frame.iloc[split_index:].copy()

        features = development_df[feature_columns]
        target = development_df["Target_1_bin"]

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            shuffle=False
        )

        origin_datasets = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }
        validation_dataset = {
            "X_validation": validation_df[feature_columns],
            "y_validation": validation_df["Target_1_bin"]
        }

        return {
            "development": origin_datasets,
            "validation": validation_dataset
        }

    def calculate_targets(self, length=1) -> pd.DataFrame:
        """
        Calculate target variables for binary classification.

        Adds target variables to the DataFrame based on the 'close'
        column:
        - 'Return': Percentage change in 'close' from the previous day.
        - 'Target_1': Shifted 'Return', representing the future day's
        return.
        - 'Target_1_bin': Binary classification of 'Target_1':
            - 1 if 'Target_1' > 1 (positive return)
            - 0 otherwise.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added target variables.

        """
        if isinstance(self.data_frame, pd.Series):
            self.data_frame = pd.DataFrame(self.data_frame)

        self.data_frame['Return'] = self.data_frame["close"].pct_change(length) + 1
        self.data_frame["Target_1"] = self.data_frame["Return"].shift(-length)
        self.data_frame["Target_1_bin"] = np.where(
            self.data_frame["Target_1"] > 1,
            1, 0)

        self.data_frame["Target_1_bin"] = np.where(
            self.data_frame['Target_1'].isna(),
            np.nan, self.data_frame['Target_1_bin']
        )
        return self.data_frame

    def model_pipeline(
        self,
        features_columns: list,
        target_column: str,
        estimator: object,
        return_series: pd.Series,
        split_location: float | int | str = 0.3,
    ) -> pd.DataFrame:
        """
        Execute a machine learning pipeline for model evaluation.

        This method performs a machine learning pipeline, including
        data splitting, training, validation, and evaluation.

        Parameters:
        -----------
        features_columns : list
            List of column names representing features used for training
            the model.
        target_column : str
            Name of the target variable column.
        estimator : object
            Machine learning model (estimator) to be trained and
            evaluated.
        return_series : pd.Series
            Series containing the target variable for the model.
        split_location : float, int, or str, optional
            Determines the location to split the dataset into training
            and validation sets.
            - Float: it represents the proportion of
            the dataset to include in the validation split.
            - Integer: it specifies the index to split the
            dataset.
            - String: it represents the label/index to split
            the dataset.

            (default: 0.3)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing model returns and validation date.

        Raises:
        -------
        ValueError
            If validation_size is outside the valid range (0.0 to 1.0).
        """
        if not isinstance(split_location, (str, float, int)):
            raise ValueError(
                "Wrong split_location type: "
                f"{split_location.__class__.__name__}"
            )
        is_percentage_location = 0 < split_location < 1

        if not (isinstance(split_location, float) and is_percentage_location):
            raise ValueError(
                "When split_location is a float, "
                "it should be between 0.0 and 1.0"
            )

        if is_percentage_location:
            split_factor = 1 - split_location
            split_index = int(self.data_frame.shape[0] * split_factor)
        else:
            split_index = split_location

        if isinstance(split_index, int):
            development = self.data_frame.iloc[:split_index].copy()
            validation = self.data_frame.iloc[split_index:].copy()
        elif isinstance(split_index, str):
            development = self.data_frame.loc[:split_index].copy()
            validation = self.data_frame.loc[split_index:].copy()

        features = development[features_columns]
        target = development[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.5,
            shuffle=False
        )

        estimator.fit(X_train, y_train)

        validation_x_test = validation[features_columns]
        validation_y_test = validation[target_column]

        x_series = pd.concat([X_test, validation_x_test], axis=0)
        y_series = pd.concat([y_test, validation_y_test], axis=0)

        model_returns = (
            ModelHandler(estimator, x_series, y_series)
            .model_returns(return_series)
        )
        model_returns['validation_date'] = str(validation.index[0])
        return model_returns

    def drop_zero_predictions(
        self,
        column: str,
    ) -> pd.Series:
        """
        Drop rows where the specified column has all zero values.

        Parameters:
        -----------
        column : str
            The column name in the DataFrame to check for zero values.

        Returns:
        --------
        pd.Series
            The Series with rows dropped where the specified column
            has all zero values.
        """

        def _is_all_zero(list_values: list) -> bool:
            return all(value == 0 for value in list_values)

        if column not in self.data_frame.columns:
            raise ValueError(
                f"Column '{column}' does not exist in the DataFrame."
            )

        mask = self.data_frame[column].apply(_is_all_zero)

        return self.data_frame[~mask]

