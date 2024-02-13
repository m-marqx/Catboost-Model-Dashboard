import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Literal


class Statistics:
    """A class for calculating strategy statistics.

    Parameters
    ----------
    dataframe : pd.Series or pd.DataFrame
        The input dataframe containing the results of the strategy. If
        `dataframe` is a pd.Series, it should contain a single column
        of results. If it is a pd.DataFrame, it should have a 'Result'
        column containing the results.

    time_span : str, optional
        The time span for resampling the returns. The default is "A"
        (annual).

    risk_free_rate : float, optional
        The risk free rate of the strategy. The default is 0.

    is_percent : bool, optional
        Whether the results are in percentage form. If True, the calculated
        statistics will be multiplied by 100. Default is False.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the `Result` column.

    Methods
    -------
    calculate_all_statistics(precision: int = 2) -> pd.DataFrame
        Calculate all strategy statistics.

    calculate_expected_value() -> pd.DataFrame
        Calculate the expected value of the strategy.

    calculate_estimated_sharpe_ratio() -> pd.Series
        Calculate the Sharpe ratio of the strategy.

    calculate_estimated_sortino_ratio() -> pd.Series
        Calculate the Sortino ratio of the strategy.

    """
    def __init__(

        self,
        dataframe: pd.Series | pd.DataFrame,
        time_span: str = "A",
        risk_free_rate: float = 0,
        is_percent: bool  = False,
    ):
        """Calculates performance metrics based on the provided data.

        Parameters
        ----------
        dataframe : pandas.Series or pandas.DataFrame
            The input data. If a Series is provided, it is converted to
            a DataFrame with a "Result" column. If a DataFrame is
            provided, it should contain a "Result" column.
        time_span : str, optional
            The time span of the data. Defaults to "A" (annual).
        risk_free_rate : float, optional
            The risk-free rate to be used in performance calculations.
            Defaults to 0 (no risk-free rate).
        is_percent : bool, optional
            Indicates whether the data is in percentage format.
            Defaults to False.

        Raises
        ------
        ValueError
            If an invalid dataframe is provided.

        Notes
        -----
        The risk-free rate should be consistent with the timeframe used
        in the dataframe. If the timeframe is annual and the risk-free
        rate is 2%, the risk_free_rate value should be set as
        `0.00007936507`  (0.02 / 252) if the asset has 252 trading days.

        """
        self.is_percent = is_percent

        if isinstance(dataframe, pd.Series):
            self.dataframe = pd.DataFrame({"Result": dataframe})
        elif "Result" in dataframe.columns:
            self.dataframe = dataframe[["Result"]].copy()
        else:
            raise ValueError(
                """
                Invalid dataframe. The dataframe should be a
                pd.Series or a pd.DataFrame with a 'Result' column.
                """
            )

        self.dataframe["Result"] = (
            self.dataframe[["Result"]]
            .query("Result != 0")
            .dropna()
        )

        self.time_span = time_span
        self.risk_free_rate = risk_free_rate

    def calculate_all_statistics(self, precision: int = 2):
        """
        Calculate all strategy statistics.

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to round the calculated
            statistics to. Defaults to 2.

        Returns
        -------
        pd.DataFrame
            A dataframe with calculated statistics, including expected
            value, Sharpe ratio, and Sortino ratio.
        """
        stats_df = pd.DataFrame()
        stats_df["Expected_Value"] = self.calculate_expected_value()["Expected_Value"]
        stats_df = stats_df.resample(self.time_span).mean()

        stats_df["Sharpe_Ratio"] = self.calculate_estimed_sharpe_ratio()
        stats_df["Sortino_Ratio"] = self.calculate_estimed_sortino_ratio()

        if self.time_span == "A":
            stats_df["Date"] = stats_df.index.year
        if self.time_span == "M":
            stats_df["Date"] = stats_df.index.strftime('%m/%Y')
        if self.time_span in ["A", "M"]:
            stats_df = stats_df.reindex(columns=["Date"] + list(stats_df.columns[:-1]))
        return round(stats_df, precision)

    def calculate_expected_value(self):
        """
        Calculate the expected value of the strategy.

        Returns
        -------
        pd.DataFrame
            A dataframe with calculated statistics, including gain count,
            loss count, mean gain, mean loss, total gain, total loss,
            total trade, win rate, loss rate, and expected value (EM).

        """
        if self.is_percent:
            self.dataframe = self.dataframe * 100

        gain = self.dataframe["Result"] > 0
        loss = self.dataframe["Result"] < 0

        self.dataframe["Gain_Count"] = np.where(gain, 1, 0)
        self.dataframe["Loss_Count"] = np.where(loss, 1, 0)

        self.dataframe["Gain_Count"] = self.dataframe["Gain_Count"].cumsum()
        self.dataframe["Loss_Count"] = self.dataframe["Loss_Count"].cumsum()

        query_gains = self.dataframe.query("Result > 0")["Result"]
        query_loss = self.dataframe.query("Result < 0")["Result"]

        self.dataframe["Mean_Gain"] = query_gains.expanding().mean()
        self.dataframe["Mean_Loss"] = query_loss.expanding().mean()

        self.dataframe["Mean_Gain"].fillna(method="ffill", inplace=True)
        self.dataframe["Mean_Loss"].fillna(method="ffill", inplace=True)

        self.dataframe["Total_Gain"] = (
            np.where(gain, self.dataframe["Result"], 0)
            .cumsum()
        )

        self.dataframe["Total_Loss"] = (
            np.where(loss, self.dataframe["Result"], 0)
            .cumsum()
        )

        total_trade = self.dataframe["Gain_Count"] + self.dataframe["Loss_Count"]
        win_rate = self.dataframe["Gain_Count"] / total_trade
        loss_rate = self.dataframe["Loss_Count"] / total_trade

        self.dataframe["Total_Trade"] = total_trade
        self.dataframe["Win_Rate"] = win_rate
        self.dataframe["Loss_Rate"] = loss_rate

        ev_gain = self.dataframe["Mean_Gain"] * self.dataframe["Win_Rate"]
        ev_loss = self.dataframe["Mean_Loss"] * self.dataframe["Loss_Rate"]
        self.dataframe["Expected_Value"] = ev_gain - abs(ev_loss)

        return self.dataframe

    def calculate_estimed_sharpe_ratio(self) -> pd.Series:
        """
        Calculate the Sharpe ratio of the strategy.

        Returns
        -------
        pd.Series
            A series containing the Sharpe ratio values.

        """
        results = self.dataframe["Result"]
        returns_annualized = (
            results
            .resample(self.time_span)
        )

        mean_excess = returns_annualized.mean() - self.risk_free_rate

        sharpe_ratio = mean_excess / returns_annualized.std()

        return sharpe_ratio

    def calculate_estimed_sortino_ratio(self) -> pd.Series:
        """
        Calculate the Sortino ratio of the strategy.

        Returns
        -------
        pd.Series
            A series containing the Sortino ratio values.

        """
        results = self.dataframe["Result"]
        returns_annualized = (
            results
            .resample(self.time_span)
        )

        negative_results = self.dataframe.query("Result < 0")["Result"]
        negative_returns_annualized = (
            negative_results
            .resample(self.time_span)
        )

        mean_excess = returns_annualized.mean() - self.risk_free_rate

        sortino_ratio = mean_excess / negative_returns_annualized.std()

        return sortino_ratio

def feat_train_qcut(
    dataset: pd.DataFrame,
    test_index: str | int,
    feat_name: str,
    bins: int = 10
) -> pd.Series:
    """
    Compute the quantile-based discretization of a feature in the
    training set and apply it to the dataset.

    Parameters:
    -----------
    dataset : pd.DataFrame
        The dataset to apply the discretization to.
    test_index : str | int
        The index or label of the test set. If it's an integer, it
        represents the number of rows to include in the training set. If
        it's a string, it represents the label of the last row to
        include in the training set.
    feat_name : str
        The name of the feature to be discretized.
    bins : int, optional
        The number of bins to divide the feature into (default is 10).

    Returns:
    --------
    pd.Series
        The discretized feature values for the dataset.
    """
    train_set = (
        dataset.iloc[:test_index].copy() if isinstance(test_index, int)
        else dataset.loc[:test_index].copy()
    )

    intervals = (
        pd.qcut(train_set[feat_name], bins)
        .value_counts().index
        .to_list()
    )

    lows = pd.Series([interval.left for interval in intervals])
    highs = pd.Series([interval.right for interval in intervals])
    lows.iloc[0] = -999999999999
    highs.iloc[-1] = 999999999999

    intervals_range = (
        pd.concat([lows.rename("lowest"), highs.rename("highest")], axis=1)
        .sort_values("highest")
        .reset_index(drop=True)
    )

    return (
        dataset[feat_name].dropna().apply(lambda x: intervals_range[
            (x >= intervals_range["lowest"])
            & (x <= intervals_range["highest"])
        ].index[0])
    )

def model_metrics(y_pred: pd.Series, target: pd.Series) -> pd.DataFrame:
    """
    Calculate various model evaluation metrics based on predicted and
    target values.

    Parameters
    ----------
    y_pred : pd.Series
        The predicted values.

    target : pd.Series
        The target values.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated model evaluation metrics.
    """
    model_metric = y_pred.rename('y_pred').to_frame()

    model_metric['y_true'] = target

    model_metric['true_pos'] = (
        np.where(
            (model_metric['y_pred'] == 1) & (model_metric['y_true'] == 1),
            1, 0
        ).cumsum()
    )

    model_metric['true_neg'] = (
        np.where(
            (model_metric['y_pred'] == 0) & (model_metric['y_true'] == 0),
            1, 0
        ).cumsum()
    )

    model_metric['false_pos'] = (
        np.where(
            (model_metric['y_pred'] == 1) & (model_metric['y_true'] == 0),
            1, 0
        ).cumsum()
    )

    model_metric['false_neg'] = (
        np.where(
            (model_metric['y_pred'] == 0) & (model_metric['y_true'] == 1),
            1, 0
        ).cumsum()
    )

    model_metric['real_support_0'] = (
        np.where(model_metric['y_true'] == 0, 1, 0).cumsum()
    )

    model_metric['real_support_1'] = (
        np.where(model_metric['y_true'] == 1, 1, 0).cumsum()
    )

    model_metric['pred_support_0'] = (
        np.where(model_metric['y_pred'] == 0, 1, 0).cumsum()
    )

    model_metric['pred_support_1'] = (
        np.where(model_metric['y_pred'] == 1, 1, 0).cumsum()
    )

    model_metric['real_support_ratio'] = (
        (model_metric['real_support_1'] / model_metric['real_support_0']) - 1
    )

    model_metric['pred_support_ratio'] = (
        (model_metric['pred_support_1'] / model_metric['pred_support_0']) - 1
    )

    model_metric['recall_pos'] = (
        model_metric['true_pos']
        / (model_metric['true_pos'] + model_metric['false_neg'])
    )
    model_metric['recall_neg'] = (
        model_metric['true_neg']
        /  (model_metric['true_neg'] + model_metric['false_pos'])
    )

    confusion_matrix = (
        model_metric['true_pos']
        + model_metric['true_neg']
        + model_metric['false_pos']
        + model_metric['false_neg']
    )

    model_metric['accuracy'] = (
        (model_metric['true_pos'] + model_metric['true_neg'])
        / confusion_matrix
    )

    model_metric['recall_diff'] = (
        model_metric[['recall_pos', 'recall_neg']]
        .diff(-1, axis=1)['recall_pos']
    )
    model_metric['support_diff'] = (
        model_metric[['real_support_ratio', 'pred_support_ratio']]
        .diff(1, axis=1)['pred_support_ratio']
    )

    return model_metric

def calculate_returns(
    gross_results: pd.Series,
    filter_trades: pd.Series,
    fee: float,
    drawdown_min_window: int
) -> pd.DataFrame:
    """
    Calculates the returns and drawdown metrics for a trading strategy.

    Parameters
    ----------
    gross_results : pd.Series
        Series containing the gross trading results.
    filter_trades : pd.Series
        Series containing the filter trades.
    fee : float
        The trading fee applied to each trade.
    drawdown_min_window : int
        The minimum window size for calculating drawdown.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated returns and drawdown metrics.
    """
    df_returns = pd.concat([filter_trades.astype(int), gross_results], axis=1)
    df_returns.columns = ["Filter", "Result"]

    df_returns["Result"] = np.where(
        (df_returns["Filter"] != 0) & (df_returns["Result"].abs() != 1),
        df_returns["Result"] - fee, 0
    )

    df_returns["Liquid_Result"] = np.where(
        (df_returns["Filter"] != 0) & (df_returns["Result"].abs() != 1),
        df_returns["Result"] - fee, 0
    )

    df_returns["Total_Return"] = df_returns["Result"].cumsum() + 1
    df_returns["Liquid_Return"] = df_returns["Liquid_Result"].cumsum() + 1

    df_returns["max_Liquid_Return"] = (
        df_returns["Liquid_Return"].expanding(drawdown_min_window).max()
    )

    df_returns["max_Liquid_Return"] = np.where(
        df_returns["max_Liquid_Return"].diff(),
        np.nan, df_returns["max_Liquid_Return"],
    )

    df_returns["drawdown"] = (
        1 - df_returns["Liquid_Return"] / df_returns["max_Liquid_Return"]
    ).fillna(0)

    drawdown_positive = df_returns["drawdown"] > 0

    df_returns["drawdown_duration"] = drawdown_positive.groupby(
        (~drawdown_positive).cumsum()
    ).cumsum()
    return df_returns

def dict_to_classification_report(report_dict: dict) -> str:
    """
    Convert a dictionary to a classification report string.

    Parameters
    ----------
    report_dict : dict
        The input dictionary containing the classification report.

    Returns
    -------
    str
        The classification report string.

    """
    header = f"{'':<18} {'precision':<10} {'recall':<10} {'f1-score':<10}"
    header += f" {'support':<10}\n"
    header += "-" * 60 + "\n"
    rows = ""
    for key, value in report_dict.items():
        if isinstance(value, dict):
            rows += f"{key:<18} {value['precision']:<10.4f}"
            rows += f" {value['recall']:<10.4f} {value['f1-score']:<10.4f}"
            rows += f" {value['support']:<10.0f}\n"
        else:
            rows += f"\n{key:<40} {value:<10.4f} {'':<10} {'':<10} {'':<10}\n"
    return header + rows

def dataset_classification_report(
    predict_all: pd.Series | pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    y_val: pd.Series | pd.DataFrame | None = None,
    ignore_value: int = 0,
    output: Literal["string", "print"] = "print",
    ) -> str | None:
    """
    Calculate the classification report for a dataset.

    Parameters
    ----------
    predict_all : pd.Series or pd.DataFrame
        The predicted values for the entire dataset.
    y_train : pd.Series or pd.DataFrame
        The true values for the training set.
    y_test : pd.Series or pd.DataFrame
        The true values for the test set.
    y_val : pd.Series or pd.DataFrame
        The true values for the validation set.
    ignore_value : int, optional
        The value to ignore in the classification report.
        (default : 0)
    output : {"string", "print"}, optional
        The output format of the classification report.
        (default :"print")

    Returns
    -------
    str or None
        The classification report string if output is "string", None
        otherwise.
    """
    data_frame = (
        pd.DataFrame(predict_all, columns=['Predict'])
        .query(f"Predict != {ignore_value}")
    )

    y_pred_values_train = (
        data_frame.where(data_frame["Predict"] == 1, 0)
        .loc[:y_train.index[-1]]["Predict"]
    )

    y_pred_values_test = (
        data_frame.where(data_frame["Predict"] == 1, 0)
        .loc[y_test.index[0]:y_test.index[-1]]["Predict"]
    )

    y_train_reindexed = y_train.reindex(y_pred_values_train.index)
    y_test_reindexed = y_test.reindex(y_pred_values_test.index)
    train_args = (y_train_reindexed, y_pred_values_train)
    test_args = (y_test_reindexed, y_pred_values_test)

    string = f"{' Train ':-^55}\n"
    string += (metrics.classification_report(*train_args))
    string += f"\n{' Test ':-^55}\n"
    string += (metrics.classification_report(*test_args))

    if y_val is not None:
        y_pred_values_val = (
            data_frame.where(data_frame["Predict"] == 1, 0)
            .loc[y_val.index[0]:]["Predict"]
        )

        y_val_reindexed = y_val.reindex(y_pred_values_val.index)
        val_args = (y_val_reindexed, y_pred_values_val)

        string += f"\n{' Validation ':-^55}\n"
        string += (metrics.classification_report(*val_args))

    if output == "string":
        return string

    print(string)

def model_reports(
    y_pred_all: pd.Series,
    target_series: pd.Series,
    train_set: pd.DataFrame | pd.Series,
    test_set: pd.DataFrame | pd.Series,
    validation_set: pd.DataFrame | pd.Series | None = None,
    metric: Literal[
        'report',
        'all_1',
        'all_0',
        'difference',
    ] = 'report',
) -> pd.DataFrame:
    """
    Generate model reports based on the specified metric.

    Parameters
    ----------
    y_pred_all : pd.Series
        Predicted target values for all samples.
    target_series : pd.Series
        True target values for all samples.
    train_set : pd.DataFrame or pd.Series
        The training set (X_train and y_train).
    test_set : pd.DataFrame or pd.Series
        The test set (X_test and y_test).
    validation_set : pd.DataFrame or pd.Series or None
        The validation set (X_val and y_val).
    metric : {'report', 'all_1', 'all_0', 'difference'}, optional
        The metric to use for generating reports.
        (default :'report')

    Returns
    -------
    pd.DataFrame
        The generated model reports.
    """
    datasets_length = len(train_set) + len(test_set) + len(validation_set)
    metric_list = [
        'report',
        'difference',
        'all_1',
        'all_0',
    ]

    all_metrics = ', '.join(metric_list)

    if isinstance(y_pred_all, pd.DataFrame):
        raise TypeError("y_pred_all should be a pd.Series")
    if isinstance(target_series, pd.DataFrame):
        raise TypeError("target_series should be a pd.Series")
    if metric not in metric_list:
        raise ValueError(f"metric should be one of {all_metrics}")
    if y_pred_all.shape[0] != target_series.shape[0]:
        raise ValueError(
            "y_pred_all and target_series should have the same length"
        )
    if y_pred_all.shape[0] != datasets_length:
        raise ValueError(
            "The sum of train_set, test_set, and validation_set should be "
            + "equal to y_pred_all")

    train_str = " Train "
    test_str = " Test "
    validation_str = " Validation "
    real_str = " Real "
    pred_str = " Predict "
    diff_str = " difference "

    if metric == 'report':
        print(f"{train_str:-^55}")
        print(
            metrics.classification_report(
                target_series.reindex(train_set.index),
                y_pred_all.reindex(train_set.index)
            )
        )
        print(f"{test_str:-^55}\n")
        print(
            metrics.classification_report(
                target_series.reindex(test_set.index),
                y_pred_all.reindex(test_set.index),
            )
        )
        if validation_set:
            print(f"{validation_str:-^55}\n")
            print(
                metrics.classification_report(
                    target_series.reindex(validation_set.index),
                    y_pred_all.reindex(validation_set.index),
                )
            )

    elif metric == "all_1":
        print(f"\n{train_str:█^55}")
        print(
            metrics.classification_report(
                target_series.reindex(train_set.index),
                np.ones(y_pred_all.reindex(train_set.index).shape),
                zero_division=0,
                digits=4,
            )
        )
        print(f"\n{test_str:█^55}")
        print(
            metrics.classification_report(
                target_series.reindex(test_set.index),
                np.ones(y_pred_all.reindex(test_set.index).shape),
                zero_division=0,
                digits=4,
            )
        )
        if validation_set is not None:
            print(f"\n{validation_str:█^55}")
            print(
                metrics.classification_report(
                    target_series.reindex(validation_set.index),
                    np.ones(y_pred_all.reindex(validation_set.index).shape),
                    zero_division=0,
                    digits=4,
                )
            )

    elif metric == "all_0":
        print(f"\n{train_str:█^55}")
        print(
            metrics.classification_report(
                target_series.reindex(train_set.index),
                np.zeros(y_pred_all.reindex(train_set.index).shape),
                zero_division=0,
                digits=4,
            )
        )
        print(f"\n{test_str:█^55}")
        print(
            metrics.classification_report(
                target_series.reindex(test_set.index),
                np.zeros(y_pred_all.reindex(test_set.index).shape),
                zero_division=0,
                digits=4,
            )
        )
        if validation_set is not None:
            print(f"\n{validation_str:█^55}")
            print(
                metrics.classification_report(
                    target_series.reindex(validation_set.index),
                    np.zeros(y_pred_all.reindex(validation_set.index).shape),
                    zero_division=0,
                    digits=4,
                )
            )

    elif metric == "difference":
        y_train_true = (
            target_series
            .reindex(train_set.index)
            .value_counts()
        )

        y_train_pred = (
            y_pred_all
            .reindex(train_set.index)
            .value_counts()
        )

        print(f"{train_str:-^55}")
        print(f"{real_str:-^35}")
        print(y_train_true.sort_index())
        print(f"\n{pred_str:-^35}")
        print(y_train_pred.sort_index())
        print(f"\n{diff_str:-^35}")
        print(y_train_true - y_train_pred)

        y_test_true = (
            target_series
            .reindex(test_set.index)
            .value_counts()
        )

        y_test_pred = y_pred_all.reindex(test_set.index).value_counts()

        print(f"\n{test_str:-^55}")
        print(f"{real_str:-^35}")
        print(y_test_true.sort_index())
        print(f"\n{pred_str:-^35}")
        print(y_test_pred.sort_index())
        print(f"\n{diff_str:-^35}")
        print(y_test_true - y_test_pred)

        if validation_set:
            y_validation_true = (
                target_series
                .reindex(validation_set.index)
                .value_counts()
            )

            y_validation_pred = (
                y_pred_all
                .reindex(validation_set.index)
                .value_counts()
            )

            print(f"\n{validation_str:-^55}")
            print(f"{real_str:-^35}")
            print(y_validation_true.sort_index())
            print(f"\n{pred_str:-^35}")
            print(y_validation_pred.sort_index())
            print(f"\n{diff_str:-^35}")
            print(y_validation_true - y_validation_pred)
