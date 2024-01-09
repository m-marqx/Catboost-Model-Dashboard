import numpy as np
import pandas as pd


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

