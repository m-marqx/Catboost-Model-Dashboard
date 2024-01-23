from typing import Literal
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn import metrics
from sklearn.model_selection import (
    learning_curve,
)

class ModelHandler:
    """
    A class for handling machine learning model evaluation.

    Parameters:
    -----------
    estimator : object
        The machine learning model to be evaluated.
    X_test : array-like of shape (n_samples, n_features)
        Testing input samples.
    y_test : array-like of shape (n_samples,)
        True target values for testing.

    Attributes:
    -----------
    x_test : array-like of shape (n_samples, n_features)
        Testing input samples.
    y_test : array-like of shape (n_samples,)
        True target values for testing.
    estimator : object
        The machine learning model.
    y_pred_probs : array-like of shape (n_samples,), optional
        Predicted class probabilities (if available).
    _has_predic_proba : bool
        Indicates whether the estimator has predict_proba method.

    Properties:
    -----------
    results_report : str
        A string containing a results report including a confusion matrix,
        a classification report, AUC, Gini index (if predict_proba is
        available), and support.
    """

    def __init__(self, estimator, X_test, y_test) -> None:
        """
        Initialize the ModelHandler object.

        Parameters:
        -----------
        estimator : object
            An instance of a scikit-learn estimator for classification or
            regression.
        X_test : array-like of shape (n_samples, n_features)
            Test input samples.
        y_test : array-like of shape (n_samples,)
            True target values for testing.
        """
        self.x_test = X_test
        self.y_test = y_test
        self.estimator = estimator
        self.y_pred_probs = None
        self.y_pred = estimator.predict(X_test)
        self._has_predic_proba = (
            hasattr(estimator, 'predict_proba')
            and callable(getattr(estimator, 'predict_proba'))
        )

        if self._has_predic_proba:
            self.y_pred_probs = estimator.predict_proba(X_test)[:, 1]

    def model_returns(
        self,
        target_series: pd.Series,
        fee: float = 0.1,
        cutoff: float = 0.5,
        step: float = 0.0,
        long_only: bool = False,
        short_only: bool = False,
    ) -> tuple[pd.DataFrame, str]:
        """
        Calculate returns and performance metrics for a trading model.

        This method calculates returns and various performance metrics
        for a trading model using predicted probabilities and actual
        returns. It takes into account transaction fees for trading.

        Parameters:
        -----------
        target_series : pd.Series
            A pandas Series containing the actual returns of the trading
            strategy.
        fee : float, optional
            The transaction fee as a percentage (e.g., 0.1% for 0.1)
            for each trade.
            (default: 0.1)

        Returns:
        --------
        tuple[pd.DataFrame, str]
            A tuple containing:
            - pd.DataFrame: A DataFrame with various columns
            representing the trading results
            - str: A message indicating the success of the operation

        Raises:
        -------
        ValueError:
            If the estimator isn't suitable for classification
            (predict_proba isn't available).
        """
        if not self._has_predic_proba:
            raise ValueError(
                "The estimator isn't suitable for classification"
                " (predict_proba isn't available)."
            )

        if target_series.min() > 0:
            target_series = target_series - 1

        fee = fee / 100
        df_returns = (
            pd.DataFrame(
                {'y_pred_probs' : self.y_pred_probs},
                index=self.x_test.index
            )
        )

        target_return = target_series.reindex(df_returns.index)

        df_returns["target_Return"] = target_return

        if long_only:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] > cutoff + step), 1, 0
            )
        elif short_only:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] < cutoff - step), -1, 0
            )
        elif step > 0 and step is not None:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] > cutoff + step), 1, 0
            )

            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] < cutoff - step),
                -1, df_returns["Predict"]
            )
        else:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] > cutoff), 1, -1
            )

        df_returns["Position"] = df_returns["Predict"].shift().fillna(0)

        df_returns["Result"] = (
            df_returns["target_Return"]
            * df_returns["Predict"]
        )

        df_returns["Liquid_Result"] = np.where(
            (df_returns["Predict"] != 0)
            & (df_returns["Result"].abs() != 1),
            df_returns["Result"] - fee, 0
        )

        df_returns["Period_Return_cum"] = (
            df_returns["target_Return"]
        ).cumsum()

        df_returns["Total_Return"] = df_returns["Result"].cumsum() + 1
        df_returns["Liquid_Return"] = df_returns["Liquid_Result"].cumsum() + 1

        df_returns["max_Liquid_Return"] = (
            df_returns["Liquid_Return"].expanding(52).max()
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

    def roc_curve(
        self,
        output: Literal["DataFrame", "Figure"] = "Figure",
    ):
        """
        Plot a Receiver Operating Characteristic (ROC) curve.

        The ROC curve is a graphical representation of the classifier's
        ability to distinguish between positive and negative classes.
        It is created by plotting the True Positive Rate (TPR) against
        the False Positive Rate (FPR) at various threshold settings.

        Parameters:
        -----------
        fpr : str, np.ndarray, or pd.Series
            An array containing the False Positive Rates for different
            classification thresholds.
        tpr : str, np.ndarray, or pd.Series
            An array containing the True Positive Rates for different
            classification thresholds.

        Returns:
        --------
        plotly.graph_objs._figure.Figure
            A Plotly figure displaying the ROC curve with AUC
            (Area Under the Curve) score.

        """
        if output not in ["DataFrame", "Figure"]:
            raise ValueError("output must be 'DataFrame' or 'Figure'")

        fpr, tpr, thresholds = (
            metrics.roc_curve(self.y_test, self.y_pred_probs)
        )

        roc_curve = pd.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            }
        )

        if output == "Figure":
            roc_auc = metrics.auc(fpr, tpr)

            fig = px.line(
                roc_curve,
                x=fpr,
                y=tpr,
                title=f"ROC Curve (AUC={roc_auc:.4f})",
                labels=dict(x="False Positive Rate", y="True Positive Rate"),
                width=700,
                height=700,
            )

            fig.add_shape(
                type="line",
                line=dict(dash="dash"),
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                opacity=0.65,
            )
            return fig

        return roc_curve

    def learning_curve(
        self,
        train_size: np.ndarray | pd.Series = None,
        k_fold: int = 5
    ) -> pd.DataFrame:
        """
        Generate a learning curve for the estimator.

        A learning curve shows the training and testing scores of an
        estimator
        for varying numbers of training samples. This can be useful to
        evaluate
        how well the estimator performs as more data is used for
        training.

        Parameters:
        -----------
        train_size : np.ndarray or pd.Series, optional
            An array of training sizes or a Series of proportions to
            use for plotting the learning curve. If None, it defaults
            to evenly spaced values between 0.1 and 1.0
            (default: None).
        k_fold : int, optional
            The number of cross-validation folds to use for computing
            scores
            (default: 5).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the mean and standard deviation of
            train and test scores for different training sizes. Columns
            include:
            - 'train_mean': Mean training score
            - 'train_std': Standard deviation of training score
            - 'test_mean': Mean test score
            - 'test_std': Standard deviation of test score

        Notes:
        ------
        The learning curve is generated using cross-validation to
        compute scores.
        """
        if train_size is None:
            train_size = np.linspace(0.1, 1.0, 20)

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.estimator,
            X=self.x_test,
            y=self.y_test,
            train_sizes=train_size,
            cv=k_fold,
        )

        train_scores_df = pd.DataFrame(train_scores, index=train_sizes)
        train_scores_concat = pd.concat(
            [
                train_scores_df.mean(axis=1),
                train_scores_df.std(axis=1)
            ], axis=1).rename(columns={0: 'train_mean', 1: 'train_std'})

        test_scores_df = pd.DataFrame(test_scores, index=train_sizes)

        test_scores_concat = pd.concat(
            [
                test_scores_df.mean(axis=1),
                test_scores_df.std(axis=1)
            ], axis=1).rename(columns={0: 'test_mean', 1: 'test_std'})

        return pd.concat(
            [
                train_scores_concat,
                test_scores_concat,
            ], axis=1)

    @property
    def results_report(self) -> str:
        """
        Generate a results report including a confusion matrix and a
        classification report.

        Returns:
        --------
        str
            A string containing the results report.
        """
        if not self._has_predic_proba:
            raise ValueError(
                "The estimator isn't suitable for classification"
                " (predict_proba isn't available)."
            )

        names = pd.Series(self.y_test).sort_values().astype(str).unique()

        confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        column_names = "predicted_" + names
        index_names = "real_" + names

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix,
            columns=column_names,
            index=index_names,
        )

        auc = metrics.roc_auc_score(self.y_test, self.y_pred_probs)
        gini = 2 * auc - 1
        support = self.y_test.shape[0]
        classification_report = metrics.classification_report(
            self.y_test, self.y_pred, digits=4
        )[:-1]

        auc_str = (
            f"\n         AUC                         {auc:.4f}"
            f"      {support}"
            f"\n        Gini                         {gini:.4f}"
            f"      {support}"
        )

        confusion_matrix_str = (
            f"Confusion matrix"
            f"\n--------------------------------------------------------------"
            f"\n{confusion_matrix_df}"
            f"\n"
            f"\n"
            f"\nClassification reports"
            f"\n--------------------------------------------------------------"
            f"\n"
            f"\n{classification_report}"
            f"{auc_str}"
        )
        return confusion_matrix_str
