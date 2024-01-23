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

