from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


def logistic_regression(
    X: NDArray[Any],
    y: NDArray[Any],
    sample_weight: NDArray[Any],
    regularization: float,
) -> LogisticRegression:
    # Assert y values are in [0,1]
    assert np.all((y == 0) | (y == 1)), "y values must be 0 or 1"

    model = LogisticRegression(C=1 / regularization)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def get_x_for_quantile(model: LogisticRegression, quantile: float) -> float:
    return (np.log(quantile / (1 - quantile)) - model.intercept_[0]) / model.coef_[0][0]  # type: ignore
