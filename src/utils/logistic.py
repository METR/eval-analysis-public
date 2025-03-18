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
    assert np.all((y >= 0) & (y <= 1)), "y values must be in [0,1]"
    original_weight_sum = np.sum(sample_weight)
    original_average = np.average(y, weights=sample_weight)

    # For any fractional y values, split into weighted 0s and 1s
    fractional_mask = (y > 0) & (y < 1)
    if np.any(fractional_mask):
        X_frac = X[fractional_mask]
        y_frac = y[fractional_mask]
        w_frac = sample_weight[fractional_mask]

        # Stack X twice for 0s and 1s
        X_split = np.vstack([X_frac, X_frac])

        # Create y array with 0s in first half, 1s in second half
        y_split = np.zeros(2 * len(y_frac))
        y_split[len(y_frac) :] = 1

        # Weight the 0s by (1-y) and 1s by y
        w_split = np.concatenate([(1 - y_frac) * w_frac, y_frac * w_frac])

        # Combine with non-fractional values
        X = np.vstack([X[~fractional_mask], X_split])
        y = np.concatenate([y[~fractional_mask], y_split])
        sample_weight = np.concatenate([sample_weight[~fractional_mask], w_split])
        assert np.allclose(np.sum(sample_weight), original_weight_sum)
        assert np.allclose(np.average(y, weights=sample_weight), original_average)

    model = LogisticRegression(C=1 / regularization)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def get_x_for_quantile(model: LogisticRegression, quantile: float) -> float:
    return (np.log(quantile / (1 - quantile)) - model.intercept_[0]) / model.coef_[0][0]  # type: ignore
