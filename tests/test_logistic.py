from typing import Any

import numpy as np
import pytest

from src.utils.logistic import logistic_regression


def synthetic_data(
    scale_true: float, coef_true: float, b_true: float, num_samples: int = 10000
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    np.random.seed(42)
    X = np.random.randn(num_samples, 1) * 5

    coef_array = np.array([coef_true])
    scale_array = np.array(scale_true)
    b_array = np.array(b_true)

    logits = X @ coef_array + b_array
    p = 1 / (1 + np.exp(-logits))  # sigmoid
    p_scaled = scale_array * p
    # Bernoulli draws
    y = np.random.binomial(n=1, p=p_scaled)  # shape (num_samples,)
    return X, y


def test_logistic_scaled_bernoulli() -> None:
    # Generate data
    X, y = synthetic_data(1.0, -3.0, 0.0)
    model = logistic_regression(X, y, sample_weight=np.ones_like(y), regularization=0.1)

    # Use pytest's built-in assert or approx
    assert model.coef_[0][0] == pytest.approx(-3.0, abs=0.1)
    assert model.intercept_[0] == pytest.approx(0.0, abs=0.1)  # type: ignore
