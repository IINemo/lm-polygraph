"""
This module contains the CenteredIsotonicRegression class.
Copied with minor modifications from
https://github.com/mathijs02/cir-model/blob/main/src/cir_model/cir_model.py
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d


class CenteredIsotonicRegression(IsotonicRegression):
    """
    Centered Isotonic Regression (CIR) model. CIR is described in [1]_ and is
    similar to Isotonic Regression (IR). CIR takes as an additional constraint,
    compared to IR, that the resulting function needs to be strictly monotonic:
    ranges of constant function values are prevented as much as possible.
    The `CenteredIsotonicRegression` class inherits all methods and attributes
    from the `scikit-learn` implementation `IsotonicRegression` and it is
    therefore compatible with the other components of the `scikit-learn`
    library, like for example pipelines.

    Parameters
    ----------
    This class takes the same parameters and has the same attributes as
    `IsotonicRegression` from `scikit-learn`.[2]_ For full documentation of
    `IsotonicRegression`, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html

    CenteredIsotonicRegression takes one additional parameter:

    non_centered_points : list, default: [0, 1]
        A list of y values that should not be collapsed in the CIR algorithm.
        In the original CIR algorithm, y values of 0 and 1 are treated
        differently by not collapsing them. This is because CIR is typically
        used for a binary target variable. The default behaviour can be
        overruled by passing a list of values for `non_centered_points`. An
        empty list means that no points are treated differently.

    References
    ----------
    .. [1] Centered Isotonic Regression: Point and Interval Estimation for
           Dose-Response Studies, Assaf P. Oron & Nancy Flournoy, Statistics in
           Biopharmaceutical Research, Volume 9, Issue 3, 258-267, 2017

    .. [2] Scikit-learn: Machine Learning in Python, Fabian Pedregosa et al.,
           JMLR 12, 2825-2830, 2011

    Examples
    --------
    >>> from cir_model import CenteredIsotonicRegression
    >>> x = [1, 2, 3, 4]
    >>> y = [1, 21, 41, 34]
    >>> model = CenteredIsotonicRegression().fit(x, y)
    >>> model.transform(x)
    array([ 1. , 21. , 32. , 37.5])
    """

    def __init__(
        self,
        non_centered_points: List[Union[float, int]] = [0, 1],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.non_centered_points = non_centered_points

    def fit(
        self,
        X: Union[np.ndarray, List],
        y: Union[np.ndarray, List],
        sample_weight: Optional[Union[np.ndarray, List]] = None,
    ) -> "CenteredIsotonicRegression":
        """
        Fit the model using X, y and optionally sample_weight as training data.
        This method takes the same parameters and returns the same objects as
        `fit` from `IsotonicRegression`. For full documentation of
        `IsotonicRegression`, see:
        https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression.fit
        """

        super().fit(X, y, sample_weight)
        x_new, y_new = self._build_cir_points(X, sample_weight)

        # Interpolate between the new points and set clipping values for
        # interpolation outside the range of the new points
        return interp1d(
            x_new,
            y_new,
            kind="linear",
            fill_value=(y_new[0], y_new[-1]),
            bounds_error=False,
        )

    def _build_cir_points(
        self,
        X: Union[np.ndarray, List],
        sample_weight: Optional[Union[np.ndarray, List]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate new x, y points for the interpolation function of Isotonic
        Regression, in line with the definition of the Centered Isotonic
        Regression model. We use here that the CIR points can be constructed
        using the outcome of IR and the training data.

        * To transform a trained IR model to a CIR model, every range of
          constant values in the interpolation function is collapsed into one
          point with as x-coordinate the weighted average of the training
          datapoints within the constant range.
        * In the CIR paper, ranges with constant values of 0 or 1 are not
          collapsed. This implementation takes the non-collapsible points as a
          parameter.
        * The original range of IR is kept in CIR. This means that ranges of
          constant values can appear at the edges of the function's domain.
        """
        points_new = []

        # Input parameters were already validated by calling `super().fit`
        X_arr = np.array(X).reshape(-1)

        order = np.argsort(X_arr)
        X_arr = X_arr[order]
        sample_weight_arr = (
            np.ones(X_arr.shape)
            if sample_weight is None
            else np.array(sample_weight)[order]
        )

        y = self.transform(X_arr)
        _, idx = np.unique(y, return_index=True)
        y_steps = y[np.sort(idx)]

        for n, y_step in enumerate(y_steps):
            idx = np.where(y == y_step)[0]
            x_step = X_arr[idx]
            x_mean = np.average(x_step, weights=sample_weight_arr[idx])

            # Points that should not be collapsed
            if y_step in self.non_centered_points:
                points_new.extend([(x_step[0], y_step), (x_step[-1], y_step)])
            # Ensure that the original range is maintained
            elif n == 0:
                # points_new.extend([(x_step[0], y_step), (x_mean, y_step)])
                points_new.extend([(x_step[0], y_step)])
            elif n == len(y_steps) - 1:
                # points_new.extend([(x_mean, y_step), (x_step[-1], y_step)])
                points_new.extend([(x_step[-1], y_step)])
            else:
                points_new.append((x_mean, y_step))

        x_new, y_new = np.array(list(set(points_new))).T
        return x_new, y_new
