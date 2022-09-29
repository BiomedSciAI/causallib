"""
Metrics to assess performance of direct-outcome (counterfactual prediction) models.
Function named as ``*_error`` return a scalar value to minimize:
the lower, the better.

Metrics' interface doesn't strictly follow skleran's metrics interface.
The outcome prediction is expected to be full potential outcome prediction
(one column for each treatment value), and also expects the true treatment assignment.
"""

import numpy as np
import pandas as pd

from causallib.utils.stat_utils import calc_weighted_standardized_mean_differences
from causallib.utils.stat_utils import robust_lookup


def abs_standardized_mean_difference(a, b, **kwargs):
    asmd = calc_weighted_standardized_mean_differences(
        a, b,
        wx=np.ones_like(a),
        wy=np.ones_like(b),
    )
    asmd = np.abs(asmd)
    return asmd


def _get_observed_outcome_prediction(potential_outcomes, a):
    # TODO: duplicated throughout causallib. move to utils or standardization module
    is_predict_proba_classification_result = isinstance(potential_outcomes.columns, pd.MultiIndex)
    if is_predict_proba_classification_result:
        # Classification `outcome_model` with `predict_proba=True` returns
        # a MultiIndex treatment-values (`a`) over outcome-values (`y`)
        # Extract the prediction for the maximal outcome class
        # (probably class `1` in binary classification):
        outcome_values = potential_outcomes.columns.get_level_values(level=-1)
        potential_outcomes = potential_outcomes.xs(
            outcome_values.max(), axis="columns", level=-1, drop_level=True,
        )
    potential_outcomes = robust_lookup(potential_outcomes, a)
    return potential_outcomes


def balanced_residuals_error(
    y_true, y_pred, a_true,
    distance_metric=abs_standardized_mean_difference,
    distance_metric_kwargs=None,
):
    """Computes how different is the residuals distribution of the control group
    from that of the treatment group.
    Residuals are based on the observed (factual) outcome prediction.

    Can plug in any uni-variate two-sample test function.

    Args:
        y_true (pd.Series): The true observed outcomes.
        y_pred (pd.DataFrame): Potential outcome prediction, the output of `estimate_individual_outcome()`.
                               A matrix of (n_samples, n_treatments), with column names as the treatment values.
        a_true (pd.Series): A vector of observed treatment assignment.
        distance_metric (callable): A two sample test function.
            First argument is the residual values of the treatment group, second is for the control group.
            Defaults to absolute standardized mean difference.
        distance_metric_kwargs (dict): Additional keyword arguments needed for the `distance_metric` function.

    Returns:
        score (float):
    """
    if distance_metric_kwargs is None:
        distance_metric_kwargs = {}

    y_pred = _get_observed_outcome_prediction(y_pred, a_true)

    residuals = y_true - y_pred
    treatment_mask = a_true == a_true.max()
    control_mask = a_true == a_true.min()

    score = distance_metric(
        residuals[treatment_mask],
        residuals[control_mask],
        **distance_metric_kwargs,
    )
    return score

