"""
Metrics to assess performance of propensity score models.
Function named as ``*_error`` return a scalar value to minimize:
the lower, the better.

Metrics' interface doesn't strictly follow skleran's metrics interface.
Names are kept as `y_true` and `y_pred` even though these regard the
treatment assignment `a` (`a_true`, `a_pred` [propensity]).
"""

import numpy as np
from numpy import interp
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm


def weighted_roc_auc_error(y_true, y_pred, sample_weight, **kwargs):
    """
    Compute the squared error between the balanced (e.g. IP-weighted) ROC AUC
    to the diagonal, i.e. AUC=0.5.

    Shimoni, Y., et al. (2019)
    An evaluation toolkit to guide model selection and cohort definition in causal inference.

    Args:
        y_true (pd.Series): True binary label assignment of size (num_subjects,)
        y_pred (pd.Series): Predicted probability of each sample being
                            the positive label of size (num_subjects,).
        sample_weight (pd.Series | None): Weights of size (num_subjects,)

    Returns:
        score (float):
    """
    weighted_auc = roc_auc_score(y_true, y_pred, sample_weight=sample_weight)
    chance_auc = 0.5
    score = (weighted_auc - chance_auc) ** 2
    return score


def expected_roc_auc_error(y_true, y_pred, **kwargs):
    """
    Compute the squared error between the expected ROC-AUC given the provided
    scores and the actual ROC-AUC they produce.

    Shimoni, Y., et al. (2019)
    An evaluation toolkit to guide model selection and cohort definition in causal inference.

    Args:
        y_true (pd.Series): True binary label assignment of size (num_subjects,)
        y_pred (pd.Series): Predicted probability of each sample being
                            the positive label of size (num_subjects,).

    Returns:
        score (float):
    """
    # Calculate expected roc auc:
    p = np.hstack((y_pred, y_pred))
    w = np.hstack((y_pred, 1 - y_pred))
    target = np.hstack((np.ones_like(y_pred), np.zeros_like(y_pred)))
    expected_auc = roc_auc_score(target, p, sample_weight=w)

    auc = roc_auc_score(y_true, y_pred)
    score = (expected_auc - auc) ** 2
    return score


def weighted_roc_curve_error(y_true, y_pred, sample_weight, agg=np.max, **kwargs):
    """Compute the absolute differences between the balanced (e.g. IP-weighted) ROC curve
    and the diagonal x=y curve.
    Since difference in curves results in a multiple values (each point along the curve),
    an aggregation function is also required.

    Shimoni, Y., et al. (2019)
    An evaluation toolkit to guide model selection and cohort definition in causal inference.

    Args:
        y_true (pd.Series): True binary label assignment of size (num_subjects,)
        y_pred (pd.Series): Predicted probability of each sample being
                            the positive label of size (num_subjects,).
        sample_weight (pd.Series | None): Weights of size (num_subjects,)
        agg (callable): A function to aggregate a vector of absolute differences
                        between the curves' points. Default is max.

    Returns:
        score (float):
    """
    weighted_roc_fpr, weighted_roc_tpr, _ = roc_curve(
        y_true, y_pred, sample_weight=sample_weight
    )
    diagonal = weighted_roc_fpr
    diff = np.abs(weighted_roc_tpr - diagonal)
    score = agg(diff)
    return score


def expected_roc_curve_error(y_true, y_pred, agg=np.max, **kwargs):
    """Compute the absolute differences between the expected ROC curve and the regular
    ROC curve of the model.
    Since difference in curves results in a multiple values (each point along the curve),
    an aggregation function is also required.

    Shimoni, Y., et al. (2019)
    An evaluation toolkit to guide model selection and cohort definition in causal inference.

    Args:
        y_true (pd.Series): True binary label assignment of size (num_subjects,)
        y_pred (pd.Series): Predicted probability of each sample being
                            the positive label of size (num_subjects,).
        agg (callable): A function to aggregate a vector of absolute differences
                        between the curves' points. Default is max.

    Returns:
        score (float):
    """
    # Calculate expected roc curve:
    p = np.hstack((y_pred, y_pred))
    w = np.hstack((y_pred, 1 - y_pred))
    target = np.hstack((np.ones_like(y_pred), np.zeros_like(y_pred)))
    expected_fpr, expected_tpr, _ = roc_curve(target, p, sample_weight=w)

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Align the FPR/TPR so the difference vector is meaningful:
    fpr_domain = np.linspace(0, 1, 100)
    tpr_interp = interp(fpr_domain, fpr, tpr)
    tpr_interp[0] = 0.0
    expected_tpr_interp = interp(fpr_domain, expected_fpr, expected_tpr)
    expected_tpr_interp[0] = 0.0

    diff = np.abs(tpr_interp - expected_tpr_interp)
    score = agg(diff)
    return score


def ici_error(y_true, y_pred, agg=np.mean, lowess_kwargs=None, **kwargs):
    """Integrated calibration index metric.
    Fits a lowess model between binary treatment assignment and the predicted probabilities.
    This generates a smooth calibration curve,
    which we can subtract from the diagonal, and examine magnitude.
    To make a single-number score the absolute difference is then aggregated using mean, median, max,
    or whatever collapses a vector into a single number.

    See: Austin and Steyerberg (2019): The Integrated Calibration Index (ICI)
    and related metrics for quantifying the calibration of logistic regression models.
    """
    if lowess_kwargs is None:
        lowess_kwargs = {}

    smooth_curve = sm.nonparametric.lowess(
        y_true, y_pred,
        **lowess_kwargs
        # TODO: maybe force prediction domain xval=linspace(0, 1, num=0) or linspace(y_true.min(), y_true.max(), num=0)
    )
    # See `out` in doc: https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    if len(smooth_curve.shape) == 2:  # `return_sorted==True`
        y_pred = smooth_curve[:, 0]
        smooth_curve = smooth_curve[:, 1]

    diff = np.abs(smooth_curve - y_pred)
    score = agg(diff)
    return score


