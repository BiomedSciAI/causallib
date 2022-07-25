"""General metric functions for causal evaluation."""
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics

from ..utils.stat_utils import (
    calc_weighted_ks2samp,
    calc_weighted_standardized_mean_differences,
)

NUMERICAL_CLASSIFICATION_METRICS = {
    "accuracy": metrics.accuracy_score,
    "precision": metrics.precision_score,
    "recall": metrics.recall_score,
    "f1": metrics.f1_score,
    "roc_auc": metrics.roc_auc_score,
    "avg_precision": metrics.average_precision_score,
    "hinge": metrics.hinge_loss,
    "matthews": metrics.matthews_corrcoef,
    "0_1": metrics.zero_one_loss,
    "brier": metrics.brier_score_loss,
}
NONNUMERICAL_CLASSIFICATION_METRICS = {
    "confusion_matrix": metrics.confusion_matrix,
    "roc_curve": metrics.roc_curve,
    "pr_curve": metrics.precision_recall_curve,
}
CLASSIFICATION_METRICS = {
    **NUMERICAL_CLASSIFICATION_METRICS,
    **NONNUMERICAL_CLASSIFICATION_METRICS,
}

REGRESSION_METRICS = {
    "expvar": metrics.explained_variance_score,
    "mae": metrics.mean_absolute_error,
    "mse": metrics.mean_squared_error,
    "msle": metrics.mean_squared_log_error,
    # Allow mdae receive sample_weight argument but ignore it. This unifies the interface:
    "mdae": lambda y_true, y_pred, **kwargs: metrics.median_absolute_error(
        y_true, y_pred
    ),
    "r2": metrics.r2_score,
}

DISTRIBUTION_DISTANCE_METRICS = {
    "smd": lambda x, y, wx, wy: calc_weighted_standardized_mean_differences(
        x, y, wx, wy
    ),
    "abs_smd": lambda x, y, wx, wy: abs(
        calc_weighted_standardized_mean_differences(x, y, wx, wy)
    ),
    "ks": lambda x, y, wx, wy: calc_weighted_ks2samp(x, y, wx, wy),
}


def evaluate_binary_metrics(
    y_true,
    y_pred_proba=None,
    y_pred=None,
    sample_weight=None,
    metrics_to_evaluate=None,
    only_numeric_metric=True,
):
    """Evaluates a binary prediction against true labels.

    Args:
        y_true (pd.Series): True labels
        y_pred_proba (pd.Series): continuous output of predictor,
            as in `predict_proba` or `decision_function`.
        y_pred (pd.Series): label (i.e., categories, decisions) predictions.
        sample_weight (pd.Series | None): weight of each sample.
        metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
            If not provided, default are used.
        only_numeric_metric (bool): If metrics_to_evaluate not provided and default is used,
            whether to use only numerical metrics. Ignored if metrics_to_evaluate is provided.
            Non-numerical metrics are for example roc_curve, that returns vectors and not scalars).

    Returns:
        pd.Series: name of metric as index and the evaluated score as value.
    """
    if metrics_to_evaluate is None:
        metrics_to_evaluate = (
            NUMERICAL_CLASSIFICATION_METRICS
            if only_numeric_metric
            else CLASSIFICATION_METRICS
        )
    scores = {}
    for metric_name, metric_func in metrics_to_evaluate.items():
        if metric_name in {
            "hinge",
            "brier",
            "roc_curve",
            "roc_auc",
            "pr_curve",
            "avg_precision",
        }:
            prediction = y_pred_proba
        else:
            prediction = y_pred

        if prediction is None:
            continue

        try:
            scores[metric_name] = metric_func(
                y_true, prediction, sample_weight=sample_weight
            )
        except ValueError as v:  # if y_true has single value
            warnings.warn(f"metric {metric_name} could not be evaluated")
            warnings.warn(str(v))
            scores[metric_name] = np.nan

    dtype = (
        float
        if all(np.isscalar(score) for score in scores.values())
        else np.dtype(object)
    )

    return pd.Series(scores, dtype=dtype)


def evaluate_regression_metrics(
    y_true, y_pred, sample_weight=None, metrics_to_evaluate=None
):
    """Evaluates continuous prediction against true labels

    Args:
        y_true (pd.Series): True label.
        y_pred (pd.Series): Predictions.
        sample_weight (pd.Series | None): weight for each sample.
        metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
            If not provided, default metrics from causallib.evaluation.metrics are used.

    Returns:
        pd.Series: name of metric as index and the evaluated score as value.
    """
    metrics_to_evaluate = metrics_to_evaluate or REGRESSION_METRICS
    evaluated_metrics = {}
    for metric_name, metric_func in metrics_to_evaluate.items():
        try:
            evaluated_metrics[metric_name] = metric_func(
                y_true, y_pred, sample_weight=sample_weight
            )
        except ValueError as v:
            evaluated_metrics[metric_name] = np.nan
            warnings.warn(f"metric {metric_name} could not be evaluated")
            warnings.warn(str(v))
    return pd.Series(evaluated_metrics)


# ################# #
# Covariate Balance #
# ################# #


def calculate_covariate_balance(X, a, w, metric="abs_smd"):
    """Calculate covariate balance table ("table 1")

    Args:
        X (pd.DataFrame): Covariates.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with
            the signature weighted_distance(x, y, wx, wy) calculating distance between the weighted
            sample x and weighted sample y (weights by wx and wy respectively).

    Returns:
        pd.DataFrame: index are covariate names (columns) from X, and columns are
            "weighted" / "unweighted" results of applying `metric` on each covariate
            to compare the two groups.
    """
    treatment_values = np.sort(np.unique(a))
    results = {}
    for treatment_value in treatment_values:
        distribution_distance_of_cur_treatment = pd.DataFrame(
            index=X.columns, columns=["weighted", "unweighted"], dtype=float
        )
        for col_name, col_data in X.items():
            weighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, w, a, treatment_value, metric
            )
            unweighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, pd.Series(1, index=w.index), a, treatment_value, metric
            )
            distribution_distance_of_cur_treatment.loc[
                col_name, ["weighted", "unweighted"]
            ] = [weighted_distance, unweighted_distance]
        results[treatment_value] = distribution_distance_of_cur_treatment
    results = pd.concat(
        results, axis="columns", names=[a.name or "a", metric]
    )  # type: pd.DataFrame
    results.index.name = "covariate"
    if len(treatment_values) == 2:
        # If there are only two treatments, the results for both are identical.
        # Therefore, we can get rid from one of them.
        # We keep the results for the higher valued treatment group (assumed treated, typically 1):
        results = results.xs(treatment_values.max(), axis="columns", level=0)
    # TODO: is there a neat expansion for multi-treatment case?
    #  maybe not current_treatment vs. the rest.
    return results


def calculate_distribution_distance_for_single_feature(
    x, w, a, group_level, metric="abs_smd"
):
    """

    Args:
        x (pd.Series): A single feature to check balancing.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        group_level: Value from `a` in order to divide the sample into one vs. rest.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with
            the signature weighted_distance(x, y, wx, wy) calculating distance between the weighted
            sample x and weighted sample y (weights by wx and wy respectively).

    Returns:
        float: weighted distance between the samples assigned to `group_level`
            and the rest of the samples.
    """
    if not callable(metric):
        metric = DISTRIBUTION_DISTANCE_METRICS[metric]
    cur_treated_mask = a == group_level
    x_treated = x.loc[cur_treated_mask]
    w_treated = w.loc[cur_treated_mask]
    x_untreated = x.loc[~cur_treated_mask]
    w_untreated = w.loc[~cur_treated_mask]
    distribution_distance = metric(x_treated, x_untreated, w_treated, w_untreated)
    return distribution_distance
