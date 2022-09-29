"""Apply machine learning metrics to causal models for evaluation."""
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics


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
    # "msle": metrics.mean_squared_log_error, #uncomment if predictions are all positive
    # Allow mdae receive sample_weight argument but ignore it. This unifies the interface:
    "mdae": lambda y_true, y_pred, **kwargs: metrics.median_absolute_error(
        y_true, y_pred
    ),
    "r2": metrics.r2_score,
}


def get_default_binary_metrics(only_numeric_metric=False):
    """Get default metrics for evaluating binary models.

    Args:
        only_numeric_metric (bool): If metrics_to_evaluate not provided and default is used,
            whether to use only numerical metrics. Ignored if metrics_to_evaluate is provided.
            Non-numerical metrics are for example roc_curve, that returns vectors and not scalars).
    Returns:
        dict [str, callable]: metrics dict with key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
    """
    if only_numeric_metric:
        return NUMERICAL_CLASSIFICATION_METRICS

    return CLASSIFICATION_METRICS


def get_default_regression_metrics():
    """Get default metrics for evaluating continuous prediction models.

    Returns:
        dict [str, callable]: metrics dict with key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
    """
    return REGRESSION_METRICS


def evaluate_metrics(
    metrics_to_evaluate,
    y_true,
    y_pred=None,
    y_pred_proba=None,
    sample_weight=None,
):
    """Evaluates the metrics against the supplied predictions and labels.

    Note that some metrics operate on proba predictions (`y_pred_proba`) and others on
    direct predictions. The function will select the correct input based on the name of the metric,
    if it knows about the metric.
    Otherwise it defaults to using the direct prediction (`y_pred`).

    Args:
        metrics_to_evaluate (dict): key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
        y_true (pd.Series): True labels
        y_pred_proba (pd.Series): continuous output of predictor,
            as in `predict_proba` or `decision_function`.
        y_pred (pd.Series): label (i.e., categories, decisions) predictions.
        sample_weight (pd.Series | None): weight of each sample.

    Returns:
        pd.Series: name of metric as index and the evaluated score as value.
    """
    evaluated_metrics = {}
    for metric_name, metric_func in metrics_to_evaluate.items():
        prediction = y_pred_proba if _metric_needs_proba(metric_name) else y_pred
        if prediction is None:
            continue

        try:
            metric_value = metric_func(y_true, prediction, sample_weight=sample_weight)
        except ValueError as v:  # if y_true has single value
            warnings.warn(f"metric {metric_name} could not be evaluated")
            warnings.warn(str(v))
            metric_value = np.nan
        evaluated_metrics[metric_name] = metric_value

    all_scalars = all(np.isscalar(v) for v in evaluated_metrics.values())
    dtype = float if all_scalars else np.dtype(object)
    

    return pd.Series(evaluated_metrics, dtype=dtype)


def _metric_needs_proba(metric_name):
    use_proba = metric_name in {
        "hinge",
        "brier",
        "roc_curve",
        "roc_auc",
        "pr_curve",
        "avg_precision",
    }

    return use_proba
