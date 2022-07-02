import numpy as np
import warnings
from sklearn import metrics

import warnings

from .plots import get_subplots, lookup_name


def plot_evaluation_results(results, X, a, y, plot_names="all"):
    if plot_names == "all":
        plot_names = results.extractor.available_plot_names
    phases = results.predictions.keys()
    all_axes = {phase: {} for phase in phases}

    for phase in phases:
        phase_fig, phase_axes = get_subplots(len(plot_names))
        phase_axes = phase_axes.ravel()
        # squeeze a vector out of the matrix-like structure of the returned fig.

        # Retrieve all indices of the different folds in the phase [idx_fold_1, idx_folds_2, ...]

        for i, name in enumerate(plot_names):
            ax = phase_axes[i]
            try:
                plot_ax = plot_single_evaluation_result(
                    results, X, a, y, name, phase, ax
                )
            except Exception as e:
                warnings.warn(f"Failed to plot {name} with error {e}")
                plot_ax = None
            all_axes[phase][name] = plot_ax
        phase_fig.suptitle("Evaluation on {} phase".format(phase))
    return all_axes


def plot_single_evaluation_result(results, X, a, y, plot_name, phase, ax=None):
    if plot_name not in results.extractor.available_plot_names:
        raise ValueError(f"Plot name '{plot_name}' not supported for this result.")
    cv_idx_folds = [
        fold_idx[0] if phase == "train" else fold_idx[1] for fold_idx in results.cv
    ]
    plot_func = lookup_name(plot_name)
    plot_data = results.get_data_for_plot(plot_name, X, a, y, phase=phase)
    # TODO: ^ consider _get_data_for_plot returning args (tuple) and kwargs (dictionary) which will be
    #       expanded when calling plot_func: plot_func(*plot_args, **plot_kwargs).
    #       This will allow more flexible specification of param-eters by the caller
    #       (For example, Propensity Distribution with kde=True and Weight Distribution with kde=False)
    return plot_func(*plot_data, cv=cv_idx_folds, ax=ax)


# Calculating ROC/PR curves:
def calculate_roc_curve(curve_data):
    """Calculates ROC curve on the folds

    Args:
        folds_predictions (list[WeightEvaluatorPredictions | OutcomeEvaluatorPredictions]):
            list of the predictions, each entry correspond to a fold.
        targets (pd.Series): True labels.
        stratify_by (pd.Series): A vector (mostly, treatment assignment) to perform groupby with.

    Returns:
        dict[str, list[np.ndarray]]: Keys being "FPR", "TPR" and "AUC" (ROC metrics) and values are a list the size
                                        of number of folds with the evaluation of each fold.
    """

    for curve_name in curve_data.keys():
        curve_data[curve_name]["FPR"] = curve_data[curve_name].pop("first_ret_value")
        curve_data[curve_name]["TPR"] = curve_data[curve_name].pop("second_ret_value")
        curve_data[curve_name]["AUC"] = curve_data[curve_name].pop("area")
    return curve_data


def calculate_pr_curve(curve_data, targets):
    """Calculates precision-recall curve on the folds

    Args:
        folds_predictions (list[WeightEvaluatorPredictions | OutcomeEvaluatorPredictions]):
            list of the predictions, each entry correspond to a fold.
        targets (pd.Series): True labels.
        stratify_by (pd.Series): A vector (mostly, treatment assignment) to perform groupby with.

    Returns:
        dict[str, list[np.ndarray]]: Keys being "Precision", "Recall" and "AP" (PR metrics) and values are a list
                                        the size of number of folds with the evaluation of each fold.
                                        Additional "prevalence" key, with positive-label prevalence is added (to be
                                        used by the chance curve).
    """

    for curve_name in curve_data.keys():
        curve_data[curve_name]["Precision"] = curve_data[curve_name].pop(
            "first_ret_value"
        )
        curve_data[curve_name]["Recall"] = curve_data[curve_name].pop(
            "second_ret_value"
        )
        curve_data[curve_name]["AP"] = curve_data[curve_name].pop("area")
    curve_data["prevalence"] = targets.value_counts(normalize=True).loc[targets.max()]
    return curve_data


def calculate_performance_curve_data_on_folds(
    folds_predictions,
    folds_targets,
    sample_weights=None,
    area_metric=metrics.roc_auc_score,
    curve_metric=metrics.roc_curve,
    pos_label=None,
):
    """Calculates performance curves (either ROC or precision-recall) of the predictions across folds.

    Args:
        folds_predictions (list[pd.Series]): Score prediction (as in continuous output of classifier,
                                                `predict_proba` or `decision_function`) for every fold.
        folds_targets (list[pd.Series]): True labels for every fold.
        sample_weights (list[pd.Series] | None): weight for each sample for every fold.
        area_metric (callable): Performance metric of the area under the curve.
        curve_metric (callable): Performance metric returning 3 output vectors - metric1, metric2 and thresholds.
                                Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
        pos_label: What label in `targets` is considered the positive label.

    Returns:
        (list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]):
            For every fold, the calculated metric1 and metric2 (the curves), the thresholds and the area calculations.
    """
    sample_weights = (
        [None] * len(folds_predictions) if sample_weights is None else sample_weights
    )
    # Scikit-learn precision_recall_curve and roc_curve do not return values in a consistent way.
    # Namely, roc_curve returns `fpr`, `tpr`, which correspond to x_axis, y_axis,
    # whereas precision_recall_curve returns `precision`, `recall`, which correspond to y_axis, x_axis.
    # That's why this function will return the values the same order as the Scikit's curves, and leave it up to the
    # caller to put labels on what those return values actually are (specifically, whether they're x_axis or y-axis)
    first_ret_folds, second_ret_folds, threshold_folds, area_folds = [], [], [], []
    for fold_prediction, fold_target, fold_weights in zip(
        folds_predictions, folds_targets, sample_weights
    ):
        first_ret_fold, second_ret_fold, threshold_fold = curve_metric(
            fold_target,
            fold_prediction,
            pos_label=pos_label,
            sample_weight=fold_weights,
        )
        try:
            area_fold = area_metric(
                fold_target, fold_prediction, sample_weight=fold_weights
            )
        except ValueError as v:  # AUC cannot be evaluated if targets are constant
            warnings.warn(f"metric {area_metric.__name__} could not be evaluated")
            warnings.warn(str(v))
            area_fold = np.nan

        first_ret_folds.append(first_ret_fold)
        second_ret_folds.append(second_ret_fold)
        threshold_folds.append(threshold_fold)
        area_folds.append(area_fold)
    return area_folds, first_ret_folds, second_ret_folds, threshold_folds
