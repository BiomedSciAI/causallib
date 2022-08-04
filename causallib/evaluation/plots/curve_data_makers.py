"""Functions that calculate curve data for cross validation plots."""
from typing import List
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics


def calculate_roc_curve(curve_data):
    """Calculates ROC curve on the folds

    Args:
        curve_data (dict) : dict of curves produced by
            BaseEvaluationPlotDataExtractor.calculate_curve_data
    Returns:
        dict[str, list[np.ndarray]]: Keys being "FPR", "TPR" and "AUC" (ROC metrics)
            and values are a list the size of number of folds with the evaluation of each fold.
    """

    for curve_name in curve_data.keys():
        curve_data[curve_name]["FPR"] = curve_data[curve_name].pop("first_ret_value")
        curve_data[curve_name]["TPR"] = curve_data[curve_name].pop("second_ret_value")
        curve_data[curve_name]["AUC"] = curve_data[curve_name].pop("area")
    return curve_data


def calculate_pr_curve(curve_data, targets):
    """Calculates precision-recall curve on the folds.

    Args:
        curve_data (dict) : dict of curves produced by
            BaseEvaluationPlotDataExtractor.calculate_curve_data
        targets (pd.Series): True labels.

    Returns:
        dict[str, list[np.ndarray]]: Keys being "Precision", "Recall" and "AP" (PR metrics)
            and values are a list the size of number of folds with the
            evaluation of each fold.
            Additional "prevalence" key, with positive-label "prevalence" is added
            to be used by the chance curve.
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
    """Calculates performance curves of the predictions across folds.

    Args:
        folds_predictions (list[pd.Series]): Score prediction (as in continuous output
            of classifier, `predict_proba` or `decision_function`) for every fold.
        folds_targets (list[pd.Series]): True labels for every fold.
        sample_weights (list[pd.Series] | None): weight for each sample for every fold.
        area_metric (callable): Performance metric of the area under the curve.
        curve_metric (callable): Performance metric returning 3 output vectors - metric1, metric2
            and thresholds.
            Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
        pos_label: What label in `targets` is considered the positive label.

    Returns:
        (list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]):
            For every fold, the calculated metric1 and metric2 (the curves), the thresholds and the
            area calculations.
    """
    sample_weights = (
        [None] * len(folds_predictions) if sample_weights is None else sample_weights
    )
    # Scikit-learn precision_recall_curve and roc_curve do not return values in a consistent way.
    # Namely, roc_curve returns `fpr`, `tpr`, which correspond to x_axis, y_axis,
    # whereas precision_recall_curve returns `precision`, `recall`,
    # which correspond to y_axis, x_axis.
    # That's why this function will return the values the same order as Scikit's curves,
    # leaving it up to the caller to put labels on what those return values actually are
    # (specifically, whether they're x_axis or y-axis)
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


def calculate_curve_data_binary_outcome(
    folds_predictions,
    targets,
    curve_metric,
    area_metric,
    stratify_by=None,
):
    """Calculate different performance (ROC or PR) curves

    Args:
        folds_predictions (list[pd.Series]): Predictions for each fold.
        targets (pd.Series): True labels
        curve_metric (callable): Performance metric returning 3 output vectors - metric1,
            metric2 and thresholds. Where metric1 and metric2 depict the curve
            when plotted on x-axis and y-axis.
        area_metric (callable): Performance metric of the area under the curve.
        stratify_by (pd.Series): Group assignment to stratify by.

    Returns:
        dict[str, dict[str, list[np.ndarray]]]: Evaluation of the metric
            for each fold and for each curve.
            One curve for each group level in `stratify_by`.
            On general: {curve_name: {metric1: [evaluation_fold_1, ...]}}.
            For example: {"Treatment=1": {"FPR": [FPR_fold_1, FPR_fold_2, FPR_fold_3]}}
    """
    # folds_targets = [targets.loc[p.index] for p in folds_predictions]
    # folds_stratify_by = [stratify_by.loc[p.index] for p in folds_predictions]

    stratify_values = sorted(set(stratify_by))
    curve_data = {}
    for stratum_level in stratify_values:
        # Slice data for that stratum level across the folds:
        folds_stratum_predictions, folds_stratum_targets = [], []
        for fold_predictions in folds_predictions:
            # Extract fold:
            fold_targets = targets.loc[fold_predictions.index]
            fold_stratify_by = stratify_by.loc[fold_predictions.index]
            # Extract stratum:
            mask = fold_stratify_by == stratum_level
            fold_predictions = fold_predictions.loc[mask]
            fold_targets = fold_targets.loc[mask]
            # Save:
            folds_stratum_predictions.append(fold_predictions)
            folds_stratum_targets.append(fold_targets)

        (
            area_folds,
            first_ret_folds,
            second_ret_folds,
            threshold_folds,
        ) = calculate_performance_curve_data_on_folds(
            folds_stratum_predictions,
            folds_stratum_targets,
            None,
            area_metric,
            curve_metric,
        )

        curve_data[f"Treatment={stratum_level}"] = {
            "first_ret_value": first_ret_folds,
            "second_ret_value": second_ret_folds,
            "Thresholds": threshold_folds,
            "area": area_folds,
        }
    return curve_data


def calculate_curve_data_propensity(
    fold_predictions: List[
        "causallib.evaluation.weight_predictor.PropensityPredictions"
    ],
    targets,
    curve_metric,
    area_metric,
):
    """Calculate different performance (ROC or PR) curves

    Args:
        fold_predictions (list[PropensityEvaluatorPredictions]):
            Predictions for each fold.
        targets (pd.Series): True labels
        curve_metric (callable): Performance metric returning 3 output vectors - metric1,
            metric2 and thresholds. Where metric1 and metric2 depict the curve when plotted
            on x-axis and y-axis.
        area_metric (callable): Performance metric of the area under the curve.
        **kwargs:

    Returns:
        dict[str, dict[str, list[np.ndarray]]]: Evaluation of the metric
            for each fold and for each curve.
            3 curves:
                * "unweighted" (regular)
                * "weighted" (weighted by inverse propensity)
                * "expected" (duplicated population, weighted by propensity)
            On general: {curve_name: {metric1: [evaluation_fold_1, ...]}}.
            For example: {"weighted": {"FPR": [FPR_fold_1, FPR_fold_2, FPR_fold3]}}
    """

    curves_sample_weights = {
        "unweighted": [None for _ in fold_predictions],
        "weighted": [
            fold_predictions.weight_by_treatment_assignment
            for fold_predictions in fold_predictions
        ],
        "expected": [
            pd.concat([fold_predictions.propensity, 1 - fold_predictions.propensity])
            for fold_predictions in fold_predictions
        ],
    }
    curves_folds_targets = [
        targets.loc[fold_predictions.weight_by_treatment_assignment.index]
        for fold_predictions in fold_predictions
    ]
    curves_folds_targets = {
        "unweighted": curves_folds_targets,
        "weighted": curves_folds_targets,
        "expected": [
            pd.concat([
                pd.Series(data=targets.max(), index=fold_predictions.propensity.index),
                pd.Series(data=targets.min(), index=fold_predictions.propensity.index)
            ])
            for fold_predictions in fold_predictions
        ],
    }
    fold_predictions = {
        "unweighted": [
            fold_predictions.propensity for fold_predictions in fold_predictions
        ],
        "weighted": [
            fold_predictions.propensity for fold_predictions in fold_predictions
        ],
        "expected": [
            pd.concat([fold_predictions.propensity, fold_predictions.propensity])
            for fold_predictions in fold_predictions
        ],
    }
    # Expected curve duplicates the population, basically concatenating so that:
    # prediction = [p, p], target = [1, 0], weights = [p, 1-p]

    curve_data = {}
    for curve_name in curves_sample_weights:
        sample_weights = curves_sample_weights[curve_name]
        folds_targets = curves_folds_targets[curve_name]
        folds_predictions = fold_predictions[curve_name]

        (
            area_folds,
            first_ret_folds,
            second_ret_folds,
            threshold_folds,
        ) = calculate_performance_curve_data_on_folds(
            folds_predictions,
            folds_targets,
            sample_weights,
            area_metric,
            curve_metric,
        )

        curve_data[curve_name] = {
            "first_ret_value": first_ret_folds,
            "second_ret_value": second_ret_folds,
            "Thresholds": threshold_folds,
            "area": area_folds,
        }

    # Rename keys (as will be presented as curve labels in legend)
    curve_data["Propensity"] = curve_data.pop("unweighted")
    curve_data["Weighted"] = curve_data.pop("weighted")
    curve_data["Expected"] = curve_data.pop("expected")
    return curve_data
