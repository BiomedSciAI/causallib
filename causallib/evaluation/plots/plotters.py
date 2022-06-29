import abc
from sklearn import metrics
from typing import List
import pandas as pd
import numpy as np

from .plots import lookup_name, get_subplots
from ..metrics import calculate_covariate_balance
from ...estimation.base_weight import PropensityEstimator, WeightEstimator
from ...estimation.base_estimator import IndividualOutcomeEstimator
import warnings
from ...utils.stat_utils import is_vector_binary
from ...utils.stat_utils import robust_lookup


# Calculating ROC/PR curves:
def _calculate_roc_curve_data(curve_data):
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


def _calculate_pr_curve_data(curve_data, targets):
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


class Plotter:
    @staticmethod
    def from_estimator(estimator):
        if isinstance(estimator, (PropensityEstimator)):
            return PropensityPlotter
        if isinstance(estimator, (PropensityEstimator, WeightEstimator)):
            return WeightPlotter
        if isinstance(estimator, IndividualOutcomeEstimator):
            return OutcomePlotter

    def plot_cv(self, predictions, X, a, y, cv, plots):
        """Plots prediction performance across different folds.

        Args:
            predictions (dict[str, list]): the output of predict_cv.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            cv (list[tuples]): list the number of folds containing tuples of indices (train_idx, validation_idx)
            plots (list[str]): list of plots to make

        Returns:
            dict[str, dict[str, plt.Axes]]:
                {phase_name: {plot_name: plot_axes}}.
                 For example: {"train": {"roc_curve": plt.Axes, "calibration": plt.Axes}}
        """
        phases = predictions.keys()
        all_axes = {phase: {} for phase in phases}

        for phase in phases:
            phase_fig, phase_axes = get_subplots(len(plots))
            phase_axes = (
                phase_axes.ravel()
            )  # squeeze a vector out of the matrix-like structure of the returned fig.

            # Retrieve all indices of the different folds in the phase [idx_fold_1, idx_folds_2, ...]
            cv_idx_folds = [
                fold_idx[0] if phase == "train" else fold_idx[1] for fold_idx in cv
            ]
            predictions_folds = predictions[phase]

            for i, plot_name in enumerate(plots):
                plot_data = self._get_data_for_plot(
                    plot_name, predictions_folds, X, a, y, cv_idx_folds
                )
                # TODO: ^ consider _get_data_for_plot returning args (tuple) and kwargs (dictionary) which will be
                #       expanded when calling plot_func: plot_func(*plot_args, **plot_kwargs).
                #       This will allow more flexible specification of parameters by the caller
                #       (For example, Propensity Distribution with kde=True and Weight Distribution with kde=False)
                plot_func = lookup_name(plot_name)

                if plot_func is None or plot_data is None:
                    plot_ax = None
                else:
                    plot_ax = plot_func(*plot_data, cv=cv_idx_folds, ax=phase_axes[i])
                all_axes[phase][plot_name] = plot_ax
            phase_fig.suptitle("Evaluation on {} phase".format(phase))
        return all_axes

    @abc.abstractmethod
    def _get_data_for_plot(self, plot_name, folds_predictions, X, a, y, cv):
        """Return a tuple containing the relevant data needed for the specific plot provided in `plot_name`"""
        raise NotImplementedError

    @abc.abstractmethod
    def _calculate_curve_data(
        self, folds_predictions, targets, curve_metric, area_metric, stratify_by=None
    ):
        """Given a list of predictions (the output of _estimator_predict by folds)
        and a vector of targets. extract (if needed)the relevant parts in each fold prediction
        and apply the curve_metric and area_metric.
        The output is nested dict: first level is the curve name and inner level dict has
        `first_ret_value` `first_ret_value`, `Thresholds` and `area` of that curve` in list
        with each entry corresponding to each fold."""
        raise NotImplementedError

    @staticmethod
    def _calculate_performance_curve_data_on_folds(
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
            [None] * len(folds_predictions)
            if sample_weights is None
            else sample_weights
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
                warnings.warn(
                    "metric {} could not be evaluated".format(area_metric.__name__)
                )
                warnings.warn(str(v))
                area_fold = np.nan

            first_ret_folds.append(first_ret_fold)
            second_ret_folds.append(second_ret_fold)
            threshold_folds.append(threshold_fold)
            area_folds.append(area_fold)
        return area_folds, first_ret_folds, second_ret_folds, threshold_folds


class WeightPlotter(Plotter):
    def _get_data_for_plot(self, plot_name, folds_predictions, X, a, y, cv):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots.py module.

        Args:
            plot_name (str): Plot name.
            folds_predictions (list[WeightEvaluatorPredictions]): Predictions for each fold.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment
            y: *IGNORED*
            cv list[np.ndarray]: Indices (in iloc positions) of each fold.

        Returns:
            tuple: Plot data
        """
        if plot_name in {"weight_distribution"}:
            folds_predictions = [
                prediction.weight_for_being_treated for prediction in folds_predictions
            ]
            return folds_predictions, a
        elif plot_name in {"roc_curve"}:
            curve_data = self._calculate_curve_data(
                folds_predictions, a, metrics.roc_curve, metrics.roc_auc_score
            )
            roc_curve_data = _calculate_roc_curve_data(curve_data)
            return (roc_curve_data,)
        elif plot_name in {"pr_curve"}:
            curve_data = self._calculate_curve_data(
                folds_predictions,
                a,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
            )
            pr_curve_data = _calculate_pr_curve_data(curve_data, a)
            return (pr_curve_data,)
        elif plot_name in {"covariate_balance_love", "covariate_balance_slope"}:
            distribution_distances = []
            for fold_prediction in folds_predictions:
                fold_w = fold_prediction.weight_by_treatment_assignment
                fold_X = X.loc[fold_w.index]
                fold_a = a.loc[fold_w.index]
                dist_dist = calculate_covariate_balance(fold_X, fold_a, fold_w)
                distribution_distances.append(dist_dist)
            return (distribution_distances,)
        else:
            return None

    def _calculate_curve_data(
        self, folds_predictions, targets, curve_metric, area_metric, **kwargs
    ):
        """Calculate different performance (ROC or PR) curves

        Args:
            folds_predictions (list[WeightEvaluatorPredictions]): Predictions for each fold.
            targets (pd.Series): True labels
            curve_metric (callable): Performance metric returning 3 output vectors - metric1, metric2 and thresholds.
                                    Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
            area_metric (callable): Performance metric of the area under the curve.
            **kwargs:

        Returns:
            dict[str, dict[str, list[np.ndarray]]]: Evaluation of the metric for each fold and for each curve.
                2 curves:
                    * "unweighted" (regular)
                    * "weighted" (weighted by weights of each sample (according to their assignment))
                On general: {curve_name: {metric1: [evaluation_fold_1, ...]}}.
                For example: {"weighted": {"FPR": [FPR_fold_1, FPR_fold_2, FPR_fold3]}}
        """
        folds_sample_weights = {
            "unweighted": [None for _ in folds_predictions],
            "weighted": [
                fold_predictions.weight_by_treatment_assignment
                for fold_predictions in folds_predictions
            ],
        }
        folds_predictions = [
            fold_predictions.weight_for_being_treated
            for fold_predictions in folds_predictions
        ]
        folds_targets = []
        for fold_predictions in folds_predictions:
            # Since this is weight estimator, which takes the inverse of a class prediction
            fold_targets = targets.loc[fold_predictions.index]
            fold_targets = fold_targets.replace(
                {
                    fold_targets.min(): fold_targets.max(),
                    fold_targets.max(): fold_targets.min(),
                }
            )
            folds_targets.append(fold_targets)

        curve_data = {}
        for curve_name, sample_weights in folds_sample_weights.items():
            (
                area_folds,
                first_ret_folds,
                second_ret_folds,
                threshold_folds,
            ) = self._calculate_performance_curve_data_on_folds(
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
        curve_data["Weights"] = curve_data.pop("unweighted")
        curve_data["Weighted"] = curve_data.pop("weighted")
        return curve_data


class PropensityPlotter(WeightPlotter):
    def _get_data_for_plot(self, plot_name, folds_predictions, X, a, y, cv):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots.py module.

        Args:
            plot_name (str): Plot name.
            folds_predictions (list[PropensityEvaluatorPredictions]): Predictions for each fold.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment
            y: *IGNORED*
            cv list[np.ndarray]: Indices (in iloc positions) of each fold.

        Returns:
            tuple: Plot data
        """
        if plot_name in {"weight_distribution"}:
            folds_predictions = [
                prediction.propensity for prediction in folds_predictions
            ]
            return folds_predictions, a
        elif plot_name in {"calibration"}:
            folds_predictions = [
                prediction.propensity for prediction in folds_predictions
            ]
            return folds_predictions, a
        else:
            # Common plots are implemented at top-most level possible.
            # Plot might be implemented by WeightEvaluator:
            return super(PropensityPlotter, self)._get_data_for_plot(
                plot_name, folds_predictions, X, a, y, cv
            )

    def _calculate_curve_data(
        self, curves_folds_predictions, targets, curve_metric, area_metric, **kwargs
    ):
        """Calculate different performance (ROC or PR) curves

        Args:
            curves_folds_predictions (list[PropensityEvaluatorPredictions]): Predictions for each fold.
            targets (pd.Series): True labels
            curve_metric (callable): Performance metric returning 3 output vectors - metric1, metric2 and thresholds.
                                    Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
            area_metric (callable): Performance metric of the area under the curve.
            **kwargs:

        Returns:
            dict[str, dict[str, list[np.ndarray]]]: Evaluation of the metric for each fold and for each curve.
                3 curves:
                    * "unweighted" (regular)
                    * "weighted" (weighted by inverse propensity)
                    * "expected" (duplicated population, weighted by propensity)
                On general: {curve_name: {metric1: [evaluation_fold_1, ...]}}.
                For example: {"weighted": {"FPR": [FPR_fold_1, FPR_fold_2, FPR_fold3]}}
        """
        curves_sample_weights = {
            "unweighted": [None for _ in curves_folds_predictions],
            "weighted": [
                fold_predictions.weight_by_treatment_assignment
                for fold_predictions in curves_folds_predictions
            ],
            "expected": [
                fold_predictions.propensity.append(1 - fold_predictions.propensity)
                for fold_predictions in curves_folds_predictions
            ],
        }
        curves_folds_targets = [
            targets.loc[fold_predictions.weight_by_treatment_assignment.index]
            for fold_predictions in curves_folds_predictions
        ]
        curves_folds_targets = {
            "unweighted": curves_folds_targets,
            "weighted": curves_folds_targets,
            "expected": [
                pd.Series(
                    data=targets.max(), index=fold_predictions.propensity.index
                ).append(
                    pd.Series(
                        data=targets.min(), index=fold_predictions.propensity.index
                    )
                )
                for fold_predictions in curves_folds_predictions
            ],
        }
        curves_folds_predictions = {
            "unweighted": [
                fold_predictions.propensity
                for fold_predictions in curves_folds_predictions
            ],
            "weighted": [
                fold_predictions.propensity
                for fold_predictions in curves_folds_predictions
            ],
            "expected": [
                fold_predictions.propensity.append(fold_predictions.propensity)
                for fold_predictions in curves_folds_predictions
            ],
        }
        # Expected curve duplicates the population, basically concatenating so that:
        # prediction = [p, p], target = [1, 0], weights = [p, 1-p]

        curve_data = {}
        for curve_name in curves_sample_weights.keys():
            sample_weights = curves_sample_weights[curve_name]
            folds_targets = curves_folds_targets[curve_name]
            folds_predictions = curves_folds_predictions[curve_name]

            (
                area_folds,
                first_ret_folds,
                second_ret_folds,
                threshold_folds,
            ) = self._calculate_performance_curve_data_on_folds(
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


class OutcomePlotter(Plotter):
    def _get_data_for_plot(self, plot_name, folds_predictions, X, a, y, cv):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots.py module.

        Args:
            plot_name (str): Plot name.
            folds_predictions (list[OutcomeEvaluatorPredictions]): Predictions for each fold.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment
            y: *IGNORED*
            cv list[np.ndarray]: Indices (in iloc positions) of each fold.

        Returns:
            tuple: Plot data
        """
        if plot_name in {"continuous_accuracy", "residuals"}:
            folds_predictions_by_actual_treatment = []
            for fold_prediction in folds_predictions:
                if fold_prediction.prediction_event_prob is not None:
                    fold_prediction = fold_prediction.prediction_event_prob
                else:
                    fold_prediction = fold_prediction.prediction
                fold_prediction_by_actual_treatment = robust_lookup(
                    fold_prediction, a[fold_prediction.index]
                )
                folds_predictions_by_actual_treatment.append(
                    fold_prediction_by_actual_treatment
                )

            return folds_predictions_by_actual_treatment, y, a

        elif plot_name in {"calibration"}:
            folds_predictions = [
                robust_lookup(
                    prediction.prediction_event_prob, a[prediction.prediction.index]
                )
                for prediction in folds_predictions
            ]
            return folds_predictions, y

        elif plot_name in {"roc_curve"}:
            folds_predictions = [
                robust_lookup(
                    prediction.prediction_event_prob, a[prediction.prediction.index]
                )
                for prediction in folds_predictions
            ]
            curve_data = self._calculate_curve_data(
                folds_predictions,
                y,
                metrics.roc_curve,
                metrics.roc_auc_score,
                stratify_by=a,
            )
            roc_curve_data = _calculate_roc_curve_data(curve_data)
            return (roc_curve_data,)

        elif plot_name in {"pr_curve"}:
            folds_predictions = [
                robust_lookup(
                    prediction.prediction_event_prob, a[prediction.prediction.index]
                )
                for prediction in folds_predictions
            ]
            curve_data = self._calculate_curve_data(
                folds_predictions,
                y,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
                stratify_by=a,
            )
            pr_curve_data = _calculate_pr_curve_data(curve_data, y)
            return (pr_curve_data,)

        elif plot_name in {"common_support"}:
            if is_vector_binary(y):
                folds_predictions = [
                    prediction.prediction_event_prob for prediction in folds_predictions
                ]
            else:
                folds_predictions = [
                    prediction.prediction for prediction in folds_predictions
                ]
            return folds_predictions, a

        else:
            return None

    def _calculate_curve_data(
        self, folds_predictions, targets, curve_metric, area_metric, stratify_by=None
    ):
        """Calculate different performance (ROC or PR) curves

        Args:
            folds_predictions (list[pd.Series]): Predictions for each fold.
            targets (pd.Series): True labels
            curve_metric (callable): Performance metric returning 3 output vectors - metric1, metric2 and thresholds.
                                    Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
            area_metric (callable): Performance metric of the area under the curve.
            stratify_by (pd.Series): Group assignment to stratify by.

        Returns:
            dict[str, dict[str, list[np.ndarray]]]: Evaluation of the metric for each fold and for each curve.
                One curve for each group level in `stratify_by`.
                On general: {curve_name: {metric1: [evaluation_fold_1, ...]}}.
                For example: {"Treatment=1": {"FPR": [FPR_fold_1, FPR_fold_2, FPR_fold_3]}}
        """
        # folds_targets = [targets.loc[fold_predictions.index] for fold_predictions in folds_predictions]
        # folds_stratify_by = [stratify_by.loc[fold_predictions.index] for fold_predictions in folds_predictions]

        stratify_values = np.sort(np.unique(stratify_by))
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
            ) = self._calculate_performance_curve_data_on_folds(
                folds_stratum_predictions,
                folds_stratum_targets,
                None,
                area_metric,
                curve_metric,
            )

            curve_data["Treatment={}".format(stratum_level)] = {
                "first_ret_value": first_ret_folds,
                "second_ret_value": second_ret_folds,
                "Thresholds": threshold_folds,
                "area": area_folds,
            }
        return curve_data
