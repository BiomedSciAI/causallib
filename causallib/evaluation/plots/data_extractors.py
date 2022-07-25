"""Plot data extractors.

The responsibility of these classes is to extract the data from the
EvaluationResults objects to match the requested plot.
"""

import abc
from typing import List

import pandas as pd
from sklearn import metrics

from . import helpers, plots
from ...utils.stat_utils import is_vector_binary
from ..metrics import calculate_covariate_balance



class BaseEvaluationPlotDataExtractor(abc.ABC):
    """Extractor to get plot data from EvaluationResults.

    Subclasses also have a `plot_names` property.
    """

    def __init__(
        self, evaluation_results: "causallib.evaluation.results.EvaluationResults"
    ):
        self.predictions = evaluation_results.predictions
        self.X = evaluation_results.X
        self.a = evaluation_results.a
        self.y = evaluation_results.y
        self.cv = evaluation_results.cv

    def cv_by_phase(self, phase="train"):
        fold_idx = 0 if phase == "train" else 1
        return [fold[fold_idx] for fold in self.cv]

    @abc.abstractmethod
    def get_data_for_plot(self, plot_name, phase="train"):
        """Get data for plot with name `plot_name`."""
        raise NotImplementedError


class WeightPlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from WeightEvaluatorPredictions."""

    plot_names = plots.WeightPlotNames()

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.

        Plot functions are in plots module.

        Args:
            plot_name (str): Plot name.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment
            y: *IGNORED*

        Returns:
            tuple: Plot data
        """

        folds_predictions = self.predictions[phase]
        if plot_name in {self.plot_names.weight_distribution}:
            return (
                [p.weight_for_being_treated for p in folds_predictions],
                self.a,
                self.cv_by_phase(phase),
            )

        elif plot_name in {self.plot_names.roc_curve}:
            curve_data = self.calculate_curve_data(
                folds_predictions, self.a, metrics.roc_curve, metrics.roc_auc_score
            )
            roc_curve = helpers.calculate_roc_curve(curve_data)
            return (roc_curve,)
        elif plot_name in {"pr_curve"}:
            curve_data = self.calculate_curve_data(
                folds_predictions,
                self.a,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
            )
            pr_curve = helpers.calculate_pr_curve(curve_data, self.a)
            return (pr_curve,)
        elif plot_name in {
            self.plot_names.covariate_balance_love,
            self.plot_names.covariate_balance_slope,
            plots.COVARIATE_BALANCE_GENERIC_PLOT,
        }:
            distribution_distances = []
            for fold_prediction in folds_predictions:
                fold_w = fold_prediction.weight_by_treatment_assignment
                fold_X = self.X.loc[fold_w.index]
                fold_a = self.a.loc[fold_w.index]
                dist_dist = calculate_covariate_balance(fold_X, fold_a, fold_w)
                distribution_distances.append(dist_dist)
            return (distribution_distances,)
        else:
            return None

    @staticmethod
    def calculate_curve_data(
        fold_predictions: List[
            "causallib.evaluation.weight_predictor.WeightPredictions"
        ],
        targets,
        curve_metric,
        area_metric,
        **kwargs,
    ):
        """Calculate different performance (ROC or PR) curves

        Args:
            fold_predictions (list[WeightEvaluatorPredictions]): Predictions for each fold.
            targets (pd.Series): True labels
            curve_metric (callable): Performance metric returning 3 output vectors - metric1,
                metric2 and thresholds.
                Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
            area_metric (callable): Performance metric of the area under the curve.
            **kwargs:

        Returns:
            dict[str, dict[str, list[np.ndarray]]]: Evaluation of the metric
                for each fold and for each curve.
                2 curves:
                    * "unweighted" regular
                    * "weighted" weighted by weights of each sample (according to their assignment)
                On general: {curve_name: {metric1: [evaluation_fold_1, ...]}}.
                For example: {"weighted": {"FPR": [FPR_fold_1, FPR_fold_2, FPR_fold3]}}
        """

        folds_treatment_weight = [p.weight_for_being_treated for p in fold_predictions]
        folds_targets = []
        for fold_predictions in folds_treatment_weight:
            # Since this is weight estimator, which takes the inverse of a class prediction
            fold_targets = targets.loc[fold_predictions.index]
            min_target, max_target = fold_targets.min(), fold_targets.max()
            fold_targets = fold_targets.replace(
                {
                    min_target: max_target,
                    max_target: min_target,
                }
            )
            folds_targets.append(fold_targets)

        folds_sample_weights = {
            "unweighted": [None for _ in fold_predictions],
            "weighted": [p.weight_by_treatment_assignment for p in fold_predictions],
        }
        curve_data = {}
        for curve_name, sample_weights in folds_sample_weights.items():
            (
                area,
                first_ret_value,
                second_ret_value,
                threshold_folds,
            ) = helpers.calculate_performance_curve_data_on_folds(
                folds_treatment_weight,
                folds_targets,
                sample_weights,
                area_metric,
                curve_metric,
            )

            curve_data[curve_name] = {
                "first_ret_value": first_ret_value,
                "second_ret_value": second_ret_value,
                "Thresholds": threshold_folds,
                "area": area,
            }

        # Rename keys (as will be presented as curve labels in legend)
        curve_data["Weights"] = curve_data.pop("unweighted")
        curve_data["Weighted"] = curve_data.pop("weighted")
        return curve_data


class PropensityPlotDataExtractor(WeightPlotDataExtractor):
    """Extractor to get plot data from PropensityEvaluatorPredictions."""

    plot_names = plots.PropensityPlotNames()

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots.py module.

        Args:
            plot_name (str): Plot name.
            fold_predictions (list[PropensityEvaluatorPredictions]): Predictions for each fold.
            cv list[np.ndarray]: Indices (in iloc positions) of each fold.

        Returns:
            tuple: Plot data
        """
        fold_predictions = self.predictions[phase]

        if plot_name in {
            self.plot_names.weight_distribution,
            self.plot_names.calibration,
        }:
            return (
                [p.propensity for p in fold_predictions],
                self.a,
                self.cv_by_phase(phase),
            )

        # Common plots are implemented at top-most level possible.
        # Plot might be implemented by WeightEvaluator:
        return super().get_data_for_plot(plot_name, phase=phase)

    @staticmethod
    def calculate_curve_data(
        fold_predictions: List[
            "causallib.evaluation.weight_predictor.PropensityPredictions"
        ],
        targets,
        curve_metric,
        area_metric,
        **kwargs,
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
                fold_predictions.propensity.append(1 - fold_predictions.propensity)
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
                pd.Series(
                    data=targets.max(), index=fold_predictions.propensity.index
                ).append(
                    pd.Series(
                        data=targets.min(), index=fold_predictions.propensity.index
                    )
                )
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
                fold_predictions.propensity.append(fold_predictions.propensity)
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
            ) = helpers.calculate_performance_curve_data_on_folds(
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


class ContinuousOutcomePlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from OutcomeEvaluatorPredictions.

    Note that the available plots are different if the outcome predictions
    are binary/classification or continuous/regression.
    """

    plot_names = plots.ContinuousOutputPlotNames()

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots module.

        Args:
            plot_name (str): Plot name.

        Returns:
            tuple: Plot data
        """
        fold_predictions = self.predictions[phase]
        if plot_name in {
            self.plot_names.continuous_accuracy,
            self.plot_names.residuals,
        }:
            return (
                [x.get_prediction_by_treatment(self.a) for x in fold_predictions],
                self.y,
                self.a,
                self.cv_by_phase(phase),
            )
        if plot_name in {self.plot_names.common_support}:
            if is_vector_binary(self.y):
                return [p.prediction_event_prob for p in fold_predictions], self.a
            else:
                return (
                    [p.prediction for p in fold_predictions],
                    self.a,
                    self.cv_by_phase(phase),
                )

        raise ValueError(f"Received unsupported plot name {plot_name}!")


class BinaryOutcomePlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from OutcomeEvaluatorPredictions.

    Note that the available plots are different if the outcome predictions
    are binary/classification or continuous/regression.
    """

    plot_names = plots.BinaryOutputPlotNames()

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots module.

        Args:
            plot_name (str): Plot name.

        Returns:
            tuple: Plot data
        """
        fold_predictions = self.predictions[phase]

        if plot_name in {self.plot_names.calibration}:
            return [x.get_proba_by_treatment(self.a) for x in fold_predictions], self.y, self.cv_by_phase(phase)
        if plot_name in {self.plot_names.roc_curve}:
            proba_list = [x.get_proba_by_treatment(self.a) for x in fold_predictions]
            curve_data = helpers.calculate_curve_data_binary(
                proba_list,
                self.y,
                metrics.roc_curve,
                metrics.roc_auc_score,
                stratify_by=self.a,
            )
            roc_curve_data = helpers.calculate_roc_curve(curve_data)
            return (roc_curve_data,)

        if plot_name in {self.plot_names.pr_curve}:
            proba_list = [x.get_proba_by_treatment(self.a) for x in fold_predictions]
            curve_data = helpers.calculate_curve_data_binary(
                proba_list,
                self.y,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
                stratify_by=self.a,
            )
            pr_curve_data = helpers.calculate_pr_curve(curve_data, self.y)
            return (pr_curve_data,)

        raise ValueError(f"Received unsupported plot name {plot_name}!")

