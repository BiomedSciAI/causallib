from dataclasses import dataclass
import pandas as pd
from typing import Any, List, Tuple, Union, Dict
from .metrics import WeightEvaluatorScores, calculate_covariate_balance
from sklearn import metrics

from ..estimation.base_weight import PropensityEstimator, WeightEstimator
from ..estimation.base_estimator import IndividualOutcomeEstimator

from ..utils.stat_utils import is_vector_binary
from .plots.helpers import (
    calculate_performance_curve_data_on_folds,
    calculate_pr_curve,
    calculate_roc_curve,
)


def make_results(evaluated_metrics, predictions, cv, models):
    
    named_args={"evaluated_metrics":evaluated_metrics, "predictions":predictions,
    "cv":cv, "models":models}
    fitted_model = models["train"][0] if isinstance(models, dict) else models[0]
    if isinstance(fitted_model, PropensityEstimator):
        return PropensityEvaluationResults(**named_args)
    if isinstance(fitted_model, WeightEstimator):
        return WeightEvaluationResults(**named_args)
    if isinstance(fitted_model, IndividualOutcomeEstimator):
        return OutcomeEvaluationResults(**named_args)


@dataclass
class EvaluationResults:
    """Data structure to hold evaluation results.
    Args:
        evaluated_metrics (pd.DataFrame or WeightEvaluatorScores):
        models (list[WeightEstimator or IndividualOutcomeEstimator]): Models trained during evaluation
        predictions (dict[str, List[Predictions]])
        cv (list[tuple[list[int], list[int]]])
    """

    evaluated_metrics: Union[pd.DataFrame, Any]
    models: Union[List[WeightEstimator], List[IndividualOutcomeEstimator]]
    predictions: Dict[
        str, List[Any]
    ]  # really Any is one of the Predictions objects and key is "train" or "valid"
    cv: List[Tuple[List[int], List[int]]]


@dataclass
class WeightEvaluationResults(EvaluationResults):
    from .weight_evaluator import WeightEvaluatorPredictions

    evaluated_metrics: WeightEvaluatorScores
    models: List[WeightEstimator]
    predictions: Dict[str, List[WeightEvaluatorPredictions]]

    def _get_data_for_plot(self, plot_name, X, a, y, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots.py module.

        Args:
            plot_name (str): Plot name.
            folds_predictions (list[WeightEvaluatorPredictions]): Predictions for each fold.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment
            y: *IGNORED*

        Returns:
            tuple: Plot data
        """

        folds_predictions = self.predictions[phase]
        if plot_name in {"weight_distribution"}:
            return [p.weight_for_being_treated for p in folds_predictions], a

        elif plot_name in {"roc_curve"}:
            curve_data = self._calculate_curve_data(
                folds_predictions, a, metrics.roc_curve, metrics.roc_auc_score
            )
            roc_curve = calculate_roc_curve(curve_data)
            return (roc_curve,)
        elif plot_name in {"pr_curve"}:
            curve_data = self._calculate_curve_data(
                folds_predictions,
                a,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
            )
            pr_curve = calculate_pr_curve(curve_data, a)
            return (pr_curve,)
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
        self,
        folds_predictions: List[WeightEvaluatorPredictions],
        targets,
        curve_metric,
        area_metric,
        **kwargs
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
        from .plots.plotters import calculate_performance_curve_data_on_folds

        folds_treatment_weight = [p.weight_for_being_treated for p in folds_predictions]
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
            "unweighted": [None for _ in folds_predictions],
            "weighted": [p.weight_by_treatment_assignment for p in folds_predictions],
        }
        curve_data = {}
        for curve_name, sample_weights in folds_sample_weights.items():
            (
                area,
                first_ret_value,
                second_ret_value,
                threshold_folds,
            ) = calculate_performance_curve_data_on_folds(
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


@dataclass
class PropensityEvaluationResults(WeightEvaluationResults):
    from .weight_evaluator import PropensityEvaluatorPredictions

    evaluated_metrics: WeightEvaluatorScores
    models: List[PropensityEstimator]
    predictions: Dict[str, List[PropensityEvaluatorPredictions]]

    def _get_data_for_plot(self, plot_name, X, a, y, phase="train"):
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
        fold_predictions = self.predictions[phase]

        if plot_name in {"weight_distribution"}:
            return [p.propensity for p in fold_predictions], a

        if plot_name in {"calibration"}:
            return [p.propensity for p in fold_predictions], a

        # Common plots are implemented at top-most level possible.
        # Plot might be implemented by WeightEvaluator:
        return super(PropensityEvaluationResults, self)._get_data_for_plot(
            plot_name, X, a, y, phase=phase
        )

    def _calculate_curve_data(
        self,
        fold_predictions: List[PropensityEvaluatorPredictions],
        targets,
        curve_metric,
        area_metric,
        **kwargs
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
        from .plots.plotters import calculate_performance_curve_data_on_folds

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
        for curve_name in curves_sample_weights.keys():
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


@dataclass
class OutcomeEvaluationResults(EvaluationResults):
    from .outcome_evaluator import OutcomeEvaluatorPredictions

    evaluated_metrics: pd.DataFrame
    models: List[IndividualOutcomeEstimator]
    predictions: Dict[str, List[OutcomeEvaluatorPredictions]]

    def _get_data_for_plot(self, plot_name, X, a, y, phase="train"):
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
        fold_predictions = self.predictions[phase]
        if plot_name in {"continuous_accuracy", "residuals"}:
            return [x.get_prediction_by_treatment(a) for x in fold_predictions], y, a
        if plot_name in {"calibration"}:
            return [x.get_calibration(a) for x in fold_predictions], y
        if plot_name in {"roc_curve"}:
            fold_predictions = [x.get_calibration(a) for x in fold_predictions]
            curve_data = self._calculate_curve_data(
                fold_predictions,
                y,
                metrics.roc_curve,
                metrics.roc_auc_score,
                stratify_by=a,
            )
            roc_curve_data = calculate_roc_curve(curve_data)
            return (roc_curve_data,)

        elif plot_name in {"pr_curve"}:
            fold_predictions = [x.get_calibration(a) for x in fold_predictions]
            curve_data = self._calculate_curve_data(
                fold_predictions,
                y,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
                stratify_by=a,
            )
            pr_curve_data = calculate_pr_curve(curve_data, y)
            return (pr_curve_data,)

        elif plot_name in {"common_support"}:
            if is_vector_binary(y):
                fold_predictions = [
                    prediction.prediction_event_prob for prediction in fold_predictions
                ]
            else:
                fold_predictions = [
                    prediction.prediction for prediction in fold_predictions
                ]
            return fold_predictions, a

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

            curve_data["Treatment={}".format(stratum_level)] = {
                "first_ret_value": first_ret_folds,
                "second_ret_value": second_ret_folds,
                "Thresholds": threshold_folds,
                "area": area_folds,
            }
        return curve_data
