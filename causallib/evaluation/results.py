"""Evaluation results objects for plotting and further analysis."""

import abc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn import metrics

from ..estimation.base_estimator import IndividualOutcomeEstimator
from ..estimation.base_weight import PropensityEstimator, WeightEstimator
from ..utils.stat_utils import is_vector_binary
from .metrics import calculate_covariate_balance
from .outcome_predictor import OutcomeEvaluatorPredictions
from .plots.helpers import (
    calculate_performance_curve_data_on_folds,
    calculate_pr_curve,
    calculate_roc_curve,
)
from .weight_predictor import (
    PropensityEvaluatorPredictions,
    WeightEvaluatorPredictions,
    WeightEvaluatorScores,
)

SingleFoldPrediction = Union[
    PropensityEvaluatorPredictions,
    WeightEvaluatorPredictions,
    OutcomeEvaluatorPredictions,
]


@dataclass
class EvaluationResults:
    """Data structure to hold evaluation results including cross-validation.

    Attrs:
        evaluated_metrics (pd.DataFrame or WeightEvaluatorScores):
        models (dict[str, Union[list[WeightEstimator], list[IndividualOutcomeEstimator]):
            Models trained during evaluation. May be dict or list or a model
            directly.
        predictions (dict[str, List[SingleFoldPredictions]]): dict with keys
            "train" and "valid" (if produced through cross-validation) and
            values of the predictions for the respective fold
        cv (list[tuple[list[int], list[int]]]): the cross validation indices,
            used to generate the results, used for constructing plots correctly
    """

    evaluated_metrics: Union[pd.DataFrame, WeightEvaluatorScores]
    models: Union[
        List[WeightEstimator],
        List[IndividualOutcomeEstimator],
        List[PropensityEstimator],
    ]
    predictions: Dict[str, List[SingleFoldPrediction]]
    cv: List[Tuple[List[int], List[int]]]

    @property
    def extractor(self):
        """Plot data extractor for these results.

        Instantiated when requested based on type of `models`.
        """
        if isinstance(self.models, dict):
            fitted_model = self.models["train"][0]
        elif isinstance(self.models, list):
            fitted_model = self.models[0]
        else:
            fitted_model = self.models

        if isinstance(fitted_model, PropensityEstimator):
            return PropensityPlotDataExtractor(self)
        if isinstance(fitted_model, WeightEstimator):
            return WeightPlotDataExtractor(self)
        if isinstance(fitted_model, IndividualOutcomeEstimator):
            return OutcomePlotDataExtractor(self)
        raise ValueError(
            "Unable to find suitable extractor "
            f"for esimator of type {type(fitted_model)}"
        )

    @property
    def available_plot_names(self):
        """Available plot names.

        Returns:
            set[str]: string names of supported plot names for these results
        """
        return self.extractor.available_plot_names

    def get_data_for_plot(self, plot_name, X, a, y, phase="train"):
        """Get data for a given plot

        Args:
            plot_name (str): plot name from `self.available_plot_names`
            X (pd.DataFrame): feature data
            a (pd.Series): treatment assignment vector
            y (pd.Series): outcome data
            phase (str, optional): phase of interest. Defaults to "train".

        Returns:
            Any: the data required for the plot in question
        """
        return self.extractor.get_data_for_plot(plot_name, X, a, y, phase)


class BaseEvaluationPlotDataExtractor(abc.ABC):
    """Extractor to get plot data from EvaluationResults."""

    available_plot_names = set()

    def __init__(self, evaluation_results: EvaluationResults):
        self.predictions = evaluation_results.predictions


    @abc.abstractmethod
    def get_data_for_plot(self, plot_name, X, a, y, phase="train"):
        """Get data for plot with name `plot_name`."""
        raise NotImplementedError


    @abc.abstractmethod
    def calculate_curve_data(
        self,
        fold_predictions: List[SingleFoldPrediction],
        targets,
        curve_metric,
        area_metric,
    ):
        """Calculate metrics to generate curve data for given list of predictions."""
        raise NotImplementedError


class WeightPlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from WeightEvaluatorPredictions."""

    available_plot_names = {
        "weight_distribution",
        "roc_curve",
        "pr_curve",
        "covariate_balance_love",
        "covariate_balance_slope",
    }

    def get_data_for_plot(self, plot_name, X, a, y, phase="train"):
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
        if plot_name in {"weight_distribution"}:
            return [p.weight_for_being_treated for p in folds_predictions], a

        elif plot_name in {"roc_curve"}:
            curve_data = self.calculate_curve_data(
                folds_predictions, a, metrics.roc_curve, metrics.roc_auc_score
            )
            roc_curve = calculate_roc_curve(curve_data)
            return (roc_curve,)
        elif plot_name in {"pr_curve"}:
            curve_data = self.calculate_curve_data(
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

    def calculate_curve_data(
        self,
        fold_predictions: List[WeightEvaluatorPredictions],
        targets,
        curve_metric,
        area_metric,
        **kwargs,
    ):
        """Calculate different performance (ROC or PR) curves

        Args:
            folds_predictions (list[WeightEvaluatorPredictions]): Predictions for each fold.
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


class PropensityPlotDataExtractor(WeightPlotDataExtractor):
    """Extractor to get plot data from PropensityEvaluatorPredictions."""

    available_plot_names = WeightPlotDataExtractor.available_plot_names | {
        "weight_distribution",
        "calibration",
    }

    def get_data_for_plot(self, plot_name, X, a, y, phase="train"):
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
        return super().get_data_for_plot(plot_name, X, a, y, phase=phase)

    def calculate_curve_data(
        self,
        fold_predictions: List[PropensityEvaluatorPredictions],
        targets,
        curve_metric,
        area_metric,
        **kwargs,
    ):
        """Calculate different performance (ROC or PR) curves

        Args:
            curves_folds_predictions (list[PropensityEvaluatorPredictions]):
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


class OutcomePlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from OutcomeEvaluatorPredictions.

    Note that the available plots are different if the outcome predictions
    are binary/classification or continuous/regression.
    """

    continuous_output_plot_names = {
        "continuous_accuracy",
        "residuals",
        "common_support",
    }
    binary_output_plot_names = {
        "calibration",
        "roc_curve",
        "pr_curve",
    }

    def __init__(self, evaluation_results):
        super().__init__(evaluation_results)
        if any(
            x and any(y.is_binary_outcome for y in x) for x in self.predictions.values()
        ):
            self.available_plot_names = self.binary_output_plot_names
        else:
            self.available_plot_names = self.continuous_output_plot_names

    def get_data_for_plot(self, plot_name, X, a, y, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots module.

        Args:
            plot_name (str): Plot name.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment
            y: *IGNORED*

        Returns:
            tuple: Plot data
        """
        fold_predictions = self.predictions[phase]
        if plot_name in {"continuous_accuracy", "residuals"}:
            return [x.get_prediction_by_treatment(a) for x in fold_predictions], y, a
        if plot_name in {"calibration"}:
            return [x.get_proba_by_treatment(a) for x in fold_predictions], y
        if plot_name in {"roc_curve"}:
            proba_list = [x.get_proba_by_treatment(a) for x in fold_predictions]
            curve_data = self.calculate_curve_data(
                proba_list,
                y,
                metrics.roc_curve,
                metrics.roc_auc_score,
                stratify_by=a,
            )
            roc_curve_data = calculate_roc_curve(curve_data)
            return (roc_curve_data,)

        elif plot_name in {"pr_curve"}:
            proba_list = [x.get_proba_by_treatment(a) for x in fold_predictions]
            curve_data = self.calculate_curve_data(
                proba_list,
                y,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
                stratify_by=a,
            )
            pr_curve_data = calculate_pr_curve(curve_data, y)
            return (pr_curve_data,)

        elif plot_name in {"common_support"}:
            if is_vector_binary(y):
                return [p.prediction_event_prob for p in fold_predictions], a
            else:
                return [p.prediction for p in fold_predictions], a
        else:
            return None

    def calculate_curve_data(
        self, folds_predictions, targets, curve_metric, area_metric, stratify_by=None
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
