"""Predictions from single folds.

Predictions are generated by predictors for causal models. They contain the estimates
for single folds and are combined in the EvaluationResults objects for further analysis.
"""
from collections import namedtuple
from typing import Union
import warnings

import pandas as pd
from ..utils.stat_utils import robust_lookup
from .metrics import (
    calculate_covariate_balance,
    evaluate_binary_metrics,
    evaluate_regression_metrics,
)

WeightEvaluatorScores = namedtuple(
    "WeightEvaluatorScores", ["prediction_scores", "covariate_balance"]
)


class WeightPredictions:
    """Data structure to hold weight-model predictions"""

    def __init__(
        self,
        weight_by_treatment_assignment,
        weight_for_being_treated,
        treatment_assignment_pred,
    ):
        self.weight_by_treatment_assignment = weight_by_treatment_assignment
        self.weight_for_being_treated = weight_for_being_treated
        self.treatment_assignment_pred = treatment_assignment_pred

    def evaluate_metrics(self, X, a_true, metrics_to_evaluate):
        """
        Evaluate metrics on prediction.

        Args:
            X (pd.DataFrame): Covariates.
            a_true (pd.Series): ground truth treatment assignment
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives
                true labels, prediction and sample_weights (the latter may be ignored).
                If not provided, default values from causallib.evaluation.metrics are used.

        Returns:
            WeightEvaluatorScores: Object with two data attributes: "predictions"
                and "covariate_balance"
        """

        prediction_scores = evaluate_binary_metrics(
            y_true=a_true,
            y_pred_proba=self.weight_for_being_treated,
            y_pred=self.treatment_assignment_pred,
            metrics_to_evaluate=metrics_to_evaluate,
        )
        # Convert single-dtype Series to a row in a DataFrame:
        prediction_scores = pd.DataFrame(prediction_scores).T
        # change dtype of each column to numerical if possible:
        prediction_scores = prediction_scores.apply(pd.to_numeric, errors="ignore")

        covariate_balance = calculate_covariate_balance(
            X, a_true, self.weight_by_treatment_assignment
        )

        results = WeightEvaluatorScores(prediction_scores, covariate_balance)
        return results


class PropensityPredictions(WeightPredictions):
    """Data structure to hold propensity-model predictions"""

    def __init__(
        self,
        weight_by_treatment_assignment,
        weight_for_being_treated,
        treatment_assignment_pred,
        propensity,
        propensity_by_treatment_assignment,
    ):
        super().__init__(
            weight_by_treatment_assignment,
            weight_for_being_treated,
            treatment_assignment_pred,
        )
        self.propensity = propensity
        self.propensity_by_treatment_assignment = propensity_by_treatment_assignment


class OutcomePredictions:
    """Data structure to hold outcome-model predictions"""

    def __init__(self, prediction, prediction_event_prob=None):
        self.prediction = prediction
        self.prediction_event_prob = self._correct_predict_proba_estimate(
            prediction, prediction_event_prob
        )
        self.is_binary_outcome = self.prediction_event_prob is not None

    @staticmethod
    def _correct_predict_proba_estimate(prediction, prediction_event_prob):
        # Estimation output for predict_proba=True has same columns as for predict_proba=False.
        # This means either base-learner has no predict_proba/decision_function
        # or problem is not classification.
        # Either way, it means there are no prediction probabilities
        if prediction_event_prob.columns.tolist() == prediction.columns.tolist():
            return None

        # predict_proba=True was able to predict probabilities. However,
        # Prediction probability evaluation is only applicable for binary outcome:
        y_values = prediction_event_prob.columns.get_level_values("y").unique()
        # Note: on pandas 23.0.0 you could do prediction_event_prob.columns.unique(level='y')
        if y_values.size == 2:
            event_value = y_values.max()
            # get the maximal value, assumes binary 0-1 (1: event, 0: non-event)
            # Extract the probability for event:
            return prediction_event_prob.xs(key=event_value, axis="columns", level="y")

        warnings.warn(
            "Multiclass probabilities are not well defined and supported for evaluation.\n"
            "Falling back to class predictions.\n"
            "Plots might be uninformative due to input being classes and not probabilities."
        )
        return None

    def evaluate_metrics(self, a, y, metrics_to_evaluate):
        """Evaluate metrics for this model prediction.

        Args:
            a (pd.Series): treatment assignment
            y (pd.Series): ground truth outcomes
            metrics_to_evaluate (Dict[str,Callable]): key: metric's name, value: callable that
                receives true labels, prediction and sample_weights (the latter may be ignored).
                If not provided, defaults from causallib.evaluation.metrics are used.

        Returns:
            pd.DataFrame: evaluated metrics
        """

        scores = {"actual": self._evaluate_metrics_overall(a, y, metrics_to_evaluate)}

        scores.update(
            {
                str(t): self._evaluate_metrics_on_treatment_value(
                    a, y, metrics_to_evaluate, t
                )
                for t in sorted(set(a))
            }
        )

        scores = pd.concat(scores, names=["model_strata"], axis="columns").T
        scores = scores.apply(pd.to_numeric, errors="ignore")
        return scores

    def _evaluate_metrics_on_treatment_value(
        self, a_true, y_true, metrics_to_evaluate, treatment_value
    ):
        # Stratify based on treatment assignment:
        y_is_binary = y_true.nunique() == 2
        treatment_value_idx = a_true == treatment_value
        y_true_strata = y_true.loc[treatment_value_idx]
        prediction_strata = self.prediction.loc[treatment_value_idx, treatment_value]
        if y_is_binary:
            prediction_prob_strata = self.prediction_event_prob.loc[
                treatment_value_idx, treatment_value
            ]
        else:
            prediction_prob_strata = None

        score = self._score_single(
            y_true_strata,
            prediction_strata,
            prediction_prob_strata,
            y_is_binary,
            metrics_to_evaluate,
        )

        return score

    def _evaluate_metrics_overall(self, a_true, y_true, metrics_to_evaluate):
        # Score overall:
        # # Extract prediction on actual treatment
        y_is_binary = y_true.nunique() == 2
        prediction_strata = robust_lookup(self.prediction, a_true)
        if y_is_binary:
            prediction_prob_strata = robust_lookup(self.prediction_event_prob, a_true)
        else:
            prediction_prob_strata = None
        score = self._score_single(
            y_true,
            prediction_strata,
            prediction_prob_strata,
            y_is_binary,
            metrics_to_evaluate,
        )

        return score

    @staticmethod
    def _score_single(
        y_true,
        prediction,
        prediction_prob,
        outcome_is_binary,
        metrics_to_evaluate,
    ):
        """Score a single prediction based on whether `y_true` is classification or regression"""
        if outcome_is_binary:
            score = evaluate_binary_metrics(
                y_true=y_true,
                y_pred=prediction,
                y_pred_proba=prediction_prob,
                metrics_to_evaluate=metrics_to_evaluate,
            )
        else:
            score = evaluate_regression_metrics(
                y_true=y_true,
                y_pred=prediction,
                metrics_to_evaluate=metrics_to_evaluate,
            )
        # score = pd.DataFrame(score).T
        # score = score.apply(pd.to_numeric, errors="ignore")
        # change dtype of each column to numerical if possible.
        return score

    def get_prediction_by_treatment(self, a: pd.Series):
        """Get proba if available else prediction"""
        if self.is_binary_outcome:
            pred = self.prediction_event_prob
        else:
            pred = self.prediction
        return robust_lookup(pred, a[pred.index])

    def get_proba_by_treatment(self, a: pd.Series):
        """Get proba of prediction"""
        return robust_lookup(self.prediction_event_prob, a[self.prediction.index])


SingleFoldPrediction = Union[
    PropensityPredictions, WeightPredictions, OutcomePredictions
]
