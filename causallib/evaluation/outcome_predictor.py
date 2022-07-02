"""
(C) Copyright 2019 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Dec 25, 2018

"""
import warnings

from .predictor import BasePredictor
from ..estimation.base_estimator import IndividualOutcomeEstimator
from ..utils.stat_utils import robust_lookup
from .metrics import evaluate_binary_metrics, evaluate_regression_metrics

import pandas as pd


class OutcomeEvaluatorPredictions:
    """Data structure to hold outcome-model predictions"""

    def __init__(self, prediction, prediction_event_prob=None):
        self.prediction = prediction
        self.prediction_event_prob = self._correct_predict_proba_estimate(prediction, prediction_event_prob)
        self.is_binary_outcome = self.prediction_event_prob is not None

    @staticmethod
    def _correct_predict_proba_estimate(prediction, prediction_event_prob):
        # Estimation output for predict_proba=True has same columns as for predict_proba=False.
        # This means either base-learner has no predict_proba/decision_function or problem is not classification.
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
        
    def calculate_metrics(self, a, y, metrics_to_evaluate):

        scores = {"actual": self.get_overall_score(a, y, metrics_to_evaluate)}

        scores.update(
            {
                str(t): self.get_treatment_value_score(a, y, metrics_to_evaluate, t)
                for t in sorted(set(a))
            }
        )

        scores = pd.concat(scores, names=["model_strata"], axis="columns").T
        scores = scores.apply(pd.to_numeric, errors="ignore")
        return scores

    def get_treatment_value_score(
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

    def get_overall_score(self, a_true, y_true, metrics_to_evaluate):
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
        # score = score.apply(pd.to_numeric, errors="ignore")  # change dtype of each column to numerical if possible.
        return score

    def get_prediction_by_treatment(self, a):
        if self.is_binary_outcome:
            pred = self.prediction_event_prob
        else:
            pred = self.prediction
        return robust_lookup(pred, a[pred.index])

    def get_calibration(self, a):
        return robust_lookup(self.prediction_event_prob, a[self.prediction.index])


class OutcomePredictor(BasePredictor):
    def __init__(self, estimator):
        """
        Args:
            estimator (IndividualOutcomeEstimator):
        """
        if not isinstance(estimator, IndividualOutcomeEstimator):
            raise TypeError(
                "OutcomeEvaluator should be initialized with IndividualOutcomeEstimator, got ({}) instead.".format(
                    type(estimator)
                )
            )
        self.estimator = estimator

    def _estimator_fit(self, X, a, y):
        """Fit estimator."""
        self.estimator.fit(X=X, a=a, y=y)

    def _estimator_predict(self, X, a):
        """Predict on data."""
        prediction = self.estimator.estimate_individual_outcome(
            X, a, predict_proba=False
        )
        # Use predict_probability if possible since it is needed for most evaluations:
        prediction_event_prob = self.estimator.estimate_individual_outcome(
            X, a, predict_proba=True
        )
        fold_prediction = OutcomeEvaluatorPredictions(prediction, prediction_event_prob)
        return fold_prediction


