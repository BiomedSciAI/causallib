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
from typing import List
import warnings

from .evaluator import Predictor
from ..estimation.base_estimator import IndividualOutcomeEstimator



class OutcomeEvaluatorPredictions:
    """Data structure to hold outcome-model predictions"""

    def __init__(self, prediction, prediction_event_prob=None):
        self.prediction = prediction
        self.prediction_event_prob = prediction_event_prob


class OutcomePredictor(Predictor):
    def __init__(self, estimator):
        """
        Args:
            estimator (IndividualOutcomeEstimator):
        """
        if not isinstance(estimator, IndividualOutcomeEstimator):
            raise TypeError("OutcomeEvaluator should be initialized with IndividualOutcomeEstimator, got ({}) instead."
                            .format(type(estimator)))
        self.estimator = estimator

    def _estimator_fit(self, X, a, y):
        """Fit estimator."""
        self.estimator.fit(X=X, a=a, y=y)

    def _estimator_predict(self, X, a):
        """Predict on data."""
        prediction = self.estimator.estimate_individual_outcome(X, a, predict_proba=False)
        # Use predict_probability if possible since it is needed for most evaluations:
        prediction_event_prob = self.estimator.estimate_individual_outcome(X, a, predict_proba=True)

        if prediction_event_prob.columns.tolist() == prediction.columns.tolist():
            # Estimation output for predict_proba=True has same columns as for predict_proba=False.
            # This means either base-learner has no predict_proba/decision_function or problem is not classification.
            # Either way, it means there are no prediction probabilities
            prediction_event_prob = None
        else:
            # predict_proba=True was able to predict probabilities. However,
            # Prediction probability evaluation is only applicable for binary outcome:
            y_values = prediction_event_prob.columns.get_level_values("y").unique()
            #   # Note: on pandas 23.0.0 you could do prediction_event_prob.columns.unique(level='y')
            if y_values.size == 2:
                event_value = y_values.max()  # get the maximal value, assumes binary 0-1 (1: event, 0: non-event)
                # Extract the probability for event:
                prediction_event_prob = prediction_event_prob.xs(key=event_value, axis="columns", level="y")
            else:
                warnings.warn("Multiclass probabilities are not well defined  and supported for evaluation.\n"
                              "Falling back to class predictions.\n"
                              "Plots might be uninformative due to input being classes and not probabilities.")
                prediction_event_prob = None

        fold_prediction = OutcomeEvaluatorPredictions(prediction, prediction_event_prob)
        return fold_prediction

