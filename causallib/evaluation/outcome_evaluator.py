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

import numpy as np
import pandas as pd

from .evaluator import BaseEvaluator
from ..estimation.base_estimator import IndividualOutcomeEstimator
from ..utils.stat_utils import is_vector_binary
from ..utils.stat_utils import robust_lookup


class OutcomeEvaluatorPredictions:
    """Data structure to hold outcome-model predictions"""

    def __init__(self, prediction, prediction_event_prob=None):
        self.prediction = prediction
        self.prediction_event_prob = prediction_event_prob


class OutcomeEvaluator(BaseEvaluator):
    def __init__(self, estimator):
        """
        Args:
            estimator (IndividualOutcomeEstimator):
        """
        if not isinstance(estimator, IndividualOutcomeEstimator):
            raise TypeError("OutcomeEvaluator should be initialized with IndividualOutcomeEstimator, got ({}) instead."
                            .format(type(estimator)))
        super(OutcomeEvaluator, self).__init__(estimator)


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

    def score_estimation(self, prediction, X, a_true, y_true, metrics_to_evaluate=None):
        """Scores a prediction against true labels.

        Args:
            prediction (OutcomeEvaluatorPredictions): Prediction on the data.
            X (pd.DataFrame): Covariates.
            a_true (pd.Series): Stratify by - treatment assignment
            y_true: Target variable - outcome.
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            pd.DataFrame: metric names (keys of `metrics_to_evaluate`) as columns and rows are stratification based
                          on `a_true`, plus unstratified results.
                          For example, binary treatment assignment weill results in three rows: scoring y^0 prediction
                          on the set {y_i | a_i = 0}, y^1 on the set {y_i | a_i = 1} and non-stratified using the
                          factual treatment assignment: y^(a_i) on `y_true`.
        """
        y_is_binary = y_true.nunique() == 2
        treatment_values = np.sort(np.unique(a_true))
        scores = {}
        for treatment_value in treatment_values:
            # Stratify based on treatment assignment:
            current_strata_mask = a_true == treatment_value
            y_true_strata = y_true.loc[current_strata_mask]
            prediction_strata = prediction.prediction.loc[current_strata_mask, treatment_value]
            if y_is_binary:
                prediction_prob_strata = prediction.prediction_event_prob.loc[current_strata_mask, treatment_value]
            else:
                prediction_prob_strata = None

            score = self._score_single(y_true_strata, prediction_strata, prediction_prob_strata,
                                       y_is_binary, metrics_to_evaluate)

            scores[str(treatment_value)] = score

        # Score overall:
        # # Extract prediction on actual treatment
        prediction_strata = robust_lookup(prediction.prediction, a_true)
        if y_is_binary:
            prediction_prob_strata = robust_lookup(prediction.prediction_event_prob, a_true)
        else:
            prediction_prob_strata = None
        score = self._score_single(y_true, prediction_strata, prediction_prob_strata,
                                   y_is_binary, metrics_to_evaluate)
        scores["actual"] = score

        scores = pd.concat(scores, names=["model_strata"], axis="columns").T
        scores = scores.apply(pd.to_numeric, errors="ignore")  # change dtype of each column to numerical if possible.
        return scores

    def _score_single(self, y_true, prediction, prediction_prob, outcome_is_binary, metrics_to_evaluate):
        """Score a single prediction based on whether `y_true` is classification or regression"""
        if outcome_is_binary:
            score = self.score_binary_prediction(
                y_true=y_true,
                y_pred=prediction,
                y_pred_proba=prediction_prob,
                metrics_to_evaluate=metrics_to_evaluate
            )
        else:
            score = self.score_regression_prediction(
                y_true=y_true,
                y_pred=prediction,
                metrics_to_evaluate=metrics_to_evaluate
            )
        # score = pd.DataFrame(score).T
        # score = score.apply(pd.to_numeric, errors="ignore")  # change dtype of each column to numerical if possible.
        return score

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
                fold_prediction_by_actual_treatment = robust_lookup(fold_prediction, a[fold_prediction.index])
                folds_predictions_by_actual_treatment.append(fold_prediction_by_actual_treatment)

            return folds_predictions_by_actual_treatment, y, a

        elif plot_name in {'calibration'}:
            folds_predictions = [robust_lookup(prediction.prediction_event_prob, a[prediction.prediction.index])
                                 for prediction in folds_predictions]
            return folds_predictions, y

        elif plot_name in {'roc_curve'}:
            folds_predictions = [robust_lookup(prediction.prediction_event_prob, a[prediction.prediction.index])
                                 for prediction in folds_predictions]
            curve_data = self._calculate_roc_curve_data(folds_predictions, y, a)
            return (curve_data,)

        elif plot_name in {'pr_curve'}:
            folds_predictions = [robust_lookup(prediction.prediction_event_prob, a[prediction.prediction.index])
                                 for prediction in folds_predictions]
            curve_data = self._calculate_pr_curve_data(folds_predictions, y, a)
            return (curve_data,)

        elif plot_name in {'common_support'}:
            if is_vector_binary(y):
                folds_predictions = [prediction.prediction_event_prob for prediction in folds_predictions]
            else:
                folds_predictions = [prediction.prediction for prediction in folds_predictions]
            return folds_predictions, a

        else:
            return None

    def _calculate_curve_data(self, folds_predictions, targets, curve_metric, area_metric, stratify_by=None):
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

            area_folds, first_ret_folds, second_ret_folds, threshold_folds = \
                self._calculate_performance_curve_data_on_folds(folds_stratum_predictions, folds_stratum_targets, None,
                                                                area_metric, curve_metric)

            curve_data["Treatment={}".format(stratum_level)] = {"first_ret_value": first_ret_folds,
                                                                "second_ret_value": second_ret_folds,
                                                                "Thresholds": threshold_folds, "area": area_folds}
        return curve_data
