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
from collections import namedtuple

import numpy as np
import pandas as pd

from .evaluator import BaseEvaluator
from ..estimation.base_weight import WeightEstimator, PropensityEstimator
from ..utils.stat_utils import calc_weighted_standardized_mean_differences, calc_weighted_ks2samp, robust_lookup


# TODO: decide what implementation stays - the one with the '2' suffix or the one without.
#       The one with is based on matrix input and does all the vector extraction by itself.
#       The one without is simpler one the receive the vectors already (more general, as not all models may have matrix.


# ################ #
# Weight Evaluator #
# ################ #

class WeightEvaluatorPredictions:
    """Data structure to hold weight-model predictions"""

    def __init__(self, weight_by_treatment_assignment, weight_for_being_treated, treatment_assignment_pred):
        self.weight_by_treatment_assignment = weight_by_treatment_assignment
        self.weight_for_being_treated = weight_for_being_treated
        self.treatment_assignment_pred = treatment_assignment_pred


class WeightEvaluatorPredictions2:
    """Data structure to hold weight-model predictions"""

    def __init__(self, weight_matrix, treatment_assignment, treatment_assignment_prediction=None):
        self.weight_matrix = weight_matrix
        self._treatment_assignment = treatment_assignment
        self.treatment_assignment_prediction = treatment_assignment_prediction

    @property
    def weight_by_treatment_assignment(self):
        weight_by_treatment_assignment = self._extract_vector_from_matrix(self.weight_matrix,
                                                                          self._treatment_assignment)
        return weight_by_treatment_assignment

    @property
    def weight_for_being_treated(self):
        weight_for_being_treated = self._extract_vector_from_matrix(self.weight_matrix,
                                                                    self._treatment_assignment.max())
        return weight_for_being_treated

    @staticmethod
    def _extract_vector_from_matrix(matrix, value):
        if np.isscalar(value):
            vector = matrix[value]
        else:
            # vector = robust_lookup(matrix, value)
            vector = matrix.lookup(value.index, value)
            vector = pd.Series(vector, index=value.index)
        return vector


WeightEvaluatorScores = namedtuple("WeightEvaluatorScores", ["prediction_scores", "covariate_balance"])


class WeightEvaluator(BaseEvaluator):
    def __init__(self, estimator):
        """
        Args:
            estimator (WeightEstimator):
        """
        if not isinstance(estimator, WeightEstimator):
            raise TypeError("WeightEvaluator should be initialized with WeightEstimator, got ({}) instead."
                            .format(type(estimator)))
        super(WeightEvaluator, self).__init__(estimator)


    def _estimator_fit(self, X, a, y=None):
        """ Fit estimator. `y` is ignored.
        """
        self.estimator.fit(X=X, a=a)

    def _estimator_predict(self, X, a):
        """Predict on data.

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment

        Returns:
            WeightEvaluatorPredictions
        """
        weight_by_treatment_assignment = self.estimator.compute_weights(X, a, treatment_values=None,
                                                                        use_stabilized=False)
        weight_for_being_treated = self.estimator.compute_weights(X, a, treatment_values=a.max(),
                                                                  use_stabilized=False)
        treatment_assignment_pred = self.estimator.learner.predict(
            X)  # TODO: maybe add predict_label to interface instead
        treatment_assignment_pred = pd.Series(treatment_assignment_pred, index=X.index)

        prediction = WeightEvaluatorPredictions(weight_by_treatment_assignment,
                                                weight_for_being_treated,
                                                treatment_assignment_pred)
        return prediction

    def _estimator_predict2(self, X, a):
        """Predict on data"""
        weight_matrix = self.estimator.compute_weight_matrix(X, a, use_stabilized=False)
        treatment_assignment_pred = self.estimator.learner.predict(
            X)  # TODO: maybe add predict_label to interface instead
        treatment_assignment_pred = pd.Series(treatment_assignment_pred, index=X.index)

        fold_prediction = WeightEvaluatorPredictions2(weight_matrix, a, treatment_assignment_pred)
        return fold_prediction

    def score_estimation(self, prediction, X, a_true, y_true=None, metrics_to_evaluate=None):
        """Scores a prediction against true labels.

        Args:
            prediction (WeightEvaluatorPredictions): Prediction on the data.
            X (pd.DataFrame): Covariates.
            a_true (pd.Series): Target variable - treatment assignment
            y_true: *IGNORED*
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            WeightEvaluatorScores: Data-structure holding scores on the predictions
                                   and covariate balancing table ("table 1")
        """
        results = self._score_estimation(X, a_true,
                                         prediction.weight_for_being_treated,
                                         prediction.treatment_assignment_pred,
                                         prediction.weight_by_treatment_assignment,
                                         metrics_to_evaluate)
        return results

    def _score_estimation(self, X, targets, predict_scores, predict_assignment, predict_weights,
                          metrics_to_evaluate=None):
        """

        Args:
            X (pd.DataFrame): Covariates.
            targets (pd.Series): Target variable - true treatment assignment
            predict_scores (pd.Series): Continuous prediction of the treatment assignment
                                        (as is `predict_proba` or `decision_function`).
            predict_assignment (pd.Series): Class prediction of the treatment assignment
                                            (i.e. prediction of the assignment it self).
            predict_weights (pd.Series): The weights derived to balance between the treatment groups
                                         (here, called `targets`).
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            WeightEvaluatorScores: Data-structure holding scores on the predictions
                                   and covariate balancing table ("table 1")
        """
        prediction_scores = self.score_binary_prediction(y_true=targets,
                                                         y_pred_proba=predict_scores,
                                                         y_pred=predict_assignment,
                                                         metrics_to_evaluate=metrics_to_evaluate)
        # Convert single-dtype Series to a row in a DataFrame:
        prediction_scores = pd.DataFrame(prediction_scores).T
        # change dtype of each column to numerical if possible:
        prediction_scores = prediction_scores.apply(pd.to_numeric, errors="ignore")

        covariate_balance = calculate_covariate_balance(X, targets, predict_weights)

        results = WeightEvaluatorScores(prediction_scores, covariate_balance)
        return results

    def _combine_fold_scores(self, scores):
        # `scores` are provided as WeightEvaluatorScores object for each fold in each phase,
        # Namely, dict[list[WeightEvaluatorScores]], which in turn hold two DataFrames components.
        # In order to combine the underlying DataFrames into a multilevel DataFrame, one must first extract them from
        # the WeightEvaluatorScores object, into two separate components.

        # Extract the two components of WeightEvaluatorScores:
        prediction_scores = {phase: [fold_score.prediction_scores for fold_score in phase_scores]
                             for phase, phase_scores in scores.items()}
        covariate_balance = {phase: [fold_score.covariate_balance for fold_score in phase_scores]
                             for phase, phase_scores in scores.items()}

        # Combine the dict[list[DataFrames]] of each component into a multilevel DataFrame separately:
        prediction_scores = super(WeightEvaluator, self)._combine_fold_scores(prediction_scores)
        covariate_balance = super(WeightEvaluator, self)._combine_fold_scores(covariate_balance)
        # TODO: consider reordering the levels, such that the covariate will be the first one and then phase and fold
        # covariate_balance = covariate_balance.reorder_levels(["covariate", "phase", "fold"])

        # Create a new WeightEvaluatorScores object with the combined (i.e., multilevel DataFrame) results:
        scores = WeightEvaluatorScores(prediction_scores, covariate_balance)
        return scores

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
        if plot_name in {'weight_distribution'}:
            folds_predictions = [prediction.weight_for_being_treated for prediction in folds_predictions]
            return folds_predictions, a
        elif plot_name in {'roc_curve'}:
            curve_data = self._calculate_roc_curve_data(folds_predictions, a)
            return (curve_data,)
        elif plot_name in {'pr_curve'}:
            curve_data = self._calculate_pr_curve_data(folds_predictions, a)
            return (curve_data,)
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

    def _calculate_curve_data(self, folds_predictions, targets, curve_metric, area_metric, **kwargs):
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
        folds_sample_weights = {"unweighted": [None for _ in folds_predictions],
                                "weighted": [fold_predictions.weight_by_treatment_assignment
                                             for fold_predictions in folds_predictions]}
        folds_predictions = [fold_predictions.weight_for_being_treated for fold_predictions in folds_predictions]
        folds_targets = []
        for fold_predictions in folds_predictions:
            # Since this is weight estimator, which takes the inverse of a class prediction
            fold_targets = targets.loc[fold_predictions.index]
            fold_targets = fold_targets.replace({fold_targets.min(): fold_targets.max(),
                                                 fold_targets.max(): fold_targets.min()})
            folds_targets.append(fold_targets)

        curve_data = {}
        for curve_name, sample_weights in folds_sample_weights.items():
            area_folds, first_ret_folds, second_ret_folds, threshold_folds = self._calculate_performance_curve_data_on_folds(
                folds_predictions, folds_targets, sample_weights, area_metric, curve_metric)

            curve_data[curve_name] = {"first_ret_value": first_ret_folds,
                                      "second_ret_value": second_ret_folds,
                                      "Thresholds": threshold_folds, "area": area_folds}

        # Rename keys (as will be presented as curve labels in legend)
        curve_data["Weights"] = curve_data.pop("unweighted")
        curve_data["Weighted"] = curve_data.pop("weighted")
        return curve_data


# #################### #
# Propensity Evaluator #
# #################### #


class PropensityEvaluatorPredictions(WeightEvaluatorPredictions):
    """Data structure to hold propensity-model predictions"""

    def __init__(self, weight_by_treatment_assignment, weight_for_being_treated, treatment_assignment_pred,
                 propensity, propensity_by_treatment_assignment):
        super(PropensityEvaluatorPredictions, self).__init__(weight_by_treatment_assignment,
                                                             weight_for_being_treated,
                                                             treatment_assignment_pred)
        self.propensity = propensity
        self.propensity_by_treatment_assignment = propensity_by_treatment_assignment


class PropensityEvaluatorPredictions2(WeightEvaluatorPredictions2):
    """Data structure to hold propensity-model predictions"""

    def __init__(self, weight_matrix, propensity_matrix, treatment_assignment, treatment_assignment_prediction=None):
        super(PropensityEvaluatorPredictions2, self).__init__(weight_matrix, treatment_assignment,
                                                              treatment_assignment_prediction)
        self.propensity_matrix = propensity_matrix

    @property
    def propensity(self):
        propensity = self._extract_vector_from_matrix(self.propensity_matrix,
                                                      self._treatment_assignment)
        return propensity

    @property
    def propensity_by_treatment_assignment(self):
        # TODO: remove propensity_by_treatment if expected-ROC is not to be used.
        propensity_by_treatment_assignment = self._extract_vector_from_matrix(self.propensity_matrix,
                                                                              self._treatment_assignment.max())
        return propensity_by_treatment_assignment


class PropensityEvaluator(WeightEvaluator):
    def __init__(self, estimator):
        """
        Args:
            estimator (PropensityEstimator):
        """
        if not isinstance(estimator, PropensityEstimator):
            raise TypeError("PropensityEvaluator should be initialized with PropensityEstimator, got ({}) instead."
                            .format(type(estimator)))
        super(PropensityEvaluator, self).__init__(estimator)

    def _estimator_predict(self, X, a):
        """Predict on data.

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment

        Returns:
            PropensityEvaluatorPredictions
        """
        propensity = self.estimator.compute_propensity(X, a, treatment_values=a.max())
        propensity_by_treatment_assignment = self.estimator.compute_propensity_matrix(X)
        propensity_by_treatment_assignment = robust_lookup(propensity_by_treatment_assignment, a)

        weight_prediction = super(PropensityEvaluator, self)._estimator_predict(X, a)
        # Do not force stabilize=False as in WeightEvaluator:
        weight_by_treatment_assignment = self.estimator.compute_weights(X, a)
        prediction = PropensityEvaluatorPredictions(weight_by_treatment_assignment,
                                                    weight_prediction.weight_for_being_treated,
                                                    weight_prediction.treatment_assignment_pred,
                                                    propensity,
                                                    propensity_by_treatment_assignment)
        return prediction

    def _estimator_predict2(self, X, a):
        """Predict on data."""
        weight_prediction = super(PropensityEvaluator, self)._estimator_predict2(X, a)
        propensity_matrix = self.estimator.compute_propensity_matrix(X)
        fold_prediction = PropensityEvaluatorPredictions2(weight_prediction.weight_matrix, propensity_matrix,
                                                          a, weight_prediction.treatment_assignment_prediction)
        return fold_prediction

    def score_estimation(self, prediction, X, a_true, y_true=None, metrics_to_evaluate=None):
        """Scores a prediction against true labels.

        Args:
            prediction (PropensityEvaluatorPredictions): Prediction on the data.
            X (pd.DataFrame): Covariates.
            a_true (pd.Series): Target variable - treatment assignment
            y_true: *IGNORED*
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            WeightEvaluatorScores: Data-structure holding scores on the predictions
                                   and covariate balancing table ("table 1")
        """
        results = self._score_estimation(X, a_true,
                                         prediction.propensity,
                                         prediction.treatment_assignment_pred,
                                         prediction.weight_by_treatment_assignment,
                                         metrics_to_evaluate)
        return results

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
        if plot_name in {'weight_distribution'}:
            folds_predictions = [prediction.propensity for prediction in folds_predictions]
            return folds_predictions, a
        elif plot_name in {'calibration'}:
            folds_predictions = [prediction.propensity for prediction in folds_predictions]
            return folds_predictions, a
        else:
            # Common plots are implemented at top-most level possible.
            # Plot might be implemented by WeightEvaluator:
            return super(PropensityEvaluator, self)._get_data_for_plot(plot_name, folds_predictions, X, a, y, cv)

    def _calculate_curve_data(self, curves_folds_predictions, targets, curve_metric, area_metric, **kwargs):
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
        curves_sample_weights = {"unweighted": [None for _ in curves_folds_predictions],
                                 "weighted": [fold_predictions.weight_by_treatment_assignment
                                              for fold_predictions in curves_folds_predictions],
                                 "expected": [fold_predictions.propensity.append(1 - fold_predictions.propensity)
                                              for fold_predictions in curves_folds_predictions]}
        curves_folds_targets = [targets.loc[fold_predictions.weight_by_treatment_assignment.index]
                                for fold_predictions in curves_folds_predictions]
        curves_folds_targets = {
            "unweighted": curves_folds_targets,
            "weighted": curves_folds_targets,
            "expected": [pd.Series(data=targets.max(), index=fold_predictions.propensity.index).append(
                pd.Series(data=targets.min(), index=fold_predictions.propensity.index))
                for fold_predictions in curves_folds_predictions]
        }
        curves_folds_predictions = {
            "unweighted": [fold_predictions.propensity for fold_predictions in curves_folds_predictions],
            "weighted": [fold_predictions.propensity for fold_predictions in curves_folds_predictions],
            "expected": [fold_predictions.propensity.append(fold_predictions.propensity)
                         for fold_predictions in curves_folds_predictions]
        }
        # Expected curve duplicates the population, basically concatenating so that:
        # prediction = [p, p], target = [1, 0], weights = [p, 1-p]

        curve_data = {}
        for curve_name in curves_sample_weights.keys():
            sample_weights = curves_sample_weights[curve_name]
            folds_targets = curves_folds_targets[curve_name]
            folds_predictions = curves_folds_predictions[curve_name]

            area_folds, first_ret_folds, second_ret_folds, threshold_folds = self._calculate_performance_curve_data_on_folds(
                folds_predictions, folds_targets, sample_weights, area_metric, curve_metric)

            curve_data[curve_name] = {"first_ret_value": first_ret_folds,
                                      "second_ret_value": second_ret_folds,
                                      "Thresholds": threshold_folds, "area": area_folds}

        # Rename keys (as will be presented as curve labels in legend)
        curve_data["Propensity"] = curve_data.pop("unweighted")
        curve_data["Weighted"] = curve_data.pop("weighted")
        curve_data["Expected"] = curve_data.pop("expected")
        return curve_data


# ################# #
# Covariate Balance #
# ################# #


DISTRIBUTION_DISTANCE_METRICS = {"smd": lambda x, y, wx, wy: calc_weighted_standardized_mean_differences(x, y, wx, wy),
                                 "abs_smd": lambda x, y, wx, wy:
                                 abs(calc_weighted_standardized_mean_differences(x, y, wx, wy)),
                                 "ks": lambda x, y, wx, wy: calc_weighted_ks2samp(x, y, wx, wy)}


def calculate_covariate_balance(X, a, w, metric="abs_smd"):
    """Calculate covariate balance table ("table 1")

    Args:
        X (pd.DataFrame): Covariates.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with the signature
                                 weighted_distance(x, y, wx, wy) calculating distance between the weighted sample x
                                 and weighted sample y (weights by wx and wy respectively).

    Returns:
        pd.DataFrame: index are covariate names (columns) from X, and columns are "weighted" / "unweighted" results
                      of applying `metric` on each covariate to compare the two groups.
    """
    treatment_values = np.sort(np.unique(a))
    results = {}
    for treatment_value in treatment_values:
        distribution_distance_of_cur_treatment = pd.DataFrame(index=X.columns, columns=["weighted", "unweighted"],
                                                              dtype=float)
        for col_name, col_data in X.items():
            weighted_distance = calculate_distribution_distance_for_single_feature(col_data, w, a,
                                                                                   treatment_value, metric)
            unweighted_distance = calculate_distribution_distance_for_single_feature(col_data,
                                                                                     pd.Series(1, index=w.index), a,
                                                                                     treatment_value, metric)
            distribution_distance_of_cur_treatment.loc[col_name, ["weighted", "unweighted"]] = \
                [weighted_distance, unweighted_distance]
        results[treatment_value] = distribution_distance_of_cur_treatment
    results = pd.concat(results, axis="columns", names=[a.name or "a", metric])  # type: pd.DataFrame
    results.index.name = "covariate"
    if len(treatment_values) == 2:
        # In case there's only two treatments, the results for treatment_value==0 and treatment_value==0 are identical.
        # Therefore, we can get rid from one of them.
        # Here we keep the results for the treated-group (noted by maximal treatment value, probably 1):
        results = results.xs(treatment_values.max(), axis="columns", level=0)
    # TODO: is there a neat expansion for multi-treatment case? maybe not current_treatment vs. the rest.
    return results


def calculate_distribution_distance_for_single_feature(x, w, a, group_level, metric="abs_smd"):
    """

    Args:
        x (pd.Series): A single feature to check balancing.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        group_level: Value from `a` in order to divide the sample into one vs. rest.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with the signature
                                 weighted_distance(x, y, wx, wy) calculating distance between the weighted sample x
                                 and weighted sample y (weights by wx and wy respectively).

    Returns:
        float: weighted distance between the samples assigned to `group_level` and the rest of the samples.
    """
    if not callable(metric):
        metric = DISTRIBUTION_DISTANCE_METRICS[metric]
    cur_treated_mask = a == group_level
    x_treated = x.loc[cur_treated_mask]
    w_treated = w.loc[cur_treated_mask]
    x_untreated = x.loc[~cur_treated_mask]
    w_untreated = w.loc[~cur_treated_mask]
    distribution_distance = metric(x_treated, x_untreated, w_treated, w_untreated)
    return distribution_distance
