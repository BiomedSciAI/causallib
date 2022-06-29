from typing import Any
import numpy as np
import pandas as pd
from sklearn import metrics
from collections import namedtuple
from ..utils.stat_utils import (
    calc_weighted_standardized_mean_differences,
    calc_weighted_ks2samp,
    robust_lookup,
)
from ..estimation.base_weight import PropensityEstimator, WeightEstimator
from ..estimation.base_estimator import IndividualOutcomeEstimator
import warnings
import abc

WeightEvaluatorScores = namedtuple(
    "WeightEvaluatorScores", ["prediction_scores", "covariate_balance"]
)


class Scorer:
    _numerical_classification_metrics = {
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "f1": metrics.f1_score,
        "roc_auc": metrics.roc_auc_score,
        "avg_precision": metrics.average_precision_score,
        "hinge": metrics.hinge_loss,
        "matthews": metrics.matthews_corrcoef,
        "0_1": metrics.zero_one_loss,
        "brier": metrics.brier_score_loss,
    }
    _nonnumerical_classification_metrics = {
        "confusion_matrix": metrics.confusion_matrix,
        "roc_curve": metrics.roc_curve,
        "pr_curve": metrics.precision_recall_curve,
    }
    _classification_metrics = {
        **_numerical_classification_metrics,
        **_nonnumerical_classification_metrics,
    }

    _regression_metrics = {
        "expvar": metrics.explained_variance_score,
        "mae": metrics.mean_absolute_error,
        "mse": metrics.mean_squared_error,
        "msle": metrics.mean_squared_log_error,
        # Allow mdae receive sample_weight argument but ignore it. This unifies the interface:
        "mdae": lambda y_true, y_pred, **kwargs: metrics.median_absolute_error(
            y_true, y_pred
        ),
        "r2": metrics.r2_score,
    }

    @staticmethod
    def from_estimator(estimator):
        if isinstance(estimator, (PropensityEstimator, WeightEstimator)):
            return WeightScorer
        if isinstance(estimator, IndividualOutcomeEstimator):
            return OutcomeScorer

    @classmethod
    def score_cv(cls, predictions, X, a, y, cv, metrics_to_evaluate=None):
        """Evaluate the prediction against the true data using evaluation score metrics.

        Args:
            predictions (dict[str, list]): the output of predict_cv.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            cv (list[tuples]): list the number of folds containing tuples of indices (train_idx, validation_idx)
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            pd.DataFrame | WeightEvaluatorScores:
                DataFrame whose columns are different metrics and each row is a product of phase x fold x strata.
                WeightEvaluatorScores also has a covariate-balance result in a DataFrame.
        """
        phases = predictions.keys()
        scores = {phase: [] for phase in phases}
        for i, (train_idx, valid_idx) in enumerate(cv):
            data = {
                "train": {
                    "X": X.iloc[train_idx],
                    "a": a.iloc[train_idx],
                    "y": y.iloc[train_idx],
                },
                "valid": {
                    "X": X.iloc[valid_idx],
                    "a": a.iloc[valid_idx],
                    "y": y.iloc[valid_idx],
                },
            }
            # TODO: use dict-comprehension to map between phases[0] to cv[0] instead writing "train" explicitly

            for phase in phases:
                X_fold, a_fold, y_fold = (
                    data[phase]["X"],
                    data[phase]["a"],
                    data[phase]["y"],
                )
                prediction = predictions[phase][i]

                fold_scores = cls.score_estimation(
                    prediction, X_fold, a_fold, y_fold, metrics_to_evaluate
                )
                scores[phase].append(fold_scores)

        scores = cls._combine_fold_scores(scores)
        return scores

    @classmethod
    def score_binary_prediction(
        cls,
        y_true,
        y_pred_proba=None,
        y_pred=None,
        sample_weight=None,
        metrics_to_evaluate=None,
        only_numeric_metric=True,
    ):
        """Evaluates a binary prediction against true labels.

        Args:
            y_true (pd.Series): True labels
            y_pred_proba (pd.Series): continuous output of predictor (as in `predict_proba` or `decision_function`).
            y_pred (pd.Series): label (i.e., categories, decisions) predictions.
            sample_weight (pd.Series | None): weight of each sample.
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.
            only_numeric_metric (bool): If metrics_to_evaluate not provided and default is used, whether to use only
                                        numerical metrics (non-numerical are for example roc_curve, that returns vectors
                                        and not scalars).
                                        Ignored if metrics_to_evaluate is provided

        Returns:
            pd.Series: name of metric as index and the evaluated score as value.
        """
        if metrics_to_evaluate is None:
            metrics_to_evaluate = (
                cls._numerical_classification_metrics
                if only_numeric_metric
                else cls._classification_metrics
            )
        scores = {}
        for metric_name, metric_func in metrics_to_evaluate.items():
            if metric_name in {
                "hinge",
                "brier",
                "roc_curve",
                "roc_auc",
                "pr_curve",
                "avg_precision",
            }:
                prediction = y_pred_proba
            else:
                prediction = y_pred

            if prediction is None:
                continue

            try:
                scores[metric_name] = metric_func(
                    y_true, prediction, sample_weight=sample_weight
                )
            except ValueError as v:  # if y_true has single value
                warnings.warn("metric {} could not be evaluated".format(metric_name))
                warnings.warn(str(v))
                scores[metric_name] = np.nan

        dtype = (
            float
            if all([np.isscalar(score) for score in scores.values()])
            else np.dtype(object)
        )
        return pd.Series(scores, dtype=dtype)

    @classmethod
    def score_regression_prediction(
        cls, y_true, y_pred, sample_weight=None, metrics_to_evaluate=None
    ):
        """Evaluates continuous prediction against true labels

        Args:
            y_true (pd.Series): True label.
            y_pred (pd.Series): Predictions.
            sample_weight (pd.Series | None): weight for each sample.
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            pd.Series: name of metric as index and the evaluated score as value.
        """
        metrics_to_evaluate = metrics_to_evaluate or cls._regression_metrics
        metrics = {}
        for metric_name, metric_func in metrics_to_evaluate.items():
            try:
                metrics[metric_name] = metric_func(
                    y_true, y_pred, sample_weight=sample_weight
                )
            except ValueError as v:
                metrics[metric_name] = np.nan
                warnings.warn("While evaluating " + metric_name + ": " + str(v))
        return pd.Series(metrics)

    @staticmethod
    def _combine_fold_scores(scores):
        """
        Combines scores of each phase and fold into a single object (DataFrame) of scores.

        Args:
            scores (dict[str, list[pd.DataFrame]]):
                scores of each fold of each phase. The structure is {phase_name: [fold_1_score, fold_2_score...]}.
                Where phase_name is usually "train" or "valid", and each fold_i_score is a DataFrame which columns are
                evaluation metrics and rows are results of that metrics in that fold.

        Returns:
            pd.DataFrame: Row-concatenated DataFrame with MultiIndex accounting for the concatenated folds and phases.
        """
        # Concatenate the scores from list of folds to DataFrame with rows as folds, keeping it by different phases:
        scores = {
            phase: pd.concat(
                scores_fold, axis="index", keys=range(len(scores_fold)), names=["fold"]
            )
            for phase, scores_fold in scores.items()
        }
        # Concatenate the train/validation DataFrame scores into DataFrame with rows as phases:
        scores = pd.concat(scores, axis="index", names=["phase"])
        return scores

    @abc.abstractmethod
    def score_estimation(self, prediction, X, a_true, y_true, metrics_to_evaluate=None):
        """Should know how to handle the _estimator_predict output provided in `prediction`.
        Can utilize any of the true values provided (covariates `X`, treatment assignment `a` or outcome `y`)."""
        raise NotImplementedError


class OutcomeScorer(Scorer):
    @classmethod
    def score_estimation(
        cls,
        prediction,
        X,
        a_true,
        y_true,
        metrics_to_evaluate=None,
    ):
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
        return prediction.calculate_metrics(a_true, y_true, metrics_to_evaluate)

class WeightScorer(Scorer):
    @classmethod
    def score_estimation(
        cls, prediction, X, a_true, y_true=None, metrics_to_evaluate=None
    ):
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
        return prediction.calculate_metrics(X, a_true, metrics_to_evaluate)

    @classmethod
    def _combine_fold_scores(cls, scores):
        # `scores` are provided as WeightEvaluatorScores object for each fold in each phase,
        # Namely, dict[list[WeightEvaluatorScores]], which in turn hold two DataFrames components.
        # In order to combine the underlying DataFrames into a multilevel DataFrame, one must first extract them from
        # the WeightEvaluatorScores object, into two separate components.

        # Extract the two components of WeightEvaluatorScores:
        prediction_scores = {
            phase: [fold_score.prediction_scores for fold_score in phase_scores]
            for phase, phase_scores in scores.items()
        }
        covariate_balance = {
            phase: [fold_score.covariate_balance for fold_score in phase_scores]
            for phase, phase_scores in scores.items()
        }

        # Combine the dict[list[DataFrames]] of each component into a multilevel DataFrame separately:
        prediction_scores = Scorer._combine_fold_scores(prediction_scores)
        covariate_balance = Scorer._combine_fold_scores(covariate_balance)
        # TODO: consider reordering the levels, such that the covariate will be the first one and then phase and fold
        # covariate_balance = covariate_balance.reorder_levels(["covariate", "phase", "fold"])

        # Create a new WeightEvaluatorScores object with the combined (i.e., multilevel DataFrame) results:
        scores = WeightEvaluatorScores(prediction_scores, covariate_balance)
        return scores


# ################# #
# Covariate Balance #
# ################# #


DISTRIBUTION_DISTANCE_METRICS = {
    "smd": lambda x, y, wx, wy: calc_weighted_standardized_mean_differences(
        x, y, wx, wy
    ),
    "abs_smd": lambda x, y, wx, wy: abs(
        calc_weighted_standardized_mean_differences(x, y, wx, wy)
    ),
    "ks": lambda x, y, wx, wy: calc_weighted_ks2samp(x, y, wx, wy),
}


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
        distribution_distance_of_cur_treatment = pd.DataFrame(
            index=X.columns, columns=["weighted", "unweighted"], dtype=float
        )
        for col_name, col_data in X.items():
            weighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, w, a, treatment_value, metric
            )
            unweighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, pd.Series(1, index=w.index), a, treatment_value, metric
            )
            distribution_distance_of_cur_treatment.loc[
                col_name, ["weighted", "unweighted"]
            ] = [weighted_distance, unweighted_distance]
        results[treatment_value] = distribution_distance_of_cur_treatment
    results = pd.concat(
        results, axis="columns", names=[a.name or "a", metric]
    )  # type: pd.DataFrame
    results.index.name = "covariate"
    if len(treatment_values) == 2:
        # In case there's only two treatments, the results for treatment_value==0 and treatment_value==0 are identical.
        # Therefore, we can get rid from one of them.
        # Here we keep the results for the treated-group (noted by maximal treatment value, probably 1):
        results = results.xs(treatment_values.max(), axis="columns", level=0)
    # TODO: is there a neat expansion for multi-treatment case? maybe not current_treatment vs. the rest.
    return results


def calculate_distribution_distance_for_single_feature(
    x, w, a, group_level, metric="abs_smd"
):
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
