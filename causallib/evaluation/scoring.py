"""Scoring functions that operate on the evaluation results objects.

These functions depend on the causallib.evalutation results objects and are
less reusable than the functions in metrics.py.
"""
import pandas as pd

from .predictions import (
    OutcomePredictions,
    PropensityPredictions,
    WeightPredictions,
    PropensityEvaluatorScores,
)

from .metrics import get_default_binary_metrics, get_default_regression_metrics

def score_cv(predictions, X, a, y, cv, metrics_to_evaluate="defaults"):
    """Evaluate the prediction against the true data using evaluation score metrics.

    Args:
        predictions (dict[str, list]): the output of predict_cv.
        X (pd.DataFrame): Covariates.
        a (pd.Series): Treatment assignment.
        y (pd.Series): Outcome.
        cv (list[tuples]): list the number of folds containing tuples of indices:
            (train_idx, validation_idx)
        metrics_to_evaluate (dict | "defaults"): key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
            If `"defaults"`, default metrics are selected.
    Returns:
        pd.DataFrame | WeightEvaluatorScores:
            DataFrame whose columns are different metrics and each row is a
            product of phase x fold x strata.
            PropensityEvaluatorScores also has a covariate-balance result in a DataFrame.
    """
    if metrics_to_evaluate == "defaults":
        metrics_to_evaluate = _get_default_metrics_to_evaluate(predictions["train"][0])

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
        # TODO: use dict-comprehension to map between phases[0] to cv[0]
        # instead of writing "train" explicitly

        for phase in phases:
            X_fold, a_fold, y_fold = (
                data[phase]["X"],
                data[phase]["a"],
                data[phase]["y"],
            )
            prediction = predictions[phase][i]

            fold_scores = score_estimation(
                prediction, X_fold, a_fold, y_fold, metrics_to_evaluate
            )
            scores[phase].append(fold_scores)

    if isinstance(fold_scores, PropensityEvaluatorScores):
        return _combine_weight_evaluator_fold_scores(scores)
    return _combine_fold_scores(scores)

def _get_default_metrics_to_evaluate(first_prediction):
    if isinstance(first_prediction, OutcomePredictions) and not first_prediction.is_binary_outcome:
        metrics_to_evaluate = get_default_regression_metrics()
    else:
        metrics_to_evaluate = get_default_binary_metrics()
    return metrics_to_evaluate


def score_estimation(prediction, X, a_true, y_true, metrics_to_evaluate=None):
    """Should know how to handle the _estimator_predict output provided in `prediction`.
    Can utilize any of the true values provided:
        covariates `X`, treatment assignment `a` or outcome `y`.
    """

    if isinstance(prediction, OutcomePredictions):
        return prediction.evaluate_metrics(a_true, y_true, metrics_to_evaluate)

    # propensity and weight both have the same interface
    # no need to differentiate
    if isinstance(prediction, (PropensityPredictions, WeightPredictions)):
        return prediction.evaluate_metrics(X, a_true, metrics_to_evaluate)
    raise ValueError(f"Invalid type for prediciton: {type(prediction)}")


def _combine_fold_scores(scores):
    """
    Combines scores of each phase and fold into a single object (DataFrame) of scores.

    Args:
        scores (dict[str, list[pd.DataFrame]]): scores of each fold of each phase.
            The structure is {phase_name: [fold_1_score, fold_2_score...]}.
            Where phase_name is usually "train" or "valid", and each fold_i_score
            is a DataFrame which columns are evaluation metrics and rows are
            results of that metrics in that fold.

    Returns:
        pd.DataFrame: Row-concatenated DataFrame with MultiIndex accounting for the
            concatenated folds and phases.
    """
    # Concatenate the scores from list of folds to DataFrame with rows as folds,
    # keeping it by different phases:
    scores = {
        phase: pd.concat(
            scores_fold, axis="index", keys=range(len(scores_fold)), names=["fold"]
        )
        for phase, scores_fold in scores.items()
    }
    # Concatenate the train/validation DataFrame scores into DataFrame with rows as phases:
    scores = pd.concat(scores, axis="index", names=["phase"])
    return scores


def _combine_weight_evaluator_fold_scores(scores):
    # `scores` are provided as PropensityEvaluatorScores object for each fold in each phase,
    # Namely, dict[list[PropensityEvaluatorScores]], which in turn hold two DataFrames components.
    # In order to combine the underlying DataFrames into a multilevel DataFrame,
    # one must first extract them from the PropensityEvaluatorScores object, then recombine.

    # Extract the two components of PropensityEvaluatorScores:
    prediction_scores_unfolded = {
        phase: [fold_score.prediction_scores for fold_score in phase_scores]
        for phase, phase_scores in scores.items()
    }
    prediction_scores = _combine_fold_scores(prediction_scores_unfolded)

    covariate_balance_unfolded = {
        phase: [fold_score.covariate_balance for fold_score in phase_scores]
        for phase, phase_scores in scores.items()
    }
    covariate_balance = _combine_fold_scores(covariate_balance_unfolded)

    # Combine the dict[list[DataFrames]] of each component into a multilevel DataFrame separately:
    # TODO: consider reordering the levels, such that the covariate will be the first one and then
    # phase and fold
    # covariate_balance = covariate_balance.reorder_levels(["covariate", "phase", "fold"])

    # Create a new PropensityEvaluatorScores object
    # with the combined (i.e., multilevel DataFrame) results:
    scores = PropensityEvaluatorScores(prediction_scores, covariate_balance)
    return scores
