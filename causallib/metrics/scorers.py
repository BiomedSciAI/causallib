"""
This submodule implements a scorer interface for the
various causal metrics implemented.
These scorers can then be incorporated into model selection objects
to select the models (or hyperparameters) that optimizes these scores.
For example, as a `scoring` parameter to a `causallib.model_selection.GridSearchCV
object.

The signature of the call is ``(estimator, X, a, y)`` where ``estimator``
is the causal model to be evaluated, ``X`` are the covariates,
`a` is the treatment assignment, and ``y`` is the ground truth target
"""
# from sklearn.metrics._scorer import _BaseScorer
import abc

from . import (
    propensity_metrics,
    weight_metrics,
    outcome_metrics,
)


class _BaseCausalScorer:
    def __init__(self, score_func, sign, **kwargs):
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs

    def __call__(self, estimator, X, a_true, y_true, sample_weight=None, **kwargs):
        return self._sign * self._score(
            estimator,
            X,
            a_true,
            y_true,
            sample_weight=sample_weight,
            **{**self._kwargs, **kwargs},
        )

    @abc.abstractmethod
    def _score(self, estimator, X, a, y, sample_weight=None, **kwargs):
        raise NotImplementedError


class PropensityScorerBase(_BaseCausalScorer):
    def _score(self, estimator, X, a, y, sample_weight=None, **kwargs):
        propensities = estimator.compute_propensity(X, a)
        weights = estimator.compute_weights(X, a)
        score = self._score_func(
            a, propensities, sample_weight=weights,
            **kwargs
        )
        return score


weighted_roc_auc_error_scorer = PropensityScorerBase(
    propensity_metrics.weighted_roc_auc_error, -1,
)
weighted_roc_curve_error_scorer = PropensityScorerBase(
    propensity_metrics.weighted_roc_curve_error, -1,
)
expected_roc_auc_error_scorer = PropensityScorerBase(
    propensity_metrics.expected_roc_auc_error, -1,
)
expected_roc_curve_error_scorer = PropensityScorerBase(
    propensity_metrics.expected_roc_curve_error, -1
)
ici_error_scorer = PropensityScorerBase(
    propensity_metrics.ici_error, -1,
)

_PROPENSITY_SCORERS = dict(
    weighted_roc_auc_error=weighted_roc_auc_error_scorer,
    weighted_roc_curve_error=weighted_roc_curve_error_scorer,
    expected_roc_auc_error=expected_roc_auc_error_scorer,
    expected_roc_curve_error=expected_roc_curve_error_scorer,
    ici_error=ici_error_scorer,

)


class WeightScorerBase(_BaseCausalScorer):
    def _score(self, estimator, X, a, y, sample_weight=None, **kwargs):
        weights = estimator.compute_weights(X, a)
        score = self._score_func(
            X, a, sample_weight=weights,
            **kwargs
        )
        return score


covariate_balancing_error_scorer = WeightScorerBase(
    weight_metrics.covariate_balancing_error, -1,
)

_WEIGHT_SCORERS = dict(
    covariate_balancing_error=covariate_balancing_error_scorer,
)


class OutcomeScorerBase(_BaseCausalScorer):
    def _score(self, estimator, X, a, y, sample_weight=None, **kwargs):
        potential_outcomes_pred = estimator.estimate_individual_outcome(X, a)
        score = self._score_func(
            y, potential_outcomes_pred, a,
            **kwargs
            # Is this a good generic API call to the outcome metrics?
        )
        return score


balanced_residuals_error_scorer = OutcomeScorerBase(
    outcome_metrics.balanced_residuals_error, -1,
)

_OUTCOME_SCORERS = dict(
    balanced_residuals_error=balanced_residuals_error_scorer,
)

_SCORERS = {
    **_PROPENSITY_SCORERS,
    **_WEIGHT_SCORERS,
    **_OUTCOME_SCORERS
}


def get_scorer(scoring):
    """Gets a scorer callable from string.
    see `causallib.metrics.get_scorer_names` to retrieve available score names.
    """
    if callable(scoring):
        return scoring
    try:
        return _SCORERS.get(scoring)
    except KeyError:
        raise ValueError(
            f"Scoring name {scoring} is not a valid scoring name."
            f"use the `causallib.metrics.get_scorer_names` to get all valid names."
        )


def get_scorer_names(score_type="all"):
    """Get the name of all available scorers.
    These names can be passed to `causallib.metrics.get_scorer` to retrieve a scorer object.

    Args:
        score_type (str): any of {"all", "propensity", "weight", "outcome"}.
            Returns only scorers relevant to the `score_type` type of model.

    Returns:

    """
    scores_types_map = {
        "all": _SCORERS,
        "propensity": _PROPENSITY_SCORERS,
        "weight": _WEIGHT_SCORERS,
        "outcome": _OUTCOME_SCORERS,
    }
    try:
        return sorted(scores_types_map[score_type])
    except KeyError:
        raise ValueError(
            f"`score_type` {score_type} is not valid."
            f"Please use one of {scores_types_map.keys()}."
        )

