"""
Interfacing between causallib and sklearn to take advantage of sklearn's
hyperparameter search machinery (e.g., GridSearchCV).
Wraps causallib's models and scorers to look like sklearn's,
and wraps sklearn's hyperparameter search models to look like causallib models.
"""
from typing import Type

from sklearn.model_selection._search import BaseSearchCV
import pandas as pd

from ..metrics.scorers import get_scorer


def _adapt_causal_scorer_to_sklearn(scorer):
    """Wraps a Causal Scorer, whose interface is `estimator, X, a, y, **kwargs`,
    with a sklearn-compatible interface of `estimator, Xa, y, **kwargs`.

    Args:
        scorer (callable): a causallib scorer.

    Returns:
        score (callable): a scikit-learn score interface for `scorer`.
    """
    def score(estimator, joinedXa, y_true, sample_weight=None, **kwargs):
        a = joinedXa.iloc[:, -1]
        X = joinedXa.iloc[:, :-1]
        score_value = scorer(estimator, X, a, y_true, sample_weight=sample_weight, **kwargs)
        return score_value
    return score


def _adapt_causal_scorers_to_sklearn(scorers):
    """Wraps each causallib scorer in a possible dict/list of them.
    Only supports causallib's scorers.
    Mostly compatible with `scoring` parameter in:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Args:
        scorers (str, callable, list, tuple, dict):
            A strategy to evaluate the performance a causallib model. can be either single or multiple scores.

            If `scorers` represents a single score, one can use:
            * A string (see causallib.metrics.get_scorer_names for all available names).
            * A callable following the API in causallib.metrics.scorers: `estimator, X, a, y, **kwargs`.

            If `scorers` represents multiple scores, one can use:
            * A list or tuple of scorer names.
            * A dict mapping between metric names and callable scorers (following the API in causallib.metrics.scorers).

    Returns:
        scores (callable, list, dict): a callable or a list/dict of callables (depending on the input),
            of causal scorers compatible with sklearn's scorers' API.
    """
    # TODO: adapt only if isinstance of BaseCausalScorer and leave sklearn's scorers untouched?
    if isinstance(scorers, dict):
        scorers = {name: _adapt_causal_scorer_to_sklearn(scorer)
                   for name, scorer in scorers.items()}
    elif isinstance(scorers, (list, tuple, set)):
        scorers = {name: _adapt_causal_scorer_to_sklearn(get_scorer(name))
                   for name in scorers}
    elif isinstance(scorers, str):
        scorers = _adapt_causal_scorer_to_sklearn(get_scorer(scorers))
    elif callable(scorers):
        scorers = _adapt_causal_scorer_to_sklearn(scorers)
    else:
        raise ValueError(
            f"`scoring` is invalid (got {scorers})."
            f"Please provide either: "
            f"a callable compatible with causallib.metrics.scorers API,"
            f"a scorer name from causallib.metrics.get_scorer_names,"
            f"a list or tuple of scorer names,"
            f"or a dictionary mapping between metric names and valid scorers"
        )
    return scorers


def _adapt_causal_estimator_to_sklearn(estimator):
    """Wraps a causallib model (type) with interface `fit(X, a, y)`
    with a sklearn's `fit(X', y)` interface.
    Other than that, it has the same (inference) capabilities as the causallib estimator.
    """
    class SklearnCompatibleEstimator(estimator.__class__):
        def fit(self, joinedXa, y, *args, **kwargs):
            a = joinedXa.iloc[:, -1]
            X = joinedXa.iloc[:, :-1]
            return super().fit(X, a, y, *args, **kwargs)

        @property
        def estimator(self):
            params = self.get_params()
            return estimator.set_params(**params)

    SklearnCompatibleEstimator.__name__ = f"SklearnCompatible{estimator.__class__.__name__}"
    SklearnCompatibleEstimator.__qualname__ = f"SklearnCompatible{estimator.__class__.__qualname__}"
    params = estimator.get_params()
    return SklearnCompatibleEstimator(learner=params["learner"]).set_params(**params)


def causalize_searcher(searcher_type: Type[BaseSearchCV]):
    """wraps a hyperparameter search algorithm (like sklearn's GridSearchCV)
    with a causallib model interface.

    Args:
        searcher_type: A class of hyperparameter search algorithm
            (e.g., sklearn's GridSearchCV)

    Returns:
        searcher(searcher_type): a class definition of the provided searcher
            with a causallib `fit(X, a, y)` interface and the underlying estimator capabilities.

    Examples:
        >>> from sklearn.model_selection import GridSearchCV
        >>> from sklearn.linear_model import LogisticRegression
        >>> from causallib.estimation import IPW
        >>> from causallib.metrics import get_scorer
        >>> from causallib.datasets import load_nhefs
        >>> data = load_nhefs()
        >>> CausalGridSearchCV = causalize_searcher(GridSearchCV)
        >>> model = IPW(LogisticRegression())
        >>> scorer = get_scorer("weighted_roc_auc_error")
        >>> param_grid = dict(clip_min=[0.2, 0.3])
        >>> grid_model = CausalGridSearchCV(model, param_grid=param_grid, scoring=scorer)  # GridSearchCV parameters
        >>> grid_model.fit(data.X, data.a, data.y)  # causallib interface
        >>> grid_model.estimate_population_outcome(data.X, data.a, data.y)
        >>> grid_model.compute_propensity(data.X, data.a)  # IPW capabilities

    """
    class CausalSearcher(searcher_type):
        def __init__(self, estimator, *args, **kwargs):
            estimator = _adapt_causal_estimator_to_sklearn(estimator)
            kwargs["scoring"] = _adapt_causal_scorers_to_sklearn(kwargs["scoring"])
            super().__init__(estimator, *args, **kwargs)

        def _set_methods_from_estimator(self):
            """Exposes all the methods from the internal `best_estimator_`,
            so that the `CausalSearcher` behaves like it's internal causallib model.
            """
            # Avoid re-setting and overwriting existing methods:
            unique_estimator_attributes = set(dir(self.best_estimator_)) - set(dir(self))
            for attr_name in dir(self.best_estimator_):
                attr_value = getattr(self.best_estimator_, attr_name, None)
                if (
                    attr_name in unique_estimator_attributes  # An estimator attribute that is not unique to searcher
                    and not attr_name.startswith("__")  # not internal method
                    and callable(attr_value)  # the current attribute is a method
                    and attr_name != "fit"  # don't overwrite the GridSearch-like fit below.
                ):
                    setattr(self, attr_name, attr_value)

        def fit(self, X, a, y, *, groups=None, **fit_params):
            joinedXa = pd.concat([X, a], axis=1)
            super().fit(joinedXa, y, groups=groups, **fit_params)
            # if hasattr(self, "best_estimator_"):
            #     self.best_estimator_ = self.estimator.estimator
            # # Should `best_estimator_` be original causallib model or adapted?
            self._set_methods_from_estimator()
            return self

    CausalSearcher.__name__ = f"Causal{searcher_type.__name__}"
    CausalSearcher.__qualname__ = f"Causal{searcher_type.__qualname__}"
    return CausalSearcher


