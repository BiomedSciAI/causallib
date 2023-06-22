from causallib.metrics.scorers import PropensityScorerBase


class SKLearnScorerWrapper(PropensityScorerBase):
    def __init__(self, score_func, sign=None, **kwargs):
        super().__init__(
            score_func=score_func,
            sign=1,  # This keeps original scorer sign
            **kwargs
        )

    def _score(self, estimator, X, a, y=None, sample_weight=None, **kwargs):
        learner = self._extract_sklearn_estimator(estimator)
        score = self._score_func(learner, X, a, sample_weight=sample_weight)
        return score

    @staticmethod
    def _extract_sklearn_estimator(estimator):
        if hasattr(estimator, "best_estimator_"):
            # Causallib's wrapper around GridSearchCV
            return estimator.best_estimator_.learner
        if hasattr(estimator, "learner"):
            return estimator.learner
        raise AttributeError(
            f"Could not extract an sklearn estimator from {estimator},"
            f"which has the following attributes:\n"
            f"{list(estimator.__dict__.keys())}"
        )