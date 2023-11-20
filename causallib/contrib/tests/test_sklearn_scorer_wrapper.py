import unittest

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.utils import Bunch
from sklearn.metrics import get_scorer

from causallib.estimation import IPW
from causallib.model_selection import GridSearchCV

from causallib.contrib.sklearn_scorer_wrapper import SKLearnScorerWrapper


class TestSKLearnScorerWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N = 500
        X, a = make_classification(
            n_samples=N,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            random_state=42,
        )
        X = pd.DataFrame(X)
        a = pd.Series(a)
        cls.data = Bunch(X=X, a=a, y=a)

        learner = LogisticRegression()
        ipw = IPW(learner)
        ipw.fit(X, a)
        # cls.learner = learner
        cls.estimator = ipw

    def test_agreement_with_sklearn(self):
        scorer_names = [
            "accuracy",
            "average_precision",
            "neg_brier_score",
            "f1",
            "neg_log_loss",
            "precision",
            "recall",
            "roc_auc",
        ]
        for scorer_name in scorer_names:
            with self.subTest(f"Test scorer {scorer_name}"):
                scorer = get_scorer(scorer_name)
                score = scorer(self.estimator.learner, self.data.X, self.data.a)

                causallib_adapted_scorer = SKLearnScorerWrapper(scorer)
                causallib_score = causallib_adapted_scorer(
                    self.estimator, self.data.X, self.data.a, self.data.y
                )

                self.assertAlmostEqual(causallib_score, score)

    def test_hyperparameter_search_model(self):
        scorer = SKLearnScorerWrapper(get_scorer("roc_auc"))
        param_grid = dict(
            clip_min=[0.2, 0.3],
            learner__C=[0.1, 1],
        )
        model = GridSearchCV(
            self.estimator,
            param_grid=param_grid,
            scoring=scorer,
            cv=3,
        )
        model.fit(self.data.X, self.data.a, self.data.y)

        score = scorer(model, self.data.X, self.data.a, self.data.y)
        self.assertGreaterEqual(score, model.best_score_)
