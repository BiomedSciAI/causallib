import abc
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from causallib.estimation import IPW, StratifiedStandardization

from causallib.metrics import get_scorer, get_scorer_names

import sklearn
LR_NO_PENALTY = None if sklearn.__version__ >= "1.2" else "none"


class BaseTestScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        # Data:
        # X, a = make_classification(
        #     n_features=1, n_informative=1, n_redundant=0, n_repeated=0,
        #     n_classes=2, n_clusters_per_class=1, flip_y=0.0, class_sep=0.5,
        #     random_state=0,
        # )
        n = 1000
        d = 2
        X = np.random.normal(scale=3, size=(n, d))
        beta_Xa = np.random.normal(size=d)
        a_logit = X @ beta_Xa  # + np.random.normal(size=n)
        a_propensity = 1 / (1 + np.exp(-a_logit))
        a = np.random.binomial(1, a_propensity)
        beta_Xy = np.random.normal(size=d)
        beta_a = 5
        y = X @ beta_Xy + a * beta_a + np.random.normal(size=n)
        cls.data = {
            "X": pd.DataFrame(X),
            "a": pd.Series(a),
            "y": pd.Series(y),
        }

        # RCT Data:
        np.random.seed(0)
        X = np.random.normal(0, 1, 1000)
        a = np.random.binomial(1, 0.5, 1000)
        y = X * 1 + a * 5 + np.random.normal(size=X.shape[0])
        cls.data_rct = {
            "X": pd.DataFrame(X),
            "a": pd.Series(a),
            "y": pd.Series(y),
        }

        # # Avoids regularization of the model:
        cls.estimator = IPW(LogisticRegression(penalty=LR_NO_PENALTY, solver='sag', max_iter=2000))

    def ensure_single_scoring_does_not_fail(self, scoring_name, data):
        scorer = get_scorer(scoring_name)
        score = scorer(
            self.estimator,
            data['X'], data['a'], data['y'],
        )
        self.assertIsInstance(score, float)
        self.assertLess(score, 0)

    def ensure_multiple_scoring_do_not_fail(self, score_type):
        self.estimator.fit(self.data['X'], self.data['a'], self.data['y'])
        scoring_names = get_scorer_names(score_type)
        for scoring_name in scoring_names:
            with self.subTest(f"Testing {scoring_name} does not fail."):
                self.ensure_single_scoring_does_not_fail(scoring_name, self.data)

    @abc.abstractmethod
    @unittest.skip
    def test_multiple_scoring_do_not_fail(self):
        pass


class TestPropensityScorer(BaseTestScorer):

    def test_multiple_scoring_do_not_fail(self):
        self.ensure_multiple_scoring_do_not_fail("propensity")

    def test_scoring_with_kwargs(self):
        self.estimator.fit(self.data['X'], self.data['a'], self.data['y'])

        with self.subTest("single known kwarg:"):
            scorer = get_scorer("weighted_roc_curve_error")
            score_default = scorer(
                self.estimator,
                self.data['X'], self.data['a'], self.data['y'],
            )
            score_agg_min = scorer(
                self.estimator,
                self.data['X'], self.data['a'], self.data['y'],
                agg=np.min,
            )
            self.assertLess(abs(score_agg_min), abs(score_default))
            # Default is max. scores are negative metrics values due to lower-is-better

        with self.subTest("multiple known kwargs"):
            scorer = get_scorer("ici_error")
            score_default = scorer(
                self.estimator,
                self.data['X'], self.data['a'], self.data['y'],
            )
            score_kwargs = scorer(
                self.estimator,
                self.data['X'], self.data['a'], self.data['y'],
                agg=np.max,
                lowess_kwargs=dict(frac=0.2),
            )
            self.assertLess(abs(score_default), abs(score_kwargs))
            # Default is mean. scores are negative metrics values due to lower-is-better

        # @unittest.expectedFailure
        # with self.subTest("passing unknown kwargs"):
        #     propensity_scoring_names = get_scorer_names("propensity")
        #     for scoring_name in propensity_scoring_names:
        #         with self.subTest(f"Testing unknown kwarg in {scoring_name}."):
        #             scorer = get_scorer(scoring_name)
        #             score = scorer(
        #                 self.estimator,
        #                 self.data['X'], self.data['a'], self.data['y'],
        #                 nonexistingkwarg="shouldn't_exist!",
        #             )
        #             self.assertLess(0, score)


class TestWeightScorer(BaseTestScorer):
    def test_multiple_scoring_do_not_fail(self):
        self.ensure_multiple_scoring_do_not_fail("weight")

    def test_scoring_with_kwargs(self):
        self.estimator.fit(self.data['X'], self.data['a'], self.data['y'])
        scorer = get_scorer("covariate_balancing_error")
        score_default = scorer(
            self.estimator,
            self.data['X'], self.data['a'], self.data['y'],
        )
        score_agg_min = scorer(
            self.estimator,
            self.data['X'], self.data['a'], self.data['y'],
            agg=np.min,
        )
        self.assertLess(-score_agg_min, -score_default)  # Default is max. Scores are negative metrics values

    def test_covariate_imbalance_count_error(self):
        X = pd.DataFrame(
            {
                "imbalanced": [5, 5, 5, 5, 4, 6, 0, 0, 0, 0, -1, 1],
                "balanced": [5, 5, 5, 5, 4, 6, 5, 5, 5, 5, 4, 6],
            }
        )
        a = pd.Series([1] * 6 + [0] * 6)
        ipw = IPW(LogisticRegression())
        ipw.fit(X, a)

        with self.subTest("Count score"):
            scorer = get_scorer("covariate_imbalance_count_error")
            score = scorer(ipw, X, a, y_true=None, fraction=False)
            self.assertEqual(score, -1)

        with self.subTest("Fractional score"):
            scorer = get_scorer("covariate_imbalance_count_error")
            score = scorer(ipw, X, a, y_true=None, fraction=True)
            self.assertEqual(score, -0.5)

        with self.subTest("Non-default threshold"):
            threshold = 10  # Should result in not violating features
            scorer = get_scorer("covariate_imbalance_count_error")
            score = scorer(ipw, X, a, y_true=None, threshold=threshold)
            self.assertEqual(score, 0)


class TestOutcomeScorer(BaseTestScorer):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.estimator = StratifiedStandardization(LinearRegression())

    def test_multiple_scoring_do_not_fail(self):
        self.ensure_multiple_scoring_do_not_fail("outcome")

    def test_scoring_with_kwargs(self):
        self.estimator.fit(self.data['X'], self.data['a'], self.data['y'])
        scorer = get_scorer("balanced_residuals_error")
        score_default = scorer(
            self.estimator,
            self.data['X'], self.data['a'], self.data['y'],
        )
        score_mean_diff = scorer(
            self.estimator,
            self.data['X'], self.data['a'], self.data['y'],
            distance_metric=lambda y0, y1: np.abs(y1.mean() - y0.mean())
        )
        self.assertLess(-score_default, -score_mean_diff)  # Default is standardized, scores are negatives

    def test_multiple_outcome_models(self):
        from causallib.estimation import (
            Standardization, StratifiedStandardization,
            PropensityFeatureStandardization,
            WeightedStandardization,
            TMLE,
            RLearner,
        )
        models = [
            Standardization(LinearRegression()),
            StratifiedStandardization(LinearRegression()),
            PropensityFeatureStandardization(
                Standardization(LinearRegression()),
                IPW(LogisticRegression())
            ),
            WeightedStandardization(
                Standardization(LinearRegression()),
                IPW(LogisticRegression())
            ),
            TMLE(
                Standardization(LinearRegression()),
                IPW(LogisticRegression())
            ),
            RLearner(
                LinearRegression(),
                LinearRegression(),
                LogisticRegression(),
            )
        ]
        for model in models:
            with self.subTest(f"Test scoring of model"):
                model.fit(self.data['X'], self.data['a'], self.data['y'])
                scorer = get_scorer("balanced_residuals_error")
                score = scorer(
                    model,
                    self.data['X'], self.data['a'], self.data['y'],
                )
                self.assertIsInstance(score, float)

