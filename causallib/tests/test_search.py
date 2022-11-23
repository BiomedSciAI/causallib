import unittest
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from causallib.estimation import IPW
from causallib.estimation import AIPW
from causallib.estimation import StratifiedStandardization, Standardization
from causallib.metrics import get_scorer

from causallib.model_selection import causalize_searcher


class TestGridSearch(unittest.TestCase):
    @classmethod
    def _generate_data(cls, n=100, d=4):
        np.random.seed(1)
        priors = np.random.uniform(0.1, 0.9, size=d)
        X = np.random.binomial(1, priors, size=(n, d))

        beta_a = np.random.normal(size=d)
        intercept_a = np.random.normal()
        a_logit = intercept_a + X @ beta_a + np.random.normal(size=n)
        a_propensity = 1 / (1 + np.exp(-a_logit))
        a = np.random.binomial(1, a_propensity)

        beta_y = np.random.normal(size=d)
        intercept_y = np.random.normal()
        treatment_effect = -3
        y = intercept_y + X @ beta_y + treatment_effect * a + np.random.normal(size=n)

        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        a = pd.Series(a, name="treatment")
        y = pd.Series(y, name="outcome")
        data = dict(X=X, a=a, y=y)
        return data

    @classmethod
    def setUpClass(cls):
        cls.data = cls._generate_data()

    def _fit_search_model(self, estimator, scoring, param_grid, data=None, cv=2, **search_kwargs):
        if data is None:
            data = self.data
        CausalGridSearchCV = causalize_searcher(GridSearchCV)
        grid_model = CausalGridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            **search_kwargs,
        )
        grid_model.fit(data['X'], data['a'], data['y'])
        return grid_model

    def test_single_propensity_scorer(self):
        data = self.data
        model = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000))
        scorer = get_scorer("weighted_roc_auc_error")
        grid_model = self._fit_search_model(model, scorer, dict(clip_min=[0.2, 0.3]))

        self.assertIsInstance(grid_model.best_estimator_, model.__class__)
        grid_model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y'])
        pd.testing.assert_series_equal(
            grid_model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y']),
            grid_model.estimate_population_outcome(data['X'], data['a'], data['y'])
        )

        with self.subTest("Grid model has its inner estimator's methods that work as expected"):
            propensities = grid_model.compute_propensity(data['X'], data['a'])
            self.assertLessEqual(0.2, propensities.min())

            weights = grid_model.compute_weights(data['X'], data['a'])
            self.assertIsInstance(weights, pd.Series)
            weights = grid_model.compute_weight_matrix(data['X'], data['a'])
            self.assertIsInstance(weights, pd.DataFrame)

    def test_single_outcome_scorer(self):
        data = self.data
        model = StratifiedStandardization(GradientBoostingRegressor())
        # Can handle the `learner` being a dict because before `fit` it is not a dict, but a single model type
        scorer = get_scorer("balanced_residuals_error")
        grid_model = self._fit_search_model(model, scorer, dict(learner__n_estimators=[10, 20]))

        self.assertIsInstance(grid_model.best_estimator_, model.__class__)
        pd.testing.assert_series_equal(
            grid_model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y']),
            grid_model.estimate_population_outcome(data['X'], data['a'], data['y'])
        )

        with self.subTest("Estimator behaves as inner estimator"):
            ind_outcomes = grid_model.estimate_individual_outcome(data['X'], data['a'])
            self.assertEqual(data['X'].shape[0], ind_outcomes.shape[0])
            self.assertEqual(2, ind_outcomes.shape[1])

    # @unittest.expectedFailure
    def test_different_stratified_estimators(self):
        """Should fail because can't assign the params once the `learner` is a dictionary."""
        data = self.data
        model = StratifiedStandardization({
            0: GradientBoostingRegressor(), 1: LinearRegression(),
        })
        scorer = get_scorer("balanced_residuals_error")
        with self.assertRaises(AttributeError):
            grid_model = self._fit_search_model(model, scorer, dict(learner__n_estimators=[10, 20]))
        # self.assertIsInstance(grid_model.best_estimator_, model.__class__)
        # grid_model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y'])

    def test_single_propensity_scorer_name(self):
        data = self.data
        model = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000))
        grid_model = self._fit_search_model(model, "weighted_roc_auc_error", dict(clip_min=[0.2, 0.3]))

        self.assertIsInstance(grid_model.best_estimator_, model.__class__)
        grid_model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y'])
        pd.testing.assert_series_equal(
            grid_model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y']),
            grid_model.estimate_population_outcome(data['X'], data['a'], data['y'])
        )

    def test_multiple_scorers(self):
        data = self.data
        model = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000, random_state=0))
        scorers_dict = {
            "weighted_roc_auc_error": get_scorer("weighted_roc_auc_error"),
            "covariate_balancing_error": get_scorer("covariate_balancing_error"),
        }
        scorers_list = ["weighted_roc_auc_error", "covariate_balancing_error"]
        scorers_tuple = ("weighted_roc_auc_error", "covariate_balancing_error")
        scorers_set = {"weighted_roc_auc_error", "covariate_balancing_error"}
        grid_model_dict = self._fit_search_model(
            model, scorers_dict,
            dict(clip_min=[0.05, 0.2]),
            refit="weighted_roc_auc_error",
        )
        grid_model_list = self._fit_search_model(
            model, scorers_list,
            dict(clip_min=[0.05, 0.2]),
            refit="weighted_roc_auc_error"
        )
        grid_model_tuple = self._fit_search_model(
            model, scorers_tuple,
            dict(clip_min=[0.05, 0.2]),
            refit="weighted_roc_auc_error"
        )
        grid_model_set = self._fit_search_model(
            model, scorers_set,
            dict(clip_min=[0.05, 0.2]),
            refit="weighted_roc_auc_error"
        )

        # dict with list
        self.assertEqual(
            grid_model_dict.best_estimator_.clip_min,
            grid_model_list.best_estimator_.clip_min,
        )
        np.testing.assert_array_almost_equal(
            grid_model_dict.best_estimator_.learner.coef_,
            grid_model_list.best_estimator_.learner.coef_,
        )
        pd.testing.assert_series_equal(
            grid_model_dict.estimate_population_outcome(data['X'], data['a'], data['y']),
            grid_model_list.estimate_population_outcome(data['X'], data['a'], data['y']),
            check_exact=False,
        )
        # tuple with list
        self.assertEqual(
            grid_model_tuple.best_estimator_.clip_min,
            grid_model_list.best_estimator_.clip_min,
        )
        np.testing.assert_array_almost_equal(
            grid_model_tuple.best_estimator_.learner.coef_,
            grid_model_list.best_estimator_.learner.coef_,
        )
        pd.testing.assert_series_equal(
            grid_model_tuple.estimate_population_outcome(data['X'], data['a'], data['y']),
            grid_model_list.estimate_population_outcome(data['X'], data['a'], data['y']),
            check_exact=False,
        )
        # set with list
        self.assertEqual(
            grid_model_set.best_estimator_.clip_min,
            grid_model_list.best_estimator_.clip_min,
        )
        np.testing.assert_array_almost_equal(
            grid_model_set.best_estimator_.learner.coef_,
            grid_model_list.best_estimator_.learner.coef_,
        )
        pd.testing.assert_series_equal(
            grid_model_set.estimate_population_outcome(data['X'], data['a'], data['y']),
            grid_model_list.estimate_population_outcome(data['X'], data['a'], data['y']),
            check_exact=False,
        )

    def test_bad_scorer(self):
        model = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000))
        CausalGridSearchCV = causalize_searcher(GridSearchCV)
        with self.assertRaises(ValueError):
            grid_model = CausalGridSearchCV(
                model,
                param_grid=dict(clip_min=[0.2, 0.3]),
                scoring=3,
            )

    def test_doubly_robust(self):
        data = self.data
        CausalGridSearchCV = causalize_searcher(GridSearchCV)

        ipw = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000))
        ipw_grid_model = CausalGridSearchCV(
            ipw,
            param_grid=dict(clip_min=[0.2, 0.3]), cv=2,
            scoring=get_scorer("weighted_roc_auc_error"),
        )

        std = Standardization(GradientBoostingRegressor())
        CausalGridSearchCV = causalize_searcher(GridSearchCV)
        std_grid_model = CausalGridSearchCV(
            std,
            param_grid=dict(learner__n_estimators=[10, 20]), cv=3,
            scoring=get_scorer("balanced_residuals_error"),
        )

        model = AIPW(
            outcome_model=std_grid_model,
            weight_model=ipw_grid_model,
        )

        model.fit(data['X'], data['a'], data['y'])
        propensities = model.weight_model.compute_propensity(data['X'], data['a'])
        self.assertLessEqual(0.2, propensities.min())
        model.estimate_population_outcome(data['X'], data['a'], data['y'])
        pd.testing.assert_frame_equal(
            model.estimate_individual_outcome(data['X'], data['a']),
            model.outcome_model.estimate_individual_outcome(data['X'], data['a'])
            # AIPW's individual outcome is the internal standardization individual outcome
        )

    def test_with_survival_ipw(self):
        from causallib.survival import WeightedSurvival, WeightedStandardizedSurvival
        from causallib.datasets import load_nhefs_survival
        from sklearn.dummy import DummyClassifier

        data = load_nhefs_survival()
        idx = data.X.sample(n=100, random_state=0).index
        X, a, t, y = data.X.loc[idx], data.a.loc[idx], data.t.loc[idx], data.y.loc[idx]
        CausalGridSearchCV = causalize_searcher(GridSearchCV)

        ipw = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000))
        ipw_grid_model = CausalGridSearchCV(
            ipw,
            param_grid=dict(clip_min=[0.2, 0.3]), cv=2,
            scoring="weighted_roc_auc_error",
        )

        for Model in [WeightedSurvival, WeightedStandardizedSurvival]:
            model = Model(
                weight_model=ipw_grid_model,
                survival_model=DummyClassifier(strategy="prior")
            )
            model.fit(X, a, t, y)

            self.assertIsInstance(model.weight_model, CausalGridSearchCV)

            propensities = model.weight_model.compute_propensity(X, a)
            self.assertLessEqual(0.2, propensities.min())

            outcomes = model.estimate_population_outcome(X, a, t, y, timeline_start=1)
            self.assertEqual(outcomes.shape, (t.max(), a.nunique()))

    def test_selecting_different_core_estimators(self):
        from sklearn.dummy import DummyClassifier
        data = self.data
        model = IPW(DummyClassifier())  # DummyClassifier used just for initialization
        scorer = get_scorer("weighted_roc_auc_error")
        param_grid = dict(
            learner=[
                LogisticRegression(penalty="none", solver="saga", max_iter=10000, random_state=0),
                LogisticRegression(penalty="l1", solver="saga", max_iter=10000, random_state=0),
                LogisticRegression(penalty="l2", solver="saga", max_iter=10000, random_state=0),
                GradientBoostingClassifier(random_state=0),
            ]
        )
        grid_model = self._fit_search_model(
            model,
            scorer,
            param_grid
        )

        self.assertIsInstance(grid_model.best_estimator_, model.__class__)
        self.assertNotIsInstance(grid_model.best_estimator_, DummyClassifier)
        self.assertIsInstance(grid_model.best_estimator_.learner, LogisticRegression)  # Best base estimator
        outcomes = grid_model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y'])
        self.assertIsInstance(outcomes, pd.Series)

    def test_different_search_algorithm(self):
        from causallib.model_selection import RandomizedSearchCV
        from scipy.stats import uniform

        data = self.data
        ipw = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000))
        model = RandomizedSearchCV(
            ipw,
            param_distributions=dict(clip_min=uniform(loc=0, scale=0.5)), cv=2,
            scoring=get_scorer("weighted_roc_auc_error"),
        )
        model.fit(data['X'], data['a'], data['y'])
        self.assertIsInstance(model.best_estimator_, ipw.__class__)
        model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y'])
        pd.testing.assert_series_equal(
            model.best_estimator_.estimate_population_outcome(data['X'], data['a'], data['y']),
            model.estimate_population_outcome(data['X'], data['a'], data['y'])
        )

    def test_with_custom_kfold(self):
        from causallib.model_selection import TreatmentOutcomeStratifiedKFold
        from causallib.model_selection import TreatmentStratifiedKFold

        data = self.data
        y = np.random.binomial(2, 0.5, size=data['y'].shape[0])  # binary y, for doubly stratification
        y = pd.Series(y, index=data['y'].index)
        model = IPW(LogisticRegression(penalty="none", solver="saga", max_iter=10000))
        scorer = get_scorer("weighted_roc_auc_error")

        with self.subTest("Doubly Stratified K-fold"):
            cv = TreatmentOutcomeStratifiedKFold(n_splits=2)
            treatment_outcome_grid_model = self._fit_search_model(
                model, scorer,
                dict(clip_min=[0.2, 0.3]),
                cv=cv,
                data={**data, "y": y},
            )
            outcomes = treatment_outcome_grid_model.estimate_population_outcome(data['X'], data['a'], data['y'])
            self.assertIsInstance(outcomes, pd.Series)

        with self.subTest("Treatment-stratified k-fold"):
            cv = TreatmentStratifiedKFold(n_splits=2)
            treatment_grid_model = self._fit_search_model(
                model, scorer,
                dict(clip_min=[0.2, 0.3]),
                cv=cv,
                data={**data, "y": y},
            )
            outcomes = treatment_grid_model.estimate_population_outcome(data['X'], data['a'], data['y'])
            self.assertIsInstance(outcomes, pd.Series)
