"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on April 4, 2021
"""
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from causallib.estimation.rlearner import RLearner

from causallib.utils.exceptions import ColumnNameChangeWarning

import sklearn
LR_NO_PENALTY = None if sklearn.__version__ >= "1.2" else "none"


class TestRlearner(unittest.TestCase):
    def setUp(self):
        # Avoid column concatination
        warnings.simplefilter("ignore", category=ColumnNameChangeWarning)

    @staticmethod
    def create_uninformative_tx_dataset():
        n = 100
        beta = 0.4
        X = pd.Series(np.random.normal(size=n))
        a = pd.Series([0] * (n // 2) + [1] * (n // 2))
        y = X.mul(beta)
        return {"X": X.to_frame(), "a": a, "y": y, "beta": beta}

    @staticmethod
    def create_uninformative_ox_dataset():
        n = 100
        beta = 0.4
        X = pd.DataFrame(np.random.normal(size=(n, 5)))
        a = pd.Series([0] * (n // 2) + [1] * (n // 2))
        y = a.mul(beta)
        return {"X": X, "a": a, "y": y, "beta": beta}

    @staticmethod
    def create_complex_dataset_nie_wagner():
        """
        following setup A in the paper of
          Nie, X., & Wager, S.(2017).
          Quasi - oracle estimation of heterogeneous treatment effects

        """
        from sklearn.preprocessing import PolynomialFeatures

        n = 500
        d = 6
        sigma = 0.5
        X = pd.DataFrame(np.random.uniform(size=(n, d)))
        propensity = np.clip(np.sin(np.pi * X.iloc[:, 0] * X.iloc[:, 1]), 0.1,
                             0.9)
        a = pd.Series(np.random.binomial(1, propensity, n))
        baseline_outcome = (
                np.sin(np.pi * X.iloc[:, 0] * X.iloc[:, 1])
                + 2 * (X.iloc[:, 2] - 0.5) ** 2
                + X.iloc[:, 3]
                + 0.5 * X.iloc[:, 4]
        )
        tau = (X.iloc[:, 0] + X.iloc[:, 1]) / 2
        noise = pd.Series(np.random.normal(size=(n,)))
        y = baseline_outcome + (a - 0.5) * tau + sigma * noise

        # pairwise interactions with 7 degrees of freedom
        poly = PolynomialFeatures(7, interaction_only=True, include_bias=False)
        X_pn = pd.DataFrame(poly.fit_transform(X))
        return {"X": X, "X_pn": X_pn, "a": a, "y": y, "tau": tau}

    @staticmethod
    def create_complex_data_for_ate_victor():
        """
        following the simulation at:
        https://www.r-bloggers.com/2017/06/cross-fitting-double-machine-learning-estimator/
        http://aeturrell.com/2018/02/10/econometrics-in-python-partI-ML/

        y = a * \tau + G(x,b) + u
        a = M(x,b) + v

        where u, v are noise terms and M() and G() are function of the covariates
        for the treatment assignment and the outcome surface, respectively.
        """
        from sklearn.datasets import make_spd_matrix

        def g(x):
            return np.power(np.sin(x), 2)

        def m(x, nu=0.0, gamma=1.0):
            return 0.5 / np.pi * (np.sinh(gamma)) / (
                    np.cosh(gamma) - np.cos(x - nu))

        n = 500
        tau = 0.5
        k = 10
        b = 1 / np.arange(1, k + 1)
        sigma = make_spd_matrix(k, random_state=1234)

        X = pd.DataFrame(
            np.random.multivariate_normal(np.ones(k), sigma, size=(n,))
        )
        G = g(np.dot(X, b))
        M = m(np.dot(X, b))
        a = pd.Series(
            M + np.random.standard_normal(size=(n,))
        )
        y = pd.Series(
            np.dot(tau, a) + G + np.random.standard_normal(size=(n,))
        )

        return {"X": X, "a": a, "y": y, "tau": tau}

    def ensure_prepare_data(self, estimator):
        data = self.create_uninformative_ox_dataset()

        with self.subTest(
                "Test that the dimension of the feature space is "
                "equal to the input space"
        ):
            X_outcome, X_treatment, X_effect = estimator._prepare_data(
                data["X"])
            self.assertListEqual(
                [data["X"].shape[1]] * 2,
                [X_outcome.shape[1], X_treatment.shape[1]]
            )
            self.assertEqual(data["X"].shape[1] + 1, X_effect.shape[1])

        with self.subTest("Test smaller feature space for the outcome model"):
            estimator.outcome_covariates = [0, 1, 2]
            X_outcome, X_treatment, X_effect = estimator._prepare_data(
                data["X"])
            self.assertEqual(data["X"].shape[1], X_treatment.shape[1])
            self.assertEqual(3, X_outcome.shape[1])

        with self.subTest("Test no covariate in the effect model"):
            estimator.effect_covariates = list()
            X_outcome, X_treatment, X_effect = estimator._prepare_data(
                data["X"])
            self.assertEqual(1, X_effect.shape[1])

    def ensure_nuisance_models_are_fitted(self, estimator):
        data = self.create_uninformative_ox_dataset()
        estimator.fit(data["X"], data["a"], data["y"])
        self.assertTrue(hasattr(estimator.outcome_model, "coef_"))
        self.assertTrue(hasattr(estimator.treatment_model, "coef_"))

    def ensure_homogeneous_effect_for_none_X_effect(self, estimator):
        """
        when the effect_covariates is equal to list(), we require same effect
        across different covariates.
        """
        data = self.create_uninformative_tx_dataset()
        estimator = estimator.__class__(
            outcome_model=estimator.outcome_model,
            treatment_model=estimator.treatment_model,
            effect_model=estimator.effect_model,
            effect_covariates=list(),
        )

        estimator.fit(data["X"], data["a"], data["y"])
        estimated_effect = estimator.estimate_individual_effect(data["X"])
        np.testing.assert_array_equal(  # Array is constant
            estimated_effect[0], estimated_effect,
        )

    def ensure_estimate_effect_continuous_treatment(self, estimator):
        data = self.create_complex_data_for_ate_victor()
        estimator = estimator.__class__(
            outcome_model=estimator.outcome_model,
            treatment_model=LinearRegression(),
            effect_model=estimator.effect_model,
        )
        with self.subTest("Test fit"):
            estimator.fit(data["X"], data["a"], data["y"])
            self.assertTrue(True)  # Dummy assert, didn't crash
        with self.subTest("Check prediction"):
            individual_cf_outcome = estimator.estimate_individual_outcome(
                data["X"], data["a"], treatment_values=[-0.2, 0.32, 1.1]
            )
            self.assertEqual(3, individual_cf_outcome.shape[1])

    def ensure_naive_model_for_uninformative_nuisance_models_rlearnerlinear(
            self, estimator
    ):
        """
        without any ability to predict the response surface and the treatment
        assignment, the rlearner narrows to a simple naive estimator
        """
        data = self.create_uninformative_ox_dataset()
        X_effect = estimator._extract_effect_model_data(data["X"])

        estimator._fit_linear_effect_model(X_effect, res_a=data["a"],
                                           res_y=data["y"])
        estimated_effect = estimator.estimate_individual_effect(
            data["X"]).mean()
        self.assertAlmostEqual(estimated_effect, data["beta"])
        np.testing.assert_array_almost_equal(
            estimator.effect_model.coef_[1:], [0] * data["X"].shape[1]
        )

    def ensure_cross_fitting_is_not_overfitting(self, estimator):
        """
        In cross-fitting the prediction are done on held-out data and
        """
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, roc_auc_score

        decision_tree_rg = DecisionTreeRegressor(max_depth=12)
        decision_tree_clf = DecisionTreeClassifier(max_depth=12)

        data = self.create_complex_dataset_nie_wagner()
        # fit and predict on the same data
        decision_tree_rg.fit(data["X"], data["y"])
        decision_tree_clf.fit(data["X"], data["a"])
        pred_y_decision_tree = decision_tree_rg.predict(data["X"])
        pred_a_decision_tree = decision_tree_clf.predict_proba(data["X"])[:, 1]

        estimator = estimator.__class__(
            outcome_model=decision_tree_rg,
            treatment_model=decision_tree_clf,
            effect_model=estimator.effect_model,
        )
        X_outcome, X_treatment, X_effect = estimator._prepare_data(data["X"])
        # cross-fitting
        pred_y, _ = estimator._fit_and_predict_model(
            estimator.outcome_model, X_outcome, data["y"], predict_proba=False
        )
        pred_a, _ = estimator._fit_and_predict_model(
            estimator.treatment_model, X_treatment, data["a"],
            predict_proba=True
        )
        self.assertLess(
            mean_squared_error(data["y"], pred_y_decision_tree),
            mean_squared_error(data["y"], pred_y)
        )

        self.assertLess(
            roc_auc_score(data["a"], np.round(pred_a)),
            roc_auc_score(data["a"], np.round(pred_a_decision_tree))
        )

    def ensure_enforce_fit_intercept_false(self, estimator):
        data = self.create_uninformative_ox_dataset()

        with self.assertWarns(UserWarning):
            # Trigger a warning.
            estimator_fit_intercept = estimator.__class__(
                outcome_model=estimator.outcome_model,
                treatment_model=estimator.treatment_model,
                effect_model=LinearRegression(fit_intercept=True),
            )

            estimator_fit_intercept.fit(data["X"], data["a"], data["y"])
            self.assertFalse(estimator_fit_intercept.effect_model.fit_intercept)

            estimator.fit(data["X"], data["a"], data["y"])
            self.assertAlmostEqual(
                estimator.effect_model.coef_[0],
                estimator_fit_intercept.effect_model.coef_[0],
                places=3,
            )

    def ensure_estimate_ate_as_in_literature(self, estimator):
        from sklearn.ensemble import RandomForestRegressor

        estimator = estimator.__class__(
            outcome_model=RandomForestRegressor(max_depth=2),
            treatment_model=RandomForestRegressor(max_depth=2),
            effect_model=LinearRegression(fit_intercept=False),
            effect_covariates=list(),
        )

        res = []
        for i in range(100):
            data = self.create_complex_data_for_ate_victor()
            estimator.fit(data["X"], data["a"], data["y"])
            res.append(np.mean(estimator.estimate_individual_effect(data["X"])))
        pred_ate = np.mean(res)
        self.assertAlmostEqual(pred_ate, 0.5, places=1)

    def ensure_pipeline_linear_gridsearch(self, estimator):
        from sklearn.linear_model import LassoCV
        from sklearn.model_selection import GridSearchCV

        data = self.create_complex_dataset_nie_wagner()

        parameters = {"max_depth": [3, 7, 10]}
        gs = GridSearchCV(RandomForestClassifier(), parameters)
        with self.subTest("Test initialization with models and gridsearch"):
            estimator = estimator.__class__(
                outcome_model=LassoCV(),
                treatment_model=gs,
                effect_model=LassoCV(fit_intercept=False),
                n_splits=2,
            )
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test fit with pipeline estimator"):
            estimator.fit(data["X"], data["a"], data["y"])
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test 'predict' with pipeline learner"):
            estimator.estimate_individual_outcome(data["X"], data["a"])
            self.assertTrue(True)  # Dummy assert for not thrown exception

    def ensure_warning_when_linear_model_and_non_parametric_is_true(self):
        estimator = RLearner(
            outcome_model=LinearRegression(),
            treatment_model=LogisticRegression(),
            effect_model=LinearRegression(fit_intercept=False),
            non_parametric=True,
        )
        data = self.create_uninformative_ox_dataset()
        with self.assertWarns(UserWarning):
            estimator.fit(data["X"], data["a"], data["y"])

    def ensure_rlearner_can_be_evaluated(self):
        """ensure that Rlearner can be evaluated"""
        from causallib.evaluation import evaluate
        from causallib.evaluation.results import ContinuousOutcomeEvaluationResults
        data = self.create_complex_dataset_nie_wagner()
        self.estimator.fit(data["X"], data["a"], data["y"])
        with self.subTest("Test evaluate R-learner"):
            evaluation_results =evaluate(self.estimator, data['X'], data['a'], data['y'])
            self.assertIsNotNone(evaluation_results)  # Dummy assert for not thrown exception
            self.assertIsInstance(evaluation_results, ContinuousOutcomeEvaluationResults)

class TestRLearnerLinear(TestRlearner):
    @classmethod
    def setUpClass(cls):
        TestRlearner.setUpClass()
        # Avoids regularization of the model:
        treatment_model = LogisticRegression(solver="sag", penalty=LR_NO_PENALTY)
        outcome_model = LinearRegression()
        effect_model = LinearRegression(fit_intercept=False)
        cls.estimator = RLearner(
            outcome_model=outcome_model,
            treatment_model=treatment_model,
            effect_model=effect_model,
            non_parametric=False,
        )

    def test_prepare_data(self):
        self.ensure_prepare_data(self.estimator)

    def test_is_fitted(self):
        self.ensure_nuisance_models_are_fitted(self.estimator)

    def test_homogeneous_effect_for_none_X_effect(self):
        self.ensure_homogeneous_effect_for_none_X_effect(self.estimator)

    def test_cross_fitting_is_not_overfitting(self):
        self.ensure_cross_fitting_is_not_overfitting(self.estimator)

    def test_naive_estimator_for_uninformative_nuisance_models(self):
        self.ensure_naive_model_for_uninformative_nuisance_models_rlearnerlinear(
            self.estimator
        )

    def test_enforce_fit_intercept_false(self):
        self.ensure_enforce_fit_intercept_false(self.estimator)

    def test_estimate_effect_continuous_treatment(self):
        self.ensure_estimate_effect_continuous_treatment(self.estimator)

    def test_pipeline_linear_gridsearch(self):
        self.ensure_pipeline_linear_gridsearch(self.estimator)

    def test_warning_when_linear_model_and_non_parametric_is_true(self):
        self.ensure_warning_when_linear_model_and_non_parametric_is_true()

    def test_rlearner_can_be_evaluated(self):
        self.ensure_rlearner_can_be_evaluated()

    @unittest.skip("long testing procedure")
    def test_compare_literature_ate_results(self):
        self.ensure_estimate_ate_as_in_literature(self.estimator)


class TestRlearnerNonparam(TestRlearner):
    @classmethod
    def setUpClass(cls):
        TestRlearner.setUpClass()
        treatment_model = LogisticRegression(solver="sag", penalty=LR_NO_PENALTY)
        outcome_model = LinearRegression()
        effect_model = RandomForestRegressor()
        cls.estimator = RLearner(
            outcome_model=outcome_model,
            treatment_model=treatment_model,
            effect_model=effect_model,
            non_parametric=True,
        )

    def test_prepare_data(self):
        self.ensure_prepare_data(self.estimator)

    def test_homogeneous_effect_for_none_X_effect(self):
        self.ensure_homogeneous_effect_for_none_X_effect(self.estimator)

    def test_is_fitted(self):
        self.ensure_nuisance_models_are_fitted(self.estimator)

    def test_cross_fitting_is_not_overfitting(self):
        self.ensure_cross_fitting_is_not_overfitting(self.estimator)

    def test_estimate_effect_continuous_treatment(self):
        self.ensure_estimate_effect_continuous_treatment(self.estimator)

    @unittest.skip("long testing procedure")
    def test_compare_literature_ate_results(self):
        self.ensure_estimate_ate_as_in_literature(self.estimator)
