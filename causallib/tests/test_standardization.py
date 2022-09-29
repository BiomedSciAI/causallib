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

Created on Aug 08, 2018

"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC

from causallib.estimation import Standardization, StratifiedStandardization


class TestStandardizationCommon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alpha, beta = 5.0, 0.4
        X = pd.Series(np.random.normal(2.0, 1.0, size=100))
        a = pd.Series([0] * 50 + [1] * 50, dtype=np.dtype(int))
        y = X.mul(alpha) + a.mul(beta)
        cls.data_lin = {"X": X.to_frame(), "a": a, "y": y, "alpha": alpha, "beta": beta}
        cls.estimator = None

    def ensure_observed_prediction(self):
        ind_outcome = self.estimator.estimate_individual_outcome(self.data_lin["X"], self.data_lin["a"])
        obs_outcome = np.concatenate((ind_outcome[0][self.data_lin["a"] == 0], ind_outcome[1][self.data_lin["a"] == 1]))
        np.testing.assert_almost_equal(self.data_lin["y"], obs_outcome)

    def ensure_counterfactual_outcomes(self):
        ind_outcome = self.estimator.estimate_individual_outcome(self.data_lin["X"], self.data_lin["a"])
        with self.subTest("Treatment value 0:"):
            np.testing.assert_array_almost_equal(self.data_lin["X"] * self.data_lin["alpha"],
                                                 ind_outcome[0].to_frame())
        with self.subTest("Treatment value 1:"):
            np.testing.assert_array_almost_equal((self.data_lin["X"] * self.data_lin["alpha"]) + self.data_lin["beta"],
                                                 ind_outcome[1].to_frame())

    def ensure_effect_estimation(self):
        with self.subTest("Check by individual effect:"):
            ind_outcome = self.estimator.estimate_individual_outcome(self.data_lin["X"], self.data_lin["a"])
            effect_est = self.estimator.estimate_effect(ind_outcome[1], ind_outcome[0], agg="individual")["diff"]
            np.testing.assert_array_almost_equal(effect_est, np.full_like(effect_est, self.data_lin["beta"]))

        with self.subTest("Check by population effect:"):
            pop_outcome = self.estimator.estimate_population_outcome(self.data_lin["X"], self.data_lin["a"],
                                                                     agg_func="mean")
            effect_est = self.estimator.estimate_effect(pop_outcome[1], pop_outcome[0], agg="population")["diff"]
            self.assertAlmostEqual(effect_est, np.full_like(effect_est, self.data_lin["beta"]), places=5)

    def ensure_pipeline_learner(self):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.pipeline import make_pipeline
        learner = make_pipeline(StandardScaler(), MinMaxScaler(), LinearRegression())
        with self.subTest("Test initialization with pipeline learner"):
            self.estimator = self.estimator.__class__(learner)
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test fit with pipeline learner"):
            self.estimator.fit(self.data_lin["X"], self.data_lin["a"], self.data_lin["y"])
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test 'predict' with pipeline learner"):
            self.estimator.estimate_individual_outcome(self.data_lin["X"], self.data_lin["a"])
            self.assertTrue(True)  # Dummy assert for not thrown exception

    def ensure_many_models(self):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import ElasticNet, RANSACRegressor, HuberRegressor, PassiveAggressiveRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR, LinearSVR

        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        for learner in [GradientBoostingRegressor, RandomForestRegressor, MLPRegressor,
                        ElasticNet, RANSACRegressor, HuberRegressor, PassiveAggressiveRegressor,
                        KNeighborsRegressor, SVR, LinearSVR]:
            learner = learner()
            learner_name = str(learner).split("(", maxsplit=1)[0]
            with self.subTest("Test fit using {learner}".format(learner=learner_name)):
                model = self.estimator.__class__(learner)
                model.fit(self.data_lin["X"], self.data_lin["a"], self.data_lin["y"])
                self.assertTrue(True)  # Fit did not crash


class TestStandardization(TestStandardizationCommon):
    @classmethod
    def setUpClass(cls):
        TestStandardizationCommon.setUpClass()
        # Avoids regularization of the model:
        cls.estimator = Standardization(LinearRegression(normalize=True))

    def setUp(self):
        self.estimator.fit(self.data_lin["X"], self.data_lin["a"], self.data_lin["y"])

    def test_is_fitted(self):
        self.assertTrue(hasattr(self.estimator.learner, "coef_"))

    def test_effect_estimation(self):
        with self.subTest("Check by model coefficient:"):
            self.assertAlmostEqual(self.estimator.learner.coef_[0], self.data_lin["beta"], places=5)
        self.ensure_effect_estimation()

    def test_observed_prediction(self):
        self.ensure_observed_prediction()

    def test_counterfactual_outcomes(self):
        self.ensure_counterfactual_outcomes()

    def test_treatment_encoding(self):
        self.estimator = Standardization(LinearRegression(), encode_treatment=True)
        a = self.data_lin["a"].replace({0: "p", 1: "q"})
        self.estimator.fit(self.data_lin["X"], a, self.data_lin["y"])
        with self.subTest("Treatment encoder created:"):
            self.assertTrue(hasattr(self.estimator, "treatment_encoder_"))
        with self.subTest("Treatment categories properly encoded"):
            self.assertSetEqual({"p", "q"}, set(*self.estimator.treatment_encoder_.categories_))
        with self.subTest("Fitted model has the right size"):
            self.assertEqual(len(self.estimator.learner.coef_), self.data_lin["X"].shape[1] + a.nunique())

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()

    def test_many_models(self):
        self.ensure_many_models()


class TestStandardizationStratified(TestStandardizationCommon):
    @classmethod
    def setUpClass(cls):
        TestStandardizationCommon.setUpClass()
        # Avoids regularization of the model:
        cls.estimator = StratifiedStandardization(LinearRegression(), [0, 1])

    def setUp(self):
        self.estimator.fit(self.data_lin["X"], self.data_lin["a"], self.data_lin["y"])

    def test_is_fitted(self):
        with self.subTest("Test fit for model 0:"):
            self.assertTrue(hasattr(self.estimator.learner[0], "coef_"))
        with self.subTest("Test fit for model 1:"):
            self.assertTrue(hasattr(self.estimator.learner[1], "coef_"))

    def test_effect_estimation(self):
        with self.subTest("Check by model coefficient:"):
            self.assertAlmostEqual(self.estimator.learner[1].intercept_, self.data_lin["beta"], places=5)
            self.assertAlmostEqual(self.estimator.learner[0].intercept_, 0, places=5)
        self.ensure_effect_estimation()

    def test_observed_prediction(self):
        self.ensure_observed_prediction()

    def test_counterfactual_outcomes(self):
        self.ensure_counterfactual_outcomes()

    def test_initialization_without_treatment_values(self):
        estimator = StratifiedStandardization(LinearRegression())
        with self.subTest("Test no-treatment initialization before fit"):
            self.assertIsInstance(estimator.learner, LinearRegression)
        estimator.fit(self.data_lin["X"], self.data_lin["a"], self.data_lin["y"])
        with self.subTest("Test no-treatment initialization after fit"):
            self.assertIsInstance(estimator.learner, dict)
            self.assertSetEqual(set(estimator.learner.keys()), set(self.data_lin["a"]))
        with self.subTest("Test treatment values were added"):
            self.assertIsNone(estimator.treatment_values)  # No treatment values in initiation
            self.assertListEqual([0, 1], estimator.treatment_values_)  # Treatment values added after fit

    def test_initialization_with_dict_of_learners(self):
        learners = {0: LinearRegression(normalize=False),
                    1: Ridge(alpha=5.0)}
        estimator = StratifiedStandardization(learners)
        with self.subTest("Test dictionary-learners initialization before fit"):
            self.assertIsInstance(estimator.learner, dict)
        estimator.fit(self.data_lin["X"], self.data_lin["a"], self.data_lin["y"])
        with self.subTest("Test dictionary-learners keys as expected"):
            self.assertSetEqual(set(estimator.learner.keys()), set(self.data_lin["a"]))
        with self.subTest("Test dictionary-learners types as expected"):
            self.assertIsInstance(estimator.learner, dict)
            self.assertIsInstance(estimator.learner[0], LinearRegression)
            self.assertIsInstance(estimator.learner[1], Ridge)
        with self.subTest("Test dictionary-learners fitted properly"):
            self.assertTrue(hasattr(estimator.learner[0], "coef_"))
            self.assertTrue(hasattr(estimator.learner[1], "coef_"))

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()

    def test_many_models(self):
        self.ensure_many_models()


class TestStandardizationClassification(TestStandardizationCommon):
    @classmethod
    def setUpClass(cls):
        # Three-class outcome, since decision_function might return a vector when n_classes=2, and we wish to check the
        # matrix form of the output behaves as expected:
        X, y = make_classification(n_features=3, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                                   n_clusters_per_class=1, flip_y=0.0, class_sep=10.0)
        X, a = X[:, :-1], X[:, -1]
        a = (a > np.median(a)).astype(int)
        cls.data_3cls = {"X": pd.DataFrame(X), "a": pd.Series(a), "y": pd.Series(y)}

        # X, y = make_classification(n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_classes=2,
        #                            n_clusters_per_class=1, flip_y=0.0, class_sep=10.0)
        # X, a = X[:, :-1], X[:, -1]
        # a = (a > np.median(a)).astype(int)
        # cls.data_2cls = {"X": pd.DataFrame(X), "a": pd.Series(a), "y": pd.Series(y)}

    def verify_individual_multiclass_output(self):
        self.estimator.fit(self.data_3cls["X"], self.data_3cls["a"], self.data_3cls["y"])
        ind_outcome = self.estimator.estimate_individual_outcome(self.data_3cls["X"], self.data_3cls["a"])

        with self.subTest("Output size, # samples:"):
            self.assertEqual(self.data_3cls["X"].shape[0], ind_outcome.shape[0])
        with self.subTest("Output size, # predictions:"):
            with self.subTest("Output's multiindex level names are describing treatment and outcome"):
                self.assertEqual(["a", "y"], ind_outcome.columns.names)
            with self.subTest("Output's number of predictions is the same as number of outcome and treatment values"):
                self.assertEqual(self.data_3cls["a"].nunique() * self.data_3cls["y"].nunique(), ind_outcome.shape[1])
                self.assertEqual(self.data_3cls["a"].nunique(), ind_outcome.columns.get_level_values("a").unique().size)
                self.assertEqual(self.data_3cls["y"].nunique(), ind_outcome.columns.get_level_values("y").unique().size)
        return ind_outcome

    def test_predict_proba(self):
        self.estimator = Standardization(LogisticRegression(C=1e6, solver='lbfgs'), predict_proba=True)
        ind_outcome = self.verify_individual_multiclass_output()
        with self.subTest("Test results are probabilities - sum to 1:"):
            for treatment_value, y_pred in ind_outcome.groupby(level="a", axis="columns"):
                pd.testing.assert_series_equal(pd.Series(1.0, index=y_pred.index), y_pred.sum(axis="columns"))

    def test_decision_function(self):
        self.estimator = Standardization(SVC(decision_function_shape='ovr'), predict_proba=True)
        self.verify_individual_multiclass_output()

    def test_predict(self):
        self.estimator = Standardization(LogisticRegression(C=1e6, solver='lbfgs'), predict_proba=False)
        self.estimator.fit(self.data_3cls["X"], self.data_3cls["a"], self.data_3cls["y"])
        ind_outcome = self.estimator.estimate_individual_outcome(self.data_3cls["X"], self.data_3cls["a"])
        with self.subTest("Output size, # predictions:"):
            self.assertEqual(self.data_3cls["a"].nunique(), ind_outcome.shape[1])
            self.assertNotEqual(self.data_3cls["y"].nunique(), ind_outcome.shape[1])
