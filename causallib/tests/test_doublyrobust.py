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
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from causallib.estimation import DoublyRobustVanilla, DoublyRobustIpFeature, DoublyRobustJoffe
from causallib.estimation import IPW
from causallib.estimation import Standardization, StratifiedStandardization


class TestDoublyRobustBase(unittest.TestCase):
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

    def fit_and_predict_all_learners(self, data, estimator):
        estimator.fit(data["X"], data["a"], data["y"])
        doubly_res = estimator.estimate_population_outcome(data["X"], data["a"], data["y"])
        std_res = estimator.outcome_model.estimate_population_outcome(data["X"], data["a"])
        ipw_res = estimator.weight_model.estimate_population_outcome(data["X"], data["a"], data["y"])
        return doubly_res, std_res, ipw_res

    def ensure_uninformative_tx_leads_to_std_like_results(self, estimator):
        data = self.create_uninformative_tx_dataset()
        doubly_res, std_res, ipw_res = self.fit_and_predict_all_learners(data, estimator)

        with self.subTest("Compare population outcome with Standardization"):
            self.assertAlmostEqual(doubly_res[0], std_res[0])
            self.assertAlmostEqual(doubly_res[1], std_res[1])
        with self.subTest("Compare population outcome with IPW"):
            self.assertNotAlmostEqual(doubly_res[0], ipw_res[0])
            self.assertNotAlmostEqual(doubly_res[1], ipw_res[1])

    def ensure_uninformative_ox_leads_to_ipw_like_results(self, estimator):
        data = self.create_uninformative_ox_dataset()
        doubly_res, std_res, ipw_res = self.fit_and_predict_all_learners(data, estimator)

        with self.subTest("Compare population outcome with Standardization"):
            self.assertAlmostEqual(doubly_res[0], std_res[0])
            self.assertAlmostEqual(doubly_res[1], std_res[1])
        with self.subTest("Compare population outcome with IPW"):
            self.assertAlmostEqual(doubly_res[0], ipw_res[0])
            self.assertAlmostEqual(doubly_res[1], ipw_res[1])

    def ensure_is_fitted(self, estimator):
        data = self.create_uninformative_ox_dataset()
        estimator.fit(data["X"], data["a"], data["y"])
        self.assertTrue(hasattr(estimator.weight_model.learner, "coef_"))
        self.assertTrue(hasattr(estimator.outcome_model.learner, "coef_"))

    def ensure_data_is_separated_between_models(self, estimator, n_added_outcome_model_features):
        data = self.create_uninformative_ox_dataset()
        # Reinitialize estimator:
        estimator = estimator.__class__(estimator.outcome_model, estimator.weight_model,
                                        outcome_covariates=[0, 1, 2, 3], weight_covariates=[3, 4])
        estimator.fit(data["X"], data["a"], data["y"])
        self.assertEqual(estimator.outcome_model.learner.coef_.size,
                         len(estimator.outcome_covariates) + n_added_outcome_model_features)
        self.assertEqual(estimator.weight_model.learner.coef_.size, len(estimator.weight_covariates))

    def ensure_weight_refitting_refits(self, estimator):
        data = self.create_uninformative_ox_dataset()
        with self.subTest("Test first fit of weight_model did fit the model"):
            estimator.fit(data["X"], data["a"], data["y"])
            self.assertEqual(estimator.weight_model.learner.coef_.size, data["X"].shape[1])
        with self.subTest("Test no-refitting does not refit"):
            estimator.weight_model.learner.coef_ = np.zeros_like(estimator.weight_model.learner.coef_)
            estimator.fit(data["X"], data["a"], data["y"], refit_weight_model=False)
            np.testing.assert_array_equal(estimator.weight_model.learner.coef_,
                                          np.zeros_like(estimator.weight_model.learner.coef_))
        with self.subTest("Test yes-refitting does indeed fit"):
            estimator.fit(data["X"], data["a"], data["y"], refit_weight_model=True)
            self.assertTrue(np.any(np.not_equal(estimator.weight_model.learner.coef_,
                                                np.zeros_like(estimator.weight_model.learner.coef_))))

    def ensure_model_combinations_work(self, estimator_class):
        data = self.create_uninformative_ox_dataset()
        for ipw, std in product([IPW], [Standardization, StratifiedStandardization]):
            with self.subTest("Test combination of {} and {} does not crash".format(ipw, std)):
                ipw = ipw(LogisticRegression())
                std = std(LinearRegression())
                dr = estimator_class(std, ipw)
                with self.subTest("Test fit"):
                    dr.fit(data["X"], data["a"], data["y"])
                    self.assertTrue(True)  # Dummy assert, didn't crash
                with self.subTest("Check prediction"):
                    ind_outcome = dr.estimate_individual_outcome(data["X"], data["a"])
                    y = data["y"] if isinstance(dr, DoublyRobustVanilla) else None  # Avoid warnings
                    pop_outcome = dr.estimate_population_outcome(data["X"], data["a"], y)
                    dr.estimate_effect(ind_outcome[1], ind_outcome[0], agg="individual")
                    dr.estimate_effect(pop_outcome[1], pop_outcome[0])
                    self.assertTrue(True)  # Dummy assert, didn't crash

    def ensure_pipeline_learner(self):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.pipeline import make_pipeline
        data = self.create_uninformative_ox_dataset()
        weight_learner = make_pipeline(StandardScaler(), MinMaxScaler(), LogisticRegression())
        outcome_learner = make_pipeline(StandardScaler(), MinMaxScaler(), LinearRegression())

        for ipw, std in product([IPW], [Standardization, StratifiedStandardization]):
            with self.subTest("Test combination of {} and {} does not crash".format(ipw, std)):
                ipw_model = ipw(weight_learner)
                std_model = std(outcome_learner)
                with self.subTest("Test initialization with pipeline learner"):
                    self.estimator = self.estimator.__class__(std_model, ipw_model)
                    self.assertTrue(True)  # Dummy assert for not thrown exception

                with self.subTest("Test fit with pipeline learner"):
                    self.estimator.fit(data["X"], data["a"], data["y"])
                    self.assertTrue(True)  # Dummy assert for not thrown exception

                with self.subTest("Test 'predict' with pipeline learner"):
                    self.estimator.estimate_individual_outcome(data["X"], data["a"])
                    self.assertTrue(True)  # Dummy assert for not thrown exception


class TestDoublyRobustVanilla(TestDoublyRobustBase):
    @classmethod
    def setUpClass(cls):
        TestDoublyRobustBase.setUpClass()
        # Avoids regularization of the model:
        ipw = IPW(LogisticRegression(C=1e6, solver='lbfgs'), use_stabilized=False)
        std = Standardization(LinearRegression(normalize=True))
        cls.estimator = DoublyRobustVanilla(std, ipw)

    def test_uninformative_tx_leads_to_std_like_results(self):
        self.ensure_uninformative_tx_leads_to_std_like_results(self.estimator)

    def test_uninformative_ox_leads_to_ipw_like_results(self):
        self.ensure_uninformative_ox_leads_to_ipw_like_results(self.estimator)

    def test_is_fitted(self):
        self.ensure_is_fitted(self.estimator)

    def test_data_is_separated_between_models(self):
        self.ensure_data_is_separated_between_models(self.estimator, 1)  # 1 treatment assignment feature

    def test_weight_refitting_refits(self):
        self.ensure_weight_refitting_refits(self.estimator)

    def test_model_combinations_work(self):
        self.ensure_model_combinations_work(DoublyRobustVanilla)

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()


class TestDoublyRobustJoffe(TestDoublyRobustBase):
    @classmethod
    def setUpClass(cls):
        TestDoublyRobustBase.setUpClass()
        # Avoids regularization of the model:
        ipw = IPW(LogisticRegression(C=1e6, solver='lbfgs'), use_stabilized=False)
        std = Standardization(LinearRegression(normalize=True))
        cls.estimator = DoublyRobustJoffe(std, ipw)

    def test_uninformative_tx_leads_to_std_like_results(self):
        self.ensure_uninformative_tx_leads_to_std_like_results(self.estimator)

    def test_uninformative_ox_leads_to_ipw_like_results(self):
        self.ensure_uninformative_ox_leads_to_ipw_like_results(self.estimator)

    def test_is_fitted(self):
        self.ensure_is_fitted(self.estimator)

    def test_data_is_separated_between_models(self):
        self.ensure_data_is_separated_between_models(self.estimator, 1)  # 1 treatment assignment feature

    def test_weight_refitting_refits(self):
        self.ensure_weight_refitting_refits(self.estimator)

    def test_model_combinations_work(self):
        self.ensure_model_combinations_work(DoublyRobustJoffe)

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()


class TestDoublyRobustIPFeature(TestDoublyRobustBase):
    @classmethod
    def setUpClass(cls):
        TestDoublyRobustBase.setUpClass()
        # Avoids regularization of the model:
        ipw = IPW(LogisticRegression(C=1e6, solver='lbfgs'), use_stabilized=False)
        std = Standardization(LinearRegression(normalize=True))
        cls.estimator = DoublyRobustIpFeature(std, ipw)

    def fit_and_predict_all_learners(self, data, estimator):
        X, a, y = data["X"], data["a"], data["y"]
        self.estimator.fit(X, a, y)
        doubly_res = self.estimator.estimate_population_outcome(X, a)
        std_res = Standardization(LinearRegression(normalize=True)).fit(X, a, y).estimate_population_outcome(X, a)
        ipw_res = self.estimator.weight_model.estimate_population_outcome(X, a, y)
        return doubly_res, std_res, ipw_res

    def test_uninformative_tx_leads_to_std_like_results(self):
        self.ensure_uninformative_tx_leads_to_std_like_results(self.estimator)

    def test_uninformative_ox_leads_to_ipw_like_results(self):
        self.ensure_uninformative_ox_leads_to_ipw_like_results(self.estimator)

    def test_is_fitted(self):
        self.ensure_is_fitted(self.estimator)

    def test_data_is_separated_between_models(self):
        self.ensure_data_is_separated_between_models(self.estimator, 2 + 1)  # 2 ip-features + 1 treatment assignment

    def test_weight_refitting_refits(self):
        self.ensure_weight_refitting_refits(self.estimator)

    def test_model_combinations_work(self):
        self.ensure_model_combinations_work(DoublyRobustIpFeature)

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()
