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
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from warnings import simplefilter, catch_warnings

from causallib.estimation import AIPW, PropensityFeatureStandardization, WeightedStandardization
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
        with catch_warnings():
            simplefilter(action='ignore', category=UserWarning)  # for some of the models using y throws a UserWarning
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

    def ensure_effect_recovery(self, n=1100):
        use_tmle_data = True
        if use_tmle_data:  # Align the datasets to the same attributes
            from causallib.tests.test_tmle import generate_data
            data = generate_data(n, 2, 0, 1, 1, seed=1)
            data['y'] = data['y_cont']
        else:
            data = self.create_uninformative_ox_dataset()
            data['treatment_effect'] = data['beta']

        self.estimator.fit(data['X'], data['a'], data['y'])
        y = data["y"] if isinstance(self.estimator, AIPW) else None  # Avoid warnings
        pop_outcomes = self.estimator.estimate_population_outcome(data['X'], data['a'], y)
        effect = pop_outcomes[1] - pop_outcomes[0]
        np.testing.assert_allclose(
            data['treatment_effect'], effect,
            atol=0.05
        )
        return data

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
                    y = data["y"] if isinstance(dr, AIPW) else None  # Avoid warnings
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

    def ensure_many_models(self, clip_min=None, clip_max=None):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import ElasticNet, RANSACRegressor, HuberRegressor, PassiveAggressiveRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR, LinearSVR

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier

        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        data = self.create_uninformative_ox_dataset()
        for propensity_learner in [GradientBoostingClassifier(n_estimators=10),
                                   RandomForestClassifier(n_estimators=100),
                                   MLPClassifier(hidden_layer_sizes=(5,)),
                                   KNeighborsClassifier(n_neighbors=20)]:
            weight_model = IPW(propensity_learner, clip_min=clip_min, clip_max=clip_max)
            propensity_learner_name = str(propensity_learner).split("(", maxsplit=1)[0]
            for outcome_learner in [GradientBoostingRegressor(n_estimators=10), RandomForestRegressor(n_estimators=10),
                                    MLPRegressor(hidden_layer_sizes=(5,)),
                                    ElasticNet(), RANSACRegressor(), HuberRegressor(), PassiveAggressiveRegressor(),
                                    KNeighborsRegressor(), SVR(), LinearSVR()]:
                outcome_learner_name = str(outcome_learner).split("(", maxsplit=1)[0]
                outcome_model = Standardization(outcome_learner)

                with self.subTest("Test fit & predict using {} & {}".format(propensity_learner_name,
                                                                            outcome_learner_name)):
                    model = self.estimator.__class__(outcome_model, weight_model)
                    model.fit(data["X"], data["a"], data["y"], refit_weight_model=False)
                    model.estimate_individual_outcome(data["X"], data["a"])
                    self.assertTrue(True)  # Fit did not crash


class TestAIPW(TestDoublyRobustBase):
    @classmethod
    def setUpClass(cls):
        TestDoublyRobustBase.setUpClass()
        # Avoids regularization of the model:
        ipw = IPW(LogisticRegression(C=1e6, solver='lbfgs', max_iter=500), use_stabilized=False)
        std = Standardization(LinearRegression(normalize=True))
        cls.estimator = AIPW(std, ipw)

    def test_uninformative_tx_leads_to_std_like_results(self):
        with self.subTest("`overlap_weighting=False`"):
            self.ensure_uninformative_tx_leads_to_std_like_results(self.estimator)

        with self.subTest("`overlap_weighting=True`"):
            self.estimator.overlap_weighting = True
            self.ensure_uninformative_tx_leads_to_std_like_results(self.estimator)
            self.estimator.overlap_weighting = False

    def test_uninformative_ox_leads_to_ipw_like_results(self):
        with self.subTest("`overlap_weighting=False`"):
            self.ensure_uninformative_ox_leads_to_ipw_like_results(self.estimator)

        with self.subTest("`overlap_weighting=True`"):
            self.estimator.overlap_weighting = True
            self.ensure_uninformative_ox_leads_to_ipw_like_results(self.estimator)
            self.estimator.overlap_weighting = False

    def test_is_fitted(self):
        self.ensure_is_fitted(self.estimator)

    def test_data_is_separated_between_models(self):
        self.ensure_data_is_separated_between_models(self.estimator, 1)  # 1 treatment assignment feature

    def test_weight_refitting_refits(self):
        self.ensure_weight_refitting_refits(self.estimator)

    def test_model_combinations_work(self):
        self.ensure_model_combinations_work(AIPW)

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()

    def test_many_models(self):
        self.ensure_many_models()

    def test_effect_recovery(self):
        with self.subTest("`overlap_weighting=False`"):
            self.ensure_effect_recovery()

        with self.subTest("`overlap_weighting=True`"):
            self.estimator.overlap_weighting = True
            self.ensure_effect_recovery()
            self.estimator.overlap_weighting = False

    def test_effect_calculation_against_direct_effect_formula(self):
        from causallib.datasets import load_nhefs
        data = load_nhefs()
        X, a, y = data.X, data.a, data.y
        a = a.astype(float)  # Test the propensity lookup for non-integer values
        estimator = AIPW(
            self.estimator.outcome_model, self.estimator.weight_model,
            overlap_weighting=True,
        )
        estimator.fit(X, a, y)

        # Estimate the effect from the model:
        effect_from_model = estimator.estimate_population_outcome(X, a, y)   # type:pd.Series
        effect_from_model = estimator.estimate_effect(effect_from_model[1], effect_from_model[0])["diff"]

        # Estimate the effect manually using the direct-effect formula
        # (not mitigated by counterfactual outcomes)
        ps = estimator.weight_model.learner.predict_proba(X)[:, 1]
        y_pred = estimator.outcome_model.estimate_individual_outcome(X, a)
        ey0, ey1 = y_pred[0].values, y_pred[1].values
        effect_from_formula = np.mean(
            ((a*y)/ps - (1-a)*y/(1-ps))  # IPW
            - (a-ps)/(ps*(1-ps)) * ((1 - ps)*ey1 + ps*ey0)  # Correction
        )

        np.testing.assert_allclose(effect_from_formula, effect_from_model)

    def test_binary_outcome_effect_recovery(self):
        from causallib.tests.test_tmle import generate_data
        data = generate_data(1100, 2, 0, seed=0)
        data['y'] = data['y_bin']

        for overlap_weights in [False, True]:
            estimator = AIPW(
                Standardization(LogisticRegression(), predict_proba=True),
                IPW(LogisticRegression()),
                overlap_weighting=overlap_weights,
            )
            estimator.fit(data['X'], data['a'], data['y'])
            pop_outcomes = estimator.estimate_population_outcome(data['X'], data['a'], data['y'])
            effect = estimator.estimate_effect(pop_outcomes[1], pop_outcomes[0])['diff']
            np.testing.assert_allclose(
                data["y_propensity"][data['a'] == 1].mean() - data["y_propensity"][data['a'] == 0].mean(),
                effect,
                atol=0.1,
            )

    def test_multiple_treatments_error(self):
        estimator = AIPW(
            self.estimator.outcome_model, self.estimator.weight_model,
            overlap_weighting=True,
        )
        data = self.create_uninformative_tx_dataset()
        data["a"].iloc[-data["a"].shape[0] // 4:] += 1  # Create a dummy third class
        with self.assertRaises(AssertionError):
            estimator.fit(data["X"], data["a"], data["a"])


class TestWeightedStandardization(TestDoublyRobustBase):
    @classmethod
    def setUpClass(cls):
        TestDoublyRobustBase.setUpClass()
        # Avoids regularization of the model:
        ipw = IPW(LogisticRegression(C=1e6, solver='lbfgs'), use_stabilized=False)
        std = Standardization(LinearRegression(normalize=True))
        cls.estimator = WeightedStandardization(std, ipw)

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
        self.ensure_model_combinations_work(WeightedStandardization)

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()

    def test_many_models(self):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import ElasticNet, RANSACRegressor, HuberRegressor, PassiveAggressiveRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR, LinearSVR

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier

        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        data = self.create_uninformative_ox_dataset()

        for propensity_learner in [GradientBoostingClassifier(n_estimators=10),
                                   RandomForestClassifier(n_estimators=100),
                                   MLPClassifier(hidden_layer_sizes=(5,)),
                                   KNeighborsClassifier(n_neighbors=20)]:
            weight_model = IPW(propensity_learner)
            propensity_learner_name = str(propensity_learner).split("(", maxsplit=1)[0]
            for outcome_learner in [GradientBoostingRegressor(n_estimators=10),
                                    RandomForestRegressor(n_estimators=10),
                                    RANSACRegressor(), HuberRegressor(), SVR(), LinearSVR()]:
                outcome_learner_name = str(outcome_learner).split("(", maxsplit=1)[0]
                outcome_model = Standardization(outcome_learner)

                with self.subTest("Test fit using {} & {}".format(propensity_learner_name, outcome_learner_name)):
                    model = self.estimator.__class__(outcome_model, weight_model)
                    model.fit(data["X"], data["a"], data["y"], refit_weight_model=False)
                    self.assertTrue(True)  # Fit did not crash

            for outcome_learner in [MLPRegressor(hidden_layer_sizes=(5,)),
                                    # ElasticNet(),  # supports sample_weights since v0.23, remove to support v<0.23
                                    PassiveAggressiveRegressor(), KNeighborsRegressor()]:
                outcome_learner_name = str(outcome_learner).split("(", maxsplit=1)[0]
                outcome_model = Standardization(outcome_learner)

                with self.subTest("Test fit using {} & {}".format(propensity_learner_name, outcome_learner_name)):
                    model = self.estimator.__class__(outcome_model, weight_model)
                    with self.assertRaises(TypeError):
                        # Joffe forces learning with sample_weights,
                        # not all ML models support that and so calling should fail
                        model.fit(data["X"], data["a"], data["y"], refit_weight_model=False)

    def test_effect_recovery(self):
        self.ensure_effect_recovery()


class TestPropensityFeatureStandardization(TestDoublyRobustBase):
    @classmethod
    def setUpClass(cls):
        TestDoublyRobustBase.setUpClass()
        # Avoids regularization of the model:
        ipw = IPW(LogisticRegression(C=1e6, solver='lbfgs'), use_stabilized=False)
        std = Standardization(LinearRegression(normalize=True))
        cls.estimator = PropensityFeatureStandardization(std, ipw)

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
        self.ensure_data_is_separated_between_models(self.estimator, 1 + 1)  # 1 ip-feature + 1 treatment assignment

    def test_weight_refitting_refits(self):
        self.ensure_weight_refitting_refits(self.estimator)

    def test_model_combinations_work(self):
        self.ensure_model_combinations_work(PropensityFeatureStandardization)

    def test_pipeline_learner(self):
        self.ensure_pipeline_learner()

    def test_many_models(self):
        self.ensure_many_models(clip_min=0.001, clip_max=1-0.001)

    def test_many_feature_types(self):
        with self.subTest("Ensure all expected feature types are supported"):
            feature_types = [
                "weight_vector",  "signed_weight_vector",
                "weight_matrix",  "masked_weight_matrix",
                "propensity_vector", "propensity_matrix",
                "logit_propensity_vector",
            ]
            assert all(self.estimator._get_feature_function(name) for name in feature_types)

        # These two options are from Bang and Robins, and should be theoretically sound,
        # however, they do seem to be less efficient (greater variance) than the other methods.
        sample_size = defaultdict(lambda: 1100)
        sample_size["signed_weight_vector"] = 20000
        sample_size["masked_weight_matrix"] = 20000
        for feature_type in feature_types:
            with self.subTest(f"Testing {feature_type}"):
                self.estimator.feature_type = feature_type

                data = self.ensure_effect_recovery(sample_size[feature_type])

                # Test added covariates:
                X_size = data['X'].shape[1]
                added_covariates = 1 if "vector" in feature_type else 2  # Else it's a matrix
                n_coefs = self.estimator.outcome_model.learner.coef_.size
                self.assertEqual(n_coefs, X_size + added_covariates + 1)  # 1 for treatment assignment

        # with self.subTest("Test signed_weight_vector takes only binary", skip=True):
        #     a = data['a'].copy()
        #     a.iloc[-a.shape[0] // 4:] += 1
        #     self.estimator.feature_type = "signed_weight_vector"
        #     with self.assertRaises(AssertionError):
        #         self.estimator.fit(data['X'], a, data['y'])

    def test_can_fit_after_deepcopy(self):
        # added following https://github.ibm.com/CausalDev/CausalInference/issues/101
        from copy import deepcopy
        estimator_copy = deepcopy(self.estimator)
        data = self.create_uninformative_ox_dataset()
        estimator_copy.fit(data['X'], data['a'], data['y'])
