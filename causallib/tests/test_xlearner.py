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

Created on Sep 9, 2021

"""

import unittest
import itertools
import operator
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from causallib.estimation import StratifiedStandardization, XLearner
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class TestXLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = make_classification(n_samples=500,
                                   n_features=3,
                                   n_informative=2,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_classes=2,
                                   n_clusters_per_class=1,
                                   flip_y=0.01,
                                   class_sep=1.0)
        X, a = X[:, :-1], X[:, -1]
        # so at least a quarter of the samples will be positive or negative
        q = np.random.rand() * 1 / 2 + 1 / 4
        # the following is a random treatment assignment
        a = (a > np.percentile(a, q * 100)).astype(int)
        sum_x = np.sum(X[:, :-1], axis=1)
        # The following is a dependent treatment assignment
        dependent_a = sum_x + np.random.rand() * np.max(sum_x) * 0.1
        dependent_a = (dependent_a > np.percentile(dependent_a, q * 100)).astype(int)
        cls.data_2cls = {"X": pd.DataFrame(X),
                         "a": pd.Series(a),
                         "dependent_a": pd.Series(dependent_a),
                         "y": pd.Series(y),
                         "quantile": q}

    def setUp(self):
        self.estimator = XLearner(
            outcome_model=StratifiedStandardization(LinearRegression()),
            effect_model=StratifiedStandardization(LinearRegression())
        )

    def default_fit(self):
        self.estimator.fit(self.data_2cls["X"], self.data_2cls["a"], self.data_2cls["y"])

    def test_dummy(self):
        self.default_fit()
        pred = self.estimator.treatment_model.predict_proba(self.data_2cls["X"])
        expect = np.repeat(np.mean(self.data_2cls["a"]), self.data_2cls["X"].shape[0])
        np.testing.assert_array_almost_equal(pred[:, 1], expect)
        np.testing.assert_array_almost_equal(pred[:, 0], 1 - expect)

    def fit_logistic_treatment_model(self, effect_types='diff'):
        estimator = XLearner(
            outcome_model=StratifiedStandardization(LinearRegression()),
            effect_model=StratifiedStandardization(LinearRegression()),
            treatment_model=LogisticRegression(),
            effect_types=effect_types
        )
        estimator.fit(self.data_2cls["X"], self.data_2cls["a"], self.data_2cls["y"])
        return estimator

    def obtain_dummy_treat_model(self, effect_types='diff'):
        return XLearner(
            outcome_model=StratifiedStandardization(LinearRegression()),
            effect_model=StratifiedStandardization(LinearRegression()),
            effect_types=effect_types
        )

    def test_logistic_fit(self):
        estimator = self.fit_logistic_treatment_model()
        self.assertIsInstance(estimator.treatment_model, LogisticRegression)
        self.assertIsNone(check_is_fitted(estimator.treatment_model, attributes="coef_"))

    def test_logistic_treatment_values(self):
        estimator = self.fit_logistic_treatment_model()
        pred = estimator.treatment_model.predict_proba(self.data_2cls["X"])
        new_model = LogisticRegression().fit(self.data_2cls["X"], self.data_2cls["a"])
        expect = new_model.predict_proba(self.data_2cls["X"])
        np.testing.assert_array_almost_equal(pred, expect)

    def test_outcome_fit(self):
        self.default_fit()
        # Following method throws an exeption if not fitted otherwise returns none
        self.assertIsNone(check_is_fitted(self.estimator.outcome_model.learner[0], attributes="coef_"))
        self.assertIsNone(check_is_fitted(self.estimator.outcome_model.learner[1], attributes="coef_"))

    def test_cate_fit(self):
        self.default_fit()
        # Following method throws an exeption if not fitted otherwise returns none
        self.assertIsNone(check_is_fitted(self.estimator.effect_model.learner[0], attributes="coef_"))
        self.assertIsNone(check_is_fitted(self.estimator.effect_model.learner[1], attributes="coef_"))

    def test_dummy_fit(self):
        self.default_fit()
        # The dummy model does not have an indicator attribute
        try:
            self.estimator.treatment_model.predict([1])
        except NotFittedError:
            self.assertTrue(False)
        self.assertTrue(True)

    def test_population_estimation(self):
        self.default_fit()
        eff = self.estimator.estimate_effect(
            self.data_2cls["X"], self.data_2cls["a"], agg="population"
        )
        self.assertEqual(len(eff), 1)

    def test_individual_estimation(self):
        self.default_fit()
        eff = self.estimator.estimate_effect(
            self.data_2cls["X"],
            self.data_2cls["a"],
            agg="individual"
        )
        self.assertEqual(len(eff), len(self.data_2cls["a"]))

    def test_fixed_effect_diff(self):
        est = self.obtain_dummy_treat_model(effect_types='diff')
        est.fit(
            self.data_2cls["X"],
            self.data_2cls["a"],
            self.data_2cls["a"] * 2
        )
        eff = est.estimate_effect(
            self.data_2cls["X"],
            self.data_2cls["a"],
            agg="individual"
        )
        self.assertEqual(np.mean(eff.values), 2)

    def test_fixed_effect_ratio(self):
        est = self.obtain_dummy_treat_model(effect_types='ratio')
        est.fit(
            self.data_2cls["X"],
            self.data_2cls["a"],
            self.data_2cls["a"] * 2 + 1
        )
        eff = est.estimate_effect(
            self.data_2cls["X"],
            self.data_2cls["a"],
            agg="individual"
        )
        self.assertAlmostEqual(np.mean(eff.values), 3)

    def test_fixed_effect_or(self):
        est = self.obtain_dummy_treat_model(effect_types='or')
        est.fit(
            self.data_2cls["X"],
            self.data_2cls["a"],
            self.data_2cls["a"] * 2 + 2
        )
        eff = est.estimate_effect(
            self.data_2cls["X"],
            self.data_2cls["a"],
            agg="individual"
        )
        self.assertAlmostEqual(np.mean(eff.values), 2 / 3)

    def test_estimate_effect(self):
        self.default_fit()
        eff = self.estimator.estimate_effect(
            self.data_2cls["X"],
            self.data_2cls["a"]
        )
        self.assertEqual(np.mean(np.isnan(eff)), 0)

    def _obtain_zero_effect_set(self):
        big_X = pd.DataFrame(
            np.concatenate((self.data_2cls["X"], self.data_2cls["X"]), axis=0)
        )
        big_y = pd.Series(
            np.concatenate((self.data_2cls["y"], self.data_2cls["y"]))
        )
        big_a = pd.Series(
            np.concatenate((np.ones(len(self.data_2cls["y"])), np.zeros(len(self.data_2cls["y"]))))
        )
        return big_X, big_a, big_y

    def test_zero_effect(self):
        big_X, big_a, big_y = self._obtain_zero_effect_set()
        self.estimator.fit(big_X, big_a, big_y)
        eff = self.estimator.estimate_effect(big_X, big_a)
        self.assertAlmostEqual(eff.values, 0, places=5)

    def test_zero_effect_forests(self):
        big_X, big_a, big_y = self._obtain_zero_effect_set()
        estimator_forest = XLearner(
            outcome_model=StratifiedStandardization(RandomForestClassifier()),
            effect_model=StratifiedStandardization(RandomForestRegressor()),
        )
        estimator_forest.fit(big_X, big_a, big_y)
        eff = estimator_forest.estimate_effect(big_X, big_a)
        np.testing.assert_array_almost_equal(eff.values, 0, decimal=2) # Random forest is more noisy

    @staticmethod
    def _obtain_simple_set():
        y_pred = pd.DataFrame({
            0: [1, 2, 3, 4],
            1: [10, 20, 30, 40],
        })
        y_true = pd.Series([100, 200, 300, 400])
        a = pd.Series([0, 0, 1, 1])
        return y_pred, y_true, a

    def test_diff_imputed_treatment_effect(self):
        y_pred, y_true, a = self._obtain_simple_set()
        imp_effect = self.estimator._obtain_imputed_treatment_effect(
            y_pred, a, y_true,
        )
        expected = pd.Series([
            10 - 100,  # y1 - mu0(X1)
            20 - 200,  # y1 - mu0(X1)
            300 - 3,  # mu1(X0) - y0
            400 - 4,  # mu1(X0) - y0
        ])
        pd.testing.assert_series_equal(imp_effect, expected)

    def test_ratio_imputed_treatment_effect(self):
        y_pred, y_true, a = self._obtain_simple_set()
        est = self.obtain_dummy_treat_model(effect_types='ratio')
        imp_effect = est._obtain_imputed_treatment_effect(
            y_pred, a, y_true
        )
        expected = pd.Series([
            10 / 100,  # y1 - mu0(X1)
            20 / 200,  # y1 - mu0(X1)
            300 / 3,  # mu1(X0) - y0
            400 / 4,  # mu1(X0) - y0
        ])
        pd.testing.assert_series_equal(imp_effect, expected)

    def test_or_imputed_treatment_effect(self):
        y_pred, y_true, a = self._obtain_simple_set()
        est = self.obtain_dummy_treat_model(effect_types='or')
        imp_effect = est._obtain_imputed_treatment_effect(
            y_pred, a, y_true
        )
        or_func = self.estimator.CALCULATE_EFFECT['or']
        expected = pd.Series([
            or_func(10, 100),  # y1 - mu0(X1)
            or_func(20, 200),  # y1 - mu0(X1)
            or_func(300, 3),  # mu1(X0) - y0
            or_func(400, 4)  # mu1(X0) - y0
        ])
        pd.testing.assert_series_equal(imp_effect, expected)

    def test_individual_outcome_warning(self):
        self.default_fit()

        self.assertWarns(
            Warning,
            self.estimator.estimate_individual_outcome, # The method to test for warning
            self.data_2cls['X'], self.data_2cls['a'] # The arguments for assertWarns are passed to the tested method
        )

    def test_individual_outcome_xlearner_outcome_model(self):
        self.default_fit()
        by_xlearner = self.estimator.estimate_individual_outcome(self.data_2cls['X'], self.data_2cls['a'])
        by_outcome = self.estimator.outcome_model.estimate_individual_outcome(self.data_2cls['X'], self.data_2cls['a'])
        np.testing.assert_array_almost_equal(by_xlearner, by_outcome)

    def _obtain_dual_indivdual_estimation(self, effect_types_first='diff', effect_types_second='diff'):
        estimator_first = XLearner(
            outcome_model=StratifiedStandardization(LinearRegression()),
            effect_model=StratifiedStandardization(LinearRegression()),
            effect_types=effect_types_first
        )
        estimator_second = XLearner(
            outcome_model=StratifiedStandardization(LinearRegression()),
            effect_model=StratifiedStandardization(LinearRegression()),
            effect_types=effect_types_second
        )
        mod_y = self.data_2cls['y'] + 2 # the plus two is for the ratio so we will divide by zero
        estimator_first.fit(self.data_2cls['X'], self.data_2cls['a'], mod_y)
        estimator_second.fit(self.data_2cls['X'], self.data_2cls['a'], mod_y)
        est_ef_first = estimator_first.estimate_effect(self.data_2cls['X'], self.data_2cls['a'])
        est_ef_second = estimator_second.estimate_effect(self.data_2cls['X'], self.data_2cls['a'])
        return est_ef_first, est_ef_second

    def test_same_effect_type_diff(self):
        for eff in self.estimator.CALCULATE_EFFECT.keys():
            np.testing.assert_array_almost_equal(
                *self._obtain_dual_indivdual_estimation(effect_types_first=eff, effect_types_second=eff)
            )

    def test_different_effect_type_diff(self):
        eff_list = self.estimator.CALCULATE_EFFECT.keys()
        for eff_frst, eff_scnd in itertools.permutations(eff_list, 2):
            est_ef_fit, est_ef_init = self._obtain_dual_indivdual_estimation(
                effect_types_first=eff_frst, effect_types_second=eff_scnd)
            # numpy does not have assert array not equal
            np.testing.assert_array_compare(operator.__ne__, est_ef_fit, est_ef_init)

    def test_deep_copy_at_fit(self):
        estimator = XLearner(outcome_model=StratifiedStandardization(LinearRegression()))
        self.assertIsNone(estimator.effect_model)
        estimator.fit(self.data_2cls['X'], self.data_2cls['a'], self.data_2cls['y'])
        self.assertIsInstance(estimator.effect_model, StratifiedStandardization)
        self.assertIsNot(estimator.effect_model, estimator.outcome_model)

    def test_predict_proba_miss_match(self):
        estimator_miss_match = XLearner(
            outcome_model=StratifiedStandardization(LogisticRegression(), predict_proba=False),
            effect_model=StratifiedStandardization(LinearRegression(), predict_proba=True),
            # following is the default value it is written to make it more explicit
            predict_proba=True,
        )
        estimator_miss_match.fit(self.data_2cls['X'], self.data_2cls['a'], self.data_2cls['y'])

        estimator_match = XLearner(
            outcome_model=StratifiedStandardization(LogisticRegression(), predict_proba=True),
            effect_model=StratifiedStandardization(LinearRegression(), predict_proba=True),
            # following is the default value it is written to make it more explicit
            predict_proba=True,
        )
        estimator_match.fit(self.data_2cls['X'], self.data_2cls['a'], self.data_2cls['y'])
        miss_match_est = estimator_miss_match.estimate_effect(self.data_2cls['X'], self.data_2cls['a'])
        match_est = estimator_match.estimate_effect(self.data_2cls['X'], self.data_2cls['a'])
        np.testing.assert_array_almost_equal(miss_match_est, match_est)

    def test_predict_proba_False(self):
        estimator_miss_match = XLearner(
            outcome_model=StratifiedStandardization(LogisticRegression(), predict_proba=False),
            effect_model=StratifiedStandardization(LinearRegression()),
            # following is the default value it is written to make it more explicit
            predict_proba=False,
        )
        estimator_miss_match.fit(
            self.data_2cls['X'], self.data_2cls['a'], self.data_2cls['y'])
        outcomes = estimator_miss_match._obtain_ind_outcomes(
            self.data_2cls['X'], self.data_2cls['a'], self.data_2cls['y'])
        np.testing.assert_array_equal(np.sort(np.unique(outcomes)), np.array([0, 1]))

    def test_predict_proba_false_overide(self):
        estimator_miss_match = XLearner(
            outcome_model=StratifiedStandardization(LogisticRegression(), predict_proba=False),
            effect_model=StratifiedStandardization(LinearRegression()),
            # following is the default value it is written to make it more explicit
            predict_proba=True,
        )
        estimator_miss_match.fit(
            self.data_2cls['X'], self.data_2cls['a'], self.data_2cls['y'])
        outcomes = estimator_miss_match._obtain_ind_outcomes(
            self.data_2cls['X'], self.data_2cls['a'], self.data_2cls['y'])
        self.assertFalse(self._is_binary_values_after_rounding(outcomes))

    def test_is_binary_values_after_rounding(self):
        sample_size = 3000
        decimals_number = 5
        decimal_fraction = 10**(-decimals_number)
        binary_vals = np.random.choice([0, 1], size=sample_size)
        noise_values = np.random.rand(sample_size)*decimal_fraction
        to_test_vec = binary_vals+noise_values
        for decimal_test in np.arange(1, decimals_number*2):
            if decimal_test >= decimals_number:
                self.assertFalse(
                    self._is_binary_values_after_rounding(to_test_vec, digit=decimal_test))
            else:
                self.assertTrue(
                    self._is_binary_values_after_rounding(to_test_vec, digit=decimal_test))


    @staticmethod
    def _is_binary_values_after_rounding(vec, digit=3):
        un = np.sort(np.unique(np.round(vec, digit)))
        return np.array_equal(un, [0, 1])

    def test_init_effect(self):
        for effect in ['diff', 'ratio', 'or']:
            est = self.obtain_dummy_treat_model(effect_types=effect)
            self.assertEqual(est.effect_types, effect)

    def test_effect_type_name_population(self):
        agg = "population"
        for effect in ['ratio', 'or', 'diff']:
            est = self.obtain_dummy_treat_model(effect_types=effect)
            est.fit(
                self.data_2cls["X"],
                self.data_2cls["a"],
                self.data_2cls["a"] * 2 + 1
            )
            eff = est.estimate_effect(
                self.data_2cls["X"],
                self.data_2cls["a"],
                agg=agg
            )
        self.assertEqual(eff.index[0], effect)

    def test_effect_type_name_individual(self):
        agg = "individual"
        for effect in ['ratio', 'or', 'diff']:
            est = self.obtain_dummy_treat_model(effect_types=effect)
            est.fit(
                self.data_2cls["X"],
                self.data_2cls["a"],
                self.data_2cls["a"] * 2 + 1
            )
            eff = est.estimate_effect(
                self.data_2cls["X"],
                self.data_2cls["a"],
                agg=agg
            )
            self.assertEqual(eff.name, effect)