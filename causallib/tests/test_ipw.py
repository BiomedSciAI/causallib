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

Created on Jul 22, 2018

"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from causallib.estimation import IPW


class TestIPW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Data:
        X, a = make_classification(n_features=1, n_informative=1, n_redundant=0, n_repeated=0, n_classes=2,
                                   n_clusters_per_class=1, flip_y=0.0, class_sep=10.0)
        cls.data_r_100 = {"X": pd.DataFrame(X), "a": pd.Series(a)}
        X, a = make_classification(n_features=1, n_informative=1, n_redundant=0, n_repeated=0, n_classes=2,
                                   n_clusters_per_class=1, flip_y=0.2, class_sep=10.0)
        cls.data_r_80 = {"X": pd.DataFrame(X), "a": pd.Series(a)}

        # Data that maps x=0->a=0 and x=1->a=1:
        X = pd.Series([0] * 50 + [1] * 50)
        cls.data_cat_r_100 = {"X": X.to_frame(), "a": X}

        # Data that maps x=0->a=0 and x=1->a=1, but 10% of x=0->a=1 and 10% of x=1->a=0:
        X = pd.Series([0] * 40 + [1] * 10 + [1] * 40 + [0] * 10).to_frame()
        a = pd.Series([0] * 50 + [1] * 50)
        cls.data_cat_r_80 = {"X": X, "a": a}

        # Avoids regularization of the model:
        cls.estimator = IPW(LogisticRegression(C=1e6, solver='lbfgs'), clip_min=0.05, clip_max=0.95,
                            use_stabilized=False)

    def setUp(self):
        self.estimator.fit(self.data_r_100["X"], self.data_r_100["a"])

    def test_is_fitted(self):
        self.assertTrue(hasattr(self.estimator.learner, "coef_"))

    def test_weight_matrix_vector_matching(self):
        a = self.data_r_100["a"]
        p_vec = self.estimator.compute_weights(self.data_r_100["X"], a)
        p_mat = self.estimator.compute_weight_matrix(self.data_r_100["X"], a)
        self.assertEqual(p_vec.size, p_mat.shape[0])
        for i in range(a.shape[0]):
            self.assertAlmostEqual(p_mat.loc[i, a[i]], p_vec[i])

    def test_weight_sizes(self):
        a = self.data_r_100["a"]
        with self.subTest("Weight vector size"):
            p = self.estimator.compute_weights(self.data_r_100["X"], a)
            self.assertEqual(len(p.shape), 1)  # vector has no second axis
            self.assertEqual(p.shape[0], a.shape[0])

        with self.subTest("Weight matrix size"):
            p = self.estimator.compute_weight_matrix(self.data_r_100["X"], a)
            self.assertEqual(len(p.shape), 2)  # Matrix has two dimensions
            self.assertEqual(p.shape[0], a.shape[0])
            self.assertEqual(p.shape[1], np.unique(a).size)

    def ensure_truncation(self, test_weights):
        with self.subTest("Estimator initialization parameters"):
            p = self.estimator.compute_propensity(self.data_r_80["X"], self.data_r_80["a"])
            if test_weights:
                p = self.estimator.compute_weights(self.data_r_80["X"], self.data_r_80["a"]).pow(-1)

            self.assertAlmostEqual(p.min(), 0.05)
            self.assertAlmostEqual(p.max(), 1 - 0.05)

        with self.subTest("Overwrite parameters in compute_weights"):
            p = self.estimator.compute_propensity(self.data_r_80["X"], self.data_r_80["a"], clip_min=0.1, clip_max = 0.9)
            if test_weights:
                p = self.estimator.compute_weights(self.data_r_80["X"], self.data_r_80["a"], clip_min=0.1, clip_max=0.9).pow(-1)
            self.assertAlmostEqual(p.min(), 0.1)
            self.assertAlmostEqual(p.max(), 1 - 0.1)

        with self.subTest("Test asymmetric clipping"):
            p = self.estimator.compute_propensity(self.data_r_80["X"], self.data_r_80["a"], clip_min=0.2,
                                                  clip_max=0.9)
            if test_weights:
                p = self.estimator.compute_weights(self.data_r_80["X"], self.data_r_80["a"], clip_min=0.2,
                                                   clip_max=0.9).pow(-1)
            self.assertAlmostEqual(p.min(), 0.2)
            self.assertAlmostEqual(p.max(), 0.9)

        with self.subTest("Test calculation of fraction of clipped observations"):
            probabilities = pd.DataFrame()
            probabilities['col1'] = [0.01, 0.02, 0.03, 0.05, 0.3, 0.6, 0.9, 0.95, 0.99, 0.99]
            probabilities['col2'] = [0.99, 0.98, 0.97, 0.95, 0.7, 0.4, 0.1, 0.05, 0.01, 0.01]
            frac = self.estimator._IPW__count_truncated(probabilities, clip_min=0.05, clip_max=0.95)
            self.assertAlmostEqual(frac, 0.5)

        with self.subTest("Test calculation of fraction of clipped observations - no clipping"):
            probabilities = pd.DataFrame()
            probabilities['col1'] = [0.0, 0.0, 0.0, 1.0, 1.0]
            probabilities['col2'] = [1.0, 1.0, 1.0, 0.0, 0.0]
            frac = self.estimator._IPW__count_truncated(probabilities, clip_min=0.0, clip_max=1.0)
            self.assertAlmostEqual(frac, 0.0)

    def test_weight_truncation(self):
        self.ensure_truncation(test_weights=True)

    def test_propensity_truncation(self):
        self.ensure_truncation(test_weights=False)

        with self.subTest("Illegal truncation values assertion on compute"):
            with self.assertRaises(AssertionError):
                self.estimator.compute_propensity(self.data_r_80["X"], self.data_r_80["a"], clip_min=0.6)
            with self.assertRaises(AssertionError):
                self.estimator.compute_propensity(self.data_r_80["X"], self.data_r_80["a"], clip_max=0.4)
            with self.assertRaises(AssertionError):
                self.estimator.compute_propensity(self.data_r_80["X"], self.data_r_80["a"], clip_min=0.6,
                                                  clip_max=0.9)
            with self.assertRaises(AssertionError):
                self.estimator.compute_propensity(self.data_r_80["X"], self.data_r_80["a"], clip_min=0.1,
                                                  clip_max=0.4)

        with self.subTest("Illegal truncation values assertion on initialization"):
            with self.assertRaises(AssertionError):
                IPW(LogisticRegression(), clip_min=0.6)
            with self.assertRaises(AssertionError):
                IPW(LogisticRegression(), clip_max=0.4)
            with self.assertRaises(AssertionError):
                IPW(LogisticRegression(), clip_min=0.1, clip_max=0.4)
            with self.assertRaises(AssertionError):
                IPW(LogisticRegression(), clip_min=0.6, clip_max=0.9)

    def test_weights_sanity_check(self):
        with self.subTest("Linearly separable X should have perfectly predicted propensity score"):
            p = self.estimator.compute_weights(self.data_r_100["X"], self.data_r_100["a"], clip_min=0.0,
                                               clip_max=1.0).pow(-1)
            np.testing.assert_array_almost_equal(p, np.ones_like(p), decimal=3)

        with self.subTest("Train on bijection X|a data and predict on data where q% are flipped"):
            # Train on data that maps x=0->a=0 and x=1->a=1:
            self.estimator.fit(self.data_cat_r_100["X"], self.data_cat_r_100["a"])
            # Predict on a set with mis-mapping: 10% of x=0 have a=1 and 10% of x=1 have a=0:
            p = self.estimator.compute_weights(self.data_cat_r_80["X"], self.data_cat_r_80["a"], clip_min=0.0, clip_max=1.0).pow(
                -1)
            # Extract subjects with mismatching X-a values:
            mis_assigned = np.logical_xor(self.data_cat_r_80["X"].iloc[:, 0], self.data_cat_r_80["a"])
            # See they have the same rate:
            self.assertAlmostEqual(p.mean(), 1.0 - mis_assigned.mean(), 4)
            np.testing.assert_almost_equal(p.mean(), 1.0 - mis_assigned.mean(), decimal=4)

    def test_forcing_probability_learner(self):
        from sklearn.svm import SVC  # Arbitrary model with decision_function instead of predict_proba
        with self.assertRaises(AttributeError):
            IPW(SVC())

    def test_pipeline_learner(self):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.pipeline import make_pipeline
        learner = make_pipeline(StandardScaler(), MinMaxScaler(), LogisticRegression(solver='lbfgs'))
        with self.subTest("Test initialization with pipeline learner"):
            self.estimator = IPW(learner)
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test fit with pipeline learner"):
            self.estimator.fit(self.data_r_100["X"], self.data_r_100["a"])
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test 'predict' with pipeline learner"):
            self.estimator.compute_weights(self.data_r_100["X"], self.data_r_100["a"])
            self.assertTrue(True)  # Dummy assert for not thrown exception
