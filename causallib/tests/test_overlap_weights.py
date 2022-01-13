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

Created on Jun 09, 2021

"""

import unittest
# from unittest import TestCase

# import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from causallib.estimation.overlap_weights import OverlapWeights


class TestOverlapWeights(unittest.TestCase):
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
        cls.estimator = OverlapWeights(LogisticRegression(C=1e6, solver='lbfgs'), use_stabilized=False)

    def setUp(self):
        self.estimator.fit(self.data_r_100["X"], self.data_r_100["a"])

    def test_classes_number_is_two(self):
        with self.subTest("OW check error arise if single class"):
            a = pd.Series(0, index=self.data_r_100["X"].index)
            with self.assertRaises(AssertionError):
                self.estimator.compute_weight_matrix(self.data_r_100["X"], a)
        with self.subTest("OW check error arise if more than two classes"):
            a = pd.Series([0] * 30 + [1] * 30 + [2] * 40, index=self.data_r_100["X"].index)
            with self.assertRaises(AssertionError):
                self.estimator.compute_weight_matrix(self.data_r_100["X"], a)

    def test_truncate_values_not_none(self):
        with self.assertWarns(RuntimeWarning):
            self.estimator.compute_weight_matrix(
                self.data_r_100["X"], self.data_r_100["a"],
                clip_min=0.2, clip_max=0.8, use_stabilized=None)

    def test_categorical_classes_df_col_names(self):
        a = pd.Series(["a"] * 50 + ["b"] * 50, index=self.data_r_100["X"].index)
        w = self.estimator.compute_weight_matrix(self.data_r_100["X"], a)
        cols_w = w.columns.values.tolist()
        self.assertTrue(cols_w, ["a", "b"])

    def test_ow_weights_reversed_to_propensity(self):
        propensity = self.estimator.learner.predict_proba(self.data_r_100["X"])
        propensity = pd.DataFrame(propensity)
        ow_weights = self.estimator.compute_weight_matrix(self.data_r_100["X"], self.data_r_100["a"],
                                                          clip_min=None, clip_max=None)
        pd.testing.assert_series_equal(propensity.loc[:, 0], ow_weights.loc[:, 1], check_names=False)
        pd.testing.assert_series_equal(propensity.loc[:, 1], ow_weights.loc[:, 0], check_names=False)
        pd.testing.assert_index_equal(propensity.columns, ow_weights.columns)