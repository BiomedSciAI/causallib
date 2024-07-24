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
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from causallib.preprocessing.confounder_selection import DoubleLASSO, RecursiveConfounderElimination


class _TestConfounderSelection(unittest.TestCase):
    def ensure_covariate_subset(self, selector, X, a, y, true_subset_confounders):
        # covariates_subset = ['x_0', 'x_2', f'x_{X.shape[1] - 1}', f'x_{X.shape[1] - 3}']
        covariates_subset = selector.covariates
        covariates_subset = list(set(covariates_subset) | set(true_subset_confounders))  # Ensure true-covariate in set

        selector.fit(X, a, y)
        X_trans = selector.transform(X)
        self.assertEqual(selector.n_features_, len(true_subset_confounders))
        for element in true_subset_confounders:
            self.assertIn(element, X_trans.columns)
        for element in set(covariates_subset) - set(true_subset_confounders):
            self.assertNotIn(element, X_trans.columns)
        self.assertGreater(X_trans.columns.difference(covariates_subset).size, 0)
        self.assertEqual(
            X_trans.shape[1],
            X.shape[1] - len(covariates_subset) + len(true_subset_confounders)
        )
        return selector

    def ensure_covariate_subset_binary(self, selector, X, a, y, true_subset_confounders):
        covariates_subset = selector.covariates | true_subset_confounders  # Ensure true-covariate in set

        selector.fit(X, a, y)
        X_trans = selector.transform(X)
        self.assertEqual(selector.n_features_, true_subset_confounders.sum())
        for element in X.columns[true_subset_confounders]:
            self.assertIn(element, X_trans.columns)
        for element in X.columns[covariates_subset].difference(X.columns[true_subset_confounders]):
            self.assertNotIn(element, X_trans.columns)
        self.assertGreater(X_trans.columns.difference(X.columns[covariates_subset]).size, 0)
        self.assertEqual(
            X_trans.shape[1],
            X.shape[1] - covariates_subset.sum() + true_subset_confounders.sum()
        )
        return selector


class TestDoubleLasso(_TestConfounderSelection):

    def setUp(self):
        # X is partitioned into two sets of xay_cols (parameter below)
        # First set is related to a and second set is related to y.
        # We generate each of this set using make_classification and
        # then merge them.
        self.xay_cols = 10
        self.min_accuracy = 0.95

    def make_xay(self, n_true_confounders_a, n_true_confounders_y, n_samples, seed=None):
        if seed is None:
            seed_a = random.randint(0, 2**16)
            seed_y = random.randint(0, 2**16)
        else:
            seed_a = seed
            seed_y = seed + 1
        xa, a = make_classification(n_samples=n_samples,
                                    n_features=self.xay_cols + 1,
                                    n_informative=min(n_true_confounders_a, self.xay_cols),
                                    n_redundant=0, n_repeated=0, class_sep=10.0,
                                    n_clusters_per_class=1,
                                    shuffle=False, random_state=seed_a)
        xy, y = make_classification(n_samples=n_samples,
                                    n_features=self.xay_cols + 1,
                                    n_informative=min(n_true_confounders_y, self.xay_cols),
                                    n_redundant=0, n_repeated=0, class_sep=10.0,
                                    n_clusters_per_class=1,
                                    shuffle=False, random_state=seed_y)
        x = np.concatenate((xa, xy), axis=1)
        x = StandardScaler().fit_transform(x)
        xdf = pd.DataFrame(x, columns=["x_" + str(i) for i in range(x.shape[1])])
        adf = pd.Series(a)
        ydf = pd.Series(y)
        return xdf, adf, ydf

    @staticmethod
    def make_simple_data():
        data = [[1, 1, 1, 1, 0, 0] * 5,     # x1 column: same as a
                [0, 0, 1, 0, 1, 1] * 5,     # x2 column: same as y
                [1, 2, 1, 2, 1, 2] * 5,     # x3 column: unrelated
                [1, 1, 1, 1, 0, 0] * 5,     # a (treatment)
                [0, 0, 1, 0, 1, 1] * 5,     # y
                ]
        col_names = ['x1', 'x2', 'x3', 'a', 'y']
        df = pd.DataFrame(list(zip(*data)), columns=col_names)
        X, a, y = df[['x1', 'x2', 'x3']], pd.Series(df['a']), pd.Series(df['y'])
        return X, a, y

    def test_correct_recovery_simple_data(self):
        X, a, y = self.make_simple_data()
        treatment_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        outcome_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        d = DoubleLASSO(treatment_lasso=treatment_lasso,
                        outcome_lasso=outcome_lasso,
                        threshold=0.1)
        d.fit(X, a, y)
        result = d.transform(X)
        treatment_cols = [X.columns[i] for i, c in enumerate(treatment_lasso.coef_.flatten())
                          if abs(c) > 1e-4]
        outcome_cols = [X.columns[i] for i, c in enumerate(outcome_lasso.coef_.flatten())
                        if abs(c) > 1e-4]
        self.assertEqual(len(treatment_cols), 1)
        self.assertEqual(len(outcome_cols), 1)
        self.assertListEqual(['x1', 'x2'], result.columns.tolist())

    @staticmethod
    def fit_transform(X, a, y, threshold=1e-6):
        treatment_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        outcome_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        dl = DoubleLASSO(treatment_lasso, outcome_lasso, threshold=threshold)
        dl.fit(X, a, y)
        result = dl.transform(X)
        score_a = treatment_lasso.score(X, a)
        score_y = outcome_lasso.score(X, y)
        return result, score_a, score_y

    def run_config(self, n_true_confounders_a, n_true_confounders_y, n_samples=100, threshold=1e-6):
        X, a, y = self.make_xay(n_true_confounders_a, n_true_confounders_y, n_samples)
        result, score_a, score_y = self.fit_transform(X, a, y, threshold)
        return result, score_a, score_y

    def test_basic(self):
        # Basic test by varying how many confounders are related to "a" and "y".
        # We also check for accuracy of treatment_lasso and outcome_lasso with
        # retained confounders as a proxy for how well we recover the confounders.
        for i in range(10):
            n_true_confounders_a = i + 1
            n_true_confounders_y = i + 1
            _, score_a, score_y = self.run_config(n_true_confounders_a, n_true_confounders_y)
            self.assertGreaterEqual(score_a, self.min_accuracy)
            self.assertGreaterEqual(score_y, self.min_accuracy)

    def test_threshold(self):
        # Vary zero-threshold and check that we select confounders as expected.
        n_true_confounders_a = 5
        n_true_confounders_y = 3
        for threshold in np.logspace(-6, 2, 9):     # Vary threshold from 1e-6 to 100 in powers of 10
            x, a, y = self.make_xay(n_true_confounders_a, n_true_confounders_y, n_samples=100)
            treatment_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
            outcome_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
            dl = DoubleLASSO(treatment_lasso, outcome_lasso, threshold=threshold)
            dl.fit(x, a, y)
            expected = ((np.abs(treatment_lasso.coef_.flatten()) >= threshold)
                        | (np.abs(outcome_lasso.coef_.flatten()) >= threshold))
            np.testing.assert_array_equal(dl.support_, expected)

    def test_mask_fn(self):

        # Custom merge function 1: True whenever array index modulo 3 is 0.
        def custom_mask_fn_and(tlasso, ylasso):
            t_coef = np.array([True if j % 3 != 2 else False
                               for j in range(len(tlasso.coef_.flatten()))], dtype=bool)
            y_coef = np.array([True if j % 3 != 1 else False
                               for j in range(len(ylasso.coef_.flatten()))], dtype=bool)
            return np.logical_and(t_coef, y_coef)

        # Custom merge function 2: True whenever array index modulo 3 is 0 or 1.
        def custom_mask_fn_or(tlasso, ylasso):
            t_coef = np.array([True if j % 3 == 0 else False
                               for j in range(len(tlasso.coef_.flatten()))], dtype=bool)
            y_coef = np.array([True if j % 3 == 1 else False
                               for j in range(len(ylasso.coef_.flatten()))], dtype=bool)
            return np.logical_or(t_coef, y_coef)

        # Vary number of confounders. Supply custom merge function.
        # Check at every step that the support_ attribute
        # of the fitted DoubleLasso is the same as expected from custom merge function.
        for i in range(10):
            n_true_confounders_a = i + 1
            n_true_confounders_y = i + 1
            X, a, y = self.make_xay(n_true_confounders_a, n_true_confounders_y, n_samples=100)
            treatment_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
            outcome_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)

            dl = DoubleLASSO(treatment_lasso, outcome_lasso, mask_fn=custom_mask_fn_and)
            dl.fit(X, a, y)
            np.testing.assert_array_equal(
                dl.support_, custom_mask_fn_and(treatment_lasso, outcome_lasso)
            )

            dl = DoubleLASSO(treatment_lasso, outcome_lasso, mask_fn=custom_mask_fn_or)
            dl.fit(X, a, y)
            np.testing.assert_array_equal(
                dl.support_, custom_mask_fn_or(treatment_lasso, outcome_lasso)
            )

    def test_covariate_subset(self):
        X, a, y = self.make_xay(2, 2, n_samples=100, seed=6)
        true_subset_confounders = ['x_0']  # Matches random seed: 6
        covariates_subset = ['x_0', 'x_2', f'x_{X.shape[1] - 1}', f'x_{X.shape[1] - 3}']

        treatment_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        outcome_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        dl = DoubleLASSO(
            treatment_lasso, outcome_lasso,
            covariates=covariates_subset,
        )

        dl = self.ensure_covariate_subset(dl, X, a, y, true_subset_confounders)

        np.testing.assert_array_equal(covariates_subset, dl.covariates)
        self.assertEqual(len(covariates_subset), dl.treatment_lasso.coef_.shape[1])
        self.assertEqual(len(covariates_subset), dl.outcome_lasso.coef_.shape[1])

    def test_covariate_subset_binary(self):
        X, a, y = self.make_xay(2, 2, n_samples=100, seed=6)
        true_subset_confounders = ['x_0']  # Matches random seed: 6
        covariates_subset = ['x_0', 'x_2', f'x_{X.shape[1] - 1}', f'x_{X.shape[1] - 3}']
        # Convert to binary:
        true_subset_confounders = X.columns.isin(true_subset_confounders)
        covariates_subset = X.columns.isin(covariates_subset)

        treatment_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        outcome_lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        dl = DoubleLASSO(
            treatment_lasso, outcome_lasso,
            covariates=covariates_subset,
        )

        dl = self.ensure_covariate_subset_binary(dl, X, a, y, true_subset_confounders)

        np.testing.assert_array_equal(covariates_subset, dl.covariates)
        self.assertEqual(dl.covariates.sum(), dl.treatment_lasso.coef_.shape[1])
        self.assertEqual(dl.covariates.sum(), dl.outcome_lasso.coef_.shape[1])

    def test_auto_estimator_initialization(self):
        from sklearn.linear_model import LogisticRegressionCV, LassoCV

        X, a, y = self.make_xay(2, 2, n_samples=100, seed=6)
        y += np.random.normal(size=y.shape)  # Make `y` continuous to check auto-target-detection
        dl = DoubleLASSO()
        dl.fit(X, a, y)

        self.assertIsInstance(dl.treatment_lasso, LogisticRegressionCV)
        self.assertIsInstance(dl.outcome_lasso, LassoCV)

        self.assertIsNone(check_is_fitted(dl.treatment_lasso, "coef_"))
        self.assertIsNone(check_is_fitted(dl.outcome_lasso, "coef_"))

        self.assertEqual(dl.treatment_lasso.coef_.shape[1], X.shape[1])
        self.assertEqual(dl.outcome_lasso.coef_.shape[0], X.shape[1])  # Lasso different shape than LogisticRegression


class TestRecursiveConfounderElimination(_TestConfounderSelection):

    def setUp(self):
        self.max_x_cols = 10
        self.min_accuracy = 0.95

    def make_xay(self, n_true_confounders, n_samples, seed=None):
        xa, y = make_classification(n_samples=n_samples,
                                    n_features=self.max_x_cols + 1,
                                    n_informative=min(n_true_confounders, self.max_x_cols),
                                    n_redundant=0, n_repeated=0, class_sep=10.0,
                                    n_clusters_per_class=1,
                                    shuffle=False, random_state=seed)
        xa = StandardScaler().fit_transform(xa)
        a, x = xa[:, -1], xa[:, :-1]
        xdf = pd.DataFrame(x, columns=["x_" + str(i) for i in range(x.shape[1])])
        adf = pd.Series(a, name="a")
        ydf = pd.Series(y, name="y")
        return xdf, adf, ydf

    @staticmethod
    def fit_transform(X, a, y, n_true_confounders, step=1):
        estimator = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        rfe = RecursiveConfounderElimination(estimator=estimator,
                                             n_features_to_select=n_true_confounders,
                                             step=step)
        rfe.fit(X, a, y)
        score = estimator.score(a.to_frame().join(X.iloc[:, rfe.support_]), y)
        result = rfe.transform(X)
        return result, score

    def run_config(self, n_true_confounders, n_samples=100, step=1):
        X, a, y = self.make_xay(n_true_confounders, n_samples)
        result, score = self.fit_transform(X, a, y, n_true_confounders, step)
        return result, score

    def test_basic(self):
        # Basic test with increasing number of confounders.
        # We check that the required number of confounders are returned.
        # We also check for accuracy of classifier with retained confounders
        # as a proxy for how well we recover the confounders.
        for i in range(self.max_x_cols):
            n_true_confounders = i + 1
            result, score = self.run_config(n_true_confounders=n_true_confounders)
            self.assertEqual(len(result.columns), n_true_confounders)
            self.assertGreaterEqual(score, self.min_accuracy)

    def test_more_features_than_confounders(self):
        # Test for n_features_to_select > X.columns: in this case
        # we should return the entire X as answer.
        # We also check for accuracy of classifier with retained confounders
        # as a proxy for how well we recover the confounders.
        for i in range(10):
            n_true_confounders = self.max_x_cols + i + 1
            result, score = self.run_config(n_true_confounders=n_true_confounders)
            self.assertEqual(len(result.columns), self.max_x_cols)
            self.assertGreaterEqual(score, self.min_accuracy)

    def test_step(self):
        # Test for different step sizes (i.e., confounders to eliminate in each step).
        # We also check for accuracy of classifier with retained confounders
        # as a proxy for how well we recover the confounders.
        n_true_confounders = 3
        for i in range(self.max_x_cols + 5):    # test from step = 1 to > max columns
            step = i + 1
            result, score = self.run_config(n_true_confounders=n_true_confounders, step=step)
            self.assertEqual(len(result.columns), n_true_confounders)
            self.assertGreaterEqual(score, self.min_accuracy)

    def test_covariates_subset(self):
        X, a, y = self.make_xay(2, n_samples=100, seed=6)
        true_subset_confounders = ['x_0']  # Matches random seed: 6
        covariates_subset = ['x_0', 'x_2', f'x_{X.shape[1] - 1}', f'x_{X.shape[1] - 3}']

        rfe = RecursiveConfounderElimination(
            estimator=LogisticRegression(),
            covariates=covariates_subset,
        )

        rfe = self.ensure_covariate_subset(rfe, X, a, y, true_subset_confounders)

        np.testing.assert_array_equal(covariates_subset, rfe.covariates)
        self.assertEqual(len(true_subset_confounders), rfe.n_features_)

    def test_covariate_subset_binary(self):
        X, a, y = self.make_xay(2, n_samples=100, seed=6)
        true_subset_confounders = ['x_0']  # Matches random seed: 6
        covariates_subset = ['x_0', 'x_2', f'x_{X.shape[1] - 1}', f'x_{X.shape[1] - 3}']
        # Convert to binary:
        true_subset_confounders = X.columns.isin(true_subset_confounders)
        covariates_subset = X.columns.isin(covariates_subset)

        rfe = RecursiveConfounderElimination(
            estimator=LogisticRegression(),
            covariates=covariates_subset,
        )

        rfe = self.ensure_covariate_subset_binary(rfe, X, a, y, true_subset_confounders)

        np.testing.assert_array_equal(covariates_subset, rfe.covariates)
        self.assertEqual(true_subset_confounders.sum(), rfe.n_features_)
