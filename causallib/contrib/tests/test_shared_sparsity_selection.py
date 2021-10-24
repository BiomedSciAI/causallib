import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from causallib.contrib.shared_sparsity_selection import SharedSparsityConfounderSelection
from causallib.tests.test_confounder_selection import _TestConfounderSelection


class TestSharedSparsitySelection(_TestConfounderSelection):

    def make_xay(self, n_confounders_a, n_max_confounders_y, n_samples, xay_cols=10, seed=None):
        # rng = np.random.default_rng(seed)
        if seed:
            np.random.seed(seed)
        X, a = make_classification(
            n_samples=n_samples,
            n_features=xay_cols + 1,
            n_informative=int(min(n_confounders_a, xay_cols)),
            n_redundant=0, n_repeated=0, class_sep=10.0,
            n_clusters_per_class=1,
            shuffle=False,  # random_state=seed
        )
        y_confounder_indicator = np.zeros(X.shape[1], dtype=bool)
        y_confounder_indicator[:int(min(n_max_confounders_y, xay_cols))] = 1
        np.random.shuffle(y_confounder_indicator)
        y = X[:, y_confounder_indicator] @ np.random.normal(size=y_confounder_indicator.sum())

        X = StandardScaler().fit_transform(X)
        X = pd.DataFrame(X, columns=["x_" + str(i) for i in range(X.shape[1])])
        a = pd.Series(a)
        y = pd.Series(y)
        return X, a, y

    def test_covariate_subset(self):
        X, a, y = self.make_xay(6, 4, n_samples=100, seed=1)
        true_subset_confounders = ['x_0', 'x_2']  # Matches random seed: 6
        covariates_subset = ['x_0', 'x_2', f'x_{X.shape[1] - 1}', f'x_{X.shape[1] - 3}']

        sss = SharedSparsityConfounderSelection(covariates=covariates_subset)
        sss = self.ensure_covariate_subset(sss, X, a, y, true_subset_confounders)

        np.testing.assert_array_equal(covariates_subset, sss.covariates)
        self.assertEqual(len(covariates_subset), sss.selector_.theta_.shape[0])
        self.assertEqual(2, sss.selector_.theta_.shape[1])  # Two treatments
        self.assertEqual(len(true_subset_confounders), np.sum(np.abs(sss.selector_.theta_[:, 0]) > 0))
        self.assertEqual(len(true_subset_confounders), np.sum(np.abs(sss.selector_.theta_[:, 1]) > 0))

    def test_covariate_subset_binary(self):
        X, a, y = self.make_xay(6, 4, n_samples=100, seed=1)
        true_subset_confounders = ['x_0', 'x_2']  # Matches random seed: 6
        covariates_subset = ['x_0', 'x_2', f'x_{X.shape[1] - 1}', f'x_{X.shape[1] - 3}']
        # Convert to binary:
        true_subset_confounders = X.columns.isin(true_subset_confounders)
        covariates_subset = X.columns.isin(covariates_subset)

        sss = SharedSparsityConfounderSelection(covariates=covariates_subset)
        sss = self.ensure_covariate_subset_binary(sss, X, a, y, true_subset_confounders)

        np.testing.assert_array_equal(covariates_subset, sss.covariates)
        self.assertEqual(covariates_subset.sum(), sss.selector_.theta_.shape[0])
        self.assertEqual(2, sss.selector_.theta_.shape[1])  # Two treatments
        self.assertEqual(sum(true_subset_confounders), np.sum(np.abs(sss.selector_.theta_[:, 0]) > 0))
        self.assertEqual(sum(true_subset_confounders), np.sum(np.abs(sss.selector_.theta_[:, 1]) > 0))

    def test_alphas(self):
        X, a, y = self.make_xay(6, 4, n_samples=100, seed=1)
        alphas = [0, 1]
        for alpha in alphas:
            sss = SharedSparsityConfounderSelection(mcp_alpha=alpha)
            sss.fit(X, a, y)
            Xt = sss.transform(X)
            self.assertSetEqual(set(Xt.columns), {'x_0', 'x_2'})

        with self.assertRaises(AssertionError):
            sss = SharedSparsityConfounderSelection(mcp_alpha=-1)
            sss.fit(X, a, y)

        with self.subTest("shrinkage"):
            strong = SharedSparsityConfounderSelection(mcp_alpha=0.1).fit(X, a, y).selector_.theta_
            weak = SharedSparsityConfounderSelection(mcp_alpha=100).fit(X, a, y).selector_.theta_
            self.assertLess(np.linalg.norm(strong), np.linalg.norm(weak))

    def test_lambdas(self):
        X, a, y = self.make_xay(6, 4, n_samples=100, seed=1)

        with self.subTest("Automatic (default) lambda"):
            sss = SharedSparsityConfounderSelection(mcp_lambda="auto")
            sss.fit(X, a, y)
            expected = 0.2 * np.sqrt(2 * np.log(X.shape[1]) / (X.shape[0] / 2))
            self.assertAlmostEqual(sss.selector_.lmda_, expected)

        with self.subTest("Pre-specified lambda"):
            lmda = 2.1
            sss = SharedSparsityConfounderSelection(mcp_lambda=lmda)
            sss.fit(X, a, y)
            self.assertEqual(sss.selector_.lmda_, lmda)

        with self.subTest("Illegal lambda"):
            with self.assertRaises(AssertionError):
                sss = SharedSparsityConfounderSelection(mcp_lambda=-1)
                sss.fit(X, a, y)

        with self.subTest("shrinkage"):
            weak = SharedSparsityConfounderSelection(mcp_lambda=0.1).fit(X, a, y).selector_.theta_
            strong = SharedSparsityConfounderSelection(mcp_lambda=1).fit(X, a, y).selector_.theta_
            self.assertLess(np.linalg.norm(strong), np.linalg.norm(weak))

    def test_max_iter(self):
        X, a, y = self.make_xay(6, 4, n_samples=100, seed=1)

        with self.subTest("Force convergence warning"):
            sss = SharedSparsityConfounderSelection(max_iter=2)
            with self.assertWarns(ConvergenceWarning):
                sss.fit(X, a, y)

        # with self.subTest("Convergence happens in less than max_iter"):
        #     import timeit
        #     n_repeats = 50
        #     times = []
        #     for max_iter in [10000, 100000]:
        #         # Algorithm will converge long before exceeding `max_iter` and so time should remain similar
        #         sss = SharedSparsityConfounderSelection(max_iter=max_iter)
        #         avg_time = timeit.timeit(lambda: sss.fit(X, a, y), number=n_repeats)
        #         times.append(avg_time)
        #     self.assertAlmostEqual(times[0], times[1], places=1)

    def test_final_selection(self):
        """Test against current implementation to allow for refactoring"""
        X, a, y = self.make_xay(6, 4, n_samples=100, seed=1)
        sss = SharedSparsityConfounderSelection()
        sss.fit(X, a, y)
        Xt = sss.transform(X)
        self.assertSetEqual(set(Xt.columns), {'x_0', 'x_2'})

    def test_importance_getter(self):
        from causallib.preprocessing.confounder_selection import _get_feature_importances

        X, a, y = self.make_xay(2, 2, xay_cols=2, n_samples=100, seed=1)
        sss = SharedSparsityConfounderSelection()
        sss.fit(X, a, y)

        importance = _get_feature_importances(sss, sss.importance_getter)
        expected = np.array([[0.0, 0.0],
                             [5.86299046, 5.94375083],
                             [0.0, 0.0]
                             ])
        np.testing.assert_array_almost_equal(expected.transpose(), importance)
        np.testing.assert_array_almost_equal(sss.selector_.theta_.transpose(), importance)

