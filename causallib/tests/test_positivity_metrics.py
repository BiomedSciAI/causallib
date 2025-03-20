import unittest
from causallib.positivity import Trimming
from causallib.positivity.metrics import cross_covariance, cross_covariance_score
from causallib.positivity.datasets.positivity_data_simulator import make_multivariate_normal_data
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


class TestScoring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Propensity model without regularization
        cls.trimming = Trimming(
            learner=LogisticRegression(solver="sag", penalty=None),
            threshold="crump"
        )
        # set mean and covariance params
        cls.treatment_2d_params = ([0, 0], [[2, -1], [-1, 2]])
        cls.control_2d_params = ([2, 2], [[1, 0], [0, 1]])
        cls.data_normal_2d = make_multivariate_normal_data(
            treatment_params=cls.treatment_2d_params,
            control_params=cls.control_2d_params)

    def _check_shape_of_cross_cov(self, X, cross_cov):
        for axis in cross_cov.shape:
            self.assertEqual(X.shape[1], axis)

    def _check_output_float_of_a_scorer(self, cross_cov_scorer):
        self.assertTrue(isinstance(cross_cov_scorer, float))

    def _check_output_list_of_a_scorer(self, cross_cov_scorer):
        self.assertTrue(isinstance(cross_cov_scorer, list))

    def test_cross_cov_shapes(self):
        X, a = self.data_normal_2d
        cross_cov_control = cross_covariance(X, a, mean_group=0)
        cross_cov_treatment = cross_covariance(X, a, mean_group=1)
        self._check_shape_of_cross_cov(X, cross_cov_control)
        self._check_shape_of_cross_cov(X, cross_cov_treatment)

    def test_raises_value_error_on_bad_value(self):
        X, a = self.data_normal_2d
        with self.assertRaises(ValueError):
            cross_covariance(X, a, mean_group=3)

    def test_cross_cov_scorer_reducing_functions_to_float(self):
        X, a = self.data_normal_2d
        cross_cov_max_score = cross_covariance_score(X, a, func=np.max)
        cross_cov_mean_trace_score = cross_covariance_score(
            X, a, func=lambda x: np.mean(np.diag(x)))
        self._check_output_float_of_a_scorer(cross_cov_max_score)
        self._check_output_float_of_a_scorer(cross_cov_mean_trace_score)

    def test_cross_cov_scorer_without_summing_scores(self):
        X, a = self.data_normal_2d
        cross_cov_scorer = cross_covariance_score(X, a, sum_scores=False)
        self._check_output_list_of_a_scorer(cross_cov_scorer)
        cross_cov_control_max = np.max(cross_covariance(X, a, mean_group=0))
        self.assertEqual(cross_cov_scorer[0], cross_cov_control_max)

    def test_cross_cov_scorer_off_diagonal(self):
        X, a = self.data_normal_2d
        cross_cov_off_diagonal = cross_covariance_score(
            X, a, off_diagonal_only=True, sum_scores=False, func=lambda x: x)
        self._check_output_list_of_a_scorer(cross_cov_off_diagonal)
        for i in range(len(cross_cov_off_diagonal)):
            self.assertEqual(np.trace(cross_cov_off_diagonal[i]), 0)
            self.assertEqual(np.argmax(cross_cov_off_diagonal[1], axis=0)[0], 1)

    def test_cross_cov_scorer_normalization(self):
        X, a = self.data_normal_2d
        cross_cov_score = cross_covariance_score(X, a)
        cross_cov_normalized_score = cross_covariance_score(
            X, a, normalize=True)
        self._check_output_float_of_a_scorer(cross_cov_score)
        self.assertTrue(cross_cov_normalized_score < cross_cov_score)

    def test_cross_cov_equiv_to_cov_if_done_on_the_same_group(self):
        """
        test that the cross-covariance is equal to covariance
        when applied on the same group
        """
        X, a = self.data_normal_2d
        X_duplicated_treated = pd.concat(
            [X.loc[a == 1, :], X.loc[a == 1, :]], axis=0).reset_index(drop=True)
        a = pd.Series(np.concatenate([np.ones_like(a.loc[a == 1]),
                                      np.zeros_like(a.loc[a == 1])]))
        cross_cov = cross_covariance(X_duplicated_treated, a)
        # true "regular" covariance matrix
        cov = np.cov(X_duplicated_treated.loc[a == 1, :], rowvar=False)

        self._check_shape_of_cross_cov(cov, cross_cov)
        np.testing.assert_array_almost_equal(cross_cov, cov, decimal=6)
        # test variance is close to 1 when normalizing similar groups
        # (not exactly 1 because only part of data is used for cross-cov)
        cross_cov_normalized = cross_covariance_score(
            X_duplicated_treated, a, normalize=True,
            func=lambda x: np.max(np.diag(x)), sum_scores=False)
        self.assertAlmostEqual(1, np.mean(cross_cov_normalized), 1)

    def test_trimming_cross_cov_score(self):
        """test scoring with trimming"""
        X, a = self.data_normal_2d
        self.trimming.fit(X, a)

        with self.subTest("Trimming score"):
            score_max = self.trimming.score(X, a)
            self._check_output_float_of_a_scorer(score_max)

        with self.subTest("Trimming score with mean function"):
            score_mean = self.trimming.score(X, a, func=np.mean)
            self.assertTrue(score_mean < score_max)

        with self.subTest("Trimming score without summing results"):
            score_mean_groups = self.trimming.score(X, a, func=np.mean,
                                                    sum_scores=False)
            self._check_output_list_of_a_scorer(score_mean_groups)
            self.assertTrue(np.max(score_mean_groups) < score_mean)

        with self.subTest("Trimming score off-diagonal"):
            score_max_off_diagonal = self.trimming.score(X, a,
                                                         off_diagonal_only=True)
            self.assertTrue(score_max_off_diagonal < score_max)

    def test_lower_score_with_trimming_in_non_overlap_case(self):
        X, a = self.data_normal_2d
        self.trimming.fit(X, a) # fit as default with crump method
        score_trimming = self.trimming.score(X, a)
        self.trimming.threshold_ = 0
        score_without_trimming = self.trimming.score(X, a)

        self.assertTrue(score_trimming < score_without_trimming)
