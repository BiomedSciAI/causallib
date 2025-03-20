import unittest
from causallib.positivity import Trimming
from causallib.positivity.datasets.positivity_data_simulator import make_1d_normal_distribution_overlap_data
from sklearn.linear_model import LogisticRegression


# todo:
#   - check the minimization criterion - following the paper

class TestTrimming(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Propensity model without regularization
        cls.trimming = Trimming(
            learner=LogisticRegression(solver="sag", penalty=None),
            threshold="auto"
        )

    def test_is_fitted(self):
        X, a = make_1d_normal_distribution_overlap_data()
        self.trimming.fit(X, a)
        self.assertTrue(hasattr(self.trimming.learner, "coef_"))
        self.assertTrue(isinstance(self.trimming.threshold_, float))

    def test_not_trimming_1d_same_distribution(self):
        # Treated and control are drawn from the same distribution with
        # probability to be treated = 0.5
        X, a = make_1d_normal_distribution_overlap_data()
        [X_trimmed, a_trimmed] = self.trimming.fit_transform(X, a)
        self.assertEqual(a_trimmed.shape[0], a.shape[0])

        # Treated and control are drawn from the same distribution with
        # probability to be treated = 0.2
        X, a = make_1d_normal_distribution_overlap_data(probability_treated=0.2)
        [X_trimmed, a_trimmed] = self.trimming.fit_transform(X, a)
        self.assertEqual(a_trimmed.shape[0], a.shape[0])

    def test_trimming_different_distribution(self):
        X, a = make_1d_normal_distribution_overlap_data(
            treatment_params=(-2, 2),
            control_params=(2, 1),
            probability_treated=0.4,
            n_samples=1000
        )
        [X_trimmed_crump, a_trimmed_crump] = self.trimming.fit_transform(X, a)
        n_untrimmed_crump = a_trimmed_crump.shape[0]
        with self.subTest("Trimmed samples with crump"):
            self.assertTrue(n_untrimmed_crump < a.shape[0])

        n_untrimmed_big_threshold = \
            self.trimming.predict(X, a, threshold=0.25).sum()
        with self.subTest("Trimmed samples with large threshold"):
            self.assertTrue(n_untrimmed_big_threshold < a.shape[0])

        with self.subTest("Test crump has more sample than large threshold"):
            self.assertTrue(n_untrimmed_big_threshold < n_untrimmed_crump)

    def test_pipeline(self):
        X, a = make_1d_normal_distribution_overlap_data()

        with self.subTest("Test fit with propensity model"):
            self.trimming.fit(X, a)
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test predict of the filtered data"):
            self.trimming.predict(X, a)
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test transform"):
            self.trimming.transform(X, a)
            self.assertTrue(True)  # Dummy assert for not thrown exception

    def test_hardcoded_threshold_initialization_chooses_no_method(self):
        self.trimming = Trimming(threshold=0.1)
        self.assertFalse(hasattr(self.trimming, 'threshold'))

    def test_default_initialization_chooses_crump(self):
        self.trimming = Trimming()
        self.assertEqual(self.trimming.threshold, "crump")
