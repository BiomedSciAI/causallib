import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression

from causallib.metrics import weighted_roc_auc_error, expected_roc_auc_error
from causallib.metrics import weighted_roc_curve_error, expected_roc_curve_error
from causallib.metrics import ici_error
from causallib.metrics import covariate_balancing_error
from causallib.metrics import covariate_imbalance_count_error
from causallib.metrics import balanced_residuals_error

import sklearn
LR_NO_PENALTY = None if sklearn.__version__ >= "1.2" else "none"


class TestPropensityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Data:
        X, a = make_classification(
            n_features=1, n_informative=1, n_redundant=0, n_repeated=0,
            n_classes=2, n_clusters_per_class=1, flip_y=0.0, class_sep=0.5,
            random_state=0,
        )
        cls.data_r_100 = {"X": pd.DataFrame(X), "a": pd.Series(a)}

        # RCT Data:
        np.random.seed(0)
        X = np.random.normal(0, 1, 10000)
        a = np.random.binomial(1, 0.5, 10000)
        cls.data_rct = {"X": pd.DataFrame(X), "a": pd.Series(a)}

        # Hard-coded Data:
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = pd.Series([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 1])
        y_chance = pd.Series(10 * [0.5])
        w_uniform = pd.Series(10 * [0.1])
        w = pd.Series([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 0, 0, 0])

        cls.hard_coded_data = {
            "y": y,
            "y_chance": y_chance,
            "y_pred": y_pred,
            "w_uniform": w_uniform,
            "w": w
        }

        # # Avoids regularization of the model:
        cls.estimator = LogisticRegression(penalty=LR_NO_PENALTY, solver='sag', max_iter=2000)

    def test_weighted_roc_auc(self):
        with self.subTest("Chance predictions"):
            expected = 0
            observed = weighted_roc_auc_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_chance'],
                sample_weight=self.hard_coded_data['w_uniform']
            )
            self.assertEqual(observed, expected)

        with self.subTest("Random label fitting"):
            self.estimator.fit(self.data_rct['X'], self.data_rct['a'])
            a_pred = self.estimator.predict_proba(self.data_rct['X'])
            ip_weights = 1 / a_pred[range(a_pred.shape[0]), self.data_rct['a']]
            expected = 0
            observed = weighted_roc_auc_error(
                y_true=self.data_rct['a'],
                y_pred=a_pred[:, 1],
                sample_weight=ip_weights
            )
            self.assertAlmostEqual(observed, expected, places=6)

        with self.subTest("Good separation"):
            self.estimator.fit(self.data_r_100['X'], self.data_r_100['a'])
            a_pred = self.estimator.predict_proba(self.data_r_100['X'])
            ip_weights = 1 / a_pred[range(a_pred.shape[0]), self.data_r_100['a']]
            expected = 0.25  # The result for this data
            observed = weighted_roc_auc_error(
                y_true=self.data_r_100['a'],
                y_pred=a_pred[:, 1],
                sample_weight=ip_weights
            )
            self.assertAlmostEqual(observed, expected, places=4)

        with self.subTest("Perfect prediction but uniform weights"):
            # When the weights are uniform, i.e. weight_sample=1/n, W-AUC is
            # similar to AUC. Here, the prediction is perfect and the
            # W-AUC=AUC=1. The squared error is (W-AUC-0.5)**2=0.25
            expected = 0.25
            observed = weighted_roc_auc_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y'],
                sample_weight=self.hard_coded_data['w_uniform']
            )
            self.assertEqual(observed, expected)

        with self.subTest("Uniform weighted AUC"):
            # W-AUC and AUC are similar, but smaller than 1.
            weighted_auc_uniform = weighted_roc_auc_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_pred'],
                sample_weight=self.hard_coded_data['w_uniform']
            )
            self.assertLess(weighted_auc_uniform, 0.25)

        with self.subTest("Non-uniform weights should be closer to the diagonal"):
            # The weights are not uniform, and the W-AUC is closer to the diagonal
            weighted_auc = weighted_roc_auc_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_pred'],
                sample_weight=self.hard_coded_data['w']
            )
            self.assertLess(weighted_auc, weighted_auc_uniform)

    def test_expected_roc_auc_error(self):
        with self.subTest("Identical prediction, perfect alignment"):
            expected_roc_auc = expected_roc_auc_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y']
            )
            self.assertEqual(expected_roc_auc, 0.0)

        with self.subTest("Good calibration"):
            expected_roc_auc = expected_roc_auc_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_pred']
            )
            self.assertAlmostEqual(0, expected_roc_auc, places=2)

    def test_weighted_roc_curve(self):
        with self.subTest("Chance predictions"):
            expected = 0
            observed = weighted_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_chance'],
                sample_weight=self.hard_coded_data['w_uniform']
            )
            self.assertEqual(observed, expected)

        with self.subTest("Random label fitting"):
            self.estimator.fit(self.data_rct['X'], self.data_rct['a'])
            a_pred = self.estimator.predict_proba(self.data_rct['X'])
            ip_weights = 1 / a_pred[range(a_pred.shape[0]), self.data_rct['a']]
            expected = 0  # 0.0130
            observed = weighted_roc_curve_error(
                y_true=self.data_rct['a'],
                y_pred=a_pred[:, 1],
                sample_weight=ip_weights
            )
            self.assertAlmostEqual(observed, expected, places=1)

        with self.subTest("Perfect separation"):
            self.estimator.fit(self.data_r_100['X'], self.data_r_100['a'])
            a_pred = self.estimator.predict_proba(self.data_r_100['X'])
            ip_weights = 1 / a_pred[range(a_pred.shape[0]), self.data_r_100['a']]
            expected = 1.0
            observed = weighted_roc_curve_error(
                y_true=self.data_r_100['a'],
                y_pred=a_pred[:, 1],
                sample_weight=ip_weights
            )
            self.assertAlmostEqual(observed, expected, places=4)

        with self.subTest("Perfect prediction but uniform weights"):
            # perfect prediction so the difference from diagonal is 1 (top-left minus bottom-left corner)
            expected = 1.0
            observed = weighted_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y'],
                sample_weight=self.hard_coded_data['w_uniform']
            )
            self.assertEqual(observed, expected)

        with self.subTest("Uniform weighted ROC curve"):
            expected = 0.4  # The result from this `y_pred` assignment
            weighted_auc_uniform = weighted_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_pred'],
                sample_weight=self.hard_coded_data['w_uniform']
            )
            self.assertEqual(weighted_auc_uniform, expected)

        with self.subTest("Non-uniform weights should be closer to the diagonal"):
            # The weights are not uniform, and the W-AUC is closer to the diagonal
            weighted_auc = weighted_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_pred'],
                sample_weight=self.hard_coded_data['w']
            )
            self.assertLess(weighted_auc, weighted_auc_uniform)

    def test_expected_roc_curve_error(self):
        with self.subTest("Identical prediction, perfect alignment"):
            expected_roc_auc = expected_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y']
            )
            self.assertEqual(expected_roc_auc, 0.0)

        with self.subTest("Good calibration"):
            expected_roc_auc = expected_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y_pred']
            )
            expected = 0.0539
            self.assertAlmostEqual(expected_roc_auc, expected, places=3)

    def test_different_agg_function_in_curve(self):
        with self.subTest("Weighted ROC curve"):
            # perfect prediction so the difference from diagonal is 1 (top-left minus bottom-left corner)
            expected = 1/3  # 3-point curve: bottom-left (0 difference), top-left (1), top-right (0)
            observed = weighted_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y'],
                sample_weight=self.hard_coded_data['w_uniform'],
                agg=np.mean
            )
            self.assertEqual(observed, expected)

        with self.subTest("Expected ROC curve"):
            expected = 0.0  # Equal curve so diff=0 all the way
            observed = expected_roc_curve_error(
                y_true=self.hard_coded_data['y'],
                y_pred=self.hard_coded_data['y'],
                agg=np.mean,
            )
            self.assertEqual(observed, expected)

    def test_ici(self):
        data = {  # Reduce size since lowess can be slow
            'X': self.data_rct['X'].loc[:200],
            'a': self.data_rct['a'].loc[:200],
        }
        self.estimator.fit(data['X'], data['a'])
        a_pred = self.estimator.predict_proba(data['X'])[:, 1]

        with self.subTest("Default parameters"):
            ici_score = ici_error(data['a'], a_pred)
            expected = 0.0239
            self.assertAlmostEqual(ici_score, expected, places=3)

        with self.subTest("`return_sorted==False`"):
            ici_score = ici_error(
                data['a'], a_pred,
                lowess_kwargs={"return_sorted": False}
            )
            expected = 0.0239
            self.assertAlmostEqual(ici_score, expected, places=3)


class TestWeightMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a = pd.Series([0]*6 + [1]*6)
        X = pd.DataFrame(
            {
                "x1": [1, 0] * 6,  # Perfectly balanced
                "x2": [0]*5 + [1, 0] + [1]*5,  # Almost perfectly imbalance (pure imbalance crashes when standardizing)
            }
        )
        w = pd.Series(data=1/6, index=a.index)
        cls.data = {"X": X, "a": a, "w": w}

        # # Avoids regularization of the model:
        cls.estimator = LogisticRegression(penalty=LR_NO_PENALTY, solver='sag', max_iter=2000)

    def test_covariate_balancing(self):
        score = covariate_balancing_error(self.data["X"], self.data["a"], self.data["w"])
        expected = (5/6 - 1/6) / np.sqrt(0.13888 + 0.13888)
        self.assertAlmostEqual(score, expected, places=4)

        with self.subTest("Test `mean` agg function"):
            score = covariate_balancing_error(
                self.data["X"], self.data["a"], self.data["w"],
                agg=np.mean,
            )
            expected /= 2  # Two features, the second has 0 ASMD
            self.assertAlmostEqual(score, expected, places=4)

    def test_covariate_imbalance_count(self):
        with self.subTest("High violation threshold"):
            score = covariate_imbalance_count_error(
                self.data["X"], self.data["a"], self.data["w"],
                threshold=10,
            )
            self.assertEqual(score, 0)

        with self.subTest("Low violation threshold"):
            score = covariate_imbalance_count_error(
                self.data["X"], self.data["a"], self.data["w"],
                threshold=-0.1, fraction=False,
            )
            self.assertEqual(score, self.data["X"].shape[1])

        with self.subTest("Fraction violation threshold"):
            score = covariate_imbalance_count_error(
                self.data["X"], self.data["a"], self.data["w"],
                threshold=0.1, fraction=True,
            )
            self.assertEqual(score, 1/2)

        with self.subTest("Doesn't fail on unrelated kwargs"):
            covariate_imbalance_count_error(
                self.data["X"], self.data["a"], self.data["w"],
                nonexistingkwarg=1,
            )
            self.assertTrue(True)


class TestOutcomeMetrics(unittest.TestCase):
    def test_balanced_residuals(self):
        a = pd.Series([0]*5 + [1]*6)  # Unequal group sizes
        y = pd.Series([8, 9, 10, 11, 12,
                       18, 19, 20, 20, 21, 22])
        y_pred = pd.DataFrame({  # Will only use the observed predictions
            0: [10]*5 + [np.nan]*6,
            1: [np.nan]*5 + [20]*6,
        })

        score = balanced_residuals_error(y, y_pred, a)
        expected = 0
        self.assertAlmostEqual(score, expected)

        with self.subTest("Custom distance metric"):
            from scipy.stats import ks_2samp
            score = balanced_residuals_error(
                y, y_pred, a,
                distance_metric=lambda y0, y1: ks_2samp(y1, y0)[0]  # Return KS statistic
            )
            expected = 0.06666666666  # Result per KS
            self.assertAlmostEqual(score, expected)

        with self.subTest("Biased residuals"):
            bias = 5
            y_pred_bias = y_pred.copy()
            y_pred_bias.loc[a == 1] += bias
            # Simple un-standardized mean-difference should reproduce the bias
            score = balanced_residuals_error(
                y, y_pred_bias, a,
                distance_metric=lambda y1, y0: abs(y1.mean() - y0.mean())
            )
            expected = bias
            self.assertEqual(score, expected)


# if __name__ == '__main__':
#     unittest.main()
