import unittest
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

from causallib.contrib.adversarial_balancing import AdversarialBalancing
from causallib.evaluation import evaluate


class TestAdversarialBalancing(unittest.TestCase):
    @staticmethod
    def create_identical_treatment_groups_data(n=100):
        np.random.seed(42)
        X = np.random.rand(n, 3)
        X = np.row_stack((X, X))  # Duplicate identical samples
        a = np.array([1] * n + [0] * n)  # Give duplicated samples different treatment assignment
        X, a = pd.DataFrame(X), pd.Series(a)
        return X, a

    @staticmethod
    def create_randomly_assigned_treatment(n=100, p=0.3):
        np.random.seed(42)
        X = np.random.rand(n, 3)
        a = np.random.binomial(1, p, n)
        X, a = pd.DataFrame(X), pd.Series(a)
        return X, a

    def test_identical_treatment_groups(self):
        """An identical patient in both groups should have the same weight in both groups"""
        X, a = self.create_identical_treatment_groups_data()

        estimator = AdversarialBalancing(LogisticRegression(solver='lbfgs', max_iter=500),
                                         loss_type="01")
        estimator.fit(X, a)
        w = estimator.compute_weights(X, a)

        with self.subTest("Equal weights to the same individual under both treatment groups"):
            w_treated = w[a == 1].values
            w_control = w[a == 0].values
            # # w_treated[i] and w_control[i] refer to the same sample at X[i]
            np.testing.assert_allclose(w_control, w_treated)

        with self.subTest("Identical weights for all samples"):
            # Slightly stronger test than the test above of the same weights *given an individual*
            np.testing.assert_allclose(w, w.mean())
            self.assertAlmostEqual(w.std(), 0)

        with self.subTest("Test discriminator zero-one loss is 0.5"):
            np.testing.assert_allclose(estimator.discriminator_loss_,
                                       np.full_like(estimator.discriminator_loss_, 0.5))

    def test_effect_estimation(self):
        X, a = self.create_randomly_assigned_treatment()
        y = X @ np.random.rand(X.shape[1]) + a * 2
        estimator = AdversarialBalancing(LogisticRegression(solver='lbfgs', max_iter=500))
        estimator.fit(X, a)
        outcomes = estimator.estimate_population_outcome(X, a, y)
        effect = estimator.estimate_effect(outcomes[1], outcomes[0])

        self.assertAlmostEqual(effect["diff"], 2, delta=1e-2)

    def test_generated_attributes(self):
        X, a = self.create_identical_treatment_groups_data()
        n_iter = 10
        n_a = len(np.unique(a))
        estimator = AdversarialBalancing(LogisticRegression(solver='lbfgs', max_iter=500),
                                         iterations=n_iter)
        estimator.fit(X, a)

        self.assertEqual(estimator.iterative_models_.shape[1], n_iter)
        self.assertEqual(estimator.iterative_normalizing_consts_.shape[1], n_iter)
        self.assertEqual(estimator.discriminator_loss_.shape[1], n_iter)

        self.assertEqual(estimator.iterative_models_.shape[0], n_a)
        self.assertEqual(estimator.iterative_normalizing_consts_.shape[0], n_a)
        self.assertEqual(estimator.discriminator_loss_.shape[0], n_a)

    def test_stabilization(self):
        X, a = self.create_identical_treatment_groups_data()
        estimator = AdversarialBalancing(LogisticRegression(solver='lbfgs', max_iter=500),
                                         use_stabilized=True)
        estimator.fit(X, a)

        with self.subTest("Test frequency attributes is created and correct"):
            self.assertEqual(estimator.treatments_frequency_, {0: 0.5, 1: 0.5})

        with self.subTest("Equal weights to the same individual under both treatment groups"):
            w = estimator.compute_weights(X, a)
            w_treated = w[a == 1].values
            w_control = w[a == 0].values
            # # w_treated[i] and w_control[i] refer to the same sample at X[i]
            np.testing.assert_allclose(w_control, w_treated)

        X, a = self.create_randomly_assigned_treatment()
        estimator = AdversarialBalancing(LogisticRegression(solver='lbfgs', max_iter=500),
                                         use_stabilized=True)
        estimator.fit(X, a)
        with self.subTest("Weights of each group sum to group-sizes"):
            w = estimator.compute_weights(X, a)
            self.assertAlmostEqual(w[a == 1].sum(), sum(a == 1))
            self.assertAlmostEqual(w[a == 0].sum(), sum(a == 0))

    def test_initialization_with_search_cv(self):
        X, a = self.create_identical_treatment_groups_data()
        learner = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=500),
                               param_grid={"C": np.logspace(-3, 2)},
                               cv=5)
        estimator = AdversarialBalancing(learner)
        estimator.fit(X, a)

        self.assertIsInstance(estimator.learner, GridSearchCV)
        self.assertIsInstance(estimator.iterative_models_[0, 0], LogisticRegression)

    def test_initialization_with_classifier_list(self):
        X, a = make_classification(random_state=42)
        poor_learner = LogisticRegression(solver='lbfgs', C=1e-6)  # highly constrained model
        good_learner = RandomForestClassifier(n_estimators=10)
        estimator = AdversarialBalancing([poor_learner, good_learner])
        estimator.fit(X, a, n_splits=7, seed=2)  # non-default params for select_classifier

        self.assertIsInstance(estimator.learner, list)
        self.assertIsInstance(estimator.iterative_models_[0, 0], RandomForestClassifier)

    def test_weight_matrix(self):
        X, a = self.create_randomly_assigned_treatment()
        estimator = AdversarialBalancing(LogisticRegression(solver='lbfgs', max_iter=500),
                                         use_stabilized=True)
        estimator.fit(X, a)

        w_mat = estimator.compute_weight_matrix(X, a)
        w_1 = estimator.compute_weights(X, a, treatment_values=1)
        w_0 = estimator.compute_weights(X, a=None, treatment_values=0)
        w = estimator.compute_weights(X, a)

        pd.testing.assert_series_equal(w_mat[1], w_1, check_names=False)
        pd.testing.assert_series_equal(w_mat[0], w_0, check_names=False)
        pd.testing.assert_series_equal(w_1.loc[a == 1], w[a == 1])
        pd.testing.assert_series_equal(w_0.loc[a == 0], w[a == 0])

    def test_evaluation_can_plot_all(self):
        X, a = self.create_randomly_assigned_treatment()
        y = X @ np.random.rand(X.shape[1]) + a * 2
        estimator = AdversarialBalancing(LogisticRegression(solver='lbfgs', max_iter=500))
        estimator.fit(X, a)
        ab_evaluate = evaluate(estimator, X, a, y, metrics_to_evaluate=None)
        ab_evaluate_plots = ab_evaluate.plot_all()
        self.assertEqual(set(ab_evaluate_plots["train"].keys()), ab_evaluate.all_plot_names)
