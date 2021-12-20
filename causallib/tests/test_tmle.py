import unittest
import abc

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from causallib.estimation import TMLE
from causallib.estimation import Standardization, IPW
from causallib.utils.general_tools import check_learner_is_fitted


def generate_data(n_samples, n_independent_features, n_interaction_features=None,
                  a_sparsity=0.8, y_sparsity=0.8,
                  X_normal=True,
                  seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate independent features:
    if X_normal:
        X_independent = np.random.normal(size=(n_samples, n_independent_features))
    else:
        X_independent = np.random.binomial(1, 0.4, size=(n_samples, n_independent_features))
    X_independent = pd.DataFrame(
        X_independent,
        columns=[f"x_{i}" for i in range(X_independent.shape[1])]
    )

    # Generate treatment assignment:
    a_assignment, a_propensity, a_logit, a_beta = generate_vector(X_independent, a_sparsity)

    # Generate interactions:
    if n_interaction_features is None:
        n_interaction_features = n_independent_features
    assert n_interaction_features <= n_independent_features
    X_interactions = X_independent.sample(
        n=n_interaction_features,
        axis="columns",
    )
    X_interactions = X_interactions.multiply(a_assignment, axis="index")
    X_interactions = X_interactions.rename(columns=lambda s: f"{s}:a")
    # Generate outcome:
    X = X_independent.join(X_interactions)
    treatment_effect = 2
    y_binary, y_propensity, y_continuous, y_beta = generate_vector(X, y_sparsity, a_assignment, treatment_effect)

    data = {
        "X": X,
        "a": a_assignment,
        "y_cont": y_continuous,
        "y_bin": y_binary,
        "treatment_effect": treatment_effect,
        "y_beta": y_beta,
    }
    return data


def generate_vector(X, sparsity, a=None, treatment_effect=None):
    beta = np.random.normal(size=X.shape[1])
    beta_mask = np.random.binomial(1, sparsity, size=beta.shape)
    beta_masked = beta * beta_mask
    logit = X @ beta_masked
    logit += np.random.normal(size=X.shape[0])  # Add noise
    if a is not None and treatment_effect is not None:
        logit += treatment_effect * a
    propensity = 1 / (1 + np.exp(-logit))
    classes = np.random.binomial(1, propensity)

    logit = pd.Series(logit, index=X.index)
    propensity = pd.Series(propensity, index=X.index)
    classes = pd.Series(classes, index=X.index)
    return classes, propensity, logit, beta_masked


class BaseTestTMLE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        random_seed = 0
        cls.data = generate_data(200, 3, 3, seed=random_seed)

        cls.treatment_model = GradientBoostingClassifier()
        cls.outcome_model_bin = GradientBoostingClassifier()
        cls.outcome_model_cont = GradientBoostingRegressor()

    @abc.abstractmethod
    def init(self, reduced, importance_sampling):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def estimator(self):
        raise NotImplementedError

    def ensure_fit(self):
        self.estimator.fit(
            self.data['X'], self.data['a'], self.data['y'],
        )
        self.assertTrue(check_learner_is_fitted(self.estimator.outcome_model.learner))
        self.assertTrue(check_learner_is_fitted(self.estimator.weight_model.learner))
        self.assertTrue(hasattr(self.estimator, "targeted_outcome_model_"))
        self.assertTrue(hasattr(self.estimator.targeted_outcome_model_, "params"))

    def ensure_estimate_individual_outcome(self):
        self.estimator.fit(
            self.data['X'], self.data['a'], self.data['y'],
        )
        ind_outcomes = self.estimator.estimate_individual_outcome(self.data['X'], self.data['a'])
        self.assertFalse(ind_outcomes.isna().any().any())


class BaseTestTMLEBinary(BaseTestTMLE):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.data['y'] = cls.data['y_bin']

    def init(self, reduced, importance_sampling):
        self._estimator = TMLE(
            Standardization(self.outcome_model_bin, predict_proba=True),
            IPW(self.treatment_model),
            reduced=reduced, importance_sampling=importance_sampling,
        )

    @property
    def estimator(self):
        return self._estimator

    def ensure_warning_if_not_predict_proba(self):
        self.estimator.outcome_model.predict_proba = False
        with self.assertWarns(UserWarning):
            self.estimator.fit(self.data['X'], self.data['a'], self.data['y'])
        self.estimator.outcome_model.predict_proba = True

    def ensure_positive_label_prediction_is_used(self):
        self.estimator.fit(self.data['X'], self.data['a'], self.data['y'])
        tmle_output = self.estimator._outcome_model_estimate_individual_outcome(
            self.data['X'], self.data['a']
        )
        outcome_model_output = self.estimator.outcome_model.estimate_individual_outcome(
            self.data['X'], self.data['a']
        )
        outcome_model_output = outcome_model_output.loc[:, pd.IndexSlice[:, 1]]  # Take prediction for `1` class
        outcome_model_output = outcome_model_output.droplevel(level=-1, axis="columns")  # Drop redundant inner level
        pd.testing.assert_frame_equal(tmle_output, outcome_model_output)


class BaseTestTMLEContinuous(BaseTestTMLE):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.data['y'] = cls.data['y_cont']

    def init(self, reduced, importance_sampling):
        self._estimator = TMLE(
            Standardization(self.outcome_model_cont),
            IPW(self.treatment_model),
            reduced=reduced, importance_sampling=importance_sampling,
        )

    @property
    def estimator(self):
        return self._estimator

    def ensure_target_scaling(self):
        self.estimator.fit(self.data['X'], self.data['a'], self.data['y'])
        with self.subTest("Test `target_scaler_` fit"):
            self.assertTrue(hasattr(self.estimator, "target_scaler_"))
            self.assertEqual(self.estimator.target_scaler_.data_min_[0], self.data['y'].min())
            self.assertEqual(self.estimator.target_scaler_.data_max_[0], self.data['y'].max())

        with self.subTest("Test `target_scaler_` inverse transform"):
            # Test predictions are not in a zero-one scale used for fitting
            ind_outcomes = self.estimator.estimate_individual_outcome(self.data['X'], self.data['a'])
            self.assertLess(ind_outcomes.min().min(), 0)
            self.assertLess(1, ind_outcomes.max().max())

    def ensure_average_effect(self):
        data = generate_data(20000, 3, 0, seed=0)
        self.estimator.fit(data['X'], data['a'], data['y_cont'])
        pop_outcome = self.estimator.estimate_population_outcome(data['X'], data['a'])
        effect = self.estimator.estimate_effect(pop_outcome[1], pop_outcome[0])
        self.assertAlmostEqual(data['treatment_effect'], effect['diff'], places=1)

    def ensure_conditional_effect(self):
        n_samples = 11000  # TODO: is it really that data inefficient to get within 0.1 of true parameters?
        data = generate_data(n_samples, 1, 1, a_sparsity=1.0, y_sparsity=1.0, X_normal=False, seed=0)
        self.estimator.fit(data['X'], data['a'], data['y_cont'])
        ind_outcome = self.estimator.estimate_individual_outcome(data['X'], data['a'])
        ind_effect = self.estimator.estimate_effect(ind_outcome[1], ind_outcome[0], agg="individual")
        ind_effect = ind_effect["diff"]

        # The modified effect should the added interaction term to the true effect
        np.testing.assert_allclose(
            ind_effect.loc[data['X'].iloc[:, -1] == 1].mean(),
            data['treatment_effect'] + data['y_beta'][-1],
            atol=0.6, rtol=1e-5,
        )

        # The effect under no modification should the coefficient of the treatment assignment
        np.testing.assert_allclose(
            ind_effect.loc[data['X'].iloc[:, -1] == 0].mean(),
            data['treatment_effect'],
            # decimal=1,
            atol=0.5, rtol=1e-5,
        )

        # The average effect should be the weighted mean between the two modifications
        np.testing.assert_almost_equal(
            ind_effect.mean(),
            np.average(
                [ind_effect.loc[data['X'].iloc[:, -1] == 0].mean(),
                 ind_effect.loc[data['X'].iloc[:, -1] == 1].mean()],
                weights=[sum(data['X'].iloc[:, -1] == 0),
                         sum(data['X'].iloc[:, -1] == 1)]
            ),
            decimal=1
        )


class TestTMLEMatrixFeatureBinary(BaseTestTMLEBinary):
    def setUp(self) -> None:
        self.init(reduced=False, importance_sampling=False)

    def test_fit(self):
        self.ensure_fit()

    def test_warning_if_not_predict_proba(self):
        self.ensure_warning_if_not_predict_proba()

    def test_positive_label_prediction_is_used(self):
        self.ensure_positive_label_prediction_is_used()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()


class TestTMLEVectorFeatureBinary(BaseTestTMLEBinary):
    def setUp(self) -> None:
        self.init(reduced=True, importance_sampling=False)

    def test_fit(self):
        self.ensure_fit()

    def test_warning_if_not_predict_proba(self):
        self.ensure_warning_if_not_predict_proba()

    def test_positive_label_prediction_is_used(self):
        self.ensure_positive_label_prediction_is_used()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()


class TestTMLEMatrixImportanceSamplingBinary(BaseTestTMLEBinary):
    def setUp(self) -> None:
        self.init(reduced=False, importance_sampling=True)

    def test_fit(self):
        self.ensure_fit()

    def test_warning_if_not_predict_proba(self):
        self.ensure_warning_if_not_predict_proba()

    def test_positive_label_prediction_is_used(self):
        self.ensure_positive_label_prediction_is_used()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()


class TestTMLEVectorImportanceSamplingBinary(BaseTestTMLEBinary):
    def setUp(self) -> None:
        self.init(reduced=True, importance_sampling=True)

    def test_fit(self):
        self.ensure_fit()

    def test_warning_if_not_predict_proba(self):
        self.ensure_warning_if_not_predict_proba()

    def test_positive_label_prediction_is_used(self):
        self.ensure_positive_label_prediction_is_used()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()


class TestTMLEMatrixFeatureContinuous(BaseTestTMLEContinuous):
    def setUp(self) -> None:
        self.init(reduced=False, importance_sampling=False)

    def test_fit(self):
        self.ensure_fit()

    def test_target_scaling(self):
        self.ensure_target_scaling()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()

    def test_average_effect(self):
        self.ensure_average_effect()

    def test_conditional_effect(self):
        self.ensure_conditional_effect()


class TestTMLEVectorFeatureContinuous(BaseTestTMLEContinuous):
    def setUp(self) -> None:
        self.init(reduced=True, importance_sampling=False)

    def test_fit(self):
        self.ensure_fit()

    def test_target_scaling(self):
        self.ensure_target_scaling()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()

    def test_average_effect(self):
        self.ensure_average_effect()

    def test_conditional_effect(self):
        self.ensure_conditional_effect()


class TestTMLEMatrixImportanceSamplingContinuous(BaseTestTMLEContinuous):
    def setUp(self) -> None:
        self.init(reduced=False, importance_sampling=True)

    def test_fit(self):
        self.ensure_fit()

    def test_target_scaling(self):
        self.ensure_target_scaling()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()

    def test_average_effect(self):
        self.ensure_average_effect()

    def test_conditional_effect(self):
        self.ensure_conditional_effect()


class TestTMLEVectorImportanceSamplingContinuous(BaseTestTMLEContinuous):
    def setUp(self) -> None:
        self.init(reduced=True, importance_sampling=True)

    def test_fit(self):
        self.ensure_fit()

    def test_target_scaling(self):
        self.ensure_target_scaling()

    def test_estimate_individual_outcome(self):
        self.ensure_estimate_individual_outcome()

    def test_average_effect(self):
        self.ensure_average_effect()

    def test_conditional_effect(self):
        self.ensure_conditional_effect()
