import unittest
import abc

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from causallib.estimation import TMLE
from causallib.estimation import Standardization, IPW


def generate_data(n_samples, n_independent_features, n_interaction_features=None,
                  a_sparsity=0.8, y_sparsity=0.8,
                  seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate independent features:
    X_independent = pd.DataFrame(
        np.random.normal(size=(n_samples, n_independent_features)),
        columns=[f"x_{i}" for i in range(n_independent_features)]
    )

    # Generate treatment assignment:
    a_assignment, a_propensity, a_logit = generate_vector(X_independent, a_sparsity)

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
    y_binary, y_propensity, y_continuous = generate_vector(X, y_sparsity, a_assignment, treatment_effect)

    data = {
        "X": X,
        "a": a_assignment,
        "y_cont": y_continuous,
        "y_bin": y_binary
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
    return classes, propensity, logit


class BaseTestTMLE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        random_seed = 0
        cls.data = generate_data(200, 3, 3, seed=random_seed)

        cls.treatment_model = GradientBoostingClassifier()
        cls.outcome_model_bin = GradientBoostingClassifier()
        cls.outcome_model_cont = GradientBoostingRegressor()

    @property
    @abc.abstractmethod
    def estimator(self):
        raise NotImplementedError


class TestTMLEMatrixFeature(BaseTestTMLE):
    @property
    def estimator(self):
        return self._estimator

    def setUp(self) -> None:
        self._estimator = TMLE(
            self.outcome_model_bin, self.treatment_model,
            reduced=False, importance_sampling=False,
        )


class TestTMLEVectorFeature(BaseTestTMLE):
    @property
    def estimator(self):
        return self._estimator

    def setUp(self) -> None:
        self._estimator = TMLE(
            self.outcome_model_bin, self.treatment_model,
            reduced=True, importance_sampling=False,
        )


class TestTMLEMatrixImportanceSampling(BaseTestTMLE):
    @property
    def estimator(self):
        return self._estimator

    def setUp(self) -> None:
        self._estimator = TMLE(
            self.outcome_model_bin, self.treatment_model,
            reduced=False, importance_sampling=True,
        )


class TestTMLEVectorImportanceSampling(BaseTestTMLE):
    @property
    def estimator(self):
        return self._estimator

    def setUp(self) -> None:
        self._estimator = TMLE(
            self.outcome_model_bin, self.treatment_model,
            reduced=True, importance_sampling=True,
        )

