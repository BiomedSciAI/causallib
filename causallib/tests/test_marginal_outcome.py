"""
(C) IBM Corp, 2019, All rights reserved
Created on Aug 25, 2019

@author: EHUD KARAVANI
"""

import unittest
import pandas as pd
import numpy as np
from causallib.estimation import MarginalOutcomeEstimator


class TestMarginalOutcomeEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = pd.DataFrame([[1, 1, 0, 0, 1, 0, 0, 0, 1, 1]])
        cls.y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        cls.a = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    def setUp(self):
        self.model = MarginalOutcomeEstimator(learner=None)

    def test_fit_return(self):
        model = self.model.fit(self.X, self.a, self.y)
        self.assertTrue(isinstance(model, MarginalOutcomeEstimator))

    def test_outcome_estimation(self):
        self.model.fit(self.X, self.a, self.y)
        outcomes = self.model.estimate_population_outcome(self.X, self.a, self.y)
        truth = pd.Series([1 / 5, 4 / 5], index=[0, 1])
        pd.testing.assert_series_equal(truth, outcomes)

        with self.subTest("Change covariate and see no change in estimation"):
            X = pd.DataFrame(np.arange(20).reshape(4, 5))  # Different values and shape
            outcomes = self.model.estimate_population_outcome(X, self.a, self.y)
            truth = pd.Series([1/5, 4/5], index=[0, 1])
            pd.testing.assert_series_equal(truth, outcomes)



