"""
(C) IBM Corp, 2019, All rights reserved
Created on Aug 25, 2019

@author: EHUD KARAVANI
"""

import unittest
import pandas as pd
from causallib.estimation.base_weight import WeightEstimator


class TestBaseWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        cls.w = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5])
        cls.a = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    def setUp(self):
        self.model = WeightEstimator(learner=None)

    def test_no_weighting_no_stratification(self):
        result = self.model._compute_stratified_weighted_aggregate(self.y, None, None)
        truth = pd.Series(5/10, index=[0])
        pd.testing.assert_series_equal(truth, result)

    def test_weighting_no_stratification(self):
        result = self.model._compute_stratified_weighted_aggregate(self.y, self.w, None)
        truth = pd.Series(4/9, index=[0])
        pd.testing.assert_series_equal(truth, result)

    def test_no_weighting_stratification(self):
        result = self.model._compute_stratified_weighted_aggregate(self.y, None, self.a)
        truth = pd.Series([1/5, 4/5], index=[0, 1])
        pd.testing.assert_series_equal(truth, result)

    def test_weighting_stratification(self):
        result = self.model._compute_stratified_weighted_aggregate(self.y, self.w, self.a)
        truth = pd.Series([1/5, 3/4], index=[0, 1])
        pd.testing.assert_series_equal(truth, result)

    def test_subset_treatment_values(self):
        with self.subTest("Subset of treatment values exist in treatment"):
            result = self.model._compute_stratified_weighted_aggregate(self.y, None, self.a, [0])
            truth = pd.Series([1/5], index=[0])
            pd.testing.assert_series_equal(truth, result)

        with self.subTest("Subset of treatment values does not exist in treatment"):
            with self.assertRaises(ZeroDivisionError):  # Since the group is empty its weights' sum is zero
                self.model._compute_stratified_weighted_aggregate(self.y, None, self.a, [3])


