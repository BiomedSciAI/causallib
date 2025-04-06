"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Oct 26, 2021

"""

import unittest

import numpy as np
import pandas as pd
import math
from collections import deque

from causallib.datasets.data_loader import load_nhefs
from causallib.estimation import IPW, StratifiedStandardization
from sklearn.linear_model import LinearRegression, LogisticRegression

from causallib.contrib.bicause_tree import BICauseTree
from causallib.contrib.bicause_tree import PropensityBICauseTree
from causallib.contrib.bicause_tree.bicause_tree import BalancingTree
from causallib.contrib.bicause_tree.overlap_utils import prevalence_symmetric


class _BaseTestBICauseTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 60, name="a")
        # constant feature - should never induce a split
        cls.feature0 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 60, name="x0")
        # biased features - should induce independent splits
        cls.feature1 = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 0] * 60, name="x1")
        cls.feature2 = pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 1, 0] * 60, name="x2")
        # outcomes with effect of 1
        cls.y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 60, name="y")
        cls.y_diverse = pd.Series([0, 0, 0, 0, 1, 2, 2, 2, 2, 1] * 60, name="yy")

    @staticmethod
    def load_nhefs_sample(n=50):
        nhefs = load_nhefs()
        X = nhefs.X.drop(columns=nhefs.X.filter(regex="education"))
        a = nhefs.a
        y = nhefs.y
        return X.iloc[:n], a.iloc[:n], y.iloc[:n]


class TestBICauseTreeDataStructure(_BaseTestBICauseTree):

    def test_large_two_level_input_splits(self):
        X = pd.DataFrame({"feature1": self.feature1})
        tree = BalancingTree(min_split_size=599).fit(X, self.a)
        self.assertEqual(tree.split_feature_, "feature1")
        self.assertEqual(tree.split_value_, 0)

    def test_two_splits(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree().fit(X, self.a)
        self.assertIsNotNone(tree.subtree_)
        self.assertIsInstance(tree.subtree_, tuple)
        self.assertEqual(len(tree.subtree_), 2)
        self.assertEqual(tree.subtree_[0].split_feature_, "feature2")
        self.assertFalse(hasattr(tree.subtree_[1], "split_feature_"))

    def test_split_size(self):
        X = pd.DataFrame({"feature1": self.feature1})
        tree = BalancingTree(min_split_size=600).fit(X, self.a)
        self.assertFalse(hasattr(tree, "split_feature_"))

    def test_stopping_max_depth(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree(max_depth=1).fit(X, self.a)
        self.assertIsNone(tree.subtree_)

    def test_stopping_min_split_size(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree(min_split_size=1200).fit(X, self.a)
        self.assertIsNone(tree.subtree_)

    def test_stopping_min_leaf_size(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree(min_leaf_size=1200).fit(X, self.a)
        self.assertIsNone(tree.subtree_)

    def test_stopping_min_treat_group_size(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree(min_treat_group_size=1200).fit(X, self.a)
        self.assertIsNone(tree.subtree_)

    def test_stopping_asmd_threshold(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree(asmd_violation_threshold=10 ** 4).fit(X, self.a)
        self.assertIsNone(tree.subtree_)

    def test_stopping_custom_function(self):
        def function_stop_immediately(tree, X, a):
            return True

        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree(stopping_criterion=function_stop_immediately).fit(X, self.a)
        self.assertIsNone(tree.subtree_)

    def test_prevalence_symmetric(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BalancingTree().fit(X, self.a)
        node_idx = tree._find_non_positivity_violating_leaves(
            prevalence_symmetric,
            positivity_filtering_kwargs={'alpha': 0.5}
        )
        self.assertEqual(len(node_idx), 0)

    def test_multiple_hypothesis_bonferroni(self):
        tree = BalancingTree()
        tree.p_value_ = 0.01
        tree.subtree_ = []
        tree.subtree_.append(BalancingTree())
        tree.subtree_[0].p_value_ = 0.03
        tree.subtree_.append(BalancingTree())
        tree.subtree_[1].p_value_ = 0.02
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].p_value_ = 0.04
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[1].p_value_ = 0.07
        tree._enumerate_nodes()
        set_node_attributes_helper(tree)
        corrected_non_leaf_summary = tree._multiple_hypothesis_correction(
            alpha=0.1, method='bonferroni'
        )
        tree._set_corrected_p_value_to_nodes(corrected_non_leaf_summary)
        self.assertEqual(tree.corrected_p_value_, 0.02)
        self.assertEqual(tree.subtree_[0].corrected_p_value_, 0.06)
        self.assertTrue(tree.corrected_p_value_is_significant_)
        self.assertTrue(tree.subtree_[0].corrected_p_value_is_significant_)

    def test_multiple_hypothesis_holm(self):
        tree = BalancingTree()
        tree.p_value_ = 0.01
        tree.subtree_ = []
        tree.subtree_.append(BalancingTree())
        tree.subtree_[0].p_value_ = 0.005
        tree.subtree_.append(BalancingTree())
        tree.subtree_[1].p_value_ = 0.02
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].p_value_ = 0.03
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[1].p_value_ = 0.001
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[0].p_value_ = 0.009
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[1].p_value_ = 0.018
        tree._enumerate_nodes()
        set_node_attributes_helper(tree)
        corrected_non_leaf_summary = tree._multiple_hypothesis_correction(
            alpha=0.1, method='holm'
        )
        tree._set_corrected_p_value_to_nodes(corrected_non_leaf_summary)
        self.assertEqual(tree.corrected_p_value_, 0.02)
        self.assertEqual(tree.subtree_[0].corrected_p_value_, 0.015)
        self.assertEqual(tree.subtree_[1].corrected_p_value_, 0.02)
        self.assertTrue(tree.corrected_p_value_is_significant_)
        self.assertTrue(tree.subtree_[0].corrected_p_value_is_significant_)
        self.assertTrue(tree.subtree_[1].corrected_p_value_is_significant_)

    def test_pruning_single_root_node(self):
        tree = BalancingTree()
        tree.corrected_p_value_is_significant_ = False
        tree._parent_ = None
        tree.subtree_ = None

        with self.subTest("Test pruning"):
            tree._recurse_over_prune()
            self.assertTrue(tree.keep_)

        with self.subTest("Test deletion after pruning"):
            tree._delete_post_pruning()
            self.assertIsNone(tree.subtree_)

    def test_no_pruning_needed(self):
        tree = BalancingTree()
        tree._parent_ = None
        tree.corrected_p_value_is_significant_ = False
        tree.subtree_ = []

        tree.subtree_.append(BalancingTree())
        tree.subtree_[0].corrected_p_value_is_significant_ = True
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(BalancingTree())
        tree.subtree_[1].corrected_p_value_is_significant_ = False
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        with self.subTest("Test pruning"):
            tree._recurse_over_prune()
            self.assertTrue(tree.keep_)
            self.assertTrue(tree.subtree_[0].keep_)
            self.assertTrue(tree.subtree_[0].subtree_[0].keep_)
            self.assertTrue(tree.subtree_[0].subtree_[1].keep_)
            self.assertTrue(tree.subtree_[1].keep_)

        with self.subTest("Test deletion after pruning"):
            tree._delete_post_pruning()
            self.assertIsNone(tree.subtree_[0].subtree_[0].subtree_)
            self.assertIsNone(tree.subtree_[0].subtree_[1].subtree_)
            self.assertIsNone(tree.subtree_[1].subtree_)

    def test_all_insignificant_p_values(self):
        tree = BalancingTree()
        tree._parent_ = None
        tree.corrected_p_value_is_significant_ = False
        tree.subtree_ = []

        tree.subtree_.append(BalancingTree())
        tree.subtree_[0].corrected_p_value_is_significant_ = False
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(BalancingTree())
        tree.subtree_[1].corrected_p_value_is_significant_ = False
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        with self.subTest("Test pruning"):
            tree._recurse_over_prune()
            self.assertTrue(tree.keep_)
            self.assertFalse(tree.subtree_[0].keep_)
            self.assertFalse(tree.subtree_[0].subtree_[0].keep_)
            self.assertFalse(tree.subtree_[0].subtree_[1].keep_)
            self.assertFalse(tree.subtree_[1].keep_)
            self.assertFalse(tree.subtree_[1].subtree_[0].keep_)
            self.assertFalse(tree.subtree_[1].subtree_[1].keep_)

        with self.subTest("Test deletion after pruning"):
            tree._delete_post_pruning()
            self.assertIsNone(tree.subtree_)

    def test_all_significant_p_values(self):
        tree = BalancingTree()
        tree._parent_ = None
        tree.corrected_p_value_is_significant_ = True
        tree.subtree_ = []

        tree.subtree_.append(BalancingTree())
        tree.subtree_[0].corrected_p_value_is_significant_ = True
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(BalancingTree())
        tree.subtree_[1].corrected_p_value_is_significant_ = True
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        with self.subTest("Test pruning"):
            tree._recurse_over_prune()
            self.assertTrue(tree.keep_)
            self.assertTrue(tree.subtree_[0].keep_)
            self.assertTrue(tree.subtree_[0].subtree_[0].keep_)
            self.assertTrue(tree.subtree_[0].subtree_[1].keep_)
            self.assertTrue(tree.subtree_[1].keep_)
            self.assertTrue(tree.subtree_[1].subtree_[0].keep_)
            self.assertTrue(tree.subtree_[1].subtree_[1].keep_)

        with self.subTest("Test deletion after pruning"):
            tree._delete_post_pruning()
            self.assertIsNone(tree.subtree_[0].subtree_[0].subtree_)
            self.assertIsNone(tree.subtree_[0].subtree_[1].subtree_)
            self.assertIsNone(tree.subtree_[1].subtree_[0].subtree_)
            self.assertIsNone(tree.subtree_[1].subtree_[1].subtree_)

    def test_mark_as_keep_and_prune1(self):
        tree = BalancingTree()
        tree._parent_ = None
        tree.corrected_p_value_is_significant_ = True
        tree.subtree_ = []

        tree.subtree_.append(BalancingTree())
        tree.subtree_[0].corrected_p_value_is_significant_ = False
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(BalancingTree())
        tree.subtree_[1].corrected_p_value_is_significant_ = False
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        with self.subTest("Test pruning"):
            tree._recurse_over_prune()
            self.assertTrue(tree.keep_)
            self.assertTrue(tree.subtree_[0].keep_)
            self.assertFalse(tree.subtree_[0].subtree_[0].keep_)
            self.assertFalse(tree.subtree_[0].subtree_[1].keep_)
            self.assertTrue(tree.subtree_[1].keep_)
            self.assertFalse(tree.subtree_[1].subtree_[0].keep_)
            self.assertFalse(tree.subtree_[1].subtree_[1].keep_)

        with self.subTest("Test deletion after pruning"):
            tree._delete_post_pruning()
            self.assertIsNone(tree.subtree_[0].subtree_)
            self.assertIsNone(tree.subtree_[1].subtree_)

    def test_mark_as_keep_and_prune2(self):
        tree = BalancingTree()
        tree.corrected_p_value_is_significant_ = False
        tree._parent_ = None
        tree.subtree_ = []

        tree.subtree_.append(BalancingTree())
        tree.subtree_[0].corrected_p_value_is_significant_ = False
        tree.subtree_.append(BalancingTree())
        tree.subtree_[1].corrected_p_value_is_significant_ = False

        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].corrected_p_value_is_significant_ = True

        tree.subtree_[0].subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[0].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_[0].subtree_.append(BalancingTree())
        tree.subtree_[0].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[0].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(BalancingTree())
        tree.subtree_[1].subtree_[1].corrected_p_value_is_significant_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        with self.subTest("Test pruning"):
            tree._recurse_over_prune()
            self.assertTrue(tree.keep_)
            self.assertTrue(tree.subtree_[0].keep_)
            self.assertTrue(tree.subtree_[1].keep_)
            self.assertTrue(tree.subtree_[0].subtree_[0].keep_)
            self.assertTrue(tree.subtree_[0].subtree_[1].keep_)
            self.assertFalse(tree.subtree_[1].subtree_[0].keep_)
            self.assertFalse(tree.subtree_[1].subtree_[1].keep_)

        with self.subTest("Test deletion after pruning"):
            tree._delete_post_pruning()
            self.assertIsNone(tree.subtree_[0].subtree_[0].subtree_[0].subtree_)
            self.assertIsNone(tree.subtree_[0].subtree_[0].subtree_[1].subtree_)
            self.assertIsNone(tree.subtree_[0].subtree_[1].subtree_)
            self.assertIsNone(tree.subtree_[1].subtree_)


class TestBICauseTree(_BaseTestBICauseTree):
    def test_outcome_learners(self):
        from causallib.estimation import IPW, Standardization, MarginalOutcomeEstimator, TMLE, AIPW
        from sklearn.linear_model import LogisticRegression, LinearRegression
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        learners = [
            None,
            IPW(LogisticRegression()),
            Standardization(LinearRegression()),
            MarginalOutcomeEstimator(learner=None),
            # TMLE(Standardization(LinearRegression()), IPW(LogisticRegression())),
            AIPW(Standardization(LinearRegression()), IPW(LogisticRegression()))
        ]
        for learner in learners:
            learner_name = str(learner).split("(", maxsplit=1)[0]
            with self.subTest(f"Test fit using {learner_name}"), warnings.catch_warnings():
                warnings.simplefilter('ignore', category=ConvergenceWarning)
                tree = BICauseTree(learner)
                tree.fit(X, self.a, self.y)
                tree.estimate_population_outcome(X, self.a, self.y)
                if learner is None:  # To match expected default learner
                    learner = MarginalOutcomeEstimator(None)
                self.assertIsInstance(
                    list(tree.node_models_.values())[0], learner.__class__
                )

    def test_standardization_in_nodes_with_individual_prediction(self):
        # standardization + individual -> works
        X, a, y = self.load_nhefs_sample()

        tree = BICauseTree(
            outcome_model=StratifiedStandardization(LinearRegression()),
            individual=True
        )
        tree.fit(X, a, y)
        outcomes = tree.estimate_individual_outcome(X, a)  # y=None
        outcomes = tree.estimate_individual_outcome(X, a, y)
        leaf_count = len(tree.node_models_.keys())
        num_unique_predictions = outcomes.nunique(dropna=False)[0]
        self.assertGreater(num_unique_predictions, leaf_count)

    def test_standardization_in_nodes_with_population_prediction(self):
        # standardization + population -> works
        X, a, y = self.load_nhefs_sample()

        tree = BICauseTree(
            outcome_model=StratifiedStandardization(LinearRegression()),
            individual=False
        )
        tree.fit(X, a, y)
        outcomes = tree.estimate_individual_outcome(X, a)  # y=None
        outcomes = tree.estimate_individual_outcome(X, a, y)
        num_unique_predictions = outcomes.nunique(dropna=False)[0]
        leaf_count = len(tree.node_models_.keys())
        self.assertEqual(num_unique_predictions, leaf_count)

    def test_weight_model_in_nodes_with_population_prediction(self):
        # IPW + population -> works
        X, a, y = self.load_nhefs_sample()

        tree = BICauseTree(
            outcome_model=IPW(LogisticRegression()),
            individual=False
        )
        tree.fit(X, a, y)
        outcomes = tree.estimate_individual_outcome(X, a, y)
        # each node has the same values for each treatment group for all subjects.
        # so the number of values is the number of nodes times two.
        num_unique_predictions = outcomes.nunique(dropna=False)[0]
        leaf_count = len(tree.node_models_.keys())
        self.assertEqual(num_unique_predictions, leaf_count)

        with self.subTest("Not providing an outcome to pop-out-est raises an exception"):
            with self.assertRaises(TypeError):
                tree.estimate_individual_outcome(X, a, y=None)

    def test_weight_model_in_nodes_with_individual_prediction(self):
        # IPW + individual -> raise exception
        X, a, y = self.load_nhefs_sample()

        tree = BICauseTree(
            outcome_model=IPW(LogisticRegression()),
            individual=True
        )
        tree.fit(X, a, y)

        with self.assertRaises(Exception):
            tree.estimate_individual_outcome(X, a, y)

    def test_split_outcome(self):
        X = pd.DataFrame({"feature1": self.feature1})
        tree = BICauseTree(min_split_size=599).fit(
            X, self.a, self.y_diverse
        )

        self.assertListEqual(
            list(tree.estimate_population_outcome(X, self.a, self.y_diverse)),
            [0.5, 1.5],
        )

    def test_positivity_violation_creates_nans(self):
        X = pd.DataFrame({"featureA": self.a})
        tree = BICauseTree().fit(X, self.a, self.y)
        po = tree.estimate_population_outcome(X, self.a, self.y)

        self.assertSetEqual(set(po.index), set(self.a))
        np.testing.assert_array_equal(po, np.nan)
        np.testing.assert_equal(tree.tree.subtree_[0].max_feature_asmd_, np.nan)
        np.testing.assert_equal(tree.tree.subtree_[1].max_feature_asmd_, np.nan)

    def test_that_positivity_does_not_get_into_outcome(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BICauseTree().fit(X, self.a, self.y)
        tree.fit_outcome_models(X, self.a, self.y)
        self.assertListEqual(
            list(tree.estimate_population_outcome(X, self.a, self.feature1)),
            [0.625, 0.625],
        )

    def test_large_constant_input_unchanged_and_returns_no_feature(self):
        X = pd.DataFrame({"feature0": self.feature0})
        tree = BICauseTree().fit(X, self.a, self.y)
        outcomes = tree.estimate_population_outcome(X, self.a, self.y)
        self.assertListEqual(list([outcomes[0], outcomes[1]]), [0, 1])
        self.assertFalse(hasattr(tree.tree, "split_feature_"))
        self.assertFalse(hasattr(tree.tree, "split_value_"))

    def test_parameter_propagation(self):
        from causallib.contrib.bicause_tree.overlap_utils import crump

        def function_stop_immediately(tree, X, a):
            return True
        # Define different-than-default values:
        kwargs = dict(
            min_leaf_size=5,
            min_split_size=3,
            min_treat_group_size=5,
            asmd_violation_threshold=0.3,
            max_depth=5,
            max_splitting_values=80,
            multiple_hypothesis_test_alpha=0.05,
            multiple_hypothesis_test_method='bonferroni',
            positivity_filtering_kwargs=dict(segments=500),
            stopping_criterion=function_stop_immediately,
            positivity_filtering_method=crump,
        )
        tree = BICauseTree(**kwargs)
        for k, v in kwargs.items():
            with self.subTest(f"Parameter propagation {k}"):
                tree_attr = getattr(tree.tree, k)
                self.assertEqual(v, tree_attr)


class TestPropensityBICauseTree(_BaseTestBICauseTree):
    def test_parameter_propagation(self):
        from causallib.contrib.bicause_tree.overlap_utils import crump

        def function_stop_immediately(tree, X, a):
            return True

        # Define different-than-default values:
        kwargs = dict(
            min_leaf_size=5,
            min_split_size=3,
            min_treat_group_size=5,
            asmd_violation_threshold=0.3,
            max_depth=5,
            max_splitting_values=80,
            multiple_hypothesis_test_alpha=0.05,
            multiple_hypothesis_test_method='bonferroni',
            positivity_filtering_kwargs=dict(segments=500),
            stopping_criterion=function_stop_immediately,
            positivity_filtering_method=crump,
        )
        tree = PropensityBICauseTree(**kwargs)
        for k, v in kwargs.items():
            with self.subTest(f"Parameter propagation {k}"):
                tree_attr = getattr(tree.tree, k)
                self.assertEqual(v, tree_attr)

    def test_prediction(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityBICauseTree().fit(X, self.a)
        a_pred = tree.predict_proba(X)
        # Compatible with sklearn's prediction:
        self.assertIsInstance(a_pred, np.ndarray)
        # Compatible with sklearn's `predict_proba` with column-per-class:
        self.assertTupleEqual(a_pred.shape, (X.shape[0], 2))
        # Result is a valid probability:
        self.assertLessEqual(a_pred.max(), 1)
        self.assertGreaterEqual(a_pred.min(), 0)

    def test_fit(self):
        from sklearn.dummy import DummyClassifier
        # Single level
        X = pd.DataFrame({"feature1": self.feature1})
        tree = PropensityBICauseTree().fit(X, self.a)
        self.assertSetEqual(set(tree.treatment_values_), set(self.a))
        self.assertEqual(len(tree.node_models_), 2)
        self.assertIsInstance(tree.node_models_[1], tree.learner.__class__)
        self.assertIsInstance(tree.node_models_[1], DummyClassifier)
        self.assertSetEqual(set(tree.node_models_[1].class_prior_), {0.2, 0.8})

    def test_non_default_learner(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import LinearSVC

        X = pd.DataFrame({"feature1": self.feature1})
        learners = [
            LogisticRegression(), GradientBoostingClassifier(),
        ]
        for learner in learners:
            with self.subTest(f"Test `PropensityBICauseTree` with {learner}"):
                tree = PropensityBICauseTree(learner).fit(X, self.a)
                self.assertSetEqual(set(tree.treatment_values_), set(self.a))
                self.assertEqual(len(tree.node_models_), 2)
                self.assertIsInstance(tree.node_models_[1], tree.learner.__class__)
                self.assertIsInstance(tree.node_models_[1], learner.__class__)
                a_pred = tree.predict_proba(X)
                # Compatible with sklearn's prediction:
                self.assertIsInstance(a_pred, np.ndarray)
                # Compatible with sklearn's `predict_proba` with column-per-class:
                self.assertTupleEqual(a_pred.shape, (X.shape[0], 2))
                # Result is a valid probability:
                self.assertLessEqual(a_pred.max(), 1)
                self.assertGreaterEqual(a_pred.min(), 0)

        with self.subTest("Test with non-`predict_proba` learner"):
            tree = PropensityBICauseTree(LinearSVC())
            tree.fit(X, self.a)
            with self.assertRaises(AttributeError):
                tree.predict_proba(X)

    def test_fit_with_positivity_violation(self):
        from sklearn.dummy import DummyClassifier

        X = self.a.to_frame("x1")
        tree = PropensityBICauseTree().fit(X, self.a)
        a_pred = tree.predict_proba(X)
        self.assertEqual(len(tree.node_models_), 2)
        self.assertIsInstance(tree.node_models_[1], DummyClassifier)
        # Positivity violations result in extreme propensity scores:
        self.assertSetEqual(set(np.unique(a_pred)), {0, 1})
        # Verify column order so class 0 is 1st column:
        np.testing.assert_equal(a_pred[self.a == 0, 0], 1)
        np.testing.assert_equal(a_pred[self.a == 1, 1], 1)


def set_node_attributes_helper(node, default_val=0):
    # here we set the node attributes to a default value to be able to use the
    # _generate_node_summary method
    queue = deque([node])
    while queue:
        node = queue.popleft()
        node.node_sample_size_ = default_val
        node.treatment_prevalence_ = default_val
        node.potential_outcomes_ = pd.Series([default_val, default_val])
        if node.subtree_ is not None:
            queue.extend(node.subtree_)
