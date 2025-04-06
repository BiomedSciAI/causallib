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
import pandas as pd
import numpy as np
import math
from collections import deque

from causallib.estimation.explainable_causal_tree import (
    BICauseTree, PropensityImbalanceStratification, prevalence_symmetric
)

def function_True(tree,X,a):
    return True

class TestExplainableCausalTree(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.a = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 60)
        # constant feature - should never induce a split
        self.feature0 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 60)
        # biased features - should induce independent splits
        self.feature1 = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 0] * 60)
        self.feature2 = pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 1, 0] * 60)
        # outcomes with effect of 1
        self.y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 60)
        self.y_diverse = pd.Series([0, 0, 0, 0, 1, 2, 2, 2, 2, 1] * 60)

    def load_nhefs_sample(self, n=500):
        from causallib.datasets.data_loader import load_nhefs
        nhefs = load_nhefs()
        X = nhefs.X.drop(columns=nhefs.X.filter(regex="education"))
        a = nhefs.a
        y = nhefs.y
        return X.iloc[:n], a.iloc[:n], y.iloc[:n]

    def test_large_constant_input_unchanged_and_returns_no_feature(self):
        X = pd.DataFrame({"feature0": self.feature0})
        tree = BICauseTree().fit(X, self.a, self.y)
        outcomes = tree.estimate_population_outcome(X, self.a, self.y)
        self.assertListEqual(list([outcomes[0], outcomes[1]]), [0, 1])
        self.assertIsNone(tree.tree.split_feature_)
        self.assertTrue(math.isnan(tree.tree.split_value_))

    def test_large_two_level_input_splits(self):
        X = pd.DataFrame({"feature1": self.feature1})
        tree = BICauseTree(min_split_size=599).fit(
            X, self.a, self.y
        )
        self.assertEqual(tree.tree.split_feature_, "feature1")
        self.assertEqual(tree.tree.split_value_, 0)

    def test_two_splits(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BICauseTree().fit(X, self.a, self.y)
        self.assertIsNotNone(tree.tree.subtree_)
        self.assertEqual(tree.tree.subtree_[0].split_feature_, "feature2")
        self.assertIsNone(tree.tree.subtree_[1].split_feature_)

    def test_split_size(self):
        X = pd.DataFrame({"feature1": self.feature1})
        tree = BICauseTree(min_split_size=600).fit(
            X, self.a, self.y
        )
        self.assertIsNone(tree.tree.split_feature_)

    def test_split_outcome(self):
        X = pd.DataFrame({"feature1": self.feature1})
        tree = BICauseTree(min_split_size=599).fit(
            X, self.a, self.y_diverse
        )

        self.assertListEqual(
            list(tree.estimate_population_outcome(X, self.a, self.y_diverse)),
            [0.5, 1.5],
        )

    def test_that_positivity_creates_nans(self):
        X = pd.DataFrame({"featureA": self.a})
        tree = BICauseTree().fit(X, self.a, self.y)

        self.assertTrue(
            [
                math.isnan
                for val in list(tree.estimate_population_outcome(X, self.a, self.y))
            ]
        )

    def test_that_positivity_does_not_get_into_outcome(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = BICauseTree().fit(X, self.a, self.y)
        tree.fit_outcome_models(X, self.a, self.y)
        self.assertListEqual(
            list(tree.estimate_population_outcome(X, self.a, self.feature1)),
            [0.625, 0.625],
        )

    def test_stopping_max_depth(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityImbalanceStratification(max_depth=1).fit(X, self.a)
        self.assertEqual(tree.subtree_,None)

    def test_stopping_min_split_size(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityImbalanceStratification(min_split_size=1200).fit(X, self.a)
        self.assertEqual(tree.subtree_,None)

    def test_stopping_min_leaf_size(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityImbalanceStratification(min_leaf_size=1200).fit(X, self.a)
        self.assertEqual(tree.subtree_,None)

    def test_stopping_min_treat_group_size(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityImbalanceStratification(min_treat_group_size=1200).fit(X, self.a)
        self.assertEqual(tree.subtree_,None)

    def test_stopping_asmd_threshold(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityImbalanceStratification(asmd_threshold_split=10 ** 4).fit(X, self.a)
        self.assertEqual(tree.subtree_, None)

    def test_stopping_custom_function(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityImbalanceStratification(stopping_criterion=function_True).fit(X, self.a)
        self.assertEqual(tree.subtree_, None)

    def test_prevalence_symmetric(self):
        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        tree = PropensityImbalanceStratification().fit(X, self.a)
        node_idx=tree._find_non_positivity_violating_leaves(prevalence_symmetric, positivity_filtering_kwargs={'alpha':0.5})
        self.assertEqual(len(node_idx), 0)

    def test_multiple_hypothesis_bonferroni(self):
        tree=PropensityImbalanceStratification()
        tree.pval_=0.01
        tree.subtree_=[]
        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].pval_=0.03
        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].pval_ = 0.02
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].pval_ = 0.04
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[1].pval_ = 0.07
        tree._enumerate_nodes()
        set_node_attributes_helper(tree)
        corrected_non_leaf_summary=tree._multiple_hypothesis_correction(alpha=0.1, method='bonferroni')
        tree._mark_nodes_post_multiple_hyp(corrected_non_leaf_summary)
        self.assertEqual(tree.corrected_pval_, 0.02)
        self.assertEqual(tree.subtree_[0].corrected_pval_,0.06)
        self.assertEqual(tree.is_split_significant_corrected_, True)
        self.assertEqual(tree.subtree_[0].is_split_significant_corrected_, True)


    def test_multiple_hypothesis_holm(self):
        tree=PropensityImbalanceStratification()
        tree.pval_=0.01
        tree.subtree_=[]
        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].pval_=0.005
        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].pval_ = 0.02
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].pval_ = 0.03
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[1].pval_ = 0.001
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[0].pval_ = 0.009
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[1].pval_ = 0.018
        tree._enumerate_nodes()
        set_node_attributes_helper(tree)
        corrected_non_leaf_summary=tree._multiple_hypothesis_correction(alpha=0.1, method='holm')
        tree._mark_nodes_post_multiple_hyp(corrected_non_leaf_summary)
        self.assertEqual(tree.corrected_pval_, 0.02)
        self.assertEqual(tree.subtree_[0].corrected_pval_,0.015)
        self.assertEqual(tree.subtree_[1].corrected_pval_, 0.02)
        self.assertEqual(tree.is_split_significant_corrected_, True)
        self.assertEqual(tree.subtree_[0].is_split_significant_corrected_, True)
        self.assertEqual(tree.subtree_[1].is_split_significant_corrected_, True)


    def test_pruning_single_root_node(self):
        tree = PropensityImbalanceStratification()
        tree.is_split_significant_corrected_ = False
        tree._parent_ = None
        tree.subtree_=None

        #sub-test for pruning
        tree._recurse_over_prune()
        self.assertEqual(tree.keep_, True)

        #sub-test for deletion
        tree._delete_post_pruning()
        self.assertEqual(tree.subtree_, None)

    def test_no_pruning_needed(self):
        tree = PropensityImbalanceStratification()
        tree._parent_=None
        tree.is_split_significant_corrected_ = False
        tree.subtree_=[]

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].is_split_significant_corrected_= True
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].is_split_significant_corrected_ = False
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        #sub-test for pruning
        tree._recurse_over_prune()
        self.assertEqual(tree.keep_, True)
        self.assertEqual(tree.subtree_[0].keep_,True)
        self.assertEqual(tree.subtree_[0].subtree_[0].keep_, True)
        self.assertEqual(tree.subtree_[0].subtree_[1].keep_, True)
        self.assertEqual(tree.subtree_[1].keep_, True)

        #sub-test for deletion
        tree._delete_post_pruning()
        self.assertEqual(tree.subtree_[0].subtree_[0].subtree_, None)
        self.assertEqual(tree.subtree_[0].subtree_[1].subtree_, None)
        self.assertEqual(tree.subtree_[1].subtree_, None)


    def test_all_insignificant_pvals(self):
        tree = PropensityImbalanceStratification()
        tree._parent_=None
        tree.is_split_significant_corrected_ = False
        tree.subtree_=[]

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].is_split_significant_corrected_= False
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].is_split_significant_corrected_ = False
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        #sub-test for pruning
        tree._recurse_over_prune()
        self.assertEqual(tree.keep_, True)
        self.assertEqual(tree.subtree_[0].keep_,False)
        self.assertEqual(tree.subtree_[0].subtree_[0].keep_, False)
        self.assertEqual(tree.subtree_[0].subtree_[1].keep_, False)
        self.assertEqual(tree.subtree_[1].keep_, False)
        self.assertEqual(tree.subtree_[1].subtree_[0].keep_, False)
        self.assertEqual(tree.subtree_[1].subtree_[1].keep_, False)

        #sub-test for deletion
        tree._delete_post_pruning()
        self.assertEqual(tree.subtree_, None)


    def test_all_significant_pvals(self):
        tree = PropensityImbalanceStratification()
        tree._parent_=None
        tree.is_split_significant_corrected_ = True
        tree.subtree_=[]

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].is_split_significant_corrected_= True
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].is_split_significant_corrected_ = True
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        #sub-test for pruning
        tree._recurse_over_prune()
        self.assertEqual(tree.keep_, True)
        self.assertEqual(tree.subtree_[0].keep_,True)
        self.assertEqual(tree.subtree_[0].subtree_[0].keep_, True)
        self.assertEqual(tree.subtree_[0].subtree_[1].keep_, True)
        self.assertEqual(tree.subtree_[1].keep_, True)
        self.assertEqual(tree.subtree_[1].subtree_[0].keep_, True)
        self.assertEqual(tree.subtree_[1].subtree_[1].keep_, True)

        #sub-test for deletion
        tree._delete_post_pruning()
        self.assertEqual(tree.subtree_[0].subtree_[0].subtree_, None)
        self.assertEqual(tree.subtree_[0].subtree_[1].subtree_, None)
        self.assertEqual(tree.subtree_[1].subtree_[0].subtree_, None)
        self.assertEqual(tree.subtree_[1].subtree_[1].subtree_, None)


    def test_mark_as_keep_and_prune1(self):
        tree = PropensityImbalanceStratification()
        tree._parent_=None
        tree.is_split_significant_corrected_ = True
        tree.subtree_=[]

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].is_split_significant_corrected_= False
        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].is_split_significant_corrected_ = False
        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        #sub-test for pruning
        tree._recurse_over_prune()
        self.assertEqual(tree.keep_, True)
        self.assertEqual(tree.subtree_[0].keep_,True)
        self.assertEqual(tree.subtree_[0].subtree_[0].keep_, False)
        self.assertEqual(tree.subtree_[0].subtree_[1].keep_, False)
        self.assertEqual(tree.subtree_[1].keep_, True)
        self.assertEqual(tree.subtree_[1].subtree_[0].keep_, False)
        self.assertEqual(tree.subtree_[1].subtree_[1].keep_, False)

        #sub-test for deletion
        tree._delete_post_pruning()
        self.assertEqual(tree.subtree_[0].subtree_, None)
        self.assertEqual(tree.subtree_[1].subtree_, None)


    def test_mark_as_keep_and_prune2(self):
        tree = PropensityImbalanceStratification()
        tree.is_split_significant_corrected_ = False
        tree._parent_ = None
        tree.subtree_=[]

        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].is_split_significant_corrected_= False
        tree.subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].is_split_significant_corrected_ = False

        tree.subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].is_split_significant_corrected_ = True

        tree.subtree_[0].subtree_[0].subtree_ = []
        tree.subtree_[0].subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].subtree_[0].is_split_significant_corrected_=None
        tree.subtree_[0].subtree_[0].subtree_[0].subtree_ = None
        tree.subtree_[0].subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[0].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_[0].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[0].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[0].subtree_[1].subtree_ = None

        tree.subtree_[1].subtree_ = []
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[0].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[0].subtree_ = None
        tree.subtree_[1].subtree_.append(PropensityImbalanceStratification())
        tree.subtree_[1].subtree_[1].is_split_significant_corrected_ = None
        tree.subtree_[1].subtree_[1].subtree_ = None

        tree._enumerate_nodes()
        tree.subtree_[0]._parent_ = tree
        tree.subtree_[1]._parent_ = tree
        tree.subtree_[0].subtree_[0]._parent_ = tree.subtree_[0]
        tree.subtree_[0].subtree_[1]._parent_ = tree.subtree_[0]
        tree.subtree_[1].subtree_[0]._parent_ = tree.subtree_[1]
        tree.subtree_[1].subtree_[1]._parent_ = tree.subtree_[1]

        #sub-test for pruning
        tree._recurse_over_prune()
        self.assertEqual(tree.keep_, True)
        self.assertEqual(tree.subtree_[0].keep_,True)
        self.assertEqual(tree.subtree_[1].keep_, True)
        self.assertEqual(tree.subtree_[0].subtree_[0].keep_,True)
        self.assertEqual(tree.subtree_[0].subtree_[1].keep_, True)
        self.assertEqual(tree.subtree_[1].subtree_[0].keep_,False)
        self.assertEqual(tree.subtree_[1].subtree_[1].keep_, False)

        #sub-test for deletion
        tree._delete_post_pruning()
        self.assertEqual(tree.subtree_[0].subtree_[0].subtree_[0].subtree_, None)
        self.assertEqual(tree.subtree_[0].subtree_[0].subtree_[1].subtree_, None)
        self.assertEqual(tree.subtree_[0].subtree_[1].subtree_, None)
        self.assertEqual(tree.subtree_[1].subtree_, None)


    def test_outcome_learners(self):
        from causallib.estimation import IPW, Standardization, MarginalOutcomeEstimator, TMLE, AIPW
        from sklearn.linear_model import LogisticRegression, LinearRegression
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        X = pd.DataFrame({"feature1": self.feature1, "feature2": self.feature2})
        learners = [None,
                    IPW(LogisticRegression()),
                    Standardization(LinearRegression()),
                    MarginalOutcomeEstimator(learner=None),
                    TMLE(Standardization(LinearRegression()), IPW(LogisticRegression())),
                    AIPW(Standardization(LinearRegression()), IPW(LogisticRegression()))
                    ]
        for learner in learners:
                learner_name = str(learner).split("(", maxsplit=1)[0]
                with self.subTest("Test fit using {learner}".format(learner=learner_name)):
                    tree = BICauseTree().fit(X, self.a, self.y)
                    self.assertTrue(True) # Fit did not crash

    def test_standardization_in_nodes_with_individual_prediction(self):
        # standardization + individual -> works
        from causallib.estimation import StratifiedStandardization
        from sklearn.linear_model import LinearRegression
        X,a,y = self.load_nhefs_sample()

        tree = BICauseTree(outcome_model=StratifiedStandardization(LinearRegression()),individual=True)
        tree.fit(X, a, y)
        outcomes = tree.estimate_individual_outcome(X,a,y)
        leaf_count = len(tree.node_models_.keys())
        num_unique_predictions = outcomes.nunique(dropna=False)[0]
        self.assertGreater(num_unique_predictions,leaf_count)


    def test_standardization_in_nodes_with_population_prediction(self):
        # standardization + population -> works
        from causallib.estimation import StratifiedStandardization
        from sklearn.linear_model import LinearRegression

        X,a,y = self.load_nhefs_sample()
        tree = BICauseTree(outcome_model=StratifiedStandardization(LinearRegression()),individual=False)
        tree.fit(X, a, y)
        outcomes = tree.estimate_individual_outcome(X,a,y)
        num_unique_predictions = outcomes.nunique(dropna=False)[0]
        leaf_count = len(tree.node_models_.keys())
        self.assertEqual(num_unique_predictions,leaf_count)

    def test_weight_model_in_nodes_with_population_prediction(self):
        # IPW + population -> works
        from causallib.estimation import IPW
        from sklearn.linear_model import LogisticRegression
        X,a,y = self.load_nhefs_sample()

        tree = BICauseTree(outcome_model=IPW(learner=LogisticRegression(solver="liblinear")),individual=False)
        tree.fit(X, a, y)
        outcomes = tree.estimate_individual_outcome(X,a,y)
        # each node has the same values for each treatment group for all subjects.
        # so the number of values is the number of nodes times two.
        num_unique_predictions = outcomes.nunique(dropna=False)[0]
        leaf_count = len(tree.node_models_.keys())
        self.assertEqual(num_unique_predictions,leaf_count)

    def test_weight_model_in_nodes_with_individual_prediction(self):
        # IPW + individual -> raise exception
        from causallib.estimation import IPW
        from sklearn.linear_model import LogisticRegression
        X,a,y = self.load_nhefs_sample()

        tree = BICauseTree(outcome_model=IPW(learner=LogisticRegression(solver="liblinear")),individual=True)
        tree.fit(X, a, y)

        with self.assertRaises(Exception):
            outcomes = tree.estimate_individual_outcome(X,a,y)


def set_node_attributes_helper(node, default_val=0):
    # here we set the node attributes to a default value to be able to use the
    # _generate_node_summary method
    queue = deque([node])
    while queue:
        node = queue.popleft()
        node.node_sample_size_ = default_val
        node.propensity_score_ = default_val
        node.potential_outcomes_ = pd.Series([default_val,default_val])
        if node.subtree_ is not None:
            queue.extend(node.subtree_)

