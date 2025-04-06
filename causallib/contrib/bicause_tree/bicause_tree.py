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
from collections import deque
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from causallib.contrib.bicause_tree.overlap_utils import prevalence_symmetric, OverlapViolationEstimator
from causallib.utils.general_tools import get_iterable_treatment_values
from causallib.estimation.base_estimator import PopulationOutcomeEstimator, IndividualOutcomeEstimator
from causallib.estimation import MarginalOutcomeEstimator
from causallib.metrics.weight_metrics import calculate_covariate_balance


def default_stopping_criterion(tree, X: pd.DataFrame, a: pd.Series):
    """
    Args:
        tree (BalancingTree): the current node (tree) being considered
        X (pd.DataFrame): The feature matrix
        a (pd.Series): the treatment assignment vector
    Returns:
        (boolean): is the stopping criterion met
    """
    # returns True if we must stop

    asmds = calculate_asmds(X, a)
    no_exploitable_asmds = asmds.isna().all()
    # asmds can be NaN if the standard deviation is null
    if no_exploitable_asmds:
        return True
    _is_asmd_threshold_reached = asmds.max() <= tree.asmd_violation_threshold
    _is_min_treat_group_size_reached = min(a.value_counts()) <= tree.min_treat_group_size
    _is_min_split_size_reached = X.shape[0] <= tree.min_split_size
    _is_max_depth_tree_reached = tree.max_depth <= 1

    criteria = any([
        _is_asmd_threshold_reached,
        _is_min_treat_group_size_reached,
        _is_min_split_size_reached,
        _is_max_depth_tree_reached
    ])

    return criteria


def calculate_asmds(X: pd.DataFrame, a: pd.Series):
    epsilon = np.finfo(float).resolution
    treatment_values = get_iterable_treatment_values(None, a)
    if len(treatment_values) < 2:
        return pd.Series(np.nan, index=X.columns)
    X0 = X.loc[a == treatment_values[0]]
    X1 = X.loc[a != treatment_values[0]]
    asmds = (X0.mean() - X1.mean()) / ((np.sqrt(X0.var() + X1.var())) + epsilon)
    asmds = asmds.abs()
    return asmds


class PropensityBICauseTree:

    def __init__(
        self,
        learner=DummyClassifier(strategy='prior'),
        **kwargs
    ) -> None:
        """
        A propensity model based on a BICause Tree,
        scikit-learn compatible

        Args:
            learner (BaseEstimator, ClassifierMixin): The propensity model
            **kwargs: arguments for `BalancingTree`. See its docstring for details.
        """
        self.learner = learner
        self.tree = BalancingTree(**kwargs)

    def fit(self, X, a):
        """Fits a PropensityBICauseTree propensity model.

        Compatible with scikit-learn classifiers,
        which can be combined with causallib's models
        - e.g., IPW(PropensityBICauseTree())

        Args:
            X (pd.DataFrame): The feature matrix
            a (pd.Series): The treatment assignment vector

        Returns:
            PropensityBICauseTree
        """
        self.tree.fit(X, a)
        self.fit_treatment_models(X, a)
        return self

    def fit_treatment_models(self, X, a):
        """Fits the probability estimator in the leaf nodes.

        Args:
            X (pd.DataFrame): The feature matrix
            a (pd.Series): The treatment assignment vector

        Returns:
            dict[int, BaseEstimator]: a mapping between leaf index and
                the corresponding probability estimator
        """
        self.treatment_values_ = np.unique(a)

        assignment = self.tree.apply(X)
        all_leaves = assignment.unique()
        node_models = {}
        for node in all_leaves:
            cur_node_mask = assignment == node
            current_node_X = X.loc[cur_node_mask]
            current_node_a = a.loc[cur_node_mask]

            model = self.learner.fit(current_node_X, current_node_a)
            node_models[node] = deepcopy(model)

        self.node_models_ = node_models
        return node_models

    def predict_proba(self, X):
        """ Predict the individual treatment probabilities.

        Args:
            X (pd.DataFrame): The feature matrix

        Returns:
            np.array: A matrix of individual treatment probabilities,
                (n_samples, n_treatment_values)
        """
        node_assignment = self.tree.apply(X)
        res = []
        for node in np.unique(node_assignment.values):
            curr_node_index = node_assignment == node
            curr_X = X.loc[curr_node_index]
            curr_model = self.node_models_[node]

            probability = curr_model.predict_proba(curr_X)
            probability = pd.DataFrame(
                probability,
                columns=curr_model.classes_,
                index=curr_X.index,
            )
            res.append(probability)

        res = pd.concat(res)
        res = res.fillna(0.0)  # If classes don't fully overlap across leaves
        res = res.loc[X.index]
        res = res.to_numpy()

        return res


class BICauseTree(IndividualOutcomeEstimator):

    def __init__(
        self,
        outcome_model: Union[IndividualOutcomeEstimator, PopulationOutcomeEstimator] = None,
        individual=False,
        asmd_violation_threshold=0.1,
        min_leaf_size=0,
        min_treat_group_size=0,
        min_split_size=0,
        max_depth=10,
        stopping_criterion=default_stopping_criterion,
        max_splitting_values=50,
        multiple_hypothesis_test_method='holm',
        multiple_hypothesis_test_alpha=0.1,
        positivity_filtering_method=prevalence_symmetric,
        positivity_filtering_kwargs=None,
    ) -> None:
        """A causal model effect estimator built on top of a tree recursively stratifying the covariate space to
        balance between treated and untreated.

        Args:
            outcome_model (Union[IndividualOutcomeEstimator, PopulationOutcomeEstimator]):
                An outcome model for generating counterfactual predictions at each leaf node of the tree.
                Defaults to a simple `MarginalOutcomeEstimator` that just takes the average outcome for each treatment
                group in each leaf. However, it may also be any arbitrary outcome model to further adjust for the
                covariates (that the tree might leave some residual bias in the stratification).
            individual (bool): If True (and if `outcome_model` has `estimate_individual_outcomes`)
                will generate individual-level predictions for observations within each leaf.
                Otherwise, each observation takes the value of the average outcome in that leaf
                (using the `estimate_population_outcomes` method).
            asmd_violation_threshold (float): The value of Absolute Standardized Mean Difference below which a subgroup
                is considered balanced.
            min_leaf_size (int): The minimum number of samples required to split an internal node
            min_treat_group_size (int): The minimum number of samples in all treatment groups
                required to split an internal node.
            min_split_size (int): The minimum number of samples required to split an internal node.
            max_depth (int): The maximum depth of the tree.
                Will be updated for each level of nodes as the tree grows.
            stopping_criterion (callable): A function that takes the node/subtree as well as the data (`X`, `a`)
                and returns a boolean `True` if to stop splitting the tree and `False` if to continue splitting.
            max_splitting_values (int): The maximal number of unique values to consider when splitting a single feature
            multiple_hypothesis_test_method: The method for correcting p-values in multiple hypotheses testing.
                Should be compatible with statsmodels' `multipletests`.
            multiple_hypothesis_test_alpha (float): The alpha value for correcting p-values in
                multiple hypotheses testing.
                Should be compatible with statsmodels' `multipletests`.
            positivity_filtering_method (callable):
                A function that takes the current node (or subtree) as well as
                the arbitrary kwargs from `positivity_filtering_kwargs` and
                returns a list of the leaves/nodes' indices that do *not* violate positivity.
            positivity_filtering_kwargs: Keyword arguments to call the `positivity_filtering_method` with.
        """
        super().__init__(learner=None)

        if outcome_model is None:
            self.outcome_model = MarginalOutcomeEstimator(None)
        else:
            self.outcome_model = deepcopy(outcome_model)

        if positivity_filtering_kwargs is None:
            positivity_filtering_kwargs = {'alpha': 0.1}

        self.individual = individual
        self.tree = BalancingTree(
            asmd_violation_threshold=asmd_violation_threshold,
            min_leaf_size=min_leaf_size,
            min_treat_group_size=min_treat_group_size,
            min_split_size=min_split_size, max_depth=max_depth,
            stopping_criterion=stopping_criterion,
            max_splitting_values=max_splitting_values,
            multiple_hypothesis_test_method=multiple_hypothesis_test_method,
            multiple_hypothesis_test_alpha=multiple_hypothesis_test_alpha,
            positivity_filtering_method=positivity_filtering_method,
            positivity_filtering_kwargs=positivity_filtering_kwargs
        )

    def fit(self, X, a, y, sample_weight=None):
        """ Build the BICauseTree partition

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values
            sample_weight: IGNORED

        Returns:
            (class) BICauseTree
        """
        self.tree.fit(X, a)
        self.fit_outcome_models(X, a, y)
        return self

    def apply(self, X):
        """ Get node assignments based on BICauseTree partition

        Args:
            X (pd.DataFrame): The feature matrix of all samples

        Returns:
            (pd.Series) A vector of node indices indexed according to X
        """
        assignment = self.tree.apply(X)
        return assignment

    def estimate_population_outcome(
        self,
        X, a, y=None,
        discard_violating_samples=True,
        agg_func="mean",
    ):
        """ Estimates outcomes at a population-level and assigns them to individuals in X

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values
            discard_violating_samples (boolean): whether to drop the NA in the individual outcomes
            agg_func : aggregation function to go from individual to population outcome estimates

        Returns:
            A vector of potential outcomes indexed according to X
        """
        individual_cf = self.estimate_individual_outcome(X, a, y)
        if discard_violating_samples:
            individual_cf = individual_cf.dropna()
        if individual_cf.isnull().any().any():
            treatment_values = len(individual_cf.columns)
            pop_outcome = pd.Series(np.nan, index=range(0, treatment_values))
        else:
            pop_outcome = individual_cf.apply(self._aggregate_population_outcome, args=(agg_func,))
        return pop_outcome

    def estimate_individual_outcome(self, X, a, y=None, same_dim_as_input=True):
        """estimate individual-level counterfactual predictions.

        if `self.individual is True` and `self.outcome_model` has `estimate_individual_outcome()`
        then each observation will get a unique counterfactual value.
        otherwise, each individual gets the average prediction of its node,
        and this function is a way to transform it to a shape of predictions-per-observation.

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values
            same_dim_as_input (boolean): whether to return nan values for
                positivity-violating observations or exclude them

        Returns:
            pd.DataFrame: A matrix of individual outcomes indexed according to X
        """
        node_assignment = self.apply(X)
        all_df = []
        for node in np.unique(node_assignment.values):

            curr_node_index = node_assignment == node
            curr_X = X.loc[curr_node_index]
            curr_a = a[curr_node_index]
            curr_y = y[curr_node_index] if y is not None else None
            curr_model = self.node_models_[node]

            if self.individual:
                if hasattr(curr_model, "estimate_individual_outcome"):
                    outcomes = curr_model.estimate_individual_outcome(curr_X, curr_a)
                else:
                    raise AttributeError(
                        "The `outcome_model` provided can't estimate individual outcomes, "
                        "even though `self.individual` is set to True."
                        "Please change the `outcome_model` or set `self.individual` to False."
                    )
            else:
                outcomes = curr_model.estimate_population_outcome(curr_X, curr_a, curr_y)
                # we duplicate the outcome values as the len of the vector to concat to the df
                outcomes = pd.DataFrame(outcomes).transpose()
                outcomes = pd.concat([outcomes] * len(curr_a), ignore_index=True)
                outcomes = outcomes.set_index(curr_X.index)

            all_df.append(outcomes)

        # concat all df together
        df = pd.concat(all_df)
        df = df.loc[X.index]

        if not same_dim_as_input:
            df = df.dropna()

        return df

    def fit_outcome_models(self, X, a, y):
        """ Fits causal models to the nodes of a fitted (already grown) tree.

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values

        Returns:
            dict[int, Union[IndividualOutcomeEstimator, PopulationOutcomeEstimator]
        """
        positivity_violating_leaves = self.tree.get_positivity_violation_status()
        assignment = self.tree.apply(X)
        total_nodes = assignment.unique()

        node_models = {}
        for node in total_nodes:

            has_positivity_violation = positivity_violating_leaves[node]
            idx = assignment[assignment == node].index
            current_node_X = X.loc[idx]
            current_node_a = a.loc[idx]
            current_node_y = y.loc[idx]

            model = OverlapViolationEstimator() if has_positivity_violation else deepcopy(self.outcome_model)
            model.fit(current_node_X, current_node_a, current_node_y)
            node_models[node] = deepcopy(model)

        self.node_models_ = node_models
        return node_models

    def explain(self, X, a, split_condition: str = None):
        """Create a list of data frames summarizing the decision tree and the marginal effect.
            Each data-frame represents a leaf in the tree, and the list represents the tree itself.
            Each data frame exhibits several summary statistics about the path from the root to the
            leaf, including the maximal asmd at that level and the marginal outcome value.

        Args:
            X (pd.DataFrame): The feature data. Assumed to be of the same column
            structure as the training data
            a (pd.Series): The treatment assignment vector
            split_condition (str): The string representing the first condition. Default to 'All'
            to which the split explanations are added

        Returns:
            List[pd.DataFrame]: The list representing the tree, holding a data-frame for every leaf.
        """
        return self.tree.explain(X, a, split_condition)


class BalancingTree:
    def __init__(
        self,
        asmd_violation_threshold=0.1,
        min_leaf_size=0,
        min_treat_group_size=0,
        min_split_size=0,
        max_depth=10,
        stopping_criterion=default_stopping_criterion,
        max_splitting_values=50,
        multiple_hypothesis_test_method='holm',
        multiple_hypothesis_test_alpha=0.1,
        positivity_filtering_method=None,
        positivity_filtering_kwargs=None,
        _parent_=None
    ):
        """A tree node that recursively stratifies the covariate space based on ASMD value and other constraints.

        Args:
            asmd_violation_threshold (float): The value of Absolute Standardized Mean Difference below which a subgroup
                is considered balanced.
            min_leaf_size (int): The minimum number of samples required to split an internal node
            min_treat_group_size (int): The minimum number of samples in all treatment groups
                required to split an internal node.
            min_split_size (int): The minimum number of samples required to split an internal node.
            max_depth (int): The maximum depth of the tree.
                Will be updated for each level of nodes as the tree grows.
            stopping_criterion (callable): A function that takes the node/subtree as well as the data (`X`, `a`)
                and returns a boolean `True` if to stop splitting the tree and `False` if to continue splitting.
            max_splitting_values (int): The maximal number of unique values to consider when splitting a single feature
            multiple_hypothesis_test_method: The method for correcting p-values in multiple hypotheses testing.
                Should be compatible with statsmodels' `multipletests`.
            multiple_hypothesis_test_alpha (float): The alpha value for correcting p-values in
                multiple hypotheses testing.
                Should be compatible with statsmodels' `multipletests`.
            positivity_filtering_method (callable):
                A function that takes the current node (or subtree) as well as
                the arbitrary kwargs from `positivity_filtering_kwargs` and
                returns a list of the leaves/nodes' indices that do *not* violate positivity.
            positivity_filtering_kwargs: Keyword arguments to call the `positivity_filtering_method` with.
            _parent_ (BalancingTree): The parent of the current node/subtree.
        """
        self.asmd_violation_threshold = asmd_violation_threshold
        self.min_leaf_size = min_leaf_size
        self.min_treat_group_size = min_treat_group_size
        self.min_split_size = min_split_size
        self.max_depth = max_depth
        self.stopping_criterion = stopping_criterion
        self.max_splitting_values = max_splitting_values
        self.multiple_hypothesis_test_method = multiple_hypothesis_test_method
        self.multiple_hypothesis_test_alpha = multiple_hypothesis_test_alpha
        self.positivity_filtering_method = positivity_filtering_method
        self.positivity_filtering_kwargs = positivity_filtering_kwargs
        self._parent_ = _parent_

        if self.multiple_hypothesis_test_method is not None and self.multiple_hypothesis_test_alpha is None:
            raise ValueError(
                "Must be set `multiple_hypothesis_test_alpha` to a non-None value if "
                f"`multiple_hypothesis_test_method` (={multiple_hypothesis_test_method}) is provided."
            )

        # Add values that will be later initialized.from
        self.subtree_ = None
        self.node_index_ = None
        self.is_violating_positivity_ = None
        self.keep_ = False
        # TODO: I don't like the logic makes the default value of a newly generated node to "discard",
        #       Can I re-write the pruning logic that makes it possible to start with `keep_=True`?

    @property
    def _is_leaf(self):
        return self.subtree_ is None

    def fit(self, X: pd.DataFrame, a: pd.Series):
        """Builds a tree that stratifies the covariate space.

        Finds a stratification of the data space according to treatment allocation disparity by covariates in X
        The tree goes down until some stopping criteria is met, then prunes back according to the
        multiple hypothesis test results. Resulting leaf nodes are marked if their subpopulation violates positivity

        When the tree is pruned, nodes are kept if they have either or both:
            i) a descendant with a significant p-value
            ii) a direct parent with a significant p-value

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments

        Returns:
            BalancingTree:

        """
        self._build_tree(X, a)
        self._enumerate_nodes()
        if self.multiple_hypothesis_test_method is not None:
            self._prune(method=self.multiple_hypothesis_test_method, alpha=self.multiple_hypothesis_test_alpha)
        if self.positivity_filtering_method is not None:
            self._find_non_positivity_violating_leaves(
                positivity_filtering_method=self.positivity_filtering_method,
                positivity_filtering_kwargs=self.positivity_filtering_kwargs,
            )

        return self

    def _build_tree(self, X: pd.DataFrame, a: pd.Series):
        asmds = calculate_asmds(X, a)
        self.max_feature_asmd_ = asmds.max()
        self.treatment_prevalence_ = a.mean(axis=0)
        self.node_sample_size_ = len(a)
        if not self.stopping_criterion(self, X, a):
            self.split_feature_ = asmds.idxmax()
            self.split_value_ = self._find_split_value(X[self.split_feature_], a)
            if np.isnan(self.split_value_):
                self.split_feature_ = None
                return self
            self.n_asmd_violating_features_ = (asmds > self.asmd_violation_threshold).sum()
            self._recurse_over_build(X, a)

        # TODO: this building functionalities can be generalized, by generalizing the 3 main parts of this function:
        #       1. Feature selection: what feature is selected for splitting.
        #                             Inputs data (X, a, possibly y) and outputs a score per feature
        #                             (say, lower is better).
        #                             Currently ASMD, but with `y` may be Oui-ASMD, too, or something else.
        #                                   (However, will require `BalancingTree` to get optional `y`, too.)
        #       2. Feature splitting: Given a feature vector, how exactly to split it (to left and right children).
        #                             Inputs feature and `a`, and outputs a threshold.
        #                             Given how much of the pruning relies on p-values, it should also output a p-value
        #                             (or any other measure where score -> 0 is more confident and score -> 1 is less).
        #       3. Stopping criteria: Currently already implemented generically.
        #                             Inputs tree (node) and data (X, a, possibly y), and returns True if node should
        #                             not be further split.
        return self

    def _recurse_over_build(self, X, a):
        X_left, X_right, a_left, a_right = self._split_data(X, a)
        self_params = {v: getattr(self, v) for v in vars(self) if not v.endswith("_")}
        child_params = {
            **self_params,
            "max_depth": self_params["max_depth"] - 1,
            "_parent_": self,
        }
        self.subtree_ = (
            BalancingTree(**child_params)._build_tree(X_left, a_left),
            BalancingTree(**child_params)._build_tree(X_right, a_right),
        )

    def _split_data(self, X: pd.DataFrame, a: pd.Series = None):
        left = X[self.split_feature_] <= self.split_value_
        right = ~left  # note that this keeps NaNs on the right always
        res = []
        for data in [X, a]:
            if data is None:
                res.extend((None, None))
            else:
                res.extend((data[left], data[right]))
        return tuple(res)

    def _find_split_value(self, x: pd.Series, a: pd.Series):
        """Find a single split that should reduce imbalance
        by finding the value for which the mean treatment prevalence differs the most on both sides.

        Do not consider splits that result in groups that are smaller than min_leaf_size

        Args:
            x (pd.Series): The feature vector
            a (pd.Series): The treatment assignments

        Returns:
            The split value (normally a float, but could be anything that supports < comparison),
        """
        x_uniq = self._get_uniq_values(x)
        p_values = [self._fisher_test_p_value(x, a, x_val) for x_val in x_uniq]
        p_values = pd.Series(p_values, index=x_uniq).dropna()
        if len(p_values) == 0:
            return np.nan
        self.p_value_ = p_values.min()
        return p_values.idxmin()

    def _get_uniq_values(self, x):
        n = np.min([self.max_splitting_values, len(x)])
        x_uniq = x.quantile(q=np.arange(n + 1) / n, interpolation="lower").unique()
        x_uniq = x_uniq[x_uniq != x.max()]
        return x_uniq

    def _fisher_test_p_value(self, x, a, x_val):
        I = pd.cut(x, [-np.inf, x_val, np.inf], labels=[0, 1])
        if (sum(I == 0) <= self.min_leaf_size) | (sum(I == 1) <= self.min_leaf_size):
            return np.nan
        crosstab = pd.crosstab(I, a)
        if crosstab.min().min() < 20:
            _, p_value = fisher_exact(crosstab)
        else:
            _, p_value, _, _ = chi2_contingency(crosstab)
        return p_value

    def _enumerate_nodes(self):
        index = 0
        queue = deque([self])
        while queue:
            node = queue.popleft()
            node.node_index_ = index
            index += 1
            if not node._is_leaf:
                queue.extend(node.subtree_)

    def _prune(self, method: str, alpha: float):
        """Remove redundant nodes from tree.

        Redundant being their p-value does not pass multiple hypothesis correction

        1) implements the multiple hypothesis correction on all node p-values in the tree
        according to the user-defined method
        2) records the corrected p-value and test result for each node
        3) prunes back the tree, keeping only the nodes with:
            i) a descendant with a significant p-value
            ii) a direct parent with a significant p-value

        """
        # Consider using `alpha` and `method` from `self` rather than from parameters
        corrected_non_leaf_summary = self._multiple_hypothesis_correction(alpha, method)
        self._set_corrected_p_value_to_nodes(corrected_non_leaf_summary)
        self._recurse_over_prune()
        self._total_pruned_nodes_ = self._count_pruned_nodes()
        self._delete_post_pruning()

    def _multiple_hypothesis_correction(self, alpha: float, method: str):
        # take all non-leaf nodes p-values and conduct a multiple hypothesis correction
        non_leaf_summary = self._generate_non_leaf_nodes_summary().dropna(subset=["p_value"])
        if non_leaf_summary.empty:
            # Accounts for the edge case of a single-node tree
            test = (None, None)
        else:
            test = multipletests(non_leaf_summary['p_value'], alpha=alpha, method=method)

        # Record the corrected p-value and boolean result of the multiple hypothesis test:
        non_leaf_summary['corrected_pval'] = test[1]
        non_leaf_summary['multiple_test_result'] = test[0]
        return non_leaf_summary

    def _set_corrected_p_value_to_nodes(self, summary_df):
        if self._is_leaf:
            self.corrected_p_value_ = None
            self.corrected_p_value_is_significant_ = None
        else:
            self.corrected_p_value_ = summary_df['corrected_pval'].loc[
                summary_df['node_index'] == self.node_index_
            ].iloc[0]  # converting a single line Series into a float
            self.corrected_p_value_is_significant_ = (
                summary_df['multiple_test_result'].loc[summary_df['node_index'] == self.node_index_]
            ).iloc[0]
            self.subtree_[0]._set_corrected_p_value_to_nodes(summary_df)
            self.subtree_[1]._set_corrected_p_value_to_nodes(summary_df)

    def _recurse_over_prune(self):
        is_root = self._parent_ is None
        if is_root:
            self.keep_ = True
            pass
        if not self._is_leaf:
            if self.corrected_p_value_is_significant_:
                # the split is significant so we mark two children as keep
                # the ancestors should be marked as keep to eventually save this split
                self._mark_all_ancestors_and_two_children_as_keep()
            self.subtree_[0]._recurse_over_prune()
            self.subtree_[1]._recurse_over_prune()

    def _mark_all_ancestors_and_two_children_as_keep(self):
        self.keep_ = True
        self._mark_two_children_as_keep()
        is_root = self._parent_ is None
        if not is_root:
            self._parent_._mark_all_ancestors_and_two_children_as_keep()

    def _mark_two_children_as_keep(self):
        self.subtree_[0].keep_ = True
        self.subtree_[1].keep_ = True

    def _delete_post_pruning(self):
        if not self._is_leaf:
            # Two children will always have the same keep_ value
            assert self.subtree_[0].keep_ == self.subtree_[1].keep_
            if not self.subtree_[0].keep_:
                self.subtree_ = None
            else:
                self.subtree_[0]._delete_post_pruning()
                self.subtree_[1]._delete_post_pruning()

    def _count_pruned_nodes(self):
        count = 0
        queue = deque([self])
        while queue:
            node = queue.popleft()
            count += not node.keep_
            if not node._is_leaf:
                queue.extend(node.subtree_)
        return count

    def _generate_node_summary(self, X=None, a=None, y=None):
        # TODO: This function should be decomposed into two:
        #       1. `get_attributes(self)`, which gets the value of inherent node values:
        #          node_index, is_leaf, depth, positivity_violation, but also p_value (and corrected_p_value).
        #       2. `calculate_attributes(self, X, a)`, which gets data-dependent node values:
        #          treatment_prevalence, sample_size, max_asmd, and n_asmd_violating_features.
        #          These could be then calculated on new unseen data, rather than stored values seen during `fit`.
        node_summary = pd.DataFrame({
            "node_index": self.node_index_,
            "is_leaf": self._is_leaf,
            "max_depth": self.max_depth,  # Current node's depth is root.max_depth - node.max_depth
            "positivity_violation": self.is_violating_positivity_,
            # `getattr` allows to ask for a value before this post-fit attribute is set.
            # Ideally, this node summary function will work on an attribute-to-attribute requested basis,
            # and not return all possible attributes at once, avoiding the current requirement for
            # accessing attributes not yet set. but in the meantime:
            "p_value": getattr(self, "p_value_", None),
            "treatment_prevalence": getattr(self, "treatment_prevalence_", None),
            "sample_size": getattr(self, "node_sample_size_", None),
            "n_asmd_violating_features": getattr(self, "n_asmd_violating_features_", None),
            "max_asmd": getattr(self, "max_feature_asmd_", None),
        }, index=[0])
        if self._is_leaf:
            return node_summary
        else:
            left_child_summary = self.subtree_[0]._generate_node_summary()
            right_child_summary = self.subtree_[1]._generate_node_summary()
            return pd.concat([node_summary, left_child_summary, right_child_summary], axis=0, ignore_index=True)

    def generate_leaf_summary(self, X=None, a=None, y=None):
        """
        Recursively building a dataframe describing the leaf nodes with treatment_prevalence, sample_size,
        node_prevalence, node_index, average_outcome_treated/untreated, p_value, is_leaf, positivity_violation,
        max_depth, n_asmd_violating_features, max_asmd

        Args:
            X (pd.DataFrame): The feature data. Assumed to be of the same column
            structure as the training data
            a (pd.Series): The treatment assignment vector
            y (pd.Series): The outcome vector

        Returns: (DataFrame) a frame describing the leaf nodes in the fitted tree, with treatment_prevalence,
            sample_size, node_prevalence, node_index, average_outcome_treated/untreated, p_value, is_leaf,
            positivity_violation, max_depth, n_asmd_violating_features, max_asmd

        """
        all_node_summary = self._generate_node_summary(X, a, y)
        return all_node_summary.loc[all_node_summary["is_leaf"]]

    def _generate_non_leaf_nodes_summary(self, X=None, a=None, y=None):
        all_node_summary = self._generate_node_summary(X, a, y)
        return all_node_summary.loc[~all_node_summary["is_leaf"]]

    def _find_non_positivity_violating_leaves(
        self,
        positivity_filtering_method,
        positivity_filtering_kwargs: dict = None,
    ):
        """

        Finding the non-positivity-violating leaf nodes according to a user-defined criterion,
        marks the corresponding nodes, and outputs a list of non-violating node indices

        Args:
            positivity_filtering_method (function): user-defined function for determining positivity violations
            positivity_filtering_kwargs (dict): parameters for positivity criterion

        Returns: (list) node indices of leaves that do not violate positivity

        """
        non_violating_nodes_idx = positivity_filtering_method(self, **positivity_filtering_kwargs)  # returns the non-violating leaves
        self._mark_positivity_nodes(non_violating_nodes_idx)
        return non_violating_nodes_idx

    def _mark_positivity_nodes(self, non_positivity_violating_nodes):
        if not self._is_leaf:
            self.subtree_[0]._mark_positivity_nodes(non_positivity_violating_nodes)
            self.subtree_[1]._mark_positivity_nodes(non_positivity_violating_nodes)
        else:
            self.is_violating_positivity_ = self.node_index_ not in non_positivity_violating_nodes

    def get_positivity_violation_status(self) -> dict:
        """For each leaf node get whether it is violating positivity or not.

        Returns:
            dict: keys are node's index and value is boolean whether the leaf is violating.
        """
        if self._is_leaf:
            result = {self.node_index_: self.is_violating_positivity_}
        else:
            left = self.subtree_[0].get_positivity_violation_status()
            right = self.subtree_[1].get_positivity_violation_status()
            result = {**left, **right}
        return result

    def apply(self, X: pd.DataFrame):
        """
        Sends X down the fitted tree and outputs which leaf node each sample from X is assigned to

        Args:
            X: (pd.DataFrame): The feature matrix

        Returns: (pd.Series) leaf index associated with each sample indexed according to X

        """
        if self._is_leaf:
            return pd.Series(data=self.node_index_, index=X.index)  # pandas series of leaf, individual index

        else:
            X_left, X_right, *_ = self._split_data(X)
            left_child_assignment = self.subtree_[0].apply(X_left)
            right_child_assignment = self.subtree_[1].apply(X_right)
            node_assignment = pd.concat([left_child_assignment, right_child_assignment], axis="index")
            node_assignment = node_assignment.loc[X.index]  # Keeping order of X input
            return node_assignment

    def explain(
            self, X, a, split_condition: str = None, df_explanation: pd.DataFrame = None
    ):
        """Create a list of data frames summarizing the decision tree and the marginal effect.
        Each data-frame represents a leaf in the tree, and the list represents the tree itself.
        Each data frame exhibits several summary statistics about the path from the root to the
        leaf, including the maximal asmd at that level and the marginal outcome value.
        Args:
            X (pd.DataFrame): The feature data. Assumed to be of the same column
            structure as the training data
            a (pd.Series): The treatment assignment vector
            split_condition (str): The string representing the first condition. Default to 'All'
            df_explanation (pd.DataFrame): Mostly used for recursion. The initial data-frame,
            to which the split explanations are added
        Returns:
            List[pd.DataFrame]: The list representing the tree, holding a data-frame for every leaf.
        """
        line = self._create_df_line(X, a, split_condition)
        df_explanation = pd.DataFrame() if df_explanation is None else df_explanation
        df_explanation = pd.concat([df_explanation.T, line], axis=1).T
        df_explanation = self._convert_dtypes(df_explanation)
        if self._is_leaf:
            return [df_explanation]
        else:
            X_left, X_right, a_left, a_right = self._split_data(X, a)
            explanation_left = self._build_split_condition_string("<=")
            explanation_right = self._build_split_condition_string(">")
            return [
                self.subtree_[0].explain(X_left, a_left, explanation_left, df_explanation),
                self.subtree_[1].explain(X_right, a_right, explanation_right, df_explanation),
            ]

    def _build_split_condition_string(self, condition: str):
        """Create the condition string `feature <condition> value`"""
        explanation = f"{self.split_feature_} "
        explanation += condition
        explanation += f" {self.split_value_:.4g}"
        return explanation

    def _convert_dtypes(self, df_explanation: pd.DataFrame):
        df_explanation = df_explanation.astype(
            {
                "N": int,
                "Propensity": float,
                "asmd": float,
                "n_asmd_violating_features": int,
                # "effect": float,
            }
        )
        return df_explanation

    def _create_df_line(self, X, a, split_condition):
        split_condition = split_condition or "All"
        # marginal = self.outcome_learner.estimate_population_outcome(X, a, y)
        # effect = marginal.diff().values[-1]
        asmds = calculate_asmds(X, a)
        line = pd.Series(
            {
                "N": len(X.index),
                "Propensity": a.sum() / len(X.index),
                "asmd": asmds.max(),
                "n_asmd_violating_features": (asmds > self.asmd_violation_threshold).sum(),
                # "effect": effect,
            },
            name=split_condition,
        )
        return line
