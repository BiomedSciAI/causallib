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
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact, chi2_contingency
import warnings
from sklearn.dummy import DummyClassifier

from causallib.utils.general_tools import get_iterable_treatment_values
from causallib.estimation.base_estimator import PopulationOutcomeEstimator, IndividualOutcomeEstimator
from causallib.estimation import MarginalOutcomeEstimator
from causallib.metrics.weight_metrics import calculate_covariate_balance


def prevalence_symmetric_cutoff(prob, mu, alpha=0.1):
    """
    Computes a lower/upper cutoff based on the prevalence of
    treatment in the cohort. Treatment should be binary.

    Args:
        mu (float): observed prevalence of treatment in the cohort
        prob (pd.DataFrame): probability to be assigned to a group
                          (n_samples, 2)
                          For binary treatment each row is (1-p, p)
        alpha (float): the fixed cutoff to be transformed, should be
        strictly between 0 and 1

    Returns: Tuple[float, float]: upper and lower cutoff
    """
    if prob.shape[1] > 2:
        raise ValueError('This threshold selection method is applicable only '
                         'for binary treatment assignment')
    if not 0 < alpha < 1:
        raise ValueError(f"`alpha` value should be in the open interval (0, 1). Got {alpha} instead.")
    upper_cutoff = (1 - alpha)*mu / ((1-alpha)*mu + alpha*(1 - mu))
    lower_cutoff = alpha*mu / (alpha*mu + (1 - alpha)*(1 - mu))
    return lower_cutoff, upper_cutoff


def crump_cutoff(prob, segments=10000 ):
    """
    A systematic approach to find the optimal trimming cutoff, based on the
    marginal distribution of the propensity score,
    and according to a variance minimization criterion.
    "Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment effects."
    Treatment should be binary.
    Args:
        prob (pd.DataFrame): probability to be assigned to a group
                          (n_samples, 2)
        segments (int): number of exclusive segments of the interval (0, 0.5].
                        more segments results with more precise cutoff
    Returns:
        float: the optimal cutoff,
               i.e. the smallest value that satisfies the criterion.
    """
    # TODO: rethink input - probability_matrix, propensity_vector, or model + data
    if prob.shape[1] > 2:
        raise ValueError('This threshold selection method is applicable only '
                         'for binary treatment assignment')
    else:
        propensities = prob.iloc[:, 1]

    alphas = np.linspace(1e-7, 0.5, segments)
    alphas_weights = alphas * (1 - alphas)
    overlap_weights = propensities * (1 - propensities)
    for i in range(segments):
        obs_meets_criterion = overlap_weights >= alphas_weights[i]
        criterion = 2 * (np.sum(obs_meets_criterion / overlap_weights) /
                         np.maximum(np.sum(obs_meets_criterion), 1e-7))
        if (1 / alphas_weights[i]) <= criterion:
            return alphas[i], 1-alphas[i]
    return None, None #no overlap

def default_stopping_criterion(tree, X: pd.DataFrame, a: pd.Series):
    """
    Args:
        X (pd.DataFrame): The feature matrix
        a (pd.Series): the treatment assignment vector
    Returns:
        (boolean): is the stopping criterion met
    """
    # returns True if we must stop

    asmds = tree._calculate_asmds(X, a)
    no_exploitable_asmds=asmds.isna().all()
    #asmds can be NaN if the standard deviation is null
    if no_exploitable_asmds:
        return True
    _is_asmd_threshold_reached = asmds.max() <= tree.asmd_threshold_split
    _is_min_treat_group_size_reached = min(a.value_counts()) <= tree.min_treat_group_size
    _is_min_split_size_reached = X.shape[0] <= tree.min_split_size
    _is_max_depth_tree_reached = tree.max_depth <= 1

    criteria = any(
        [_is_asmd_threshold_reached, _is_min_treat_group_size_reached, _is_min_split_size_reached,
         _is_max_depth_tree_reached])

    return criteria

def prevalence_symmetric(tree, alpha=0.1):

    """
    Computes the symmetric lower/upper cutoff based on prevalence and a fixed cutoff alpha
    with p the original propensity score and mu the prevalence in the cohort
    Returns the list of non violating nodes for these cutoffs

    Args:
        tree (PropensityImbalanceStratification): the splitting tree
        alpha (float): the fixed cutoff to be transformed

    Checks for the default stopping criterion, which is:
    is the ASMD threshold reached
    OR is the minimum treatment group size reached
    OR is the minimum leaf size reached
    OR is the maximum depth of the tree reached
    Returns: (list) The node indices of non-violating leaves
    """
    leaf_summary = tree.generate_leaf_summary()
    ps_singles_list = leaf_summary['pscore'].values
    repetition_numbers = leaf_summary['sample_size'].values.tolist()
    ps_list = ps_singles_list.repeat(repetition_numbers)
    ps_vect = pd.DataFrame(np.column_stack((1 - ps_list, ps_list)))
    mu=tree.node_prevalence_ #overall prevalence of treatment
    cutoffs=prevalence_symmetric_cutoff(ps_vect, mu, alpha)
    lower_cutoff, upper_cutoff= cutoffs[0], cutoffs[1]
    non_violating_nodes = leaf_summary.loc[leaf_summary['pscore'].between(lower_cutoff, upper_cutoff), 'node_index'].tolist()

    return non_violating_nodes


def crump(tree, segments=10000):
    """
    Generates a list of non positivity violating node indices according to the crump cutoff
    i.e. nodes with propensity scores in [cutoff, 1-cutoff]

    Crump is a systematic approach to find the optimal trimming cutoff see
    "Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment effects."

    Args:
        tree (class): the splitting tree
        segments (int): number of portions used for computing the crump cutoff


    Returns: (list) The node indices of non-violating leaves
    """
    # TODO: add data to signature once we remove the saved data attributes from leaves.

    leaf_summary = tree.generate_leaf_summary()
    ps_singles_list =  leaf_summary['pscore'].values
    repetition_numbers = leaf_summary['sample_size'].values.tolist()
    ps_list = ps_singles_list.repeat(repetition_numbers)
    ps_vect = pd.DataFrame(np.column_stack((1 - ps_list, ps_list)))
    lower_cutoff, upper_cutoff = crump_cutoff(ps_vect, segments)
    if lower_cutoff is not None:
        non_violating_nodes = leaf_summary['node_index'][leaf_summary['pscore'].between(lower_cutoff,upper_cutoff)].tolist()
        return non_violating_nodes
    return [] # lower_cutoff=None means all nodes have positivity violations


class OverlapViolationEstimator():
    '''
    A causallib-compatible model returning NaNs with appropriate format and indexing.
    Meant to be used when overlap violations are detected.
    '''
    def __init__(
        self,
        value = None
    ) -> None:
        self.value = value
    def fit(self,X,a,y):
        '''

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values

        Returns:
            (class) OverlapViolationEstimator

        '''
        return self
    def estimate_population_outcome(self,X,a,y):
        '''

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values

        Returns:
            (pd.Series) A vector of NaN population outcomes indexed according to X
        '''
        treatment_values = self._get_treatment_index(a)
        return pd.Series(np.nan, index=treatment_values)
    def estimate_individual_outcome(self, X, a):
        '''

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments

        Returns:
            (pd.Series) A vector of NaN individual outcomes indexed according to X
        '''
        treatment_values = self._get_treatment_index(a)
        return pd.DataFrame(np.nan, index=X.index,columns=treatment_values)
    def _get_treatment_index(self,a):
        return pd.Index(sorted(a.unique()), name=a.name)


class PropensityBICauseTree():

    def __init__(
        self,
        learner = DummyClassifier(strategy='prior'),
            **kwargs
    ) -> None:
        """
        A propensity model based on a BICause Tree,
        scikit-learn compatible


        Args:
            learner: The propensity model
            **kwargs: The propensity model hyperparameters
        """

        self.tree = PropensityImbalanceStratification(**kwargs)
        self.learner = learner

    def fit(self,X,a):
        """Fits a PropensityBICauseTree propensity model.
        Compatible with scikit-learn classifiers, which can be combined with causallib's models
        - e.g., IPW(PropensityBICauseTree())

        Args:
            X (pd.DataFrame): The feature matrix
            a (pd.Series): The treatment assignment vector

        Returns:
            (class) PropensityBICauseTree
        """
        self.tree.fit(X, a)
        self.fit_treatment_models(X,a)
        return self

    def fit_treatment_models(self, X, a):
        '''Fits the propensity models in the leaf nodes.

        Args:
            X (pd.DataFrame): The feature matrix
            a (pd.Series): The treatment assignment vector

        Returns:
            (class) PropensityBICauseTree
        '''

        assignment = self.tree.apply(X)
        total_nodes = assignment.unique()
        self.tree._set_positivity_violation_nodes(violating_nodes={})
        self.treatment_values_ = np.unique(a)

        node_models = {}
        for node in total_nodes:

            id = assignment[assignment==node].index
            # create subset of subjects in current node

            current_node_X = X.loc[id]
            current_node_a = a.loc[id]
            has_positivity_violation = self.tree.is_violating_[node]

            # if not has_positivity_violation:
            # fit model to subset of samples
            model = self.learner.fit(current_node_X, current_node_a)


            # dictionary : key - node index , value - fitted model
            # this dictionary will be a atribute of BecauseTree class
            node_models[node] = deepcopy(model)

        self.node_models_ =  node_models
        return self

    def predict_proba(self, X):
        ''' Predict the individual treatment probabilities.
        Sends the data down the fitted tree and evaluates the leaf propensity score models.

        Args:
            X (pd.DataFrame): The feature matrix

        Returns:
            (pd.DataFrame) A vector of individual treatment probabilities
        '''
        node_assignment = self.tree.apply(X)
        all_df = np.zeros((X.shape[0], len(self.treatment_values_.astype('int'))))
        for node in np.unique(node_assignment.values):

            curr_node_index = node_assignment==node
            curr_X = X.loc[curr_node_index]
            curr_model = self.node_models_[node]

            # TODO : make df and sort by index and only at the end covert to np array
            probability = curr_model.predict_proba(curr_X)
            if probability.shape[1] == len(self.treatment_values_):
                all_df[curr_node_index, :] = probability
            else:
                existing_classes = curr_model.classes_ == np.arange(all_df.shape[1])
                shape_all_df = all_df[curr_node_index, existing_classes].shape
                all_df[curr_node_index, existing_classes] = probability.reshape(shape_all_df)
            # adding a column with in index for later sorting
            # id = np.atleast_2d(np.array(curr_X.index)).T
            # probability = np.append(probability, id, axis=1)
            # all_df.append(probability)

        # total_propensity = np.vstack(all_df)
        # sorted_array = total_propensity[np.argsort(total_propensity[:, -1])]
        # sorted_array = np.delete(sorted_array, -1, 1)

        return all_df

class BICauseTree(IndividualOutcomeEstimator):

    def __init__(
        self,
        outcome_model: PopulationOutcomeEstimator = None,
        individual=False,
        asmd_threshold_split=0.1,
        min_leaf_size=0,
        min_split_size=0,
        min_treat_group_size=0,
        asmd_violation_threshold=0.1,
        max_depth=10,
        n_values=50,
        multiple_hypothesis_test_alpha=0.1,
        multiple_hypothesis_test_method='holm',
        positivity_filtering_kwargs={'alpha':0.1},
        stopping_criterion=default_stopping_criterion,
        positivity_filtering_method=prevalence_symmetric,
        _parent_=None
    ) -> None:
        super().__init__(learner=None)

        if outcome_model is None:
            self.outcome_model = MarginalOutcomeEstimator(None)
        else:
            self.outcome_model = deepcopy(outcome_model)

        self.tree = PropensityImbalanceStratification(stopping_criterion=stopping_criterion,
                                                        min_split_size=min_split_size,
                                                        max_depth = max_depth,
                                                        min_leaf_size=min_leaf_size,
                                                        min_treat_group_size=min_treat_group_size,
                                                        asmd_threshold_split=asmd_threshold_split,
                                                        positivity_filtering_method=positivity_filtering_method,
                                                        positivity_filtering_kwargs=positivity_filtering_kwargs
                                                        )
        self.individual = individual

    def fit(self, X, a, y, sample_weight=None):
        ''' Build the BICauseTree partition

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values
            sample_weight: IGNORED

        Returns:
            (class) BICauseTree
        '''
        self.tree.fit(X,a)
        self.fit_outcome_models(X,a,y)
        return self

    def apply(self, X):
        ''' Get node assignments based on BICauseTree partition

        Args:
            X (pd.DataFrame): The feature matrix of all samples

        Returns:
            (pd.Series) A vector of node indices indexed according to X
        '''
        assignment = self.tree.apply(X)
        return assignment

    def estimate_population_outcome(self, X, a, y=None, discard_violating_samples = True, agg_func="mean"):

        ''' Estimates outcomes at a population-level and assigns them to individuals in X

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values
            discard_violating_samples (boolean): whether to drop the NA in the individual outcomes
            agg_func : aggregation function to go from individual to population outcome estimates

        Returns:
            A vector of potential outcomes indexed according to X
        '''
        individual_cf = self.estimate_individual_outcome(X, a, y)
        if discard_violating_samples:
            individual_cf = individual_cf.dropna()
        if individual_cf.isnull().any().any():
            treatment_values = len(individual_cf.columns)
            pop_outcome = pd.Series(np.nan, index=range(0,treatment_values))
        else:
            pop_outcome = individual_cf.apply(self._aggregate_population_outcome, args=(agg_func,))
        return pop_outcome


    def estimate_individual_outcome(self, X, a, y=None, same_dim_as_input=True):
        '''

        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values
            same_dim_as_input (boolean): whether or not to return Nan individual outcomes or trim them

        Returns:
            (pd.DataFrame) A vector of individual outcomes indexed according to X
        '''
        node_assignment = self.apply(X)
        all_df = []
        for node in np.unique(node_assignment.values):

            curr_node_index = node_assignment==node
            node_index_list = curr_node_index[curr_node_index == True].index
            curr_X = X.loc[curr_node_index]
            curr_a = a[curr_node_index]
            curr_y = y[node_index_list] if y is not None else None
            curr_model = self.node_models_[node]

            if not self.individual and hasattr(curr_model,'estimate_population_outcome'):
                # no individual estimation from the model but we aggregate the pop outcomes
                if y is None:
                    raise AttributeError('You must enter an outcome vector for estimating population-level outcomes\
                                                                        ...or use an individual-level outcome estimation model')
                else:
                    outcomes = curr_model.estimate_population_outcome(curr_X,curr_a,curr_y)
                    # we duplicate the outcome values as the len of the vector to concat to the df
                    outcomes = pd.DataFrame(outcomes).transpose()
                    outcomes = pd.concat([outcomes]*len(curr_a), ignore_index=True)
                    outcomes = outcomes.set_index(curr_X.index)

            elif self.individual and hasattr(curr_model,'estimate_individual_outcome'):
                outcomes = curr_model.estimate_individual_outcome(curr_X, curr_a)
            else:
                raise AttributeError('The model you have chosen for estimation doesnt have the functionality to estimate\
                                                        ...individual outcomes. Please change the bool -> population=True')

            # outcomes['node'] = node
            all_df.append(outcomes)

        # concat all df together
        df = pd.concat(all_df)
        df = df.loc[X.index]

        if not same_dim_as_input:
            df = df.dropna()

        return df


    def fit_outcome_models(self, X, a, y):
        '''
        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments
            y (pd.Series): The outcome values

        Returns:
            (class) BICauseTree
        '''
        self.tree._set_positivity_violation_nodes(violating_nodes={})
        assignment = self.tree.apply(X)
        total_nodes = assignment.unique()

        node_models = {}
        for node in total_nodes:

            has_positivity_violation = self.tree.is_violating_[node]
            id = assignment[assignment==node].index
            current_node_X = X.loc[id]
            current_node_a = a.loc[id]
            current_node_y = y.loc[id]
            if has_positivity_violation:
                model = OverlapViolationEstimator().fit(X, a, y)
            else:
            # fit model to subset of samples
                model = self.outcome_model.fit(current_node_X, current_node_a, current_node_y)

            # dictionary : key - node index , value - fitted model
            # this dictionary will be a atribute of BecauseTree class
            node_models[node] = deepcopy(model)

        self.node_models_ =  node_models
        return self

    def explain(self, X, a, split_condition: str = None, df_explanation: pd.DataFrame = None):
        """Create a list of data frames summarizing the decision tree and the marginal effect.
            Each data-frame represents a leaf in the tree, and the list represents the tree itself.
            Each data frame exhibits several summary statistics about the path from the root to the
            leaf, including the maximal asmd at that level and the marginal outcome value.

        Args:
            X (pd.DataFrame): The feature data. Assumed to be of the same column
            structure as the training data
            a (pd.Series): The treatment assignment vector
            y (pd.Series): The outcome vector
            split_condition (str): The string representing the first condition. Default to 'All'
            df_explanation (pd.DataFrame): Mostly used for recursion. The inital data-frame,
            to which the split explanations are added

        Returns:
            List[pd.DataFrame]: The list representing the tree, holding a data-frame for every leaf.
        """
        return self.tree.explain(X, a, split_condition, df_explanation)



class PropensityImbalanceStratification():
    def __init__(
        self,
        asmd_threshold_split=0.1,
        min_leaf_size=0,
        min_split_size=0,
        min_treat_group_size=0,
        asmd_violation_threshold=0.1,
        max_depth=10,
        n_values=50,
        multiple_hypothesis_test_alpha=0.1,
        multiple_hypothesis_test_method='holm',
        positivity_filtering_kwargs=None,
        stopping_criterion=default_stopping_criterion,
        positivity_filtering_method=None,
        _parent_=None,
        violating_nodes= None
    ) -> None:
        '''

        Args:
            asmd_threshold_split:
            min_leaf_size:
            min_split_size:
            min_treat_group_size:
            asmd_violation_threshold:
            max_depth:
            n_values:
            multiple_hypothesis_test_alpha:
            multiple_hypothesis_test_method:
            positivity_filtering_kwargs:
            stopping_criterion:
            positivity_filtering_method:
            _parent_:
            violating_nodes:
        '''
        super().__init__()

        self.n_values = n_values
        # values that are determined during training
        self.positivity_filtering_kwargs=positivity_filtering_kwargs
        self.positivity_filtering_method = positivity_filtering_method
        self.multiple_hypothesis_test_alpha = multiple_hypothesis_test_alpha
        self.multiple_hypothesis_test_method = multiple_hypothesis_test_method
        if self.multiple_hypothesis_test_method is not None and self.multiple_hypothesis_test_alpha is None:
            raise ValueError("multiple_hypothesis_test_alpha is None, must be set to a value if multiple_hypothesis_test_method is not None")
        self.node_sample_size_=None
        self.positivity_violation_ = None
        self.propensity_score_ = None
        self.node_prevalence_ = None
        self.pval_ = None
        self.n_asmd_violating_features_ = None
        self.node_index_ = None
        self.split_feature_ = None
        self.split_value_ = np.NaN
        self.max_feature_asmd_ = np.NaN
        self.subtree_ = None
        self.asmd_threshold_split=asmd_threshold_split
        self.asmd_violation_threshold=asmd_violation_threshold
        self.min_leaf_size=min_leaf_size
        self.min_split_size=min_split_size
        self.min_treat_group_size=min_treat_group_size
        self.stopping_criterion=stopping_criterion
        self.max_depth = max_depth
        self.corrected_pval_ = None
        self._total_pruned_nodes_=0
        self.is_split_significant_corrected_ = None
        self._parent_ = _parent_
        self.keep_=False
        self.potential_outcomes_ = None
        self.is_violating_ = None


    def fit(self, X: pd.DataFrame, a: pd.Series):
        """

        Finds a stratification of the data space according to treatment allocation disparity by covariates in X
        The tree goes down until some stopping criteria is met, then prunes back according to the
        multiple hypothesis test results. Resulting leaf nodes are marked if their subpopulation violates positivity

        When the tree is pruned nodes are kept if they have either or both:
            i) a descendant with a significant p-value
            ii) a direct parent with a significant p-value
        Outcome models are fitted in the leaf nodes of the resulting tree, and the average potential outcomes estimated.
        In case there is no overlap in the leaf, the average potential outcomes will be NaNs.



        Args:
            X (pd.DataFrame): The feature matrix of all samples
            a (pd.Series): The treatment assignments

        Returns:
            (class) PropensityImbalanceStratification

        """
        self._build_tree(X, a)
        self._enumerate_nodes()
        if self.multiple_hypothesis_test_method is not None:
            self._prune(alpha=self.multiple_hypothesis_test_alpha, method=self.multiple_hypothesis_test_method)
        if self.positivity_filtering_method is not None:
            self._find_non_positivity_violating_leaves(positivity_filtering_method=self.positivity_filtering_method,
                                                       positivity_filtering_kwargs=self.positivity_filtering_kwargs)

        return self

    def _build_tree(self, X: pd.DataFrame, a: pd.Series):
        asmds = self._calculate_asmds(X, a)
        self.max_feature_asmd_ = asmds.max()
        # TODO: avoid saving explicit summary statistics on the training data by fitting models instead.
        self.node_prevalence_ = a.mean(axis=0)
        self.propensity_score_ = a.mean(axis=0) # this redundancy is only temporary
        # in the future the propensity_score_ will be computed from a model
        self.node_sample_size_=len(a)
        if not self.stopping_criterion(self, X, a):
            self.split_feature_ = asmds.idxmax()
            self.split_value_ = self._find_split_value(X[self.split_feature_], a)
            # TODO: This method can be plugged-in to use other heuristics
            if np.isnan(self.split_value_):
                self.split_feature_ = None
                return self
            self.n_asmd_violating_features_= (asmds > self.asmd_violation_threshold).sum()
            self._recurse_over_build(X, a)
        return self


    @staticmethod
    def _calculate_asmds(X: pd.DataFrame, a: pd.Series):
        epsilon = np.finfo(float).resolution
        treatment_values = get_iterable_treatment_values(None, a)
        if len(treatment_values) == 0:
            return a.replace(a.index, np.nan)
        asmds = calculate_covariate_balance(X, a, w=pd.Series(1, index=a.index))
        return asmds.abs()

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
        pvals = [self._fisher_test_pval(x, a, x_val) for x_val in x_uniq]
        pvals = pd.Series(pvals, index=x_uniq).dropna()
        if len(pvals) == 0:
            return np.nan
        self.pval_ = pvals.min()
        return pvals.idxmin()



    def _build_split_condition_string(self, condition: str):
        """Create the condition string `feature <condition> value`"""
        explanation = f"{self.split_feature_} "
        explanation += condition
        explanation += f" {self.split_value_:.4g}"
        return explanation

    def _recurse_over_build(self, X, a):
        X_left, X_right, a_left, a_right= self._split_data(X, a)
        self_params = {v: getattr(self, v) for v in vars(self) if not v.endswith("_")}
        self_params.pop("max_depth")
        next_max_depth=self.max_depth-1
        self.subtree_ = (
            PropensityImbalanceStratification(**self_params, max_depth=next_max_depth, _parent_=self)._build_tree(X_left, a_left),
            PropensityImbalanceStratification(**self_params, max_depth=next_max_depth, _parent_=self)._build_tree(X_right, a_right),
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

    def _fisher_test_pval(self, x, a, x_val):
        I = pd.cut(x, [-np.inf, x_val, np.inf], labels=[0, 1])
        if (sum(I == 0) <= self.min_leaf_size) | (sum(I == 1) <= self.min_leaf_size):
            return np.nan
        crosstab = pd.crosstab(I, a)
        if crosstab.min().min() < 20:
            _, pval = fisher_exact(crosstab)
        else:
            _, pval, _, _ = chi2_contingency(crosstab)
        return pval

    def _get_uniq_values(self, x):
        n = np.min([self.n_values, len(x)])
        x_uniq = x.quantile(q=list(range(n + 1) / n), interpolation="lower").unique()
        x_uniq = x_uniq[x_uniq != x.max()]
        return x_uniq

    def _enumerate_nodes(self):
        index = 0
        queue = deque([self])
        while queue:
            node = queue.popleft()
            node.node_index_ = index
            index += 1
            if not node._is_leaf:
                queue.extend(node.subtree_)

    def _prune(self, alpha:float, method:str):
        """

        1) implements the multiple hypothesis correction on all node p-values in the tree
        according to the user-defined method
        2) records the corrected p-value and test result for each node
        3) prunes back the tree, keeping only the nodes with:
            i) a descendant with a significant p-value
            ii) a direct parent with a significant p-value

        Args:
            alpha (float): significance level for the hypothesis test
            (before correction). Corresponds to a Type I error in a statistical test
            method (str): method for multiple hypothesis correction from statsmodels.stats.multitest
        """
        # TODO: Make alpha and multiple hypothesis test method be provided in the constructor
        corrected_non_leaf_summary=self._multiple_hypothesis_correction(alpha, method)
        self._mark_nodes_post_multiple_hyp(corrected_non_leaf_summary)
        self._recurse_over_prune()
        self._total_pruned_nodes_=self._count_pruned_nodes()
        self._delete_post_pruning()


    def _multiple_hypothesis_correction(self, alpha:float, method:str):
        # take all non-leaf nodes p-values and conduct a multiple hypothesis correction
        non_leaf_summary=self._generate_non_leaf_nodes_summary().dropna(subset=["p_value"])
        if not non_leaf_summary.empty: # accounts for edge case of single node tree
            test=multipletests(non_leaf_summary['p_value'], alpha=alpha, method=method)
        else:
            test = (None, None)
        # record the corrected p-value and boolean result of the multiple hypothesis test
        non_leaf_summary['corrected_pval'] = test[1]
        non_leaf_summary['multiple_test_result'] = test[0]
        return non_leaf_summary

    def _mark_nodes_post_multiple_hyp(self, summary_df):
        if self.subtree_ is not None:
            self.corrected_pval_ = summary_df.loc[summary_df['node_index'] == self.node_index_, 'corrected_pval']
            self.corrected_pval_ = self.corrected_pval_.iloc[0] # converting single line series into a float
            self.is_split_significant_corrected_ = (
                summary_df['multiple_test_result'].loc[summary_df['node_index'] == self.node_index_]
            )
            self.is_split_significant_corrected_ = self.is_split_significant_corrected_.iloc[0]
            self.subtree_[0]._mark_nodes_post_multiple_hyp(summary_df)
            self.subtree_[1]._mark_nodes_post_multiple_hyp(summary_df)
        else:
            self.corrected_pval_=None
            self.is_split_significant_corrected_=None


    def _mark_all_ancestors_and_two_children_as_keep(self):
        self.keep_ = True
        self._mark_two_children_as_keep()
        is_root = self._parent_ is None
        if not is_root:
            self._parent_._mark_all_ancestors_and_two_children_as_keep()
    @property
    def _is_leaf(self):
        return self.subtree_ is None


    def _recurse_over_prune(self):
        is_root = self._parent_ is None
        if is_root:
            self.keep_ = True
            pass
        if not self._is_leaf:
            if self.is_split_significant_corrected_:
                # the split is significant so we mark two children as keep
                # the ancestors should be marked as keep to eventually save this split
                self._mark_all_ancestors_and_two_children_as_keep()
            self.subtree_[0]._recurse_over_prune()
            self.subtree_[1]._recurse_over_prune()

    def _mark_two_children_as_keep(self):
        self.subtree_[0].keep_ = True
        self.subtree_[1].keep_ = True

    def _delete_post_pruning(self):
        if not self._is_leaf:
            #two children will always have the same keep_ value
            assert self.subtree_[0].keep_ == self.subtree_[1].keep_
            if not self.subtree_[0].keep_:
                self.subtree_=None
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

    def _generate_non_leaf_nodes_summary(self, X=None, a=None, y=None):
        all_node_summary=self._generate_node_summary(X, a, y)
        return all_node_summary.loc[~all_node_summary["is_leaf"]]

    def _mark_nodes_post_multiple_hyp(self, summary_df):
        # if not self._is_leaf:
        #     correct_pval = None
        #     sig_split = None
        # else:
        #     correct_pval = summary_df['corrected_pval'].loc[summary_df['node_index'] == self.node_index_]
        #     sig_split = self.corrected_pval_.iloc[0] # converting single line series into a float
        #     self.subtree_[0]._mark_nodes_post_multiple_hyp(summary_df)
        #     self.subtree_[1]._mark_nodes_post_multiple_hyp(summary_df)
        # self.correct_pval_ = correct_pval
        # self.sig_split_ = sig_split
        if not self._is_leaf:
            self.corrected_pval_ = summary_df['corrected_pval'].loc[summary_df['node_index'] == self.node_index_]
            self.corrected_pval_ = self.corrected_pval_.iloc[0] # converting single line series into a float
            self.is_split_significant_corrected_ = (
                summary_df['multiple_test_result'].loc[summary_df['node_index'] == self.node_index_]
            )
            self.is_split_significant_corrected_ = self.is_split_significant_corrected_.iloc[0]
            self.subtree_[0]._mark_nodes_post_multiple_hyp(summary_df)
            self.subtree_[1]._mark_nodes_post_multiple_hyp(summary_df)
        else:
            self.corrected_pval_=None
            self.is_split_significant_corrected_=None

    #TODO Change this function so its mainly for p-vals, list of attrubutes and returns the vlaue of those atributes
    def _generate_node_summary(self, X=None, a=None, y=None):
        node_summary = pd.DataFrame({
                "pscore": self.propensity_score_,
                "sample_size": self.node_sample_size_,
                "node_index": self.node_index_,
                "p_value": self.pval_,
                "is_leaf": self._is_leaf,
                "positivity_violation": self.positivity_violation_,
                "max_depth": self.max_depth,
                "n_asmd_violating_features": self.n_asmd_violating_features_,
                "max_asmd": self.max_feature_asmd_
            }, index=[0])
        if self._is_leaf:
            return node_summary
        else:
            left_child_summary = self.subtree_[0]._generate_node_summary()
            right_child_summary = self.subtree_[1]._generate_node_summary()
            return pd.concat([node_summary, left_child_summary, right_child_summary], axis=0, ignore_index=True)



    def generate_leaf_summary(self, X= None, a= None, y= None):
         """
         Recursively building a dataframe describing the leaf nodes with pscore, sample_size,
         node_prevalence, node_index, average_outcome_treated/untreated, p_value, is_leaf, positivity_violation,
         max_depth, n_asmd_violating_features, max_asmd

        Args:
            X (pd.DataFrame): The feature data. Assumed to be of the same column
            structure as the training data
            a (pd.Series): The treatment assignment vector
            y (pd.Series): The outcome vector

         Returns: (DataFrame) a frame describing the leaf nodes in the fitted tree, with pscore, sample_size,
         node_prevalence, node_index, average_outcome_treated/untreated, p_value, is_leaf, positivity_violation,
         max_depth, n_asmd_violating_features, max_asmd

         """

         all_node_summary=self._generate_node_summary(X, a, y)
         return all_node_summary.loc[all_node_summary["is_leaf"]]

    def _find_non_positivity_violating_leaves(self, positivity_filtering_method, positivity_filtering_kwargs: dict=None):
        """

        Finding the non positivity violating leaf nodes according to a user-defined criterion,
        marks the corresponding nodes, and outputs a list of non-violating node indices

        Args:
            positivity_filtering_method (function): user-defined function for determining positivity violations
            positivity_filtering_kwargs (dict): parameters for positivity criterion

        Returns:(list) node indices of leaves that do not violate positivity

        """
        non_violating_nodes_idx = positivity_filtering_method(self, **positivity_filtering_kwargs)  # returns the non-violating leaves
        self._mark_positivity_nodes(non_violating_nodes_idx)
        return non_violating_nodes_idx


    def _mark_positivity_nodes(self, non_positivity_violating_nodes):
        if not self._is_leaf:
            self.subtree_[0]._mark_positivity_nodes(non_positivity_violating_nodes)
            self.subtree_[1]._mark_positivity_nodes(non_positivity_violating_nodes)
        else:
            self.positivity_violation_ = self.node_index_ not in non_positivity_violating_nodes


    def _set_positivity_violation_nodes(self, violating_nodes:dict={}):
        '''
        Creating a dictionay with all positivity violating nodes in the tree
        '''

        if not self._is_leaf:
            self.subtree_[0]._set_positivity_violation_nodes(violating_nodes)
            self.subtree_[1]._set_positivity_violation_nodes(violating_nodes)
        else:
           violating_nodes[self.node_index_] = self.positivity_violation_

        self.is_violating_ = violating_nodes


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
            node_assignment=pd.concat([left_child_assignment, right_child_assignment], axis="index")
            node_assignment=node_assignment.loc[X.index] #keeping order of X input
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
            y (pd.Series): The outcome vector
            split_condition (str): The string representing the first condition. Default to 'All'
            df_explanation (pd.DataFrame): Mostly used for recursion. The inital data-frame,
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
        asmds = self._calculate_asmds(X, a)
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