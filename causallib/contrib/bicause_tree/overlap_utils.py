import numpy as np
import pandas as pd

from causallib.estimation.base_estimator import IndividualOutcomeEstimator


class OverlapViolationEstimator(IndividualOutcomeEstimator):
    """
    A causallib-compatible model returning NaNs with appropriate format and indexing.
    Meant to be used when overlap violations are detected.
    """
    def __init__(self, value=np.nan, *args, **kwargs) -> None:
        super().__init__(None, *args, **kwargs)
        self.value = value

    def fit(self, X, a, y, sample_weight=None):
        return self

    def estimate_population_outcome(self, X, a, y=None, **kwargs):
        treatment_values = self._get_treatment_index(a)
        return pd.Series(self.value, index=treatment_values)

    def estimate_individual_outcome(self, X, a, **kwargs):
        treatment_values = self._get_treatment_index(a)
        return pd.DataFrame(self.value, index=X.index, columns=treatment_values)

    @staticmethod
    def _get_treatment_index(a):
        return pd.Index(sorted(a.unique()), name=a.name)


def prevalence_symmetric(tree, alpha=0.1):

    """
    Computes the symmetric lower/upper cutoff based on prevalence and a fixed cutoff alpha
    with p the original propensity score and mu the prevalence in the cohort
    Returns the list of non violating nodes for these cutoffs

    Args:
        tree (BalancingTree): the splitting tree
        alpha (float): the fixed cutoff to be transformed

    Checks for the default stopping criterion, which is:
    is the ASMD threshold reached
    OR is the minimum treatment group size reached
    OR is the minimum leaf size reached
    OR is the maximum depth of the tree reached
    Returns: (list) The node indices of non-violating leaves
    """
    leaf_summary = tree.generate_leaf_summary()
    ps_singles_list = leaf_summary['treatment_prevalence'].values
    repetition_numbers = leaf_summary['sample_size'].values.tolist()
    ps_list = ps_singles_list.repeat(repetition_numbers)
    ps_vect = pd.DataFrame(np.column_stack((1 - ps_list, ps_list)))
    mu = tree.treatment_prevalence_  # Overall prevalence of treatment
    cutoffs = prevalence_symmetric_cutoff(ps_vect, mu, alpha)
    lower_cutoff, upper_cutoff = cutoffs[0], cutoffs[1]
    non_violating_nodes = leaf_summary.loc[
        leaf_summary['treatment_prevalence'].between(lower_cutoff, upper_cutoff),
        'node_index'
    ].tolist()
    return non_violating_nodes


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
    ps_singles_list = leaf_summary['treatment_prevalence'].values
    repetition_numbers = leaf_summary['sample_size'].values.tolist()
    ps_list = ps_singles_list.repeat(repetition_numbers)
    ps_vect = pd.DataFrame(np.column_stack((1 - ps_list, ps_list)))
    lower_cutoff, upper_cutoff = crump_cutoff(ps_vect, segments)
    if lower_cutoff is not None:
        non_violating_nodes = leaf_summary.loc[
            leaf_summary['treatment_prevalence'].between(lower_cutoff, upper_cutoff),
            'node_index'
        ].tolist()
        return non_violating_nodes
    return []  # lower_cutoff=None means all nodes have positivity violations


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
    return None, None  # No overlap
