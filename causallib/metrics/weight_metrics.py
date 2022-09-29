"""
Metrics to assess performance of weight models.
Function named as ``*_error`` return a scalar value to minimize:
the lower, the better.

Metrics' interface doesn't strictly follow skleran's metrics interface.
"""

import pandas as pd
import numpy as np

from ..utils.stat_utils import (
    calc_weighted_ks2samp,
    calc_weighted_standardized_mean_differences,
)

DISTRIBUTION_DISTANCE_METRICS = {
    "smd": lambda x, y, wx, wy: calc_weighted_standardized_mean_differences(
        x, y, wx, wy
    ),
    "abs_smd": lambda x, y, wx, wy: abs(
        calc_weighted_standardized_mean_differences(x, y, wx, wy)
    ),
    "ks": lambda x, y, wx, wy: calc_weighted_ks2samp(x, y, wx, wy),
}


def calculate_covariate_balance(X, a, w, metric="abs_smd"):
    """Calculate covariate balance table ("table 1")

    Args:
        X (pd.DataFrame): Covariates.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with
            the signature weighted_distance(x, y, wx, wy) calculating distance between the weighted
            sample x and weighted sample y (weights by wx and wy respectively).

    Returns:
        pd.DataFrame: index are covariate names (columns) from X, and columns are
            "weighted" / "unweighted" results of applying `metric` on each covariate
            to compare the two groups.
    """
    treatment_values = np.sort(np.unique(a))
    results = {}
    for treatment_value in treatment_values:
        distribution_distance_of_cur_treatment = pd.DataFrame(
            index=X.columns, columns=["weighted", "unweighted"], dtype=float
        )
        for col_name, col_data in X.items():
            weighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, w, a, treatment_value, metric
            )
            unweighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, pd.Series(1, index=w.index), a, treatment_value, metric
            )
            distribution_distance_of_cur_treatment.loc[
                col_name, ["weighted", "unweighted"]
            ] = [weighted_distance, unweighted_distance]
        results[treatment_value] = distribution_distance_of_cur_treatment
    results = pd.concat(
        results, axis="columns", names=[a.name or "a", metric]
    )  # type: pd.DataFrame
    results.index.name = "covariate"
    if len(treatment_values) == 2:
        # If there are only two treatments, the results for both are identical.
        # Therefore, we can get rid of one of them.
        # We keep the results for the higher valued treatment group (assumed treated, typically 1):
        results = results.xs(treatment_values.max(), axis="columns", level=0)
    # TODO: is there a neat expansion for multi-treatment case?
    #  maybe not current_treatment vs. the rest.
    return results


def calculate_distribution_distance_for_single_feature(
    x, w, a, group_level, metric="abs_smd"
):
    """

    Args:
        x (pd.Series): A single feature to check balancing.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        group_level: Value from `a` in order to divide the sample into one vs. rest.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with
            the signature weighted_distance(x, y, wx, wy) calculating distance between the weighted
            sample x and weighted sample y (weights by wx and wy respectively).

    Returns:
        float: weighted distance between the samples assigned to `group_level`
            and the rest of the samples.
    """
    if not callable(metric):
        metric = DISTRIBUTION_DISTANCE_METRICS[metric]
    cur_treated_mask = a == group_level
    x_treated = x.loc[cur_treated_mask]
    w_treated = w.loc[cur_treated_mask]
    x_untreated = x.loc[~cur_treated_mask]
    w_untreated = w.loc[~cur_treated_mask]
    distribution_distance = metric(x_treated, x_untreated, w_treated, w_untreated)
    return distribution_distance



def covariate_balancing_error(X, a, sample_weight, agg=max):
    """Computes the weighted (i.e. balanced) absolute standardized mean difference
    of every covariate in X.

    Args:
        X (pd.DataFrame): Covariate matrix.
        a (pd.Series): Treatment assignment vector.
        sample_weight (pd.Series): Weights balancing between the treatment groups.
        agg (callable): A function to aggregate a vector of absolute differences
                        between the curves' points. Default is max.

    Returns:
        score (float):
    """
    asmds = calculate_covariate_balance(X, a, sample_weight, metric="abs_smd")
    weighted_asmds = asmds["weighted"]
    score = agg(weighted_asmds)
    return score
