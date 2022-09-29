"""
Metrics to assess performance of weight models.
Function named as ``*_error`` return a scalar value to minimize:
the lower, the better.

Metrics' interface doesn't strictly follow skleran's metrics interface.
"""
from causallib.evaluation.weight_evaluator import calculate_covariate_balance


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
