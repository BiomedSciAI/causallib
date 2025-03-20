"""
Metrics to asses violations of positivity
"""

import numpy as np
import pandas as pd


def _check_mean_group(mean_group):
    mean_group = int(mean_group)
    if mean_group not in {0, 1}:
        raise ValueError('mean_group needs to be equal to 0 or 1')
    return mean_group


def cross_covariance(X, a, mean_group=0):
    """
    Computing the covariance, where the mean is taken from the counter group.
    suitable only for binary treatment, i.e a \in {0, 1}
    Args:
        X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
        a (pd.Series): Treatment assignment of size (num_subjects,).
        mean_group (0 or 1): the treatment group

    Returns:
        np.array: cross-covariance square matrix
    """
    mean_group = _check_mean_group(mean_group)
    avg = X.loc[a == mean_group, :].mean(axis=0)
    X_counter_group = X.loc[a == int(not mean_group)]
    X_counter_group = (X_counter_group - avg).astype(float)
    e_xx = np.dot(X_counter_group.T, X_counter_group)

    cross_cov = e_xx / (X_counter_group.shape[0] - 1)  # unbiased cov estimate
    return cross_cov


def cross_covariance_score(X, a, normalize=False, sum_scores=True,
                           func=np.max, off_diagonal_only=False):
    """
    Reduce the cross-covariance matrix into a single score
    Args:
        X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
        a (pd.Series): Treatment assignment of size (num_subjects,).
        func (callable): function to apply on the cross-covariance matrix
        normalize (bool): If True, standardize features by removing the mean
                          and scaling to unit variances.
        off_diagonal_only (bool): The diagonal of the cross-covariance matrix
                                  represents variances, and by definition
                                  max{var(x),var(y)} >= cov(x,y).
                                  If True, set on zero the diagonal of the
                                  cross-covariance matrix, focusing on maximum
                                  of cov elements.
                                  otherwise, focusing on maximum of
                                  var elements.

        sum_scores (bool): If True, sum the scores of the different groups,
                           otherwise returns a list of scores.
    Returns:
        float|list: non-negative scores
    """
    treatment_values = sorted(pd.unique(a))
    if normalize:
        X = pd.DataFrame((X - X.mean(axis=0)) / X.std(axis=0), index=X.index)

    scores = list()
    for treatment in treatment_values:
        cross_cov = np.abs(cross_covariance(X, a, mean_group=treatment))
        if off_diagonal_only:
            cross_cov -= np.diag(np.diag(cross_cov))
        scores.append(func(cross_cov))
    return np.sum(scores) if sum_scores else scores
