import pandas as pd
import numpy as np


def make_1d_overlap_data(treatment_bounds=(0, 75),
                         control_bounds=(25, 100)):
    """Generate 1d overlap data with integer covariates

    Args:
        treatment_bounds (tuple, optional): Bounds for covariates in treatment
            group. Defaults to (0, 75).
        control_bounds (tuple, optional): Bounds for covariates in control
            group. Defaults to (25, 100).

    Returns:
        X (pd.DataFrame), a (pd.Series): covariate and treatment assignment
    """

    X_treatment = np.arange(*treatment_bounds)
    a_treatment = np.ones_like(X_treatment)

    X_control = np.arange(*control_bounds)
    a_control = np.zeros_like(X_control)

    X = pd.DataFrame(data=np.hstack((X_treatment, X_control)), columns=["X1"])
    a = pd.Series(data=np.hstack((a_treatment, a_control)), name="treatment")

    return X, a


def make_1d_normal_distribution_overlap_data(treatment_params=(0, 1),
                                             control_params=(0, 1),
                                             probability_treated=0.5,
                                             n_samples=400,
                                             random_seed=1234):
    """
    Args:
        treatment_params (tuple): loc and scale parameter of normal distribution
        control_params (tuple): loc and scale parameter of normal distribution
        probability_treated (float):
        n_samples (int):
        random_seed (int):
    Returns:
        X (pd.DataFrame), a (pd.Series):
    """
    n_treated = int(round(n_samples * probability_treated))
    n_control = n_samples - n_treated

    a_control = np.zeros(n_control)
    np.random.seed(random_seed)
    X_control = np.random.normal(
        loc=control_params[0], scale=control_params[1], size=n_control)

    a_treatment = np.ones(n_treated)
    np.random.seed(random_seed + 1)
    X_treatment = np.random.normal(
        loc=treatment_params[0], scale=treatment_params[1], size=n_treated)

    X = pd.DataFrame(data=np.hstack((X_treatment, X_control)), columns=["X"])
    a = pd.Series(data=np.hstack((a_treatment, a_control)), name="treatment")
    return X, a


def make_multivariate_normal_data(
        treatment_params=([0, 0], [[2, 1], [1, 2]]),
        control_params=([2, 2], [[1, 0], [0, 1]]),
        probability_treated=0.5,
        n_samples=400,
        random_seed=1234):
    """
    Args:
        treatment_params (tuple): loc and scale parameter of multivariate normal distribution
        control_params (tuple): loc and scale parameter of multivariate normal distribution
        probability_treated (float): the probability to be in the treatment group
        n_samples (int): number of samples for the full data set
        random_seed (int): each group receive different seeding
    Returns:
        X (pd.DataFrame), a (pd.Series):
    """
    n_treated = int(round(n_samples * probability_treated))
    n_control = n_samples - n_treated

    a_control = np.zeros(n_control)
    np.random.seed(random_seed)
    X_control = np.random.multivariate_normal(control_params[0],
                                              control_params[1],
                                              n_control)
    a_treatment = np.ones(n_treated)
    np.random.seed(random_seed + 1)
    X_treatment = np.random.multivariate_normal(treatment_params[0],
                                                treatment_params[1],
                                                n_treated)

    X = pd.DataFrame(data=np.vstack((X_treatment, X_control)))
    a = pd.Series(data=np.hstack((a_treatment, a_control)), name="treatment")
    return X, a


def make_random_y_like(a, random_seed=1234):
    np.random.seed(random_seed)
    y = pd.Series(data=np.random.random_sample(a.shape), name="outcome")
    return y
