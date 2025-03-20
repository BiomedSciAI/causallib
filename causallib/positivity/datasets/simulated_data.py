# todo:
#   adding more datasets as positivity benchmarks
#   and maybe order it nicely in a Class of Simulated_data

import numpy as np
import pandas as pd


def _rotate_data(X, angle_rotation):
    """ rotate only the first 2d """
    angle = np.deg2rad(angle_rotation)
    rot_2d = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    rot_mat = np.eye(X.shape[1])
    rot_mat[:2, :2] = rot_2d
    X_rotated = np.dot(X, rot_mat)
    return pd.DataFrame(X_rotated)


def _probability_of_being_treated(X, angle_slice):
    """ based on the amount of slicing in the data """
    angle = np.deg2rad(angle_slice)
    slope = np.sin(angle)/np.cos(angle)
    slicing_cond = (0 < X.iloc[:, 1]) & ((X.iloc[:, 1] / slope) < X.iloc[:, 0])
    p = 0.5 + 0.5 * slicing_cond
    return p


def pizza(n_dim=2, n_samples=1000,
          angle_rotation=45, angle_slice=90, seed=0):
    """
    Rotate the data and create slice with strict non-overlapping,
    i.e. the propensity score given this sub space is equal to 1.

    If n_dim>2, uniform distribution over the covariate space, where
    the non-overlapping area is the same as n_dim=2, meaning that the
    other covariates are completely overlapping.
    Args:
        n_dim (int): number of dimensions, have to be equal or bigger than 2.
        n_samples (int): number of samples
        angle_rotation (float): the angle of rotation in degrees
        angle_slice (float): the angle of sliced out area, the bigger the angle
                             the wider the non-overlapping area,
                             ranging from [0, 180]
        seed (None | int):

    Returns:

    """
    np.random.seed(seed)
    X = np.random.uniform(-np.ones(n_dim),
                          np.ones(n_dim),
                          size=(n_samples, n_dim)) / np.sqrt(n_dim)
    X_rotated = _rotate_data(X, angle_rotation)
    p = _probability_of_being_treated(X_rotated, angle_slice)
    a = pd.Series(np.random.binomial(1, p), name='a')
    return X_rotated, a
