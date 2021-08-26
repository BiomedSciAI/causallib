"""
(C) Copyright 2019 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from builtins import range

import numpy as np
import scipy.stats as stats
import sklearn.feature_selection as feature_selection
from pandas import DataFrame as pdDataFrame, Series as pdSeries  # For type hinting purposes only
from pandas.core.indexes.base import InvalidIndexError


def is_vector_binary(vec):
    return np.unique(vec).size <= 2


def which_columns_are_binary(X):
    """

    Args:
        X (pdDataFrame):

    Returns:

    """
    return X.apply(lambda x: x.nunique()).le(2)


def isBinary(x):
    """
    Asses whether a vector is binary.
    Args:
        x (pdSeries | np.ndarray):

    Returns:
        bool: True iff x is binary.
    """
    if hasattr(x, "values"):
        x = x.values
    x = x.reshape((-1, 1))
    return areColumnsBinary(x)[0]


def areColumnsBinary(X):
    """
    Assess whether all matrix columns are binary.
    Args:
        X (np.ndarray | pdDataFrame): Covariate matrix.

    Returns:
        np.ndarray: A boolean vector the length of number of features (columns). An entry is True iff the corresponding 
                    column is binary.
    """
    mins = np.nanmin(X, axis=0)  # min value for every column
    maxs = np.nanmax(X, axis=0)  # max value for every column
    X0 = X == mins[np.newaxis, :]  # TODO: newaxis is redundant?
    X1 = X == maxs[np.newaxis, :]  # TODO: newaxis is redundant?
    is_binary_features = np.all(X0 | X1, axis=0)  # binary vector stating which column is binary

    return is_binary_features


def computeCorrPvals(X, y, is_X_binary, is_y_binary, isLinear=True):
    """

    Args:
        X (pd.DataFrame): The covariate matrix
        y (pdSeries): The response
        is_X_binary (np.ndarray): Indication which columns are binary
        is_y_binary (bool): Indication whether the response vector is binary or not.
        isLinear (bool): Whether to perform a linear (slope) test (t-test) on the non-binary features or to perform
                         a two-sample Kolmogorov-Smirnov test

    Returns:
        np.array: A vector of p-values, one for every feature.
    """
    p_vals = np.empty(X.shape[1]) * np.nan
    y = np.squeeze(y)  # TODO: is it necessary?
    if is_y_binary:
        if any(
                is_X_binary):  # TODO: there's a better way, by first assigning Z=X[:, is_X_binary] and cheking whether Z is empty (or apply everything on an empty matrix)
            X2 = X.loc[:, is_X_binary]
            # xx = X2[:,[7]]
            #             t0 = timeit.default_timer()
            #             _, p1 = feature_selection.chi2(X2,y)
            #             t1 = timeit.default_timer()
            #             print 'after chi2_test_fs', (t1-t0)

            # t0 = timeit.default_timer()
            p2 = chi2_test(X2, y)
            # t1 = timeit.default_timer()
            # print 'after chi2_test', (t1-t0)

            p_vals[is_X_binary] = p2

        if (any(~is_X_binary)):
            # _, p_vals[is_X_binary] = feature_selection.f_classif(X[:, is_X_binary], y)
            for i in np.where(~is_X_binary)[0]:
                x = X.iloc[:, i]
                x0 = x[(~np.isnan(x)) & (y == 0)]  # TODO: y doesn;t have to be 0s or 1s
                x1 = x[(~np.isnan(x)) & (y == 1)]  # TODO: y doesn;t have to be 0s or 1s
                if isLinear:
                    _, p_vals[i] = stats.ttest_ind(x0, x1)
                else:
                    _, p_vals[i] = stats.ks_2samp(x0, x1)
    else:
        # if (any(~is_X_binary)):
        #    _, p_vals[~is_X_binary] = feature_selection.f_regression(X[:, ~is_X_binary], y)
        # y_mat = np.row_stack(y)
        # for i in np.where(is_X_binary)[0]:
        #    _, p_vals[i] = feature_selection.f_regression(y_mat, X[:, i])
        _, p_vals = feature_selection.f_regression(X, y)

    return p_vals


# def mycrosstab(x,y):
#     catx = np.unique(x)
#     caty = np.unique(y)
#     tab = [[sum((x==cx) & (y==cy)) for cx in catx] for cy in caty]
#     
#     #res1 = stats.chi2_contingency(tab, False)
#     res2 = stats.chi2_contingency(tab, True)
#     #return res1[1], res2[1]
#     return res2[1]

# def chi2_test_fs(X,y):
#     
#     m = X.shape[1]
#     pvals = np.empty(m)*np.NaN
#     for i in range(m):
#         Xi = X[:,[i]]
#         Xi01 = np.hstack((1-Xi, Xi))
#         _, pvals[i]= feature_selection.chi2(Xi01, y)
#     
#     return pvals


def chi2_test(X, y):
    """

    Args:
        X (np.ndarray): Binary feature matrix
        y (np.ndarray): Binary response vector

    Returns:
        np.ndarray: A vector of p-values, one for every feature.
    """
    X0 = 1 - X
    if hasattr(y, "values"):
        y = y.values
    Y = y.reshape((-1, 1))
    Y = np.append(1 - Y, Y, axis=1)
    Tbl1 = np.dot(Y.T, X)
    Tbl0 = np.dot(Y.T, X0)

    m = X.shape[1]
    pvals = np.empty(m) * np.NaN
    for i in range(m):
        if np.all([Tbl1[:, i] == 0]) or np.all([Tbl0[:, i] == 0]):
            pvals[i] = 1
        else:
            r = stats.chi2_contingency([Tbl0[:, i], Tbl1[:, i]], True)
            pvals[i] = r[1]
    return pvals


def calc_weighted_standardized_mean_differences(x, y, wx, wy, weighted_var=False):
    r"""
    Standardized mean difference: frac{\mu_1 - \mu_2 }{\sqrt{\sigma_1^2 + \sigma_2^2}}

    References:
        [1]https://cran.r-project.org/web/packages/cobalt/vignettes/cobalt_A0_basic_use.html#details-on-calculations
        [2]https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference#Concept

    Note on variance:
    - The variance is calculated on unadjusted to avoid paradoxical situation when adjustment decreases both the
      mean difference and the spread of the sample, yielding a larger smd than that prior to adjustment,
      even though the adjusted groups are now more similar [1].
    - The denominator is as depicted in the "statistical estimation" section:
      https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference#Statistical_estimation,
      namely, disregarding the covariance term [2], and is unweighted as suggested above in [1].
    """
    numerator = np.average(x, weights=wx) - np.average(y, weights=wy)
    if weighted_var:
        var = lambda vec, weights: np.average((vec - np.average(vec, weights=weights)) ** 2, weights=weights)
        denominator = np.sqrt(var(x, wx) + var(y, wy))
    else:
        denominator = np.sqrt(np.nanvar(x) + np.nanvar(y))
    if np.isfinite(denominator) and np.isfinite(numerator) and denominator != 0:
        bias = numerator / denominator
    else:
        bias = np.nan
    return bias


def calc_weighted_ks2samp(x, y, wx, wy):
    """
    Weighted Kolmogorov-Smirnov

    References:
        [1] https://stackoverflow.com/a/40059727
    """
    x_ix = np.argsort(x)
    y_ix = np.argsort(y)
    x, wx = x[x_ix], wx[x_ix]
    y, wy = y[y_ix], wy[y_ix]
    data = np.concatenate((x, y))
    wx_cum = np.hstack([0, wx.cumsum() / wx.sum()])
    wy_cum = np.hstack([0, wy.cumsum() / wy.sum()])
    # Align the "steps" between the two distribution so the differences will be well defined:
    x_align = wx_cum[[np.searchsorted(x, data, side="right")]]
    y_align = wy_cum[[np.searchsorted(y, data, side="right")]]
    stat = np.max(np.abs(x_align - y_align))
    # stat = ks_2samp(wx * x, wy * y)
    return stat


def robust_lookup(df, indexer):
    """
    Robust way to apply pandas lookup when indices are not unique

    Args:
        df (pdDataFrame):
        indexer (pdSeries): A Series whose index is either same or a subset of `df.index`
                            and whose values are values from `df.columns`.
                            If `a.index` contains values not in `df.index` 
                            they will have NaN values.

    Returns:
        pdSeries: a vector where (logically) `extracted[i] = df.loc[indexer.index[i], indexer[i]]`. 
            In most cases, when `indexer.index == df.index` this translates to 
            `extracted[i] = df.loc[i, indexer[i]]`
    """
    # Convert the index into 
    idx, col = indexer.factorize()  # convert text labels into integers
    extracted = df.reindex(col, axis=1).reindex(indexer.index, axis=0)  # make sure the columns exist and the indeces are the same
    extracted = extracted.to_numpy()[range(len(idx)), idx]  # numpy accesses by location, not by named index
    extracted = pdSeries(extracted, index=indexer.index)
    return extracted
