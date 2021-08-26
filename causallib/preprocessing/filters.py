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
import abc

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.feature_selection as feature_selection
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.stat_utils import isBinary, areColumnsBinary, computeCorrPvals


def track_selected_features(pipeline_stages, num_features):
    """

    Args:
        pipeline_stages (list [tuple[str, TransformerMixin]]): list of steps. each step is a tuple of Name and
                                                               Transformer Object.
        num_features (int):

    Returns:
        np.ndarray:
    """
    selected_features = np.arange(num_features)
    for p_name, p in pipeline_stages:
        if not isinstance(p, BaseFeatureSelector):
            continue
        p_features = p.selected_features
        selected_features = selected_features[p_features]
    return selected_features


class BaseFeatureSelector(BaseEstimator, TransformerMixin):
    """

    """

    def __init__(self):
        """

        """
        self._selected_features = None

    @property
    def selected_features(self):
        return self._selected_features

    @selected_features.setter
    def selected_features(self, features):
        if np.sum(self.selected_features) == 0:
            raise AssertionError("All features were removed by feature")
        self._selected_features = features

    def transform(self, X):
        """

        Args:
            X (pd.DataFrame):

        Returns:
            pd.DataFrame:
        """
        return X.loc[:, self.selected_features]

    @abc.abstractmethod
    def fit(self, X, y=None):
        """

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] The data used for filtering.
            y: Passthrough for ``Pipeline`` compatibility.

        Returns:
            BaseFeatureSelector
        """
        raise NotImplementedError


class ConstantFilter(BaseFeatureSelector):
    """Removes features that are almost constant"""

    def __init__(self, threshold=0.95):
        """

        Args:
            threshold (float):
        """
        super(ConstantFilter, self).__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        is_const = pd.Series(0, index=X.columns, dtype=np.dtype(bool))
        for col in X.columns:
            # NaNs are not counted using unique (since np.nan != np.nan). Fill them with a unique value:
            cur_col = X.loc[:, col]
            cur_col.loc[~np.isfinite(cur_col)] = cur_col.max() + 1
            # Get values' frequency:
            freqs = cur_col.value_counts(normalize=True)
            is_const[col] = np.any(freqs > self.threshold)

        self.selected_features = ~is_const
        return self


class SparseFilter(BaseFeatureSelector):
    """Removes features with many missing values"""

    def __init__(self, threshold=0.2):
        """

        Args:
            threshold (float):
        """
        super(SparseFilter, self).__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        nan_freqs = np.mean(np.isnan(X), axis=0)
        is_sparse = nan_freqs > self.threshold
        self.selected_features = ~is_sparse
        return self


class HrlVarFilter(BaseFeatureSelector):
    """Removes features with a small variance, while allowing for missing values"""

    def __init__(self, threshold=0.0):
        """

        Args:
            threshold (float):
        """
        super(HrlVarFilter, self).__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        vars = np.nanvar(X, axis=0)
        self.selected_features = vars > self.threshold
        return self


class UnivariateAssociationFilter(BaseFeatureSelector):
    """Removes features according to univariate association"""

    def __init__(self, is_linear=True, threshold=0.2):
        """

        Args:
            is_linear (bool):
            threshold (float):
        """
        super(BaseFeatureSelector, self).__init__()
        self.is_linear = is_linear
        self.threshold = threshold

    def fit(self, X, y=None):
        p_vals = self.compute_pvals(X, y)

        self.selected_features = p_vals < self.threshold
        return self

    def compute_pvals(self, X, y):
        # TODO: export to stats_utils?
        is_y_binary = (len(np.unique(y)) == 2)
        # is_binary_feature = np.sum(((X != np.nanmin(X, axis=0)[np.newaxis, :]) &
        #                             (X != np.nanmax(X, axis=0)[np.newaxis, :])), axis=0) == 0
        is_binary_feature = areColumnsBinary(X)
        p_vals = np.zeros(X.shape[1])
        if is_y_binary:
            # Process non-binary columns:
            for i in np.where(~is_binary_feature)[0]:
                x0 = X.loc[y == 0, i]
                x1 = X.loc[y == 1, i]
                if self.is_linear:
                    _, p_vals[i] = stats.ttest_ind(x0, x1)
                else:
                    _, p_vals[i] = stats.ks_2samp(x0, x1)

            # Process binary features:
            _, p_vals[is_binary_feature] = feature_selection.chi2(X.loc[:, is_binary_feature], y)

        else:
            # Process non-binary features:
            _, p_vals[~is_binary_feature] = feature_selection.f_regression(X.loc[:, ~is_binary_feature], y)

            # Process binary features:
            y_mat = np.row_stack(y)
            for i in np.where(is_binary_feature)[0]:
                _, p_vals[i] = feature_selection.f_regression(y_mat, X.loc[:, i])
        return p_vals


class StatisticalFilter(BaseFeatureSelector):
    """Removes features according to univariate association"""

    # TODO: isn't this the same as the above?

    def __init__(self, threshold=0.2, isLinear=True):
        """

        Args:
            isLinear (bool):
            threshold (float):
        """
        super(StatisticalFilter, self).__init__()
        self.isLinear = isLinear
        self.threshold = threshold

    def fit(self, X, y=None):
        is_y_binary = isBinary(y)
        is_binary_feature = areColumnsBinary(X)
        p_vals = computeCorrPvals(X, y, is_binary_feature, is_y_binary, self.isLinear)

        self.selected_features = p_vals < self.threshold
        return self


class CorrelationFilter(BaseFeatureSelector):
    """Removes features that are strongly correlated to other features"""

    def __init__(self, threshold=0.9):
        """

        Args:
            threshold (float):
        """
        super(CorrelationFilter, self).__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        if np.any(np.isnan(X)):
            raise AssertionError("Correlation filter cannot handle NaN values")

        stat_filter = UnivariateAssociationFilter(threshold=1.0)  # Do not remove features prior the following process
        stat_filter.fit(X, y)
        p_vals = stat_filter.compute_pvals(X, y)
        # p_vals = computeCorrPvals(X, y)

        features_sorted = np.argsort(p_vals)
        is_removed = np.zeros(X.shape[1], dtype=np.bool)
        corr_mat = np.corrcoef(X.T)
        for i in features_sorted:  # iterate by p-values to keep the most significant among the highly correlated.
            if is_removed[i]:
                continue
            is_above_threshold = np.abs(corr_mat[:, i]) > self.threshold
            is_above_threshold[i] = False  # Ignore the correlation of i with itself.
            is_removed[is_above_threshold] = True  # Mark all features with big correlation to removal.

        self.selected_features = ~is_removed
        return self
