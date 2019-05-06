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
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer as skImputer

from ..utils.stat_utils import which_columns_are_binary


class StandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize continuous features by removing the mean and scaling to unit variance while allowing nans.

        X = (X - X.mean()) / X.std()
    """

    def __init__(self, with_mean=True, with_std=True, ignore_nans=True):
        """

        Args:
            with_mean (bool): Whether to center the data before scaling.
            with_std (bool): Whether to scale the data to unit variance.
            ignore_nans (bool): Whether to ignore NaNs during calculation.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.ignore_nans = ignore_nans

    def fit(self, X, y=None):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            X (pd.DataFrame): The data used to compute the mean and standard deviation used for later scaling along the
                            features axis (axis=0).
            y: Passthrough for ``Pipeline`` compatibility.

        Returns:
            StandardScaler: A fitted standard-scaler
        """
        continuous_features = self._get_relevant_features(X)
        self._feature_mask_ = continuous_features

        if self.with_mean:
            means = X.loc[:, self._feature_mask_].mean(skipna=self.ignore_nans)
        else:
            means = pd.Series(0, index=continuous_features)
        self.mean_ = means

        if self.with_std:
            scales = X.loc[:, self._feature_mask_].std(skipna=self.ignore_nans)
        else:
            scales = pd.Series(1, index=continuous_features)
        self.scale_ = scales

        return self

    def transform(self, X, y='deprecated'):
        """
        Perform standardization by centering and scaling

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] The data used to compute the mean and standard
                            deviation used for later scaling along the features axis (axis=0).
            y: Passthrough for ``Pipeline`` compatibility.X:

        Returns:
            pd.DataFrame: Scaled dataset.
        """
        # Taken from the sklearn implementation. Will probably need adjustment when a new scikit-learn version is out:
        if not isinstance(y, str) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        X = X.copy()  # type: pd.DataFrame
        if self.with_mean:
            X.loc[:, self._feature_mask_] -= self.mean_
        if self.with_std:
            X.loc[:, self._feature_mask_] /= self.scale_
        return X

    def inverse_transform(self, X):
        """
        Scale back the data to the original representation

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] The data used to compute the mean and standard
                              deviation used for later scaling along the features axis (axis=0).

        Returns:
            pd.DataFrame: Un-scaled dataset.
        """
        X = X.copy()  # type: pd.DataFrame
        if self.with_std:
            X.loc[:, self._feature_mask_] *= self.scale_
        if self.with_mean:
            X.loc[:, self._feature_mask_] += self.mean_
        return X

    @staticmethod
    def _get_relevant_features(X):
        """
        Returns a binary mask specifying the continuous features to operate on.

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] The data used to compute the mean and standard
                              deviation used for later scaling along the features axis (axis=0).

        Returns:
            pd.Index: a pd.Index with name of columns specifying which features to apply the transformation on.
        """
        # FIXME utilize sklearn.utils.multiclass.type_of_target()
        continuous_cols = X.columns[~which_columns_are_binary(X)]
        return continuous_cols


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scales features to 0-1, allowing for NaNs.

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    """

    def __init__(self, only_binary_features=True, ignore_nans=True):
        """

        Args:
            only_binary_features (bool): Whether to apply only on binary features or across all.
            ignore_nans (bool): Whether to ignore NaNs during calculation.
        """
        self.only_binary_features = only_binary_features
        self.ignore_nans = ignore_nans

    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for later scaling.

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] The data used to compute the mean and standard
                              deviation used for later scaling along the features axis (axis=0).
            y: Passthrough for ``Pipeline`` compatibility.

        Returns:
            MinMaxScaler: a fitted MinMaxScaler
        """
        feature_mask = self._get_relevant_features(X)
        self._feature_mask_ = feature_mask

        self.min_ = X.min(skipna=self.ignore_nans)[feature_mask]
        self.max_ = X.max(skipna=self.ignore_nans)[feature_mask]
        self.scale_ = self.max_ - self.min_

        # if feature_mask.size != X.shape[1]:
        #     self.scale_[~feature_mask] = 1
        #     self.min_[~feature_mask] = 0
        #     self.max_[~feature_mask] = 1

        return self

    def inverse_transform(self, X):
        """
        Scaling chosen features of X to the range of 0 - 1.

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] Input data that will be transformed.

        Returns:
            pd.DataFrame: array-like, shape [n_samples, n_features]. Transformed data.
        """
        # No warning for y, since there's no y variable.
        # This correpsonds to function signature in scikit-learn's code base
        X = X.copy()  # type: pd.DataFrame
        X.loc[:, self._feature_mask_] *= self.scale_
        X.loc[:, self._feature_mask_] += self.min_
        return X

    def transform(self, X):
        """
        Undo the scaling of X according to feature_range.

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] Input data that will be transformed.

        Returns:
            pd.DataFrame: array-like, shape [n_samples, n_features]. Transformed data.
        """

        X = X.copy()  # type: pd.DataFrame
        X.loc[:, self._feature_mask_] -= self.min_
        X.loc[:, self._feature_mask_] /= self.scale_
        return X

    def _get_relevant_features(self, X):
        """
        Returns a binary mask specifying the features to operate on (either all features or binary features if
        self.only_binary_features is True.

        Args:
            X (pd.DataFrame): array-like, shape [n_samples, n_features] The data used to compute the mean and standard
                            deviation used for later scaling along the features axis (axis=0).

        Returns:
            pd.Index: a binary mask specifying which features to apply the transformation on.
        """
        if self.only_binary_features:
            feature_mask = which_columns_are_binary(X)
        else:
            feature_mask = np.ones(X.shape[1], dtype=bool)
        return feature_mask


class Imputer(skImputer):
    def transform(self, X):
        X_transformed = super().transform(X.values)
        X_transformed = pd.DataFrame(X_transformed, index=X.index, columns=X.columns)
        return X_transformed
