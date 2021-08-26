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
from sklearn.impute import SimpleImputer as skImputer
from ..utils.stat_utils import which_columns_are_binary
from causallib.estimation import Matching


# TODO: Entire module might be redundant, now that scikit-learn supports missing values
#       in its preprocessing: https://scikit-learn.org/stable/whats_new/v0.20.html#highlights
#       The only support now needed is:
#       1) Transforming from numpy-array to pandas DataFrame in a pipeline, before specifying a causal model.
#       2) Possible generic support for causallib's additional `a` parameter, along with `X` and `y`.


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
        X_transformed = pd.DataFrame(
            X_transformed, index=X.index, columns=X.columns)
        return X_transformed


class PropensityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, learner, include_covariates=False):
        """Transform covariates by adding/replacing with the propensity score.
    
        Args:
            learner (sklearn.estimator) : A learner implementing `fit` and 
                `predict_proba` to use for predicting the propensity score.
            include_covariates (bool) : Whether to return the original
                covariates alongside the "propensity" column.

        """
        self.include_covariates = include_covariates
        self.learner = learner

    def fit(self, X, a):
        self.learner.fit(X, a)
        return self

    def transform(self, X, treatment_values=None):
        """Append propensity or replace covariates with propensity.

        Args:
            X (pd.DataFrame): A DataFrame of samples to transform. This will be
                input to the learner trained by fit. If the columns are 
                different, the results will not be valid.
            treatment_values (Any | None): A desired value/s to extract
                propensity to (i.e. probabilities to what treatment value
                should be calculated). If not specified, then the maximal
                treatment value is chosen. This is since the usual case is of
                treatment (A=1) control (A=0) setting.
            
        Returns:
            pd.DataFrame : DataFrame with a "propensity" column. 
            If "include_covariates" is `True`, it will include all of the 
            original features plus "propensity", else it will only have the 
            "propensity" column.

        """
        treatment_values = 1 if treatment_values is None else treatment_values

        res = self.learner.predict_proba(X)[:, treatment_values]
        res = pd.DataFrame(res, index=X.index, columns=["propensity"])
        if self.include_covariates:
            res = X.join(res)
        return res


class MatchingTransformer(object):

    def __init__(
        self,
        propensity_transform=None,
        caliper=None,
        with_replacement=True,
        n_neighbors=1,
        matching_mode="both",
        metric="mahalanobis",
        knn_backend="sklearn",
    ):
        """Transform data by removing poorly matched samples.

        Args:
            propensity_transform (causallib.transformers.PropensityTransformer):
                an object for data preprocessing which adds the propensity
                score as a feature (default: None)
            caliper (float) : maximal distance for a match to be accepted. If
                not defined, all matches will be accepted. If defined, some
                samples may not be matched and their outcomes will not be
                estimated. (default: None)
            with_replacement (bool): whether samples can be used multiple times
                for matching. If set to False, the matching process will optimize
                the linear sum of distances between pairs of treatment and
                control samples and only `min(N_treatment, N_control)` samples
                will be estimated. Matching with no replacement does not make
                use of the `fit` data and is therefore not implemented for
                out-of-sample data (default: True)
            n_neighbors (int) : number of nearest neighbors to include in match.
                Must be 1 if `with_replacement` is `False.` If larger than 1, the
                estimate is calculated using the `regress_agg_function` or 
                `classify_agg_function` across the `n_neighbors`. Note that when
                the `caliper` variable is set, some samples will have fewer than
                `n_neighbors` matches. (default: 1).
            matching_mode (str) : Direction of matching: `treatment_to_control`,
                `control_to_treatment` or `both` to indicate which set should
                be matched to which. All sets are cross-matched in `match`
                and when `with_replacement` is `False` all matching modes 
                coincide. With replacement there is a difference.
            metric (str) : Distance metric string for calculating distance
                between samples. Note: if an external built `knn_backend`
                object with a different metric is supplied, `metric` needs to
                be changed to reflect that, because `Matching` will set its 
                inverse covariance matrix if "mahalanobis" is set. (default: 
                "mahalanobis", also supported: "euclidean")
            knn_backend (str or callable) : Backend to use for nearest neighbor
                search. Options are "sklearn"  or a callable  which returns an 
                object implementing `fit`, `kneighbors` and `set_params` 
                like the sklearn `NearestNeighbors` object. (default: "sklearn"). 

        """
        self.matching = Matching(
            propensity_transform=propensity_transform,
            caliper=caliper,
            with_replacement=with_replacement,
            n_neighbors=n_neighbors,
            matching_mode=matching_mode,
            metric=metric,
            knn_backend=knn_backend,
        )

    def fit(self, X, a, y):
        """Fit data to transform

        This function loads the data for matching and must be called before
        `transform`. For convenience, consider using `fit_transform`.

        Args:
            X (pd.DataFrame): DataFrame of shape (n,m) containing m covariates
                for n samples.
            a (pd.Series): Series of shape (n,) containing discrete treatment
                values for the n samples.
            y (pd.Series): Series of shape (n,) containing outcomes for
                the n samples.

        Returns:
            self (MatchingTransformer) : Fitted object
        """
        self.matching.fit(X, a, y)

        return self

    def transform(self, X, a, y):
        """Transform data by restricting it to samples which are matched

        Following a matching process, not all of the samples will find matches.
        Transforming the data by only allowing samples in treatment that have
        close matches in control, or in control that have close matches in
        treatment can make other causal methods more effective. This function 
        will call `match` on the underlying Matching object.

        The attribute `matching_mode` changes the behavior of this function.
        If set to `control_to_treatment` each control will attempt to find a
        match among the treated, hence the transformed data will have a maximum
        size of N_c + min(N_c,N_t).
        If set to `treatment_to_control`, each treatment will attempt to find a
        match among the control and the transformed data will have a maximum
        size of N_t + min(N_c,N_t).
        If set to `both`, both matching operations will be executed and if a
        sample succeeds in either direction it will be included, hence the
        maximum size of the transformed data will be `len(X)`.

        If `with_replacement` is `False`, `matching_mode` does not change the
        behavior. There will be up to `min(N_c,N_t)` samples in
        the returned DataFrame, regardless.

        Args:
            X (pd.DataFrame): DataFrame of shape (n,m) containing m covariates
                for n samples.
            a (pd.Series): Series of shape (n,) containing discrete treatment
                values for the n samples.
            y (pd.Series): Series of shape (n,) containing outcomes for
                the n samples.

        Raises:
            NotImplementedError: Raised if a value of attribute `matching_mode`
            other than the supported values is set.

        Returns:
            Xm (pd.DataFrame): Covariates of samples that were matched
            am (pd.Series): Treatment values of samples that were matched
            ym (pd.Series): Outcome values of samples that were matched

        """
        self.matching.match(X, a, use_cached_result=True)
        matched_sample_indices = self.find_indices_of_matched_samples(X, a)
        X = X.loc[matched_sample_indices]
        a = a.loc[matched_sample_indices]
        y = y.loc[matched_sample_indices]
        return X, a, y

    def find_indices_of_matched_samples(self, X, a):
        """Find indices of samples which matched successfully.

        Given a DataFrame of samples `X` and treatment assignments `a`, return
        a list of indices of samples which matched successfully.

        Args:
            X (pd.DataFrame): Covariates of samples
            a (pd.Series): Treatment assignments

        Returns:
            pd.Series: indices of matched samples to be passed to `X.loc` 
        """

        matching_weights = self.matching.matches_to_weights()
        matches_mask = self._filter_matching_weights_by_mode(matching_weights)
        return matches_mask

    def _filter_matching_weights_by_mode(self, matching_weights):
        if self.matching.matching_mode == "control_to_treatment":
            matches_mask = matching_weights.control_to_treatment
        elif self.matching.matching_mode == "treatment_to_control":
            matches_mask = matching_weights.treatment_to_control
        elif self.matching.matching_mode == "both":
            matches_mask = matching_weights.sum(axis=1)
        else:
            raise NotImplementedError("Matching mode {} not supported".format(
                    self.matching.matching_mode))
        matches_mask = matches_mask.astype(bool)
        return matches_mask

    def fit_transform(self, X, a, y):
        """Match data and return matched subset.

        This is a convenience method, calling `fit` and `transform` at once.
        For details, see documentation of each function.

        Args:
            X (pd.DataFrame): DataFrame of shape (n,m) containing m covariates
                for n samples.
            a (pd.Series): Series of shape (n,) containing discrete treatment
                values for the n samples.
            y (pd.Series): Series of shape (n,) containing outcomes for
                the n samples.

        Returns:
            Xm (pd.DataFrame): Covariates of samples that were matched
            am (pd.Series): Treatment values of samples that were matched
            ym (pd.Series): Outcome values of samples that were matched
        """
        self.fit(X, a, y)
        return self.transform(X, a, y)

    def set_params(self, **kwargs):
        """Set parameters of matching engine. Supported parameters are:

        Keyword Args:
            propensity_transform (causallib.transformers.PropensityTransformer):
                an object for data preprocessing which adds the propensity
                score as a feature (default: None)
            caliper (float) : maximal distance for a match to be accepted
                (default: None)
            with_replacement (bool): whether samples can be used multiple times
                for matching (default: True)
            n_neighbors (int) : number of nearest neighbors to include in match.
                Must be 1 if `with_replacement` is False (default: 1).
            matching_mode (str) : Direction of matching: `treatment_to_control`,
                `control_to_treatment` or `both` to indicate which set should
                be matched to which. All sets are cross-matched in `match`
                and without replacement there is no difference in outcome,
                but with replacement there is a difference and it impacts
                the results of `transform`.
            metric (str) : Distance metric string for calculating
                distance between samples (default: "mahalanobis",
                    also supported: "euclidean")
            knn_backend (str or callable) : Backend to use for nearest neighbor
                search. Options are "sklearn"  or a callable  which returns an 
                object implementing `fit`, `kneighbors` and `set_params` like
                the sklearn `NearestNeighbors` object. (default: "sklearn"). 


        Returns:
            self: (MatchingTransformer) object with new parameters set

        """
        supported_params = [
            "propensity_transform",
            "caliper",
            "n_neighbors",
            "metric",
            "with_replacement",
            "matching_mode",
            "knn_backend",
        ]
        for key, value in kwargs.items():
            if key in supported_params:
                self.matching.__setattr__(key, value)
            else:
                warnings.warn(
                    "Received unsupported parameter: {}. Nothing done.".format(key))
        return self
