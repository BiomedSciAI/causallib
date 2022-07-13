# (C) Copyright 2021 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import pandas as pd
import numpy as np
from itertools import permutations, combinations
from collections import namedtuple, Counter
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.base import clone as sk_clone
from .base_estimator import IndividualOutcomeEstimator
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


KNN = namedtuple("KNN", "learner index")
# scipy distance routine requires matrix of valid numerical distances
# we use `VERY_LARGE_NUMBER` to represent an infinite distance
VERY_LARGE_NUMBER = np.finfo('d').max


def majority_rule(x):
    return Counter(x).most_common(1)[0][0]


class Matching(IndividualOutcomeEstimator):

    def __init__(
        self,
        propensity_transform=None,
        caliper=None,
        with_replacement=True,
        n_neighbors=1,
        matching_mode="both",
        metric="mahalanobis",
        knn_backend="sklearn",
        estimate_observed_outcome=False,
    ):
        """Match treatment and control samples with similar covariates.

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
            estimate_observed_outcome (bool) : Whether to allow a match of a
                sample to a sample other than itself when looking within its own
                treatment value. If True, the estimated potential outcome for the
                observed outcome may differ from the true observed outcome.
                (default: False)

        Attributes:
            classify_agg_function (callable) : Aggregating function for outcome
                estimation when classifying. (default: majority_rule)
                Usage is determined by type of `y` during `fit`
            regress_agg_function (callable) : Aggregating function for outcome
                estimation when regressing or predicting prob_a. (default: np.mean)
                Usage is determined by type of `y` during `fit`
            treatments_ (pd.DataFrame) : DataFrame of treatments (created after `fit`)
            outcomes_ (pd.DataFrame) : DataFrame of outcomes (created after `fit`)
            match_df_ (pd.DataFrame) : Dataframe of most recently calculated
                matches. For details, see `match`. (created after `match`)
            samples_used_ (pd.Series) : Series with count of samples used
                during most recent match. Series includes a count for each
                treatment value. (created after `match`)
        """
        self.propensity_transform = propensity_transform
        self.covariance_conditioner = EmpiricalCovariance()
        self.caliper = caliper
        self.with_replacement = with_replacement
        self.n_neighbors = n_neighbors
        self.matching_mode = matching_mode
        self.metric = metric
        # if classify task, default aggregation function is majority
        self.classify_agg_function = majority_rule
        # if regress task,  default aggregation function is mean
        self.regress_agg_function = np.mean
        self.knn_backend = knn_backend
        self.estimate_observed_outcome = estimate_observed_outcome

    def fit(self, X, a, y, sample_weight=None):
        """Load the treatments and outcomes and fit search trees.

        Applies transform to covariates X, initializes search trees for each
        treatment value for performing nearest neighbor searches.
        Note: Running `fit` a second time overwrites any information from
        previous `fit or `match` and re-fits the propensity_transform object.

        Args:
            X (pd.DataFrame): DataFrame of shape (n,m) containing m covariates
                for n samples.
            a (pd.Series): Series of shape (n,) containing discrete treatment
                values for the n samples.
            y (pd.Series): Series of shape (n,) containing outcomes for
                the n samples.
            sample_weight: IGNORED In signature for compatibility with other
                estimators.


        Note: `X`, `a` and `y` must share the same index.

        Returns:
            self (Matching) the fitted object
        """
        self._clear_post_fit_variables()
        self.outcome_ = y.copy()
        self.treatments_ = a.copy()

        if self.propensity_transform:
            self.propensity_transform.fit(X, a)
            X = self.propensity_transform.transform(X)

        self.conditioned_covariance_ = self._calculate_covariance(X)

        self.treatment_knns_ = {}
        for a in self.treatments_.unique():
            haystack = X[self.treatments_ == a]
            self.treatment_knns_[a] = self._fit_sknn(haystack)

        return self

    def _execute_matching(self, X, a):
        """Execute matching of samples in X according to the treatment values in a.

        Returns a DataFrame including all the results, which is also set as
        the attribute `self.match_df_`. The arguments `X` and `a` define the
        "needle" where the "haystack" is the data that was previously passed
        to fit, for matching with replacement. As such, treatment and control 
        samples from within `X` will not be matched with each other, unless
        the same `X` and `a` were passed to `fit`. For matching without
        replacement, the `X` and `a` passed to `match` provide the "needle" and
        the "haystack". If the attribute `caliper` is set, the matches are
        limited to those with a distance less than `caliper`.

        This function ignores the existing `match_df_` and will overwrite it.
        It is thus useful for if you have changed the settings and need to
        rematch the samples. For most applications, the `match` function is
        more convenient.

        Args:
            X (pd.DataFrame): DataFrame of shape (n,m) containing m covariates
                for n samples.
            a (pd.Series): Series of shape (n,) containing discrete treatment
                values for the n samples.

        Note: The args are assumed to share the same index.

        Returns:
            match_df: The resulting matches DataFrame is indexed so that
              ` match_df.loc[treatment_value, sample_id]` has columns `matches`
               and `distances` containing lists of indices to samples and the
               respective distances for the matches discovered for `sample_id`
               from within the fitted samples with the given `treatment_value`.
               The indices in the `matches` column are from the fitted data,
               not the X argument in `match`. If `sample_id` had no match,
               `match_df.loc[treatment_value, sample_id].matches = []`.
               The DataFrame has shape (n* len(a.unique()), 2 ).

        Raises:
            NotImplementedError: Raised when with_replacement is False and
               n_neighbors is not 1.
        """
        if self.n_neighbors != 1 and not self.with_replacement:
            raise NotImplementedError(
                "Matching more than one neighbor is only implemented for"
                "no-replacement"
            )

        if self.propensity_transform:
            X = self.propensity_transform.transform(X)
        if self.with_replacement:
            self.match_df_ = self._withreplacement_match(X, a)
        else:
            self.match_df_ = self._noreplacement_match(X, a)
        sample_id_name = X.index.name if X.index.name is not None else "sample_id"
        self.match_df_.index.set_names(
            ["match_to_treatment", sample_id_name], inplace=True
        )
        # we record the number of samples that were successfully matched of
        # each treatment value
        self.samples_used_ = self._count_samples_used_by_treatment_value(a)

        return self.match_df_

    def estimate_individual_outcome(
        self, X, a, y=None, treatment_values=None, predict_proba=True, dropna=True
    ):
        """
        Calculate the potential outcome for each sample and treatment value.

        Execute match and calculate, for each treatment value and each sample,
        the expected outcome. 

        Note: Out of sample estimation for matching without replacement requires
        passing a `y` vector here. If no 'y' is passed here, the values received
        by `fit` are used, and if the estimation indices are not a subset of the 
        fitted indices, the estimation will fail.

        If the attribute `estimate_observed_outcome` is 
        `True`, estimates will be calculated for the observed outcomes as well.
        If not, then the observed outcome will be passed through from the 
        corresponding element of `y` passed to `fit`.


        Args:
            X (pd.DataFrame): DataFrame of shape (n,m) containing m covariates
                for n samples.
            a (pd.Series): Series of shape (n,) containing discrete treatment
                values for the n samples.
            y (pd.Series): Series of shape (n,) containing outcome values for
                n samples. This is only used when `with_replacemnt=False`. 
                Otherwise, the outcome values passed to `fit` are used.
            predict_proba (bool) : whether to output classifications or
                probabilties for a classification task. If set to False and
                data is non-integer, a warning is issued. (default True)
            dropna (bool) : For samples that were unmatched due to caliper
                restrictions, drop from outcome_df leading to a potentially
                smaller sized output, or include them as NaN. (default: True)
            treatment_values : IGNORED

        Note: The args are assumed to share the same index.

        Returns:
            outcome_df (pd.DataFrame)
        """
        match_df = self.match(X, a, use_cached_result=True)

        outcome_df = self._aggregate_match_df_to_generate_outcome_df(
            match_df, a, predict_proba)
        outcome_df = self._filter_outcome_df_by_matching_mode(outcome_df, a)
        if outcome_df.isna().all(axis=None):
            raise ValueError("Matching was not successful and no outcomes can"
                             "be estimated. Check caliper value.")
        if dropna:
            outcome_df = outcome_df.dropna()
        
        return outcome_df

    def match(self, X, a, use_cached_result=True, successful_matches_only=False):
        """Matching the samples in X according to the treatment values in a.

        Returns a DataFrame including all the results, which is also set as
        the attribute `self.match_df_`. The arguments `X` and `a` define the
        "needle" where the "haystack" is the data that was previously passed
        to fit, for matching with replacement. As such, treatment and control 
        samp    les from within `X` will not be matched with each other, unless
        the same `X` and `a` were passed to `fit`. For matching without
        replacement, the `X` and `a` passed to `match` provide the "needle" and
        the "haystack". If the attribute `caliper` is set, the matches are
        limited to those with a distance less than `caliper`.

        Args:
            X (pd.DataFrame): DataFrame of shape (n,m) containing m covariates
                for n samples.
            a (pd.Series): Series of shape (n,) containing discrete treatment
                values for the n samples.
            use_cached_result (bool): Whether or not to return the `match_df` 
                from the most recent matching operation. The cached result will
                only be used if the sample indices of `X` and those of `match_df`
                are identical, otherwise it will rematch.
            successful_matches_only (bool): Whether or not to filter the matches
                to those which matched successfully. If set to `False`, the
                resulting DataFrame will have shape (n* len(a.unique()), 2 ),
                otherwise it may have a smaller shape due to unsuccessful matches.

        Note: The args are assumed to share the same index.

        Returns:
            match_df: The resulting matches DataFrame is indexed so that
              ` match_df.loc[treatment_value, sample_id]` has columns `matches`
               and `distances` containing lists of indices to samples and the
               respective distances for the matches discovered for `sample_id`
               from within the fitted samples with the given `treatment_value`.
               The indices in the `matches` column are from the fitted data,
               not the X argument in `match`. If `sample_id` had no match,
               `match_df.loc[treatment_value, sample_id].matches = []`.
               The DataFrame has shape (n* len(a.unique()), 2 ), if
               `successful_matches_only` is set to `False.

        Raises:
            NotImplementedError: Raised when with_replacement is False and
               n_neighbors is not 1.
        """
        cached_result_available = (hasattr(self, "match_df_")
                                   and X.index.equals(self.match_df_.loc[0].index))
        if not (use_cached_result and cached_result_available):
            self._execute_matching(X, a)

        return self._get_match_df(successful_matches_only=successful_matches_only)


    def matches_to_weights(self, match_df=None):
        """Calculate weights based on a given set of matches.

        For each matching from one treatment value to another, a weight vector
        is generated. The weights are calculated as the number of times a
        sample was selected in a matching, with each occurrence weighted
        according to the number of other samples in that matching. The weights
        can be used to estimate outcomes or to check covariate balancing. The 
        function can only be called after `match` has been run.

        Args:
            match_df (pd.DataFrame) : a DataFrame of matches returned from
                `match`. If not supplied, will use the `match_df_` attribute if
                available, else raises ValueError. Will not execute `match` to
                generate a `match_df`.

        Returns:
            weights_df (pd.DataFrame): DataFrame of shape (n,M) where M is the
                number of permutations of `a.unique()`.
        """
        if match_df is None:
            match_df = self._get_match_df(successful_matches_only=False)

        match_permutations = sorted(permutations(self.treatments_.unique()))
        weights_df = pd.DataFrame([
            self._matches_to_weights_single_matching(s, t, match_df)
            for s, t in match_permutations],).T

        return weights_df

    def get_covariates_of_matches(self, s, t, covariates):
        """
        Look up covariates of closest matches for a given matching.

        Using `self.match_df_` and the supplied `covariates`, look up
        the covariates of the last match. The function can only be called after
        `match` has been run.

            Args:
                s (int) : source treatment value
                t (int) : target treatment value
                covariates (pd.DataFrame) : The same covariates which were
                   passed to `fit`.

            Returns:
                covariate_df (pd.DataFrame) : a DataFrame of size
                (n_matched_samples, n_covariates * 3 + 2) with the covariate
                values of the sample, covariates of its match, calculated
                distance and number of neighbors found within the given
                caliper (with no caliper this will equal self.n_neighbors )

        """
        match_df = self._get_match_df()
        subdf = match_df.loc[s][self.treatments_ == t]
        sample_id_name = subdf.index.name

        def get_covariate_difference_from_nearest_match(source_row_index):
            j = subdf.loc[source_row_index].matches[0]
            delta_series = pd.Series(
                covariates.loc[source_row_index] - covariates.loc[j])
            source_row = covariates.loc[j].copy()
            source_row.at[sample_id_name] = j
            target_row = covariates.loc[source_row_index].copy()
            target_row = target_row
            covariate_differences = pd.concat(
                {
                    t: target_row,
                    s: source_row,
                    "delta": delta_series,
                    "outcomes": pd.Series(
                        {t: self.outcome_.loc[source_row_index],
                            s: self.outcome_.loc[j]}
                    ),
                    "match": pd.Series(
                        dict(
                            n_neighbors=len(
                                subdf.loc[source_row_index].matches),
                            distance=subdf.loc[source_row_index].distances[0],
                        )
                    ),
                }
            )
            return covariate_differences

        covdf = pd.DataFrame(
            data=[get_covariate_difference_from_nearest_match(i)
                  for i in subdf.index], index=subdf.index
        )
        covdf = covdf.reset_index()
        cols = covdf.columns
        covdf.columns = pd.MultiIndex.from_tuples(
            [(t, sample_id_name)] + list(cols[1:]))
        return covdf

    def _clear_post_fit_variables(self):
        for var in list(vars(self)):
            if var[-1] == "_":
                self.__delattr__(var)

    def _calculate_covariance(self, X):
        if len(X.shape) > 1 and X.shape[1] > 1:
            V_list = []
            for a in self.treatments_.unique():
                X_at_a = X[self.treatments_ == a].copy()
                current_V = self.covariance_conditioner.fit(X_at_a).covariance_
                V_list.append(current_V)
            # following Imbens&Rubin, we average across treatment groups
            V = np.mean(V_list, axis=0)
        else:
            # for 1d data revert to euclidean metric
            V = np.array(1).reshape(1, 1)
        return V

    def _aggregate_match_df_to_generate_outcome_df(self, match_df, a, predict_proba):
        agg_function = self._get_agg_function(predict_proba)

        def outcome_from_matches_by_idx(x):
            return agg_function(self.outcome_.loc[x])

        outcomes = {}
        for i in sorted(a.unique()):
            outcomes[i] = match_df.loc[i].matches.apply(
                outcome_from_matches_by_idx)
        outcome_df = pd.DataFrame(outcomes)
        return outcome_df

    def _get_match_df(self, successful_matches_only=True):
        if not hasattr(self, "match_df_") or self.match_df_ is None:
            raise NotFittedError("You need to run `match` first")
        match_df = self.match_df_.copy()
        if successful_matches_only:
            match_df = match_df[match_df.matches.apply(bool)]
        if match_df.empty:
            raise ValueError(
                "Matching was not successful and no outcomes can be "
                "estimated. Check caliper value."
            )
        return match_df

    def _filter_outcome_df_by_matching_mode(self, outcome_df, a):
        if self.matching_mode == "treatment_to_control":
            outcome_df = outcome_df[a == 1]
        elif self.matching_mode == "control_to_treatment":
            outcome_df = outcome_df[a == 0]
        elif self.matching_mode == "both":
            pass
        else:
            raise NotImplementedError(
                "Matching mode {} is not implemented. Please select one of "
                "'treatment_to_control', 'control_to_treatment, "
                "or 'both'.".format(self.matching_mode)
            )
        return outcome_df

    def _get_agg_function(self, predict_proba):
        if predict_proba:
            agg_function = self.regress_agg_function
        else:
            agg_function = self.classify_agg_function
            try:
                isoutputinteger = np.allclose(
                    self.outcome_.apply(int), self.outcome_)
                if not isoutputinteger:
                    warnings.warn(
                        "Classifying non-categorical outcomes. "
                        "This is probably a mistake."
                    )
            except:
                warnings.warn(
                    "Unable to detect whether outcome is integer-like. ")
        return agg_function

    def _instantiate_nearest_neighbors_object(self):
        backend = self.knn_backend
        if backend == "sklearn":
            backend_instance = NearestNeighbors(algorithm="auto")
        elif callable(backend):
            backend_instance = backend()
            self.metric = backend_instance.metric
        elif hasattr(backend, "fit") and hasattr(backend, "kneighbors"):
            backend_instance = sk_clone(backend)
            self.metric = backend_instance.metric
        else:
            raise NotImplementedError(
                "`knn_backend` must be either an NearestNeighbors-like object,"
                " a callable returning such an object, or the string \"sklearn\"")
        backend_instance.set_params(**self._get_metric_dict())
        return backend_instance

    def _fit_sknn(self, target_df):
        """
        Fit scikit-learn NearestNeighbors object with samples in target_df.

        Fits object, adds metric parameters and returns namedtuple which
        also includes DataFrame indices so that identities can looked up.

        Args:
            target_df (pd.DataFrame) : DataFrame of covariates to fit

        Returns:
            KNN (namedtuple) : Namedtuple with members `learner` and `index`
            containing the fitted sklearn object and an index lookup vector,
            respectively.
        """
        target_array = target_df.values

        sknn = self._instantiate_nearest_neighbors_object()

        target_array = self._ensure_array_columnlike(target_array)

        sknn.fit(target_array)
        return KNN(sknn, target_df.index)

    @staticmethod
    def _ensure_array_columnlike(target_array):
        if len(target_array.shape) < 2 or target_array.shape[1] == 1:
            target_array = target_array.reshape(-1, 1)
        return target_array

    def _get_metric_dict(
        self,
        VI_in_metric_params=True,
    ):
        metric_dict = dict(metric=self.metric)
        if self.metric == "mahalanobis":
            VI = np.linalg.inv(self.conditioned_covariance_)
            if VI_in_metric_params:
                metric_dict["metric_params"] = {"VI": VI}
            else:
                metric_dict["VI"] = VI

        return metric_dict

    def _kneighbors(self, knn, source_df, n_neighbors):
        """Lookup neighbors in knn object.

        Args:
           knn (namedtuple) : knn named tuple to look for neighbors in. The
               object has `learner` and `index` attributes to reference the
               original df index.
           source_df (pd.DataFrame) : a DataFrame of source data points to use
               as "needles" for the knn "haystack."
           n_neighbors

        Returns:
            match_df (pd.DataFrame) : a DataFrame of matches
        """
        source_array = source_df.values
        # 1d data must be in shape (-1, 1) for sklearn.knn
        source_array = self._ensure_array_columnlike(source_array)

        distances, neighbor_array_indices = knn.learner.kneighbors(
            source_array, n_neighbors=n_neighbors
        )

        return self._generate_match_df(
            source_df, knn.index, distances, neighbor_array_indices
        )

    def _generate_match_df(
        self, source_df, target_df_index, distances, neighbor_array_indices
    ):
        """
        Take results of matching and build into match_df DataFrame.

        For clarity we'll call the samples that are being matched "needles" and
        the set of samples that they looked for matches in the "haystack".

        Args:
            source_df (pd.DataFrame) : Covariate dataframe of N "needles"
            target_df_index (np.array) : An array of M indices of the haystack
                samples in their original dataframe.
            distances (np.array) : An array of N arrays of floats of length K
                where K is `self.n_neighbors`.
            neighbor_array_indices (np.array) : An array of N arrays of ints of
                length K where K is `self.n_neighbors`.
        """
        # target is the haystack, source is the needle(s)
        # translate array indices back to original indices
        matches_dict = {}
        for source_idx, distance_row, neighbor_array_index_row in zip(
            source_df.index, distances, neighbor_array_indices
        ):
            neighbor_df_indices = \
                target_df_index[neighbor_array_index_row.flatten()]
            if self.caliper is not None:
                neighbor_df_indices = [
                    n
                    for i, n in enumerate(neighbor_df_indices)
                    if distance_row[i] < self.caliper
                ]
                distance_row = [d for d in distance_row if d < self.caliper]
            matches_dict[source_idx] = dict(
                matches=list(neighbor_df_indices), distances=list(distance_row)
            )
        # convert dict of dicts like { 1: {'matches':[], 'distances':[]}} to df
        return pd.DataFrame(matches_dict).T

    def _matches_to_weights_single_matching(self, s, t, match_df):
        """
        For a given match, calculate the resulting weight vector.

        The weight vector adds a count each time a sample is used, weighted by
        the number of other neighbors when it was used. This is necessary to
        make the weighted sum return the correct effect estimate.
        """
        weights = pd.Series(self.treatments_.copy() * 0)
        name = {0: "control", 1: "treatment"}
        weights.name = "{s}_to_{t}".format(s=name[s], t=name[t])
        s_to_t_matches = match_df.loc[t][self.treatments_ == s].matches
        for source_idx, matches_list in s_to_t_matches.iteritems():
            if matches_list:
                weights.loc[source_idx] += 1
            for match in matches_list:
                weights.loc[match] += 1 / len(matches_list)
        return weights

    def _get_distance_matrix(self, source_df, target_df):
        """
        Create distance matrix for no replacement match.

        Combines metric, caliper and source/target data into a
        precalculated distance matrix which can be passed to
        scipy.optimize.linear_sum_assignment.
        """

        cdist_args = dict(
            XA=self._ensure_array_columnlike(source_df.values),
            XB=self._ensure_array_columnlike(target_df.values),
        )
        cdist_args.update(self._get_metric_dict(False))
        distance_matrix = distance.cdist(**cdist_args)

        if self.caliper is not None:
            distance_matrix[distance_matrix > self.caliper] = VERY_LARGE_NUMBER
        return distance_matrix

    def _withreplacement_match(self, X, a):
        matches = {}  # maps treatment value to list of matches TO that value

        for treatment_value, knn in self.treatment_knns_.items():
            n_matchable = sum(a==treatment_value)
            if n_matchable < self.n_neighbors:
                n_neighbors = n_matchable
                warnings.warn(
                    f"Not enough matchable samples in treatment group {treatment_value}. "
                    f"Reducing `n_neighbors` for this direction to {n_neighbors}."
                )
            else:
                n_neighbors = self.n_neighbors

            matches[treatment_value] = self._kneighbors(knn, X, n_neighbors)
            # when producing potential outcomes we may want to force the
            # value of the observed outcome to be the actual observed
            # outcome, and not an average of the k nearest samples.
            if not self.estimate_observed_outcome:

                def limit_within_treatment_matches_to_self_only(row):
                    if (
                        a.loc[row.name] == treatment_value
                        and row.name in row.matches
                    ):
                        row.matches = [row.name]
                        row.distances = [0]
                    return row

                matches[treatment_value] = matches[treatment_value].apply(
                    limit_within_treatment_matches_to_self_only, axis=1
                )

        return pd.concat(matches, sort=True)

    def _noreplacement_match(self, X, a):

        match_combinations = sorted(combinations(a.unique(), 2))
        matches = {}

        for s, t in match_combinations:
            distance_matrix = self._get_distance_matrix(X[a == s], X[a == t])
            source_array, neighbor_array_indices, distances = \
                self._optimally_match_distance_matrix(distance_matrix)
            source_df = X[a == s].iloc[np.array(source_array)]
            target_df = X[a == t].iloc[np.array(neighbor_array_indices)]
            if t in matches or s in matches:
                warnings.warn(
                    "No-replacement matching for more than "
                    "2 treatment values is not supported"
                )

            matches[t] = self._create_match_df_for_no_replacement(
                a, source_df, target_df, distances
            )
            matches[s] = self._create_match_df_for_no_replacement(
                a, target_df, source_df, distances
            )

        match_df = pd.concat(matches, sort=True)
        return match_df

    def _optimally_match_distance_matrix(self, distance_matrix):
        source_array, neighbor_array_indices = linear_sum_assignment(
            distance_matrix
        )
        distances = [
            [distance_matrix[s_idx, t_idx]]
            for s_idx, t_idx in zip(source_array, neighbor_array_indices)
        ]
        source_array, neighbor_array_indices, distances = \
            self._filter_noreplacement_matches_using_caliper(
                source_array, neighbor_array_indices, distances)
        return source_array, neighbor_array_indices, distances

    def _filter_noreplacement_matches_using_caliper(
            self, source_array, neighbor_array_indices, distances):
        if self.caliper is None:
            return source_array, neighbor_array_indices, distances
        keep_indices = [i for i, d in enumerate(
            distances) if d[0] <= self.caliper]
        source_array = source_array[keep_indices]
        neighbor_array_indices = neighbor_array_indices[keep_indices]
        distances = [distances[i] for i in keep_indices]
        if not keep_indices:
            warnings.warn("No matches found, check caliper."
                          "No estimation possible.")
        return source_array, neighbor_array_indices, distances

    @staticmethod
    def _create_match_df_for_no_replacement(
        base_series, source_df, target_df, distances
    ):
        match_sub_df = pd.DataFrame(
            index=base_series.index,
            columns=[
                "matches",
                "distances",
            ],
            data=base_series.apply(lambda x: pd.Series([[], []])).values,
            dtype="object",
        )

        # matching from source to target: read distances
        match_sub_df.loc[source_df.index] = pd.DataFrame(
            data=dict(
                matches=[[tidx] for tidx in target_df.index],
                distances=distances,
            ),
            index=source_df.index,
        )

        # matching from target to target: fill with zeros
        match_sub_df.loc[target_df.index] = pd.DataFrame(
            data=dict(
                matches=[[tidx] for tidx in target_df.index],
                distances=[[0]] * len(distances),
            ),
            index=target_df.index,
        )
        return match_sub_df

    def _count_samples_used_by_treatment_value(self, a):
        # we record the number of samples that were successfully matched of
        # each treatment value
        samples_used = {
            treatment_value:
            self.match_df_.loc[treatment_value][a != treatment_value]
            .matches.apply(bool).sum()

            for treatment_value in sorted(a.unique(), reverse=True)
        }

        return pd.Series(samples_used)


class PropensityMatching(Matching):

    def __init__(self, learner, **kwargs):
        """Matching on propensity score only.

        This is a convenience class to execute the common task of propensity
        score matching. It shares all of the methods of the `Matching` class
        but offers a shortcut for initialization.

        Args:
            learner (sklearn.estimator) :  a trainable propensity model that
                implements `fit` and `predict_proba`. Will be passed to the
                `PropensityTransformer` object.
            **kwargs : see Matching.__init__ for supported kwargs.
        """
        from causallib.preprocessing.transformers import PropensityTransformer

        super().__init__(**kwargs)
        self.learner = learner
        self.propensity_transform = PropensityTransformer(
            include_covariates=False, learner=self.learner
        )
