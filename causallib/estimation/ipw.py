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

Created on Apr 25, 2018

"""

import warnings

import pandas as pd

from .base_estimator import PopulationOutcomeEstimator
from .base_weight import PropensityEstimator
from ..utils.stat_utils import robust_lookup


class IPW(PropensityEstimator, PopulationOutcomeEstimator):
    """
    Causal model implementing inverse probability (propensity score) weighting.
    w_i = 1 / Pr[A=a_i|Xi]
    """

    def __init__(self, learner, clip_min=None, clip_max=None, use_stabilized=False, verbose=False):
        """

        Args:
            learner: Initialized sklearn model.
            clip_min (None|float): Optional value between 0 to 0.5 to lower bound the propensity estimation by
                                   clipping it. Will clip probabilities under clip_min to this value.
            clip_max (None|float): Optional value between 0.5 to 1 to upper bound the propensity estimation by
                                   clipping it. Will clip probabilities above clip_max to this value.
            use_stabilized (bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                   See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title
            verbose (bool): Whether to print summary statistics regarding the number of samples clipped due to
                            clip_min and clip_max.
        """
        super(IPW, self).__init__(learner, use_stabilized)
        self.__check_truncation_value_is_legal(clip_min, clip_max)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.verbose = verbose

    def fit(self, X, a, y=None):
        if self.use_stabilized:
            self.treatment_prevalence_ = a.value_counts(normalize=True, sort=False)
        self.learner.fit(X, a)
        return self

    def _predict(self, X):
        # Assumes PropensityEstimator checked that learner has predict_proba during initialization:
        prediction_matrix = self.learner.predict_proba(X)
        columns = getattr(self.learner, "classes_", None)  # non-sklearn classifiers might not have the attribute
        prediction_matrix = pd.DataFrame(prediction_matrix, index=X.index, columns=columns)
        return prediction_matrix

    def compute_weights(self, X, a, treatment_values=None, clip_min=None, clip_max=None, use_stabilized=None):
        """
        Computes individual weight given the individual's treatment assignment.
        w_i = 1 / Pr[A=a_i|X_i]  for each individual i.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            treatment_values (Any | None): A desired value/s to extract weights to (i.e. weights to what treatment
                                           value should be calculated).
                                           If not specified, then the weights are chosen by the individual's actual
                                           treatment assignment.
            clip_min (None|float): Optional value between 0 to 0.5 to lower bound the propensity estimation by
                                   clipping it. Will clip probabilities under clip_min to this value.
            clip_max (None|float): Optional value between 0.5 to 1 to upper bound the propensity estimation by
                                   clipping it. Will clip probabilities above clip_max to this value.
            use_stabilized (None|bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                        This overrides the use_stabilized parameter provided at initialization.
                                        If True provided, but the model was initialized with use_stabilized=False, then
                                        prevalence is calculated from data at hand, rather than the prevalence from the
                                        training data.
                                        See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title

        Returns:
            pd.Series | pd.DataFrame: If treatment_values is not supplied (None) or is a scalar, then a vector of
                                      n_samples with a weight for each sample is returned.
                                      If treatment_values is a list/array, then a DataFrame is returned.
        """
        weight_matrix = self.compute_weight_matrix(X, a, clip_min, clip_max, use_stabilized)
        if treatment_values is None:
            weights = robust_lookup(weight_matrix, a)  # lookup table: take the column a[i] for every i in index(a).
        else:
            weights = weight_matrix[treatment_values]
        return weights

    def compute_weight_matrix(self, X, a, clip_min=None, clip_max=None, use_stabilized=None):
        """
        Computes individual weight across all possible treatment values.
        w_ij = 1 / Pr[A=a_j | X_i]  for all individual i and treatment j.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            clip_min (None|float): Optional value between 0 to 0.5 to lower bound the propensity estimation by clipping it.
                                       Will clip probabilities under clip_min to this value.
            clip_max (None|float): Optional value between 0.5 to 1 to upper bound the propensity estimation by clipping it.
                                       Will clip probabilities above clip_max to this value.
            use_stabilized (None|bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                        This overrides the use_stabilized parameter provided at initialization.
                                        If True provided, but the model was initialized with use_stabilized=False, then
                                        prevalence is calculated from data at hand, rather than the prevalence from the
                                        training data.
                                        See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title

        Returns:
            pd.DataFrame: A matrix of size (num_subjects, num_treatments) with weight for every individual and every
                          treatment.
        """
        use_stabilized = self.use_stabilized if use_stabilized is None else use_stabilized

        probabilities = self.compute_propensity_matrix(X, a, clip_min, clip_max)

        # weight_matrix = 1.0 / probabilities                                                     # type: pd.DataFrame
        weight_matrix = probabilities.rdiv(1.0)

        if use_stabilized:
            if self.use_stabilized:
                prevalence = self.treatment_prevalence_
            else:
                warnings.warn("Stabilized is asked, however, the model was not trained using stabilization, and "
                              "therefore, stabilized weights are taken from the provided treatment assignment.",
                              RuntimeWarning)
                prevalence = a.value_counts(normalize=True, sort=False)
            prevalence_per_subject = a.replace(prevalence)  # map tx-assign to prevalence
            # pointwise multiplication of each column in weights:
            weight_matrix = weight_matrix.multiply(prevalence_per_subject, axis="index")

        return weight_matrix

    def compute_propensity(self, X, a, treatment_values=None, clip_min=None, clip_max=None):
        """

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            treatment_values (Any | None): A desired value/s to extract propensity to (i.e. probabilities to what
                                          treatment value should be calculated).
                                          If not specified, then the maximal treatment value is chosen. This is since
                                          the usual case is of treatment (A=1) control (A=0) setting.
            clip_min (None|float): Optional value between 0 to 0.5 to lower bound the propensity estimation by clipping it.
                                       Will clip probabilities under clip_min to this value.
            clip_max (None|float): Optional value between 0.5 to 1 to upper bound the propensity estimation by clipping it.
                                       Will clip probabilities above clip_max to this value.

        Returns:
            pd.DataFrame | pd.Series: A matrix/vector num_subjects rows and number of columns is the number of values
                                      provided to treatment_value. The content is probabilities for every individual
                                      to have the specified treatment_value.
                                      If treatment_value is a list/vector, than a pd.DataFrame is returned.
                                      If treatment_value is sort of scalar, than a pd.Series is returned.
                                       (just like slicing a DataFrame's columns)
        """
        treatment_values = a.max() if treatment_values is None else treatment_values

        probabilities = self.compute_propensity_matrix(X, a, clip_min, clip_max)
        probabilities = probabilities[treatment_values]
        return probabilities

    def compute_propensity_matrix(self, X, a=None, clip_min=None, clip_max=None):
        """

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            clip_min (None|float): Optional value between 0 to 0.5 to lower bound the propensity estimation by clipping it.
                                       Will clip probabilities under clip_min to this value.
            clip_max (None|float): Optional value between 0.5 to 1 to upper bound the propensity estimation by clipping it.
                                       Will clip probabilities above clip_max to this value.

        Returns:
            pd.DataFrame: A matrix of size (num_subjects, num_treatments) with probability for every individual and e
                          very treatment.
        """
        clip_min = self.clip_min if clip_min is None else clip_min
        clip_max = self.clip_max if clip_max is None else clip_max
        self.__check_truncation_value_is_legal(clip_min, clip_max)

        probabilities = self._predict(X)

        clip_min = 0 if clip_min is None else clip_min
        clip_max = 1 if clip_max is None else clip_max
        self.__count_truncated(probabilities, clip_min, clip_max)
        probabilities = probabilities.clip(lower=clip_min, upper=clip_max)

        return probabilities

    def estimate_population_outcome(self, X, a, y, w=None, treatment_values=None):
        """
        Calculates weighted population outcome for each subgroup stratified by treatment assignment.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            w (pd.Series | None): Individual (sample) weights calculated. Used to achieved unbiased average outcome.
                                   If not provided, will be calculated on the data.
            treatment_values (Any): Desired treatment value/s to stratify upon.
                                    Must be a subset of values found in `a`.
                                    If not supplied, calculates for all available treatment values.

        Returns:
            pd.Series[Any, float]: Series which index are treatment values, and the values are numbers - the
                                   aggregated outcome for the strata of people whose assigned treatment is the key.
        """
        if w is None:
            w = self.compute_weights(X, a)
        res = self._compute_stratified_weighted_aggregate(y, sample_weight=w, stratify_by=a,
                                                          treatment_values=treatment_values)
        return res

    @staticmethod
    def __check_truncation_value_is_legal(clip_min, clip_max):
        if clip_min is not None and not 0 <= clip_min <= 0.5:
            raise AssertionError("Provided value for truncation (clip_min) should be between 0.0 and 0.5")
        if clip_max is not None and not 0.5 <= clip_max <= 1:
            raise AssertionError("Provided value for truncation (clip_max) should be between 0.5 and 1.0")

    def __count_truncated(self, probabilities, clip_min, clip_max):
        num_clipped = probabilities.apply(lambda x: ~x.between(clip_min, clip_max)).sum().sum()
        fraction_clipped = num_clipped/probabilities.size
        if self.verbose:
            print(f'Fraction of values being truncated: {fraction_clipped:.5f}.')  # TODO: do as log
        return fraction_clipped
