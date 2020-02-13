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

import pandas as pd

from .base_weight import WeightEstimator
from .base_estimator import PopulationOutcomeEstimator


class MarginalOutcomeEstimator(WeightEstimator, PopulationOutcomeEstimator):
    """
    A marginal outcome predictor.
    Assumes the sample is marginally exchangeable, and therefore does not correct (adjust, control) for covariates.
    Predicts the outcome/effect as if the sample came from a randomized control trial: $\\Pr[Y|A]$.
    """

    def compute_weight_matrix(self, X, a, use_stabilized=None, **kwargs):
        # Another way to view this is that Uncorrected is basically an IPW-like with all individuals equally weighted.
        treatment_values = a.unique()
        treatment_values = treatment_values.sort()
        weights = pd.DataFrame(data=1, index=a.index, columns=treatment_values)
        return weights

    def compute_weights(self, X, a, treatment_values=None, use_stabilized=None, **kwargs):
        # Another way to view this is that Uncorrected is basically an IPW-like with all individuals equally weighted.
        weights = pd.Series(data=1, index=a.index)
        return weights

    def fit(self, X=None, a=None, y=None):
        """
        Dummy implementation to match the API.
        MarginalOutcomeEstimator acts as a WeightEstimator that weights each sample as 1

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).

        Returns:
            MarginalOutcomeEstimator: a fitted model.
        """
        return self

    def estimate_population_outcome(self, X, a, y, w=None, treatment_values=None):
        """
        Calculates potential population outcome for each treatment value.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            w (pd.Series | None): Individual (sample) weights calculated. Used to achieved unbiased average outcome.
                                   If not provided, will be calculated on the data.
            treatment_values (Any): Desired treatment value/s to stratify upon before aggregating individual into
                                    population outcome.
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

