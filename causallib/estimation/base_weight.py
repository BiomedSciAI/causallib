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

import abc

import pandas as pd
import numpy as np

from ..utils.general_tools import create_repr_string, get_iterable_treatment_values


class WeightEstimator:
    """
    Interface for causal estimators balancing datasets through weighting.
    """

    def __init__(self, learner, use_stabilized=False, *args, **kwargs):
        """

        Args:
            learner: Initialized sklearn model.
            use_stabilized (bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                   See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title
        """
        self.learner = learner
        self.use_stabilized = use_stabilized

    @abc.abstractmethod
    def fit(self, X, a, y=None):
        """
        Trains a model to predict treatment assignment given the covariates: Pr[A|X].

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y: IGNORED.

        Returns:
            WeightEstimator: A causal weight model with an inner learner fitted.
        """
        raise NotImplementedError

    # def predict(self, X):
    #     raise NotImplementedError

    @abc.abstractmethod
    def compute_weights(self, X, a, treatment_values=None, use_stabilized=None, **kwargs):
        """
        Computes individual weight given the individual's treatment assignment.
        f(Pr[A=a_i | X_i])  for each individual i.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            treatment_values (Any | None): A desired value/s to extract weights to (i.e. weights to what treatment
                                           value should be calculated).
                                           If not specified, then the weights are chosen by the individual's actual
                                           treatment assignment.
            use_stabilized (bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                   This overrides the use_stabilized parameter provided at initialization.
                                   See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title
            **kwargs:

        Returns:
            pd.Series: A vector of size (num_subjects,) with a weight for each individual
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_weight_matrix(self, X, a, use_stabilized=None, **kwargs):
        """
        Computes individual weight across all possible treatment values.
        f(Pr[A=a_j | X_i])  for all individual i and treatment j.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            use_stabilized (bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                   This overrides the use_stabilized parameter provided at initialization.
                                   See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title
            **kwargs:

        Returns:
            pd.DataFrame: A matrix of size (num_subjects, num_treatments) with weight for every individual and every
                          treatment.
        """
        raise NotImplementedError

    @staticmethod
    def _compute_stratified_weighted_aggregate(y, sample_weight=None,
                                               stratify_by=None, treatment_values=None):
        """
        Calculates aggregation of `y` weighted by `sample_weights` stratified by `stratify_by` variable.

        Args:
            y (pd.Series): The variable to aggregate (num_subjects,).
            sample_weight (pd.Series|None): Individual (sample) weights calculated.
                                            Used to achieved unbiased average outcome.
                                            If not provided, gives equal weights to every example.
            stratify_by (pd.Series|None): Categorical variable to stratify according to (num_subjects,).
                                          Namely, aggregate within subgroups sharing the same values.
                                          If not provided, the aggregation is on the entire
            treatment_values (Any): Subset of values to stratify on from `stratify_by`.
                                    If not supplied, all available stratification values are used.

        Returns:
            pd.Series[Any, float]: Series which index are treatment values, and the values are numbers - the
                                   aggregated outcome for the strata of people whose assigned treatment is the key.
        """
        if sample_weight is None:
            sample_weight = pd.Series(data=1.0, index=y.index)
        if treatment_values is None and stratify_by is None:
            stratify_by = pd.Series(data=0, index=y.index)

        treatment_values = get_iterable_treatment_values(treatment_values, stratify_by)

        res = {}
        for treatment_value in treatment_values:
            subgroup_mask = stratify_by == treatment_value
            aggregated_value = np.average(y[subgroup_mask], weights=sample_weight[subgroup_mask])
            res[treatment_value] = aggregated_value
        res = pd.Series(res)
        return res

    def evaluate_balancing(self, X, a, y, w):
        pass  # TODO: implement: (1) table one with smd (2) gather lots of metric (ks, kl, smd) (3) plot CDF of each feature

    def __repr__(self):
        repr_string = create_repr_string(self)
        return repr_string


class PropensityEstimator(WeightEstimator):
    """
    Interface for causal estimators balancing datasets through propensity (i.e. treatment probability) estimation
    (e.g. inverse probability weighting).
    """

    def __init__(self, learner, use_stabilized=False, *args, **kwargs):
        """

        Args:
            learner: Initialized sklearn model.
            use_stabilized (bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                   See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title
        """
        super(PropensityEstimator, self).__init__(learner, use_stabilized=use_stabilized)
        if not hasattr(self.learner, "predict_proba"):
            raise AttributeError("Propensity Estimator must use a machine learning that can predict probabilities"
                                 "(i.e., have predict_proba method)")

    @abc.abstractmethod
    def compute_propensity(self, X, a, treatment_values=None, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_propensity_matrix(self, X, a, **kwargs):
        raise NotImplementedError
