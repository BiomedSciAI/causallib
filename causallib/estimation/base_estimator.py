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

Created on Apr 16, 2018


A module defining the various hierarchy of causal models interface.
Causal models have two main tasks - predicting counterfactual outcomes and predicting effect based on these estimated
outcomes.
On top of it there are two resolutions we can work on: the individual level (i.e. outcome and effect for each individual
in the dataset) and population level (i.e. some aggregation on the sample level).
This module defines it all with:
* EffectEstimator - can estimate both individual and population level effect
* PopulationOutcomeEstimator - estimates aggregated outcomes on different sub-groups in the dataset.
* IndividualOutcomeEstimator - estimates individual level outcomes.
"""
import abc
import warnings

import pandas as pd
from numpy import isscalar

from sklearn.base import BaseEstimator

from ..utils.general_tools import create_repr_string


class EffectEstimator(BaseEstimator):
    """
    Class-based interface for estimating either individual-level or sample-level effect.
    """
    # This is somewhat static class, but since we wish it be inherited interface, we grouped it as class rather than a
    # module.

    # Invariant to vector or scalar arithmetic:
    CALCULATE_EFFECT = {"diff": lambda x, y: x - y,
                        "ratio": lambda x, y: x / y,
                        "or": lambda x, y: (x / (1 - x)) / (y / (1 - y))
                        }

    def estimate_effect(self, outcome_1, outcome_2, effect_types="diff"):
        """
        Estimates an effect given two potential outcomes.

        Args:
            outcome_1 (pd.Series | float): A potential outcome.
            outcome_2 (pd.Series | float): A potential outcome.
            effect_types (list[str] | str): Any iterable of strings from the set of EffectEstimator.CALCULATE_EFFECT keys

        Returns:
            pd.Series | pd.DataFrame: A Series if population effect (input is scalar) with index are the effect types
                                      and values are the corresponding computed effect.
                                      A DataFrame if individual effect (input is a vector) where columns are effects
                                      types and rows are effect in each individual.
                                      Always: Value type is same is outcome_1 and outcome_2 type.
        Examples:
            >>> from causallib.estimation.base_estimator import EffectEstimator
            >>> effect_estimator = EffectEstimator()
            >>> effect_estimator.estimate_effect(0.3, 0.6)
            >>> {"diff": -0.3,    # 0.3 - 0.6
                 "ratio": 0.5,    # 0.3 / 0.6
                 "or": 0.2857}    # Odds-Ratio(0.3, 0.6)
        """
        effect_types = [effect_types] if isscalar(effect_types) else effect_types
        results = {}
        for effect_type in effect_types:
            effect = self.CALCULATE_EFFECT[effect_type](outcome_1, outcome_2)
            results[effect_type] = effect
        # Format results in pandas array:
        results = pd.Series(results) if isscalar(outcome_1) else pd.concat(results, axis="columns",
                                                                           names=["effect_type"])
        return results


class PopulationOutcomeEstimator(EffectEstimator):
    """
    Interface for estimating aggregated outcome over different subgroups in the dataset.
    """

    # def __init__(self, *args, **kwargs):
    #     super(PopulationOutcomeEstimator, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def estimate_population_outcome(self, X, a, y, treatment_values=None):
        raise NotImplementedError


class IndividualOutcomeEstimator(PopulationOutcomeEstimator, EffectEstimator):
    """
    Interface for estimating individual-level outcome for different treatment values.
    """

    def __init__(self, learner, predict_proba=False, *args, **kwargs):
        """

        Args:
            learner: Initialized sklearn model.
            predict_proba (bool): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications.
        """
        self.learner = learner
        self.predict_proba = predict_proba
        # super(IndividualOutcomeEstimator, self).__init__(*args, **kwargs)

    @staticmethod
    def _aggregate_population_outcome(y, agg_func="mean"):
        """
        Aggregates a vector (of individual outcomes) to a population outcome scalar.

        Args:
            y (pd.Series): Individual outcome prediction.
            agg_func (str): Type of aggregation function (e.g. "mean" or "median").

        Returns:
            float: A scalar (float) aggregated result on the input.
        """
        if agg_func == "mean":
            return y.mean()
        elif agg_func == "median":
            return y.median()
        # TODO: consider adding max and min aggregation
        else:
            raise LookupError("Not supported aggregation function ({})".format(agg_func))

    def estimate_population_outcome(self, X, a, y=None, treatment_values=None, agg_func="mean"):
        """
        Implements aggregation of individual outcome into population (sample) outcome.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series | None): Observed outcome of size (num_subjects,).
            treatment_values (Any): Desired treatment value/s to stratify upon before aggregating individual into
                                    population outcome.
                                    If not supplied, calculates for all available treatment values.
            agg_func (str): Type of aggregation function (e.g. "mean" or "median").

        Returns:
            pd.Series: Series which index are treatment values, and the values are numbers - the aggregated outcome for
                       the strata of people whose assigned treatment is the key.
        """
        if y is not None:
            warnings.warn("Variable y (outcome) is not used when calculating population outcome of an "
                          "IndividualOutcome estimator. Instead it utilizes the individual outcome estimation.")
        individual_cf = self.estimate_individual_outcome(X, a, treatment_values)
        pop_outcome = individual_cf.apply(self._aggregate_population_outcome, args=(agg_func,))
        return pop_outcome

    def estimate_effect(self, outcome1, outcome2, agg="population", effect_types="diff"):
        """
        Estimates an effect given two potential outcomes.

        Args:
            outcome1 (pd.Series): A potential outcome.
            outcome2 (pd.Series): A potential outcome.
            agg (str): Either "population" or "individual" - whether to calculate individual effect or population
                       effect.
            effect_types (list[str] | str): Any iterable of strings from the set of EffectEstimator.CALCULATE_EFFECT keys

        Returns:
            pd.Series | pd.DataFrame: A Series if population effect (input is scalar) with index are the effect types
                                      and values are the corresponding computed effect.
                                      A DataFrame if individual effect (input is a vector) where columns are effects
                                      types and rows are effect in each individual.
                                      Always: Value type is the same as outcome_1 and outcome_2 type.
        """
        if agg == "population":
            outcome1 = self._aggregate_population_outcome(outcome1)
            outcome2 = self._aggregate_population_outcome(outcome2)
        effect = super(IndividualOutcomeEstimator, self).estimate_effect(outcome1, outcome2, effect_types)
        return effect

    @abc.abstractmethod
    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        """
        Estimates individual outcome under different treatment values (interventions)

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            treatment_values (Any): Desired treatment value/s to use when estimating the counterfactual outcome/
                                    If not supplied, calculates for all available treatment values.
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                         operation, if True - prediction will utilize learner's `predict_proba` or
                                         `decision_function` which returns a continuous matrix of size
                                         (n_samples, n_classes).
                                         If False - `predict` will be used and return value will be based on a vector of
                                         class classifications.
                                         If None - parameter is ignored and behaviour is as specified when initializing
                                         the IndividualOutcomeEstimator.

        Returns:
            pd.DataFrame: DataFrame which columns are treatment values and rows are individuals: each column is a vector
                          size (num_samples,) that contains the estimated outcome for each individual under the
                          treatment value in the corresponding key.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, X, a, y, sample_weight=None):
        """
        Trains a causal model from observed data.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            sample_weight: To be passed to the underlining scikit-learn's fit method.

        Returns:
            IndividualOutcomeEstimator: A causal weight model with an inner learner fitted.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def predict(self, X, a):
    #     raise NotImplementedError

    def evaluate_fit(self, X, y, a=None):
        # if a is given then you can return fit on subgroups stratified by treatment values
        pass  # TODO: Implement

    def __repr__(self):
        repr_string = create_repr_string(self)
        return repr_string
