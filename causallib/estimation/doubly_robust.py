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


A module implementing several doubly-robust methods.
These methods utilize causal standardization model and causal weight models and combine them to hopefully achieve a
better model.
The exact way to combine differs from different models and is described in the class docstring.
"""

import abc
import warnings

import pandas as pd

from .base_estimator import IndividualOutcomeEstimator
from .base_weight import WeightEstimator
from ..utils import general_tools as g_tools


class BaseDoublyRobust(IndividualOutcomeEstimator):
    """
    Abstract class defining the interface and general initialization of specific doubly-robust methods.
    """

    def __init__(self, outcome_model, weight_model,
                 outcome_covariates=None, weight_covariates=None):
        """

        Args:
            outcome_model(IndividualOutcomeEstimator): A causal model that estimate on individuals level
                                                      (e.g. Standardization).
            weight_model (WeightEstimator): A causal model for weighting individuals (e.g. IPW).
            outcome_covariates (array): Covariates to use for outcome model.
                                        If None - all covariates passed will be used.
                                        Either list of column names or boolean mask.
            weight_covariates (array): Covariates to use for weight model.
                                       If None - all covariates passed will be used.
                                       Either list of column names or boolean mask.
        """
        super(BaseDoublyRobust, self).__init__(lambda **x: None)  # Dummy initialization
        delattr(self, "learner")  # To remove the learner attribute a IndividualOutcomeEstimator has
        self.outcome_model = outcome_model
        self.weight_model = weight_model
        self.outcome_covariates = outcome_covariates
        self.weight_covariates = weight_covariates

    @abc.abstractmethod
    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        raise NotImplementedError

    def _extract_outcome_model_data(self, X):
        outcome_covariates = self.outcome_covariates or X.columns
        X_outcome = X[outcome_covariates]
        return X_outcome

    def _extract_weight_model_data(self, X):
        weight_covariates = self.weight_covariates or X.columns
        X_weight = X[weight_covariates]
        return X_weight

    def _prepare_data(self, X, a):
        """
        Extract the relevant parts for outcome model and weight model for the entire data matrix

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).

        Returns:
            (pd.DataFrame, pd.DataFrame): X_outcome, X_weight
                Data matrix for outcome model and data matrix weight model
        """
        X_outcome = self._extract_outcome_model_data(X)
        X_weight = self._extract_weight_model_data(X)
        return X_outcome, X_weight

    def __repr__(self):
        repr_string = g_tools.create_repr_string(self)
        # Make a new line between outcome_model and weight_model
        repr_string = repr_string.replace(", weight_model",
                                          ",\n{spaces}weight_model".format(spaces=" " * (len(self.__class__.__name__)
                                                                                         + 1)))
        return repr_string


class DoublyRobustVanilla(BaseDoublyRobust):
    """
    Given the measured outcome Y, the assignment Y, and the coefficients X calculate a doubly-robust estimator
    of the effect of treatment

    Let e(X) be the estimated propensity score and m(X, A) is the estimated effect by an estimator,
    then the individual estimates are:

    | Y + (A-e(X))*(Y-m(X,1)) / e(X) if A==1, and
    | Y + (e(X)-A)*(Y-m(X,0)) / (1-e(X)) if A==0

    These expressions show that when e(X) is an unbiased estimator of A, or when m is an unbiased estimator of Y
    then the resulting estimator is unbiased. Note that the term for A==0 is derived from (1-A)-(1-e(X))

    Another way of writing these equation is by "correcting" the individual prediction rather than the individual
    outcome:

    | m(X,1) + A*(Y-m(X,1))/e(X), and
    | m(X,0) + (1-A)*(Y-m(X,0))/(1-e(X))
    """

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome, X_weight = self._prepare_data(X, a)
        weight_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.weight_model.learner)

        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X=X_weight, a=a)

        self.outcome_model.fit(X=X_outcome, y=y, a=a)
        return self

    # def predict(self, X, a):
    #     raise NotImplementedError("Predict is not well defined for doubly robust and thus unimplemented.")

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        """
        Estimates individual outcome under different treatment values (interventions).

        Notes:
            This method utilizes only the standardization model behind the doubly-robust model. Namely, this is an
            uncorrected outcome (that does not incorporates the weighted observed outcome).
            To get a true doubly-robust estimation use the estimate_population_outcome, rather than an individual
            outcome.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            treatment_values (Any): Desired treatment value/s to use when estimating the counterfactual outcome/
                                    If not supplied, calculates for all available treatment values.
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                         operation, if True - prediction will utilize learner's `predict_proba` or
                                         `decision_function` which returns a continuous matrix of size
                                         (n_samples, n_classes).
                                         If False - `predict` will be used and return value will be based on a vector
                                         of class classifications.

        Returns:
            pd.DataFrame: DataFrame which columns are treatment values and rows are individuals: each column is a vector
                          size (num_samples,) that contains the estimated outcome for each individual under the
                          treatment value in the corresponding key.
        """
        X_outcome = self._extract_outcome_model_data(X)
        prediction = self.outcome_model.estimate_individual_outcome(X_outcome, a, treatment_values, predict_proba)
        return prediction

    def _estimate_corrected_individual_outcome(self, X, a, y, treatment_values=None, predict_proba=None):
        """
        Estimating corrected individual counterfactual outcomes.
        This correction is an intermediate stage needed to estimate doubly-robust population outcome.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            treatment_values (Any): Desired treatment value/s to use when estimating the counterfactual outcome.
                                    If not supplied, calculates for all available treatment values.
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                         operation, if True - prediction will utilize learner's `predict_proba` or
                                         `decision_function` which returns a continuous matrix of size
                                         (n_samples, n_classes).
                                         If False - `predict` will be used and return value will be based on a vector
                                         of class classifications.

        Returns:
            pd.DataFrame: DataFrame which columns are treatment values and rows are individuals: each column is a vector
                          size (num_samples,) that contains the estimated outcome for each individual under the
                          treatment value in the corresponding key.
        """
        X_outcome, X_weight = self._prepare_data(X, a)
        individual_cf = self.estimate_individual_outcome(X_outcome, a, treatment_values, predict_proba)
        weights = self.weight_model.compute_weights(X_weight, a)
        # Correct individual-estimation for later averaging:
        for treatment_value in treatment_values:
            is_treated = a == treatment_value
            y_cur = y.loc[is_treated]
            y_pred_cur = individual_cf.loc[is_treated, treatment_value]
            w_cur = weights.loc[is_treated]
            # This is a the same as (y_cur - y_pred_cur) * w_cur, but compatible with y_pred being both Series (due to
            # predict_proba=False) and DataFrame (predict_proba=True).
            correction = y_pred_cur.mul(-1.0).add(y_cur, axis="index").mul(w_cur, axis="index")
            individual_cf.loc[is_treated, treatment_value] += correction
        # NOTE: that we do not use the corrected-counter-factual-outcome vector when we return our individual effect.
        # This is because this correction "contaminates" the model prediction with the actual outcome.
        # Causing the model results to be inconsistent with the model itself.
        return individual_cf

    def estimate_population_outcome(self, X, a, y=None, treatment_values=None, predict_proba=None, agg_func="mean"):
        """
        Doubly-robust averaging, combining the individual counterfactual predictions from the standardization model
        and the weighted observed outcomes to estimate population outcome for each treatment subgroup.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            treatment_values (Any): Desired treatment value/s to stratify upon before aggregating individual into
                                    population outcome.
                                    If not supplied, calculates for all available treatment values.
            predict_proba (bool | None): To be used when provoking estimate_individual_outcome.
                                         In case the outcome task is classification and in case `learner` supports the
                                         operation, if True - prediction will utilize learner's `predict_proba` or
                                         `decision_function` which returns a continuous matrix of size
                                         (n_samples, n_classes).
                                         If False - `predict` will be used and return value will be based on a vector
                                         of class classifications.
            agg_func (str): Type of aggregation function (e.g. "mean" or "median").

        Returns:
            pd.Series: Series which index are treatment values, and the values are numbers - the aggregated outcome for
                       the strata of people whose assigned treatment is the key.

        """
        if y is None:
            raise TypeError("Must supply outcome variable (y). Got None instead.")
        treatment_values = g_tools.get_iterable_treatment_values(treatment_values, a)

        individual_cf = self._estimate_corrected_individual_outcome(X, a, y, treatment_values)

        population_outcome = individual_cf.apply(self._aggregate_population_outcome, args=(agg_func,))
        return population_outcome

    def estimate_effect(self, outcome1, outcome2, agg="population", effect_types="diff"):
        if isinstance(outcome1, pd.Series) and isinstance(outcome2, pd.Series) and agg == "population":
            warnings.warn("Seems you're trying to calculate population (average) effect from individual outcome "
                          "prediction.\n"
                          "Note that the result might be biased since the output of estimate_individual_outcome() is "
                          "not corrected for population effect.\n"
                          "In case you want individual effect use agg='individual', or in case you want population"
                          "effect use the estimate_population_effect() output as your input to this function.")
        effect = super(DoublyRobustVanilla, self).estimate_effect(outcome1, outcome2, agg, effect_types)
        return effect


class DoublyRobustIpFeature(BaseDoublyRobust):
    """
    A doubly-robust estimator of the effect of treatment.
    This model adds the weighting (inverse probability weighting) as feature to the model.
    """

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        X_augmented = self._augment_outcome_model_data(X, a)
        prediction = self.outcome_model.estimate_individual_outcome(X_augmented, a, treatment_values, predict_proba)
        return prediction

    def _augment_outcome_model_data(self, X, a):
        """
        This method adds the features needed to the model (either weight vector or weight matrix, depending on the
        underlining weight_estimator).

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).

        Returns:
            pd.DataFrame: Concatenation of the weight features (either single or multi column) to the provided covariate
                          matrix (W | X).
        """
        X_outcome, X_weight = self._prepare_data(X, a)
        try:
            weights_feature = self.weight_model.compute_weight_matrix(X_weight, a)
            weights_feature = weights_feature.add_prefix("ipf_")
        except NotImplementedError:
            weights_feature = self.weight_model.compute_weights(X_weight, a)
            weights_feature = weights_feature.rename("ipf")
        # Let standardization deal with incorporating treatment assignment (a) into the data:
        X_augmented = pd.concat([weights_feature, X_outcome], join="outer", axis="columns")
        return X_augmented

    # def predict(self, X, a):
    #     raise NotImplementedError

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome, X_weight = self._prepare_data(X, a)
        weight_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.weight_model.learner)

        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X=X_weight, a=a)

        X_augmented = self._augment_outcome_model_data(X, a)
        self.outcome_model.fit(X=X_augmented, y=y, a=a)
        return self


class DoublyRobustJoffe(BaseDoublyRobust):
    """
    A doubly-robust estimator of the effect of treatment.
    This model uses the weights from the weight-model (e.g. inverse probability weighting) as individual weights for
    fitting the outcome model.
    """

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        prediction = self.outcome_model.estimate_individual_outcome(X=X, a=a, treatment_values=treatment_values,
                                                                    predict_proba=predict_proba)
        return prediction

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome, X_weight = self._prepare_data(X, a)
        weight_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.weight_model.learner)

        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X=X_weight, a=a)

        weights = self.weight_model.compute_weights(X_weight, a)
        self.outcome_model.fit(X=X_outcome, y=y, a=a, sample_weight=weights)
        return self

    # def predict(self, X, a):
    #     prediction = self.outcome_model.predict(X, a)
    #     return prediction
