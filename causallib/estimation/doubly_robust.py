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
import numpy as np

from .base_estimator import IndividualOutcomeEstimator
from .base_weight import WeightEstimator, PropensityEstimator
from ..utils import general_tools as g_tools
from ..utils.stat_utils import robust_lookup


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

    def _is_weight_model_fitted(self):
        try:
            weight_model_is_fitted = g_tools.check_learner_is_fitted(self.weight_model.learner)
        except AttributeError:  # `weight_model` has no `learner` attribute. fit anyway
            warnings.warn(
                f"Weight model of type {type(self.weight_model)} does not"
                f"have a `learner` attribute and cannot be tested whether"
                f"it is fitted or not. Attempting to fit anyway."
            )
            weight_model_is_fitted = False
        return weight_model_is_fitted

    def __repr__(self):
        repr_string = g_tools.create_repr_string(self)
        # Make a new line between outcome_model and weight_model
        repr_string = repr_string.replace(", weight_model",
                                          ",\n{spaces}weight_model".format(spaces=" " * (len(self.__class__.__name__)
                                                                                         + 1)))
        return repr_string


class AIPW(BaseDoublyRobust):
    def __init__(self, outcome_model, weight_model,
                 outcome_covariates=None, weight_covariates=None,
                 overlap_weighting=False):
        """
        Calculates a doubly-robust estimate of the treatment effect by performing
        potential-outcome prediction (`outcome_model`) and then correcting its
        prediction-residuals using re-weighting from a treatment model (`weight_model`, like IPW).

        It has two flavors, which slightly change the weighting of the outcome model in the correction term.
        Let e(X) be the estimated propensity score and m(X, A) is the estimated effect by an estimator,
        then the individual estimates are:

        | m(X,1) + A*(Y-m(X,1))/e(X), and
        | m(X,0) + (1-A)*(Y-m(X,0))/(1-e(X))

        Which are basically add IP-weighted residuals from the observed predictions.
        As described in Kang and Schafer (2007) section 3.1 and Robins, Rotnitzky, and Zhao (1994).

        The additional flavor when `overlap_weighting=True` is from Glynn and Quinn (2010),
        adds weighting by the propensity-of-the-other-class to the outcome model,
        so extreme example (with poor covariate overlap) will contribute less to the correction
        (i.e. rely less on their prediction value that might extrapolate too much).
        This is a similar notion used in Overlap Weights model (hence the argument name)

        | A * [Y - (1-e(X))m(X,1)]/e(X) + (1-A) * m(X,1), and
        | (1-A) * [Y - e(X)m(X,0)]/(1-e(X)) + A * m(X,0)

        Args:
            outcome_model(IndividualOutcomeEstimator): A causal model that estimate on individuals level
                                                      (e.g. Standardization).
            weight_model (WeightEstimator | PropensityEstimator): A causal model for weighting individuals (e.g. IPW).
                If `overlap_weighting=True` then must be a `PropensityEstimator` model.
            outcome_covariates (array): Covariates to use for outcome model.
                If None - all covariates passed will be used. Either list of column names or boolean mask.
            weight_covariates (array): Covariates to use for weight model.
                If None - all covariates passed will be used. Either list of column names or boolean mask.
            overlap_weighting (bool): Whether to tweak the outcome-model correction-term to rely less on
                data-points with poor covariate overlap (extreme propensity).
                if `True`, requires `weight_model` to be an instance of `PropensityEstimator`.

        References:
            * Kang and Schafer, 2007, (https://dx.doi.org/10.1214/07-STS227)
            * Kang and Schafer attribute the original method to Cassel, Särndal and Wretman.
            * Glynn and Quinn, 2010, https://doi.org/10.1093/pan/mpp036
            * Robins, Rotnitzky, and Zhao, 1994, https://doi.org/10.1080/01621459.1994.10476818
        """
        super().__init__(outcome_model, weight_model,
                         outcome_covariates, weight_covariates)
        self.overlap_weighting = overlap_weighting

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        if self.overlap_weighting and a.nunique() != 2:
            raise AssertionError(
                f"`overlap_weights=True` version can only be used with binary treatment."
                f"Instead, treatment values are {set(a)}. Try setting it to `False`."
            )

        X_outcome, X_weight = self._prepare_data(X, a)
        weight_model_is_not_fitted = not self._is_weight_model_fitted()

        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X=X_weight, a=a, y=y)

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
        weights = self.weight_model.compute_weights(X_weight, a)
        individual_cf = self.estimate_individual_outcome(X_outcome, a, treatment_values, predict_proba)
        is_predict_proba_classification_result = isinstance(individual_cf.columns, pd.MultiIndex)
        if is_predict_proba_classification_result:
            # Classification `outcome_model` with `predict_proba=True` returns a MultiIndex of treatments over outcomes.
            # Extract the prediction for the maximal outcome class (probably class `1` in binary classification):
            outcome_values = individual_cf.columns.get_level_values(level=-1)
            individual_cf = individual_cf.xs(
                outcome_values.max(), axis="columns", level=-1, drop_level=True,
            )
        factual_prediction = robust_lookup(individual_cf, a)

        if self.overlap_weighting:
            propensities = self.weight_model.compute_propensity_matrix(X_weight)
            reversed_propensities = robust_lookup(propensities, 1 - a)  # take propensities of opposite treatment group
            factual_prediction *= reversed_propensities
            corrected_outcome = (y - factual_prediction) * weights
        else:
            outcome_correction = (y - factual_prediction) * weights
            corrected_outcome = factual_prediction + outcome_correction

        for treatment_value in treatment_values:
            individual_cf.loc[a == treatment_value, treatment_value] = corrected_outcome.loc[a == treatment_value]

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
        effect = super().estimate_effect(outcome1, outcome2, agg, effect_types)
        return effect


class PropensityFeatureStandardization(BaseDoublyRobust):
    def __init__(self, outcome_model, weight_model,
                 outcome_covariates=None, weight_covariates=None,
                 feature_type="weight_vector"):
        """
        A doubly-robust estimator of the effect of treatment.
        This model adds the weighting (inverse probability weighting)
        as additional feature to the outcome model.

        References:
            * Bang and Robins, https://doi.org/10.1111/j.1541-0420.2005.00377.x
            * Kang and Schafer, section 3.3, https://dx.doi.org/10.1214/07-STS227

        Args:
            outcome_model(IndividualOutcomeEstimator): A causal model that estimate on individuals level
            weight_model (WeightEstimator | PropensityEstimator): A causal model for weighting individuals (e.g. IPW).
            outcome_covariates (array): Covariates to use for outcome model.
                If None - all covariates passed will be used. Either list of column names or boolean mask.
            weight_covariates (array): Covariates to use for weight model.
                If None - all covariates passed will be used. Either list of column names or boolean mask.
            feature_type (str): the type of covariate to add. One of the following options:
                *  "weight_vector": uses a signed weight vector. Only defined for binary treatment.
                   For example, if `weight_model` is IPW then: 1/Pr[A=a_i|X] for each sample `i`.
                   As described in Bang and Robins (2005).
                * "signed_weight_vector": as `'weight_vector'`, but negates the weights of the control group.
                  For example, if `weight_model` is IPW then: 1/Pr[A|X] for treated and 1/Pr[A|X] for controls.
                  As described in the correction for Bang and Robins (2008)
                * "weight_matrix": uses the entire weight matrix.
                   For example, if `weight_model` is IPW then: 1/Pr[A_i=a|X_i=x],
                                for all treatment values `a` and for every sample `i`.
                * "masked_weighted_matrix": uses the entire weight matrix, but masks it with a dummy-encoding
                  of the treatment assignment.
                  For example, if weight_model` is IPW then: 1/Pr[A=a_i|X=x_i] and 0 for all other `a≠a_i` columns.
                  As described in Bang and Robins (2005).
                * "propensity_vector": uses the probabilities for being in treatment group: Pr[A=1|X].
                                       Better defined for binary treatment.
                                       Equivalent to Scharfstein, Rotnitzky, and Robins (1999) that use its inverse.
                * "logit_propensity_vector": uses logit transformation of the propensity to treat Pr[A=1|X].
                                             As described in Kang and Schafer (2007)
                * "propensity_matrix": uses the probabilities for all treatment options,
                    Pr[A_i=a|X_i=x] for all treatment values `a` and samples `i`.
        """
        super().__init__(outcome_model, weight_model,
                         outcome_covariates, weight_covariates)
        self.feature_type = feature_type

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
        feature_func = self._get_feature_function(self.feature_type)
        weights_feature = feature_func(X_weight, a)
        # Let standardization deal with incorporating treatment assignment (a) into the data:
        X_augmented = pd.concat([weights_feature, X_outcome], join="outer", axis="columns")
        return X_augmented

    # def predict(self, X, a):
    #     raise NotImplementedError

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome, X_weight = self._prepare_data(X, a)
        weight_model_is_not_fitted = not self._is_weight_model_fitted()

        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X=X_weight, a=a, y=y)

        X_augmented = self._augment_outcome_model_data(X, a)
        self.outcome_model.fit(X=X_augmented, y=y, a=a)
        return self

    def _get_feature_function(self, function_name):

        def weight_vector(X, a):
            w = self.weight_model.compute_weights(X, a)
            w = w.rename("ipf")
            return w

        def signed_weight_vector(X, a):
            """Bang and Robins, 2005 / 2008"""
            if a.nunique() != 2:
                raise AssertionError(
                    f"`feature_type` 'weight_vector' can only be used with binary treatment."
                    f"Instead, treatment values are {set(a)}."
                )
            w = weight_vector(X, a)
            w[a == 0] *= -1
            return w

        def weight_matrix(X, a):
            W = self.weight_model.compute_weight_matrix(X, a)
            W = W.add_prefix("ipf_")
            return W

        def masked_weight_matrix(X, a):
            """Bang and Robins, 2005"""
            W = weight_matrix(X, a)
            A = pd.get_dummies(a)
            A = A.add_prefix("ipf_")  # To match naming of `W`
            W_masked = W * A
            return W_masked

        def propensity_vector(X, a):
            p = self.weight_model.compute_propensity(X, a)
            p = p.rename("propensity")
            return p

        def logit_propensity_vector(X, a, safe=True):
            p = propensity_vector(X, a)
            if safe:
                epsilon = np.finfo(float).eps
                p = np.clip(p, epsilon, 1 - epsilon)
            return np.log(p / (1 - p))

        def propensity_matrix(X, a):
            P = self.weight_model.compute_propensity_matrix(X)
            # P = P.iloc[:, 1:]  # Drop first column
            P = P.add_prefix("propensity_")
            return P

        feature_functions = {
            "weight_vector": weight_vector,
            "signed_weight_vector": signed_weight_vector,
            "weight_matrix": weight_matrix,
            "masked_weight_matrix": masked_weight_matrix,
            "propensity_vector": propensity_vector,
            "logit_propensity_vector": logit_propensity_vector,
            "propensity_matrix": propensity_matrix,
        }
        return feature_functions[function_name]


class WeightedStandardization(BaseDoublyRobust):
    """
    This model uses the weights from the weight-model (e.g. inverse probability weighting)
    as individual weights for fitting the outcome model.

    References:
        * Kang and Schafer, section 3.2, https://dx.doi.org/10.1214/07-STS227
    """

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        prediction = self.outcome_model.estimate_individual_outcome(X=X, a=a, treatment_values=treatment_values,
                                                                    predict_proba=predict_proba)
        return prediction

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome, X_weight = self._prepare_data(X, a)
        weight_model_is_not_fitted = not self._is_weight_model_fitted()

        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X=X_weight, a=a, y=y)

        weights = self.weight_model.compute_weights(X_weight, a)
        self.outcome_model.fit(X=X_outcome, y=y, a=a, sample_weight=weights)
        return self

    # def predict(self, X, a):
    #     prediction = self.outcome_model.predict(X, a)
    #     return prediction
