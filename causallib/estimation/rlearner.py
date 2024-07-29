"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on April 4, 2021
"""
import warnings
import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target

from .base_estimator import IndividualOutcomeEstimator

from ..utils.crossfit import cross_fitting
from ..utils import general_tools as g_tools


class VotingEstimator:
    def __init__(self, estimators):
        self.estimators = estimators

    def _stack_predictions(self, X):
        """Collect results from clf.predict_proba or ref.predict calls."""
        if hasattr(self.estimators[0], "predict_proba"):
            predictions = [est.predict_proba(X)[:, 1] for est in self.estimators]
        else:
            predictions = [est.predict(X) for est in self.estimators]
        predictions = np.stack(predictions, axis=-1)
        return predictions

    def predict(self, X):
        """Aggregate results of different estimators"""
        predicted_target = self._stack_predictions(X)
        averaged_target = np.average(predicted_target, axis=-1)
        return pd.Series(averaged_target, index=X.index)


class RLearner(IndividualOutcomeEstimator):
    """
    Given the measured outcome Y, the assignment A, and the coefficients X
    calculate an R-learner estimator of the effect of the treatment
    Let e(X) be the estimated propensity score and m(X) is the estimated outcome
    (E[Y|X]) by an estimator, then the R-learner minimize the following:
        ||Y - m(X) - (A-e(X))\tau(X)||^2_2 + lambda (\tau)
    where \tau(X) is a conditional average treatment effect and
    lambda is a regularize coefficient.

    If the effect_model is Linear, than minimizing squared loss with
    the target variable (Y-m(X)) and the features (A-e(X))X,
    otherwise it corresponds to a weighted regression problem,
    where the weights are (A-e(X))**2. This can be used with any scikit-learn
    regressor that accepts sample weights

    References:
    Nie, X., & Wager, S.(2017).
    Quasi - oracle estimation of heterogeneous treatment effects
    https://arxiv.org/abs/1712.04912

    Chernozhukov, V., et al. (2018).
    Double/debiased machine learning for treatment and structural parameters.‏
    https://academic.oup.com/ectj/article/21/1/C1/5056401
    """

    def __init__(
        self,
        effect_model,
        outcome_model,
        treatment_model,
        outcome_covariates=None,
        treatment_covariates=None,
        effect_covariates=None,
        n_splits=5,
        refit=True,
        caliper=1e-6,
        non_parametric=False,
    ):
        """
        Args:
            effect_model: An sklearn model that estimate that estimate
                the conditional average treatment effect \tau(X)
            outcome_model: An sklearn model that estimate the
                regressor Y|X (without the treatment).
                Note: it is recommended to use a regressor, even for binary outcome.
            treatment_model: An sklearn model that estimate the treatment model
                or the probability to be treated, i.e A|X or P(A=1|X)
            outcome_covariates (array): Covariates to use for the outcome model.
                If None - all covariates passed will be used.
                Either list of column names or boolean mask.
            treatment_covariates (array): Covariates to use for treatment model.
                If None - all covariates passed will be used.
                Either list of column names or boolean mask.
            effect_covariates (array): Covariates to use for the effect model.
                If None - all covariates passed will be used.
                Either list of column names or boolean mask.
            n_splits (int): number of sample-splitting in the cross-fitting procedure
            refit (bool): if True - Nuisance models are fitted over the whole
                training set, otherwise Nuisance models are fitted per folds
            non_parametric(bool): if True - the effect_model is estimated as
                weighted regression task, otherwise the effect_model is
                considered linear.
        """

        super(RLearner, self).__init__(lambda **x: None)  # Dummy initialization
        delattr(self, "learner")  # To remove the learner attribute a
        # IndividualOutcomeEstimator has
        self.effect_model = effect_model
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.outcome_covariates = outcome_covariates
        self.treatment_covariates = treatment_covariates
        self.effect_covariates = effect_covariates
        self.n_splits = n_splits
        self.refit = refit
        self.caliper = caliper
        self.non_parametric = non_parametric

    def __repr__(self):
        repr_string = (
            "{cls_name}(\n "
            "outcome_model={outcome},\n "
            "treatment_model={treatment},\n "
            "effect_model={effect})"
        ).format(
            cls_name=self.__class__.__name__,
            outcome=self.outcome_model,
            treatment=self.treatment_model,
            effect=self.effect_model,
        )
        return repr_string

    def _extract_outcome_model_data(self, X):
        outcome_covariates = self.outcome_covariates or X.columns
        X_outcome = X[outcome_covariates]
        return X_outcome

    def _extract_treatment_model_data(self, X):
        treatment_covariates = self.treatment_covariates or X.columns
        X_treatment = X[treatment_covariates]
        return X_treatment

    def _extract_effect_model_data(self, X):
        X_effect = pd.Series(data=1, index=X.index, name="intercept_").to_frame()
        if not isinstance(self.effect_covariates, list) or self.effect_covariates:
            effect_covariates = self.effect_covariates or X.columns
            X_effect = g_tools.column_name_type_safe_join(X[effect_covariates], X_effect.squeeze())
        return X_effect

    def _prepare_data(self, X):
        """
        Extract the relevant parts for outcome model, treatment model and
        effect model for the entire data matrix

        Args:
            X (pd.DataFrame): Covariate matrix of size
                              (num_subjects, num_features).
        Returns:
            (pd.DataFrame, pd.DataFrame, pd.DataFrame): X_outcome, X_treatment, X_effect
                Data matrix for outcome model, treatment model and data effect model
        """
        X_outcome = self._extract_outcome_model_data(X)
        X_treatment = self._extract_treatment_model_data(X)
        X_effect = self._extract_effect_model_data(X)
        return X_outcome, X_treatment, X_effect

    def _fit_and_predict_model(self, model, X, target, predict_proba):
        """
        fit outcome model (E[Y|X]) or treatment model (E[A|X] or P[A=1|X])
        and produce predictions on held-out data subsets
        Args:
            model (object): Treatment or outcome model
            X (pd.DataFrame): Covariate matrix of size
                              (num_subjects, num_features).
            target (pd.Series): Observed target (outcome or treatment)
                                of size (num_subjects,).
            predict_proba (bool): if True predict probabilities with predict_proba,
                                  otherwise use predict
        Returns:
            pd.Series: cross-fitted prediction of the outcome
        """
        target_pred, estimators = cross_fitting(
            model, X, target, n_splits=self.n_splits, predict_proba=predict_proba
        )
        target_pred = pd.Series(data=target_pred, index=X.index)
        if self.refit:
            estimators = [model.fit(X, target)]
        return target_pred, estimators

    def _fit_and_predict_nuisance(self, X_outcome, X_treatment, a, y):
        """
        fit the nuisance models and return residuals
        Args:
            X_outcome (pd.DataFrame): Covariate matrix of size
                (num_subjects, num_features).
            X_treatment (pd.DataFrame): Covariate matrix of size
                (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
        Returns:
            pd.Series, pd.Series:
                residuals of the outcome model, residuals of the treatment model
        """
        # residuals of the outcome model
        pred_y, outcome_model_ = self._fit_and_predict_model(
            self.outcome_model, X_outcome, y, predict_proba=False
        )
        res_y = y - pred_y
        self.outcome_model_ = VotingEstimator(outcome_model_)

        # residuals of the treatment model
        pred_a, treatment_model_ = self._fit_and_predict_model(
            self.treatment_model, X_treatment, a, predict_proba=self.binary_treatment_
        )
        res_a = a - pred_a
        self.treatment_model_ = VotingEstimator(treatment_model_)

        return res_y, res_a

    def estimate_individual_effect(self, X):
        """
        Predict the individual treatment effect
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).

        Returns:
            pd.Series: The series is a vector in size (num_subjects) that
                contains the estimated treatment effect, each row is an individual
        """
        X_effect = self._extract_effect_model_data(X)
        return pd.Series(self.effect_model.predict(X_effect), index=X.index)

    def estimate_individual_outcome(self, X, a, treatment_values=None,
                                    predict_proba=False):
        """
        Estimating corrected individual counterfactual outcomes.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            treatment_values (Any): Desired treatment value/s to use when
                estimating the counterfactual outcome.
                If not supplied, calculates for all available treatment values.
            predict_proba: IGNORED.
                Not used, present for API consistency by convention.
        Returns:
            pd.DataFrame: DataFrame which columns are treatment values and rows
                are individuals: each column is a vector size (num_samples,)
                that contains the estimated outcome for each individual under
                the treatment value in the corresponding key.
        """
        if (not self.binary_treatment_) & (treatment_values is None):
            raise ValueError(
                "The individual_outcome cannot be represented "
                "for a continuous treatment effect.\n Choose "
                'treatment in "treatment_values" '
            )
        X_outcome = self._extract_outcome_model_data(X)
        X_treatment = self._extract_treatment_model_data(X)

        outcome_pred = self.outcome_model_.predict(X_outcome)
        treatment_pred = self.treatment_model_.predict(X_treatment)
        effect_pred = self.estimate_individual_effect(X)

        treatment_values = g_tools.get_iterable_treatment_values(treatment_values, a)
        individual_cf = {
            treatment_value: (treatment_value - treatment_pred) * effect_pred
            + outcome_pred
            for treatment_value in treatment_values
        }
        return pd.DataFrame(individual_cf)

    def _fit_linear_effect_model(self, X, res_a, res_y):
        """
        fit a linear effect model without fitting an intercept
        Args:
            X (pd.DataFrame): Covariate matrix of size
                              (num_subjects, num_features).
            res_a (pd.Series): residuals of the treatment model
            res_y (pd.Series): residuals of the outcome model
        """
        if hasattr(self.effect_model, "fit_intercept"):
            if self.effect_model.fit_intercept:
                self.effect_model.fit_intercept = False
                warnings.warn(
                    "The effect model forces intercept estimation as an "
                    "additional coefficient. Therefore, the explicit "
                    "`fit_intercept` attribute was set to False."
                )
        else:
            warnings.warn(
                "`non_parametric=False` was passed and `effect_model` "
                "does not have a `fit_intercept` attribute and therefore "
                "does not seem to be scikit-learn LinearModel. "
                "Ignore this warning if you are sure `effect_model` "
                "is a linear model."
            )
        # multiply each row by a scalar
        X_tilde = X * np.array(res_a)[:, np.newaxis]
        self.effect_model.fit(X_tilde, res_y)

    def _fit_non_parametric_effect_model(self, X, res_a, res_y, caliper):
        """
        fit a non-parametric effect model as a weighted regression.
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            res_a (pd.Series): residuals of the treatment model
            res_y (pd.Series): residuals of the outcome model
            caliper (None | float): minimal value of treatment-probability residual
        """
        if hasattr(self.effect_model, "fit_intercept"):
            warnings.warn(
                "`non_parametric=True` was passed, but `effect_model` has a "
                "`fit_intercept` attributes and therefore seems to be a"
                "scikit-learn LinearModel. Ignore this warning if you're sure"
                "`effect_model` is a nonparametric model"
            )

        caliper_ = caliper or self.caliper
        if caliper_ is not None:
            cond_clipping = np.abs(res_a) < caliper_
            res_a.loc[cond_clipping] = caliper_ * np.sign(res_a.loc[cond_clipping])
        self.effect_model.fit(X, (res_y / res_a), sample_weight=res_a ** 2)

    def fit(self, X, a, y, caliper=None):
        """
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            caliper (None | float): minimal value of treatment-probability residual.
                used to avoid division by zero when fitting the effect-model.
                If None - no clipping is done.
                The caliper is irrelevant if the effect_model is Linear.
        """
        self.binary_treatment_ = type_of_target(a) == "binary"
        X_outcome, X_treatment, X_effect = self._prepare_data(X)
        res_y, res_a = self._fit_and_predict_nuisance(X_outcome, X_treatment, a, y)
        if self.non_parametric:
            self._fit_non_parametric_effect_model(X_effect, res_a, res_y, caliper)
        else:
            self._fit_linear_effect_model(X_effect, res_a, res_y)
        return self
