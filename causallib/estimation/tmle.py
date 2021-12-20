import abc
import warnings
from typing import Type

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.multiclass import type_of_target

from .doubly_robust import DoublyRobust as BaseDoublyRobust
from causallib.utils.stat_utils import robust_lookup
from causallib.utils.general_tools import get_iterable_treatment_values, check_learner_is_fitted


class TMLE(BaseDoublyRobust):

    def __init__(self, outcome_model, weight_model,
                 outcome_covariates=None, weight_covariates=None,
                 reduced=False, importance_sampling=False,
                 glm_fit_kwargs=None,
                 ):
        super().__init__(
            outcome_model=outcome_model, weight_model=weight_model,
            outcome_covariates=outcome_covariates, weight_covariates=weight_covariates,
        )
        self.reduced = reduced
        self.importance_sampling = importance_sampling
        self.glm_fit_kwargs = {} if glm_fit_kwargs is None else glm_fit_kwargs
        # TODO: doc: `reduce==True` only work on binary treatment
        # TODO: doc: glm_got_kwargs: https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit.html

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome = self._extract_outcome_model_data(X)
        self.outcome_model.fit(X_outcome, a, y)
        y_pred = self._outcome_model_estimate_individual_outcome(X, a)
        y_pred = robust_lookup(y_pred, a)  # Predictions on the observed

        # self.treatment_values_ = sorted(a.unique())
        weight_model_is_not_fitted = not check_learner_is_fitted(self.weight_model.learner)
        X_treatment = self._extract_weight_model_data(X)
        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X_treatment, a)
        self.clever_covariate_ = _clever_covariate_factory(
            self.reduced, self.importance_sampling
        )(self.weight_model)
        exog = self.clever_covariate_.clever_covariate_fit(X, a)
        sample_weights = self.clever_covariate_.sample_weights(X, a)

        y = self._scale_target(y, fit=True, inverse=False)
        y_pred = self._scale_target(y_pred, fit=False, inverse=False)
        y_pred = _logit(y_pred)  # Used as offset in logit-space
        # Statsmodels supports logistic regression with continuous (0-1 bounded) targets
        # so can be used with non-binary (but scaled) response variable (`y`)
        # targeted_outcome_model = sm.Logit(
        #     endog=y, exog=clever_covariate, offset=y_pred,
        # ).fit(method="lbfgs")
        # GLM supports weighted regression, while Logit doesn't.
        targeted_outcome_model = sm.GLM(
            endog=y, exog=exog, offset=y_pred, freq_weights=sample_weights,
            family=sm.families.Binomial(),
            # family=sm.families.Binomial(sm.genmod.families.links.logit)
        ).fit(**self.glm_fit_kwargs)
        self.targeted_outcome_model_ = targeted_outcome_model

        return self

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        potential_outcomes = self._outcome_model_estimate_individual_outcome(X, a)

        res = {}
        for treatment_value in get_iterable_treatment_values(treatment_values, a):
            potential_outcome = potential_outcomes[treatment_value]
            potential_outcome = self._scale_target(potential_outcome, fit=False, inverse=False)
            potential_outcome = _logit(potential_outcome)
            treatment_assignment = self.clever_covariate_.clever_covariate_inference(X, a, treatment_value)
            counterfactual_prediction = self.targeted_outcome_model_.predict(
                treatment_assignment, offset=potential_outcome,
            )
            counterfactual_prediction = self._scale_target(counterfactual_prediction, fit=False, inverse=True)
            res[treatment_value] = counterfactual_prediction

        res = pd.concat(res, axis="columns", names=[a.name or "a"])
        return res

    def _scale_target(self, y, fit=False, inverse=False):
        """The re-targeting of the estimation requires log loss,
        which requires the target to be bounded between 0 and 1.
        However, general continuous targets can still be used for targeted learning
        as long as they are scaled into the 0-1 interval.

        See Gruber and van der Laan 2010: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3126669/

        Function transforms response variable into [0, 1] interval when fitting,
        and inverse-transform it during inference.

        Args:
            y (pd.Series): Response variable.
            fit (bool): If True - fit and transform. If False - inverse transforms.

        Returns:
            pd.Series: a scaled response variable.
        """
        y_index, y_name = y.index, y.name  # Convert back to pandas Series later
        y = y.to_frame()  # MinMaxScaler requires a 2D array, not a vector
        if fit:
            self._validate_predict_proba_for_classification(y)
            self.target_scaler_ = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler_.fit(y)

        if inverse:
            y = self.target_scaler_.inverse_transform(y)
        else:
            y = self.target_scaler_.transform(y)

        y = pd.Series(
            y[:, 0], index=y_index, name=y_name,
        )
        return y

    def _outcome_model_estimate_individual_outcome(self, X, a):
        """Standardize output for continuous `outcome_model` with `predict` with
        binary `outcome_model` with `predict_proba`"""
        potential_outcomes = self.outcome_model.estimate_individual_outcome(X, a)

        is_predict_proba_classification_result = isinstance(potential_outcomes.columns, pd.MultiIndex)
        if is_predict_proba_classification_result:
            # Classification `outcome_model` with `predict_proba=True` returns
            # a MultiIndex treatment-values (`a`) over outcome-values (`y`)
            # Extract the prediction for the maximal outcome class
            # (probably class `1` in binary classification):
            outcome_values = potential_outcomes.columns.get_level_values(level=-1)
            potential_outcomes = potential_outcomes.xs(
                outcome_values.max(), axis="columns", level=-1, drop_level=True,
            )
        return potential_outcomes

    def _validate_predict_proba_for_classification(self, y):
        # TODO: maybe instead for `predict_proba=True` when calling `outcome_model.estimate_individual_outcome`
        if type_of_target(y) != "continuous" and not self.outcome_model.predict_proba:
            warnings.warn(
                "`predict_proba` should be used if outcome type is not continuous."
                "Hint: set `predict_proba=True` when initializing the `outcome_model`.",
                UserWarning
            )


class BaseCleverCovariate:
    def __init__(self, weight_model):
        self.weight_model = weight_model

    @abc.abstractmethod
    def clever_covariate_fit(self, X, a):
        raise NotImplementedError

    @abc.abstractmethod
    def clever_covariate_inference(self, X, a, treatment_value):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_weights(self, X, a):
        raise NotImplementedError


class CleverCovariateFeatureMatrix(BaseCleverCovariate):

    def clever_covariate_fit(self, X, a):
        w = self.weight_model.compute_weight_matrix(X, a)
        return w

    def clever_covariate_inference(self, X, a, treatment_value):
        weight_matrix = self.weight_model.compute_weight_matrix(X, a)
        w = pd.DataFrame(data=0, index=weight_matrix.index, columns=weight_matrix.columns)
        w[treatment_value] = weight_matrix[treatment_value]
        return w

    def sample_weights(self, X, a):
        return None  # pd.Series(data=1, index=a.index)


class CleverCovariateFeatureVector(BaseCleverCovariate):

    def clever_covariate_fit(self, X, a):
        if a.nunique() != 2:
            raise AssertionError("Can only apply model on a binary treatment")
        w = self.weight_model.compute_weights(X, a)
        w[a == 0] *= -1
        return w

    def clever_covariate_inference(self, X, a, treatment_value):
        weight_matrix = self.weight_model.compute_weight_matrix(X, a)
        w = weight_matrix[treatment_value]
        if treatment_value == 0:
            w *= -1
        return w

    def sample_weights(self, X, a):
        return None  # pd.Series(data=1, index=a.index)


class CleverCovariateImportanceSamplingMatrix(BaseCleverCovariate):

    def clever_covariate_fit(self, X, a):
        self.treatment_encoder_ = OneHotEncoder(sparse=False, categories="auto")
        self.treatment_encoder_.fit(a.to_frame())
        A = self.treatment_encoder_.transform(a.to_frame())
        A = pd.DataFrame(A, index=a.index, columns=self.treatment_encoder_.categories_)
        return A

    def clever_covariate_inference(self, X, a, treatment_value):
        treatment_assignment = np.full(
            shape=(a.shape[0], 1),
            fill_value=treatment_value,
        )
        A = self.treatment_encoder_.transform(treatment_assignment)
        A = pd.DataFrame(
            A, index=a.index, columns=self.treatment_encoder_.categories_
        )
        return A

    def sample_weights(self, X, a):
        w = self.weight_model.compute_weights(X, a)
        return w


class CleverCovariateImportanceSamplingVector(BaseCleverCovariate):

    def clever_covariate_fit(self, X, a):
        if a.nunique() != 2:
            raise AssertionError("Can only apply model on a binary treatment")
        a_sign = a.replace({0: -1})  # 2 * a - 1  # Convert a==0 to -1, keep a==1 as 1.
        return a_sign

    def clever_covariate_inference(self, X, a, treatment_value):
        treatment_value = -1 if treatment_value == 0 else treatment_value
        treatment_assignment = pd.Series(data=treatment_value, index=a.index)
        return treatment_assignment

    def sample_weights(self, X, a):
        w = self.weight_model.compute_weights(X, a)
        return w


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _clever_covariate_factory(reduced, importance_sampling) -> Type[BaseCleverCovariate]:
    if importance_sampling and reduced:
        return CleverCovariateImportanceSamplingVector
    elif importance_sampling and not reduced:
        return CleverCovariateImportanceSamplingMatrix
    elif not importance_sampling and reduced:
        return CleverCovariateFeatureVector
    else:  # not importance_sampling and not reduced
        return CleverCovariateFeatureMatrix
