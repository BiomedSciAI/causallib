import pandas as pd
import numpy as np
import statsmodels.api as sm

from .doubly_robust import DoublyRobust as BaseDoublyRobust
from causallib.utils.stat_utils import robust_lookup
from causallib.utils.general_tools import get_iterable_treatment_values


class BaseTMLE(BaseDoublyRobust):
    # TODO: decorator to convert continuous `y` to 0-1 range
    pass

    def _predict_observed(self, X, a):
        y_pred = self.outcome_model.estimate_individual_outcome(X, a)
        y_pred = robust_lookup(y_pred, a)  # Predictions on the observed
        return y_pred


class TMLEMatrix(BaseTMLE):  # TODO: TMLE for multiple treatments
    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        # TODO: support also just estimators?
        X_outcome = self._extract_outcome_model_data(X)
        self.outcome_model.fit(X_outcome, a, y)
        y_pred = self._predict_observed(X, a)

        X_treatment = self._extract_weight_model_data(X)
        self.weight_model.fit(X_treatment, a)
        w = self.weight_model.compute_weight_matrix(X_treatment, a)
        self.treatment_values_ = sorted(a.unique())

        # Statsmodels is the supports logistic regression with continuous (0-1 bounded) targets
        # so can be used with non-binary (but scaled) response
        targeted_outcome_model = sm.Logit(
            endog=w, exog=y, offset=y_pred,
        ).fit()
        self.targeted_outcome_model_ = targeted_outcome_model

        return self

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        y_pred = self._predict_observed(X, a)
        y_pred_logit = _logit(y_pred)
        weight_matrix = self.weight_model.compute_weight_matrix(X, a)

        res = {}
        for treatment_value in get_iterable_treatment_values(treatment_values, a):
            # TODO refactor common logic with TMLEMatrix, and just replace the assignment to vector
            treatment_assignment = pd.DataFrame(data=0, index=a.index, columns=self.treatment_values_)
            treatment_assignment[treatment_value] = weight_matrix[treatment_value]
            target_offset = self.targeted_outcome_model_.predict(treatment_assignment, linear=True)

            res[treatment_value] = _expit(y_pred_logit + target_offset)

        res = pd.concat(res, axis="columns", names=[a.name or "a"])
        return res


class TMLEVector(BaseTMLE):  # TODO: TMLE for binary treatment
    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome = self._extract_outcome_model_data(X)
        self.outcome_model.fit(X_outcome, a, y)
        y_pred = self._predict_observed(X, a)

        # TODO refactor common logic with TMLEMatrix, and just replace the final `w` into the model
        if a.nunique() != 2:
            raise AssertionError("Can only apply model on a binary treatment")
        X_treatment = self._extract_weight_model_data(X)
        self.weight_model.fit(X_treatment, a)
        w = self.weight_model.compute_weights(X_treatment, a)
        a_sign = 2 * a - 1  # Convert a==0 to -1, keep a==1 as 1.
        w *= a_sign  # w_i if a_i == 1, -w_i if a_i == 0.
        self.treatment_values_ = sorted(a.unique())

        targeted_outcome_model = sm.Logit(
            endog=w, exog=y, offset=y_pred,
        ).fit()
        self.targeted_outcome_model_ = targeted_outcome_model

        return self

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        y_pred = self._predict_observed(X, a)
        y_pred_logit = _logit(y_pred)
        weight_matrix = self.weight_model.compute_weight_matrix(X, a)

        res = {}
        for treatment_value in get_iterable_treatment_values(treatment_values, a):
            # TODO refactor common logic with TMLEMatrix, and just replace the assignment to vector
            treatment_assignment = weight_matrix[treatment_value]
            target_offset = self.targeted_outcome_model_.predict(treatment_assignment, linear=True)

            res[treatment_value] = _expit(y_pred_logit + target_offset)

        res = pd.concat(res, axis="columns", names=[a.name or "a"])
        return res


class TMLEImportanceSampling(BaseTMLE):
    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        X_outcome = self._extract_outcome_model_data(X)
        self.outcome_model.fit(X_outcome, a, y)
        y_pred = self.outcome_model.estimate_individual_outcome(X, a)
        y_pred = robust_lookup(y_pred, a)  # Predictions on the observed

        X_treatment = self._extract_weight_model_data(X)
        self.weight_model.fit(X_treatment, a)
        w = self.weight_model.compute_weights(X_treatment, a)

        # endog = a
        endog = pd.Series(1, index=y.index)  # TODO: does intercept results treatment effect (not potential outcomes)?
        targeted_outcome_model = sm.GLM(
            endog=endog, exog=y, offset=y_pred, freq_weights=w,
            family=sm.families.Binomial(),
            # family=sm.families.Binomial(sm.genmod.families.links.logit)
        ).fit()
        self.targeted_outcome_model_ = targeted_outcome_model

        return self


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))
