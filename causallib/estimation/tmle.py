import abc

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

from .doubly_robust import DoublyRobust as BaseDoublyRobust
from causallib.utils.stat_utils import robust_lookup
from causallib.utils.general_tools import get_iterable_treatment_values


class BaseTMLE(BaseDoublyRobust):

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        # TODO: support also just estimators?
        y = self._scale_target(y, fit=True)
        X_outcome = self._extract_outcome_model_data(X)
        self.outcome_model.fit(X_outcome, a, y)
        y_pred = self._predict_observed(X, a)

        # self.treatment_values_ = sorted(a.unique())
        X_treatment = self._extract_weight_model_data(X)
        self.weight_model.fit(X_treatment, a)
        w = self._get_clever_covariate_fit(X, a)

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
            treatment_assignment = self._get_clever_covariate_inference(weight_matrix, treatment_value)
            target_offset = self.targeted_outcome_model_.predict(treatment_assignment, linear=True)
            counterfactual_prediction = _expit(y_pred_logit + target_offset)
            counterfactual_prediction = self._scale_target(counterfactual_prediction, fit=False)
            res[treatment_value] = counterfactual_prediction

        res = pd.concat(res, axis="columns", names=[a.name or "a"])
        return res

    @abc.abstractmethod
    def _get_clever_covariate_fit(self, X, a):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_clever_covariate_inference(self, weight_matrix, treatment_value):
        raise NotImplementedError

    # TODO: general implementation by taking the treatment encoding -
    #       either as signed-treatment or OneHotMatrix
    #       and multiplying it with the weight-matrix?
    # TODO: do a _get_weight for fitting a weighted regression
    #       Then the TMLE has endog=clever_covariate, weight=1 / None
    #       Then the TMLEIS has endog=signed_treatment/treatment_matrix, weight=ipw

    def _scale_target(self, y, fit=False):
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
        is_binary_target = (y >= 0) & (y <= 1)
        if is_binary_target:
            return y

        y_index, y_name = y.index, y.name  # Convert back to Series later
        y = y.to_frame()  # MinMaxScaler requires a 2D array, not a vector
        if fit:
            self._target_scaler_ = MinMaxScaler(feature_range=(0, 1))
            self._target_scaler_.fit(y)
            y = self._target_scaler_.transform(y)
        else:
            y = self._target_scaler_.inverse_transform(y)

        y = pd.Series(
            y[:, 0], index=y_index, name=y_name,
        )
        return y

    def _predict_observed(self, X, a):
        y_pred = self.outcome_model.estimate_individual_outcome(X, a)
        y_pred = robust_lookup(y_pred, a)  # Predictions on the observed
        return y_pred


class TMLEMatrix(BaseTMLE):  # TODO: TMLE for multiple treatments

    def _get_clever_covariate_fit(self, X, a):
        w = self.weight_model.compute_weight_matrix(X, a)
        return w

    def _get_clever_covariate_inference(self, weight_matrix, treatment_value):
        w = pd.DataFrame(data=0, index=weight_matrix.index, columns=weight_matrix.columns)
        w[treatment_value] = weight_matrix[treatment_value]
        return w


class TMLEVector(BaseTMLE):  # TODO: TMLE for binary treatment

    def _get_clever_covariate_fit(self, X, a):
        if a.nunique() != 2:
            raise AssertionError("Can only apply model on a binary treatment")
        w = self.weight_model.compute_weights(X, a)
        a_sign = 2 * a - 1  # Convert a==0 to -1, keep a==1 as 1.
        w *= a_sign  # w_i if a_i == 1, -w_i if a_i == 0.
        return w

    def _get_clever_covariate_inference(self, weight_matrix, treatment_value):
        w = weight_matrix[treatment_value]
        a_sign = 2 * treatment_value - 1
        w *= a_sign
        return w


class TMLEImportanceSampling(BaseTMLE):
    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        y = self._scale_target(y, fit=True)
        X_outcome = self._extract_outcome_model_data(X)
        self.outcome_model.fit(X_outcome, a, y)
        y_pred = self._predict_observed(X, a)

        X_treatment = self._extract_weight_model_data(X)
        self.weight_model.fit(X_treatment, a)
        w = self.weight_model.compute_weights(X_treatment, a)

        # endog = a
        # endog = pd.Series(1, index=y.index)
        endog = pd.DataFrame(
            {f"{a.name}": a,
             f"inverse_{a.name}": 1 - a}
        )  # TODO: General One hot matrix of treatment assignment
        # TODO: an equivalent ImportanceSamplingVector class with signed treatment vector rather than matrix
        targeted_outcome_model = sm.GLM(
            endog=endog, exog=y, offset=y_pred, freq_weights=w,
            family=sm.families.Binomial(),
            # family=sm.families.Binomial(sm.genmod.families.links.logit)
        ).fit()
        # TODO: maybe include in the Base as well. Convert implementation from Logit to GLM,
        #       make the _get_weight_term functions to return endog and freq_weight
        #       (In the others it is ipw (matrix/vector) and weights of 1 / None)
        #       (In this one it is intercept/treatment and weights of ipw)
        self.targeted_outcome_model_ = targeted_outcome_model

        return self

    def _get_clever_covariate_fit(self, X, a):
        raise NotImplementedError

    def _get_clever_covariate_inference(self, weight_matrix, treatment_value):
        raise NotImplementedError



def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))
