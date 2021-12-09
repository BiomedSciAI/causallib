import abc

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
        endog = self._get_clever_covariate_fit(X, a)
        sample_weights = self._get_sample_weights(X, a)

        # Statsmodels is the supports logistic regression with continuous (0-1 bounded) targets
        # so can be used with non-binary (but scaled) response
        # targeted_outcome_model = sm.Logit(
        #     endog=clever_covariate, exog=y, offset=y_pred,
        # ).fit()
        # GLM supports weighted regression, while Logit doesn't.
        targeted_outcome_model = sm.GLM(
            endog=endog, exog=y, offset=y_pred, freq_weights=sample_weights,
            family=sm.families.Binomial(),
            # family=sm.families.Binomial(sm.genmod.families.links.logit)
        ).fit()
        # TODO: should be Quasibinomial due to weights? If so, break up the statsmodels call for
        #       regular TMLE and Imprtance Sampling TMLE
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

    @abc.abstractmethod
    def _get_sample_weights(self, X, a):
        raise NotImplementedError

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

    def _get_sample_weights(self, X, a):
        return None  # pd.Series(data=1, index=a.index)


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

    def _get_sample_weights(self, X, a):
        return None  # pd.Series(data=1, index=a.index)


class TMLEImportanceSampling(BaseTMLE):

    def _get_clever_covariate_fit(self, X, a):
        self.treatment_encoder_ = OneHotEncoder(sparse=False, categories="auto")
        self.treatment_encoder_.fit(a.to_frame())
        A = self.treatment_encoder_.transform(a.to_frame())
        A = pd.DataFrame(A, index=a.index, columns=self.treatment_encoder_.categories_)
        return A

    def _get_clever_covariate_inference(self, weight_matrix, treatment_value):
        treatment_assignment = np.full(
            shape=(weight_matrix.shape[0], 1),
            fill_value=treatment_value,
        )
        A = self.treatment_encoder_.transform(treatment_assignment)
        A = pd.DataFrame(
            A, index=weight_matrix.index, columns=self.treatment_encoder_.categories_
        )
        return A

    def _get_sample_weights(self, X, a):
        w = self.weight_model.compute_weights(X, a)
        return w


class TMLEImportanceSamplingVector(BaseTMLE):

    def _get_clever_covariate_fit(self, X, a):
        if a.nunique() != 2:
            raise AssertionError("Can only apply model on a binary treatment")
        a_sign = 2 * a - 1  # Convert a==0 to -1, keep a==1 as 1.
        return a_sign

    def _get_clever_covariate_inference(self, weight_matrix, treatment_value):
        treatment_value = -1 if treatment_value == 0 else treatment_value
        treatment_assignment = pd.Series(data=treatment_value, index=weight_matrix.index)
        return treatment_assignment

    def _get_sample_weights(self, X, a):
        w = self.weight_model.compute_weights(X, a)
        return w


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))
