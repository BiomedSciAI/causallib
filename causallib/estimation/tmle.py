import abc
import warnings
from typing import Type

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.multiclass import type_of_target

from .doubly_robust import BaseDoublyRobust
from causallib.estimation.base_estimator import IndividualOutcomeEstimator
from causallib.estimation.base_weight import PropensityEstimator
from causallib.utils.stat_utils import robust_lookup
from causallib.utils.general_tools import get_iterable_treatment_values, check_learner_is_fitted


class TMLE(BaseDoublyRobust):
    def __init__(self, outcome_model, weight_model,
                 outcome_covariates=None, weight_covariates=None,
                 reduced=False, importance_sampling=False,
                 glm_fit_kwargs=None,
                 ):
        """Targeted Maximum Likelihood Estimation.
        A model that takes an outcome model that was optimized to predict E[Y|X,A],
        and "retargets" ("updates") it to estimate E[Y^A|X] using a "clever covariate"
        constructed from the inverse propensity weights.

        Steps:
         1. Fit an outcome model Y=Q(X,A).
         2. Fit a weight model A=g(X,A).
         3. Construct a clever covariate using g(X,A).
         4. Fit a logistic regression model Q* to predict Y
            using g(X,A) as features and Q(X,A) as offset.
         5. Predict counterfactual outcome for treatment value `a` Q*(X,a)
            by plugging in Q(X,a) as offset, g(X,a) as covariate.

        Implements 4 flavours of TMLE controlled by the `reduced` and `importance_sampling` parameters.
        `importance_sampling=True` moves the clever covariate from being a feature to being a sample
        weight in the targeted regression.
        'reduced=True' use a clever covariate vector of 1s and -1s, therefore only good for binary treatment.
        Otherwise, the clever covariate are the entire IPW matrix and can be used for multiple treatments.

        References:
            * TMLE:
              Van Der Laan MJ, Rubin D. Targeted maximum likelihood learning. 2006.
              https://doi.org/10.2202/1557-4679.1043
            * TMLE with a vector version of clever covariate:
              Schuler MS, Rose S. Targeted maximum likelihood estimation for causal inference in observational studies.
              2017.
              https://doi.org/10.1093/aje/kww165
            * TMLE with a matrix version of clever covariate:
              Gruber S, van der Laan M. tmle: An R package for targeted maximum likelihood estimation. 2012.
              https://doi.org/10.18637/jss.v051.i13
            * TMLE with weighted regression and matrix of clever covariate:
              Gruber S, van der Laan M, Kennedy C. tmle: Targeted Maximum Likelihood Estimation. Cran documentation.
              https://cran.r-project.org/web/packages/tmle/index.html
            * TMLE for continuous outcomes
              Gruber S, van der Laan MJ. A targeted maximum likelihood estimator of a causal effect
              on a bounded continuous outcome. 2010.
              https://doi.org/10.2202/1557-4679.1260

        Args:
            outcome_model (IndividualOutcomeEstimator): An initial prediction of the outcome
            weight_model (PropensityEstimator): An IPW model predicting the treatment.
            outcome_covariates (array): Covariates to use for outcome model.
                                        If None - all covariates passed will be used.
                                        Either list of column names or boolean mask.
            weight_covariates (array): Covariates to use for weight model.
                                       If None - all covariates passed will be used.
                                       Either list of column names or boolean mask.
            reduced (bool): If `True` uses a vector version of the clever covariate
                            (rather than a matrix of all treatment values).
                            If `True` enforces a binary treatment assignment.
            importance_sampling (bool): If `True` moves the clever covariate from being
                                        a feature to being a weight in the regression.
            glm_fit_kwargs (dict): Additional kwargs for statsmodels' `GLM.fit()`.
              Can be used for example for refining the optimizers.
              see: https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit.html
        """
        super().__init__(
            outcome_model=outcome_model, weight_model=weight_model,
            outcome_covariates=outcome_covariates, weight_covariates=weight_covariates,
        )
        self.reduced = reduced
        self.importance_sampling = importance_sampling
        self.glm_fit_kwargs = {} if glm_fit_kwargs is None else glm_fit_kwargs

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        # Initial outcome model:
        X_outcome = self._extract_outcome_model_data(X)
        self.outcome_model.fit(X_outcome, a, y)
        y_pred = self._outcome_model_estimate_individual_outcome(X, a)
        y_pred = robust_lookup(y_pred, a)  # Predictions on the observed

        # IPW to prepare covariates to fluctuate the initial estimator:
        weight_model_is_not_fitted = not check_learner_is_fitted(self.weight_model.learner)
        X_treatment = self._extract_weight_model_data(X)
        if refit_weight_model or weight_model_is_not_fitted:
            self.weight_model.fit(X_treatment, a)
        self.clever_covariate_ = _clever_covariate_factory(
            self.reduced, self.importance_sampling
        )(self.weight_model)
        exog = self.clever_covariate_.clever_covariate_fit(X, a)
        sample_weights = self.clever_covariate_.sample_weights(X, a)

        # Update the initial estimator with the IPW:
        self._validate_predict_proba_for_classification(y)
        # The re-targeting of the estimation is done through logistic regression,
        # which requires the target to be bounded between 0 and 1.
        # We force this bounding in case target is continuous.
        # See Gruber and van der Laan 2010: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3126669/
        self.target_scaler_ = TargetMinMaxScaler(feature_range=(0, 1))
        self.target_scaler_.fit(y)
        y = self.target_scaler_.transform(y)
        y_pred = self.target_scaler_.transform(y_pred)
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
            potential_outcome = self.target_scaler_.transform(potential_outcome)
            potential_outcome = _logit(potential_outcome)
            treatment_assignment = self.clever_covariate_.clever_covariate_inference(X, a, treatment_value)
            counterfactual_prediction = self.targeted_outcome_model_.predict(
                treatment_assignment, offset=potential_outcome,
            )
            counterfactual_prediction = self.target_scaler_.inverse_transform(counterfactual_prediction)
            res[treatment_value] = counterfactual_prediction

        res = pd.concat(res, axis="columns", names=[a.name or "a"])
        return res

    def _outcome_model_estimate_individual_outcome(self, X, a):
        """Standardize output for continuous `outcome_model` with `predict` with
        binary `outcome_model` with `predict_proba`"""
        # Force `predict_proba` so if `outcome_model.learner` is a classifier,
        # It will produce continuous scores.
        # For zeros and ones, logit(y_pred) will break with `inf`
        potential_outcomes = self.outcome_model.estimate_individual_outcome(X, a, predict_proba=True)

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
        # TODO: move scale + logit here? since they always go together
        return potential_outcomes

    def _validate_predict_proba_for_classification(self, y):
        if type_of_target(y) != "continuous" and not self.outcome_model.predict_proba:
            warnings.warn(
                "`predict_proba` should be used in `outcome_model` if outcome type "
                "is not continuous. TMLE will force the use of `predict_proba`.",
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


class CleverCovariateFeatureVector(BaseCleverCovariate):
    """Clever covariate uses a signed vector of inverse propensity weights,
    with control group have their weights negated.
    The vector is then used as a predictor to the targeting regression.

    References:
        * Schuler MS, Rose S. Targeted maximum likelihood estimation for causal inference in observational studies. 2017
          https://doi.org/10.1093/aje/kww165
    """
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
    """Clever covariate of inverse propensity weight vector is used as weight for
    the targeting regression. The predictors are a one-hot (full dummy) encoding
    of the treatment assignment.

    References:
        * Gruber S, van der Laan M. tmle: An R package for targeted maximum likelihood estimation. 2012.
          https://doi.org/10.18637/jss.v051.i13
    """
    def clever_covariate_fit(self, X, a):
        self.treatment_encoder_ = OneHotEncoder(categories="auto")
        self.treatment_encoder_.fit(a.to_frame())
        A = self.treatment_encoder_.transform(a.to_frame())
        A = A.toarray()
        A = pd.DataFrame(A, index=a.index, columns=self.treatment_encoder_.categories_[0])
        return A

    def clever_covariate_inference(self, X, a, treatment_value):
        treatment_assignment = pd.DataFrame(
            data=treatment_value,
            index=a.index, columns=[a.name],
        )
        A = self.treatment_encoder_.transform(treatment_assignment)
        A = A.toarray()
        A = pd.DataFrame(
            A, index=a.index, columns=self.treatment_encoder_.categories_[0]
        )
        return A

    def sample_weights(self, X, a):
        w = self.weight_model.compute_weights(X, a)
        return w


class CleverCovariateImportanceSamplingVector(BaseCleverCovariate):
    """Clever covariate of inverse propensity weight vector is used as weight for
    the targeting regression. The predictors are a signed vector with negative 1 for
    the control group.
    """
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


class CleverCovariateFeatureMatrix(CleverCovariateImportanceSamplingMatrix):
    """Clever covariate uses a matrix of inverse propensity weights
    of all treatment values as a predictor to the targeting regression.

    References:
        * Gruber S, van der Laan M. tmle: An R package for targeted maximum likelihood estimation. 2012.
          https://doi.org/10.18637/jss.v051.i13
    """
    def clever_covariate_fit(self, X, a):
        A = super().clever_covariate_fit(X, a)
        W = self.weight_model.compute_weight_matrix(X, a)
        w = A * W
        return w

    def clever_covariate_inference(self, X, a, treatment_value):
        assignment = super().clever_covariate_inference(X, a, treatment_value)
        W = self.weight_model.compute_weight_matrix(X, a)
        w = assignment * W
        return w

    def sample_weights(self, X, a):
        return None  # pd.Series(data=1, index=a.index)


def _logit(p, safe=True):
    # TODO: move logit as a method, and do a clipped version with bounds specified in constructor
    if safe:
        epsilon = np.finfo(float).eps
        p = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p / (1 - p))


def _clever_covariate_factory(reduced, importance_sampling) -> Type[BaseCleverCovariate]:
    if importance_sampling and reduced:
        return CleverCovariateImportanceSamplingVector
    elif importance_sampling and not reduced:
        return CleverCovariateImportanceSamplingMatrix
    elif not importance_sampling and reduced:
        return CleverCovariateFeatureVector
    else:  # not importance_sampling and not reduced
        return CleverCovariateFeatureMatrix


class TargetMinMaxScaler(MinMaxScaler):
    """A MinMaxScaler that operates on a vector (Series)"""
    # @staticmethod
    def _series_to_matrix_and_back(func):
        def to_matrix_run_and_to_series(self, X):
            X_index, X_name = X.index, X.name  # Convert back to pandas Series later
            X = X.to_frame()  # MinMaxScaler requires a 2D array, not a vector
            X = func(self, X)
            X = pd.Series(
                X.ravel(), index=X_index, name=X_name,
            )
            return X
        return to_matrix_run_and_to_series

    def fit(self, X, y=None):
        X = X.to_frame()  # MinMaxScaler requires a 2D array, not a vector
        super().fit(X, y)
        return self

    @_series_to_matrix_and_back
    def transform(self, X):
        X = super().transform(X)
        return X

    @_series_to_matrix_and_back
    def inverse_transform(self, X):
        X = super().inverse_transform(X)
        return X

    # Decorator function cannot be defined as static before decorating,
    # so setting as the decorator as `staticmethod` is done after defining the functions using the decorator
    _series_to_matrix_and_back = staticmethod(_series_to_matrix_and_back)

