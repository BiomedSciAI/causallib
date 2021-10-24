import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LassoCV, LogisticRegressionCV
# Find internal implementations
try:  # Version 0.20 - 0.21
    from sklearn.feature_selection.base import SelectorMixin
except ModuleNotFoundError:
    # Version >= 0.22
    from sklearn.feature_selection._base import SelectorMixin


__all__ = ["DoubleLASSO", "RecursiveConfounderElimination"]


# noinspection PyAbstractClass
class _BaseConfounderSelection(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, importance_getter='auto', covariates=None):
        self.importance_getter = importance_getter
        self.covariates = covariates

    # TODO: move functionality for general utils and use for Estimators as well
    @staticmethod
    def _filter_covariates(func):
        def filter_covariates_and_run(self, X, *args, **kwargs):
            covariates = self.covariates
            if covariates is None:
                covariates = X.columns
            X = X.loc[:, covariates]
            return func(self, X, *args, **kwargs)
        return filter_covariates_and_run

    # @staticmethod
    def _filter_and_re_add_covariates(func):
        def filter_covariates_and_run_and_add(self, X, *args, **kwargs):
            covariates = self.covariates
            if covariates is None:
                covariates = X.columns
            sub_X = X.loc[:, covariates]
            res = func(self, sub_X, *args, **kwargs)
            complement_covariates = X.columns.difference(sub_X.columns)
            res = res.join(X.loc[:, complement_covariates])
            return res
        return filter_covariates_and_run_and_add

    @_filter_and_re_add_covariates
    def transform(self, X, a=None):
        X = X.loc[:, self.get_support()]
        return X

    # Decorator function cannot be defined as static before decorating,
    # so setting as the decorator as `staticmethod` is done after defining the `transform` using the decorator
    _filter_and_re_add_covariates = staticmethod(_filter_and_re_add_covariates)


class DoubleLASSO(_BaseConfounderSelection):
    def __init__(self, treatment_lasso=None, outcome_lasso=None,
                 mask_fn=None, threshold=1e-6,
                 importance_getter='auto', covariates=None):
        """
        A method for selecting confounders using sparse regression
        on both the treatment and the outcomes, and select for

        Implementing "Inference on Treatment Effects after Selection
        among High-Dimensional Controls"
        https://academic.oup.com/restud/article/81/2/608/1523757

        Args:
            treatment_lasso: Lasso learner to fit confounders and treatment.
                For example using scikit-learn,
                continuous treatment may use: `Lasso()`,
                discrete treatment may use: `LogisticRegression(penalty='l1')`.
                If `None` will try to automatically assign a lasso model with cross validation.
            outcome_lasso: Lasso learner to fit confounders and outcome.
                For example using scikit-learn,
                continuous outcome may use: `Lasso()`,
                discrete outcome may use: `LogisticRegression(penalty='l1')`.
                If `None` will try to automatically assign lasso model cross-validation.
            mask_fn: Function that takes input as two fitted lasso learners
                and returns a mask of the length of number of columns where True
                corresponds to columns that need to be selected. When set to None,
                the default implementation returns a mask based on non-zero
                coefficients in either learner. User can supply their own function,
                which must return a boolean array (of the length of columns of X)
                to indicate which columns are to be included.
            threshold: For default mask_fn, absolute value below which a lasso
                coefficient is treated as zero.
            importance_getter (str | callable): how to obtain feature importance.
                either a callable that inputs an estimator,
                a string of `'coef_'` or `'feature_importance_'`,
                or `'auto'` will detect `'coef_'` or `'feature_importance_'` automatically.
            covariates (list | np.ndarray): Specifying a subset of columns to perform selection on.
                Columns in `X` but not in `covariates` will be included after `transform`
                no matter the selection.
                Can be either a list of column names, or an array of boolean indicators length of `X`,
                or anything compatible with pandas `loc` function for columns.
                if `None` then all columns are participating in the selection process.
                This is similar to using sklearn's `ColumnTransformer` or `make_column_selector`.
        """
        # TODO: allowing users to provide the models follows the same design
        #       design principle throughout causallib,
        #       however, this might put some strain on the users who will need
        #       to know to supply `sklearn.linear_model.LogisticRegression(penalty='l1')
        #       for the treatment and the same for categorical outcome, but
        #       `sklearn.linear_model.Lasso` for continuous outcome.
        super().__init__(importance_getter, covariates)
        self.treatment_lasso = treatment_lasso
        self.outcome_lasso = outcome_lasso
        self.mask_fn = mask_fn
        self.threshold = threshold

    @_BaseConfounderSelection._filter_covariates
    def fit(self, X, a, y):
        self.treatment_lasso = self._data_driven_initialization(self.treatment_lasso, a)
        self.outcome_lasso = self._data_driven_initialization(self.outcome_lasso, y)

        self.treatment_lasso.fit(X, a)
        self.outcome_lasso.fit(X, y)
        mask_fn = self.mask_fn or self._get_non_zero_coef_mask
        self.support_ = mask_fn(self.treatment_lasso, self.outcome_lasso)
        self.n_features_ = self.support_.sum()
        return self

    # @_BaseConfounderSelection._filter_and_re_add_covariates
    # def transform(self, X, a=None):
    #     X = X.iloc[:, self.get_support()]
    #     return X

    def _get_support_mask(self):
        return self.support_

    def _get_non_zero_coef_mask(self, treatment_lasso, outcome_lasso):
        # Using _get_feature_importances from sklearn covers many more
        # edge cases than writing a vanilla function.
        # Specifying transform_func "norm" in the call below
        # actually calls np.abs when the coef_ attribute of treatment_lasso
        # (or output_lasso) is a one dimensional vector.
        treatment_lasso_importances = _get_feature_importances(
            treatment_lasso, self.importance_getter, transform_func="norm",
        )
        outcome_lasso_importances = _get_feature_importances(
            outcome_lasso, self.importance_getter, transform_func="norm",
        )
        treatment_mask = treatment_lasso_importances >= self.threshold
        outcome_mask = outcome_lasso_importances >= self.threshold
        return treatment_mask | outcome_mask

    @staticmethod
    def _data_driven_initialization(estimator, target):
        if estimator is not None:  # User provided an estimator
            return estimator

        if type_of_target(target) == "continuous":
            estimator = LassoCV()
        else:
            estimator = LogisticRegressionCV(penalty='l1', solver='saga', max_iter=5000)
        return estimator


class RecursiveConfounderElimination(_BaseConfounderSelection):

    def __init__(self, estimator, n_features_to_select: int = 1, step: int = 1,
                 importance_getter="auto", covariates=None):
        """Recursively eliminate confounders to prune confounders.

        Args:
            estimator: Estimator to fit for every step of recursive elimination.
            n_features_to_select (int): The number of confounders to keep.
            step (int): The number of confounders to eliminate in one iteration.
            importance_getter (str | callable): how to obtain feature importance.
              either a callable that inputs an estimator,
              a string of `'coef_'` or `'feature_importance_'`,
              or `'auto'` will detect `'coef_'` or `'feature_importance_'` automatically.
            covariates (list | np.ndarray): Specifying a subset of columns to perform selection on.
              Columns in `X` but not in `covariates` will be included after `transform`
              no matter the selection.
              Can be either a list of column names, or an array of boolean indicators length of `X`,
              or anything compatible with pandas `loc` function for columns.
              if `None` then all columns are participating in the selection process.
              This is similar to using sklearn's `ColumnTransformer` or `make_column_selector`.
        """
        super().__init__(importance_getter, covariates)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step

    @_BaseConfounderSelection._filter_covariates
    def fit(self, X, a, y):
        # This is like an abbreviated implementation of RFE in sklearn.
        # Main differences are (a) Conditioning on treatment for every iteration,
        # (b) adjusting ranking/support not to include treatment, and (c) accounting
        # for causallib data types for X and a.
        # TODO: the entire implementation may be reduced to overwriting the
        #       `importance_getter` function and rigging it to have infinity
        #       importance for the treatment assignment every time.
        n_features = len(X.columns)
        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)
        while np.sum(support_) > self.n_features_to_select:
            features = np.arange(n_features)[support_]
            estimator = clone(self.estimator)
            estimator.fit(a.to_frame().join(X.iloc[:, features]), y)
            importances = _get_feature_importances(
                estimator, self.importance_getter, transform_func="square",
            )
            importances = importances[1:]  # Do not consider "a" for dropping
            ranks = np.argsort(importances)
            ranks = np.ravel(ranks)
            threshold = min(self.step, np.sum(support_) - self.n_features_to_select)
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1
        features = np.arange(n_features)[support_]
        self.estimator.fit(a.to_frame().join(X.iloc[:, features]), y)
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        return self

    # @_BaseConfounderSelection._filter_and_re_add_covariates
    # def transform(self, X, a=None):
    #     X = X.iloc[:, self.get_support()]
    #     return X

    def _get_support_mask(self):
        return self.support_


def _get_feature_importances(estimator, getter, transform_func=None, norm_order=1):
    """
    Retrieve and aggregate (if ndim > 1) (and optionally transforms)
    the feature importances from an estimator.

    Args:
        estimator: A scikit-learn estimator from which we want to get the feature importances.
        getter (str | callable):  An attribute or a callable to get the feature importance.
                If `"auto"`, `estimator` is expected to expose `coef_` or `feature_importances_`.
        transform_func (str | None): The transform to apply to the feature importances.
                By default (`None`) no transformation is applied.
                Only "norm" and "square" are currently supported.
        norm_order (int): The norm order to apply when `transform_func="norm"`.
                Only applied when `importances.ndim > 1`.

    Returns:
        np.ndarray: The features importances, optionally transformed.
    """
    # A local version of sklearn's `_get_feature_importance`,
    # Because there have been multiple changes between version 0.20 and 0.24
    # in both API and import location, it uses the 0.24 version of the function.
    # Once dependencies move to > 0.24 this may be removed

    if isinstance(getter, str):
        if getter == "auto":
            if hasattr(estimator, "coef_"):
                getter = "coef_"
            elif hasattr(estimator, "feature_importances_"):
                getter = "feature_importances_"
            else:
                raise ValueError(
                    f"`importance_getter=='auto'` requires the estimator to have "
                    f"a `coef_` or `feature_importances_` attribute. "
                    f"If your estimator should have these attributes, "
                    f"make sure it is fitted before calling transform."
                )
        importances = getattr(estimator, getter)
    elif callable(getter):
        importances = getter(estimator)
    else:
        raise ValueError("`importance_getter` has to be a string or `callable`")

    if transform_func is None:
        pass
    elif transform_func == "norm":
        if importances.ndim == 1:
            importances = np.abs(importances)
        else:
            importances = np.linalg.norm(importances, axis=0, ord=norm_order)
    elif transform_func == "square":
        importances = importances ** 2
        if importances.ndim > 1:
            importances = importances.sum(axis=0)
    else:
        raise ValueError("`transform_func` only supports None, 'norm' and 'square'.")

    return importances
