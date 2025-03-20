from . import BasePositivity
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError


class Support:
    """Base Support class
    """

    def __init__(self, support=None):
        self.support = support

    def __str__(self):
        return "support: " + self.__repr__()

    def __repr__(self):
        if self.support is None:
            return "no support"
        return repr(self.support)

    def assert_same_type(self, other):
        if not isinstance(other, type(self)) and not isinstance(self, type(other)):
            raise ValueError(
                "Cannot intersect discrete and continuous support")


class ContinuousSupport(Support):
    """Continuous support class

    Intended for use with continuous valued data for which `min` and `max` can
    be said to characterize the support.
    """

    def fit(self, X):
        """Calculate support as min and max of continuous data `X`

        Args:
            X (pd.DataFrame|pd.Series): one dimensional continuous valued data

        Returns:
            ContinuousSupport: fitted ContinuousSupport object
        """
        self.support = [min(X), max(X)]
        return self

    def predict(self, x):
        """Predict if variable is in support

        Args:
            x (int|float): numerical value to check

        Returns:
            bool: True if `x` in support else False
        """
        if x >= self.support[0] and x <= self.support[1]:
            return True
        return False

    def intersection(self, other_support):
        """Find intersection of supports

        Args:
            other_support (ContinuousSupport): other support

        Returns:
            ContinuousSupport: intersection of this support with other support

        Raises:
            ValueError: if attempting to intersect with incompatible type
        """
        self.assert_same_type(other_support)
        if self.non_zero_overlap(other_support):
            joint_support = [max(self.support[0], other_support.support[0]),
                             min(self.support[1], other_support.support[1])]
        else:
            joint_support = None
        return ContinuousSupport(support=joint_support)

    def non_zero_overlap(self, other_support):
        if self.support is None or other_support.support is None:
            return False
        if self.support[1] >= other_support.support[0] and other_support.support[1] >= self.support[0]:
            return True

    def __sub__(self, other):
        return [self.support[0] - other.support[0], self.support[1] - other.support[1]]


class CategoricalSupport(Support):
    """Support for categorical variables based on `set`
    """

    def __init__(self, support=None):
        super().__init__(support=support)
        if self.support is not None and not isinstance(self.support, set):
            self.support = set(self.support)

    def fit(self, X):
        """Calculate support of categorical variables in `X` using `set`

        Args:
            X (pd.Series|pd.DataFrame): one dimensional categorical data

        Returns:
            CategoricalSupport: fitted discrete support object
        """
        self.support = set(X)
        return self

    def predict(self, x):
        return True if x in self.support else False

    def intersection(self, other_support):
        self.assert_same_type(other_support)
        return CategoricalSupport(support=self.support.intersection(other_support.support))

    def __sub__(self, other):
        return (self.support - other.support).union(other.support - self.support)


class QuantileContinuousSupport(ContinuousSupport):
    """Continuous support based on quantiles
    """

    def __init__(self, alpha=0.01, support=None):
        super().__init__(support=support)
        self.alpha = alpha

    def fit(self, X):
        """Calculate support based on quantiles

        Args:
            X (pd.DataFrame|pd.Series): one dimensional continuous valued data

        Returns:
            QuantileContinuousSupport: fitted quantile Continuous support object
        """
        self.support = list(np.quantile(
            X.values, [self.alpha/2, 1 - (self.alpha/2)]))
        return self


class UnivariateBoundingBox(BasePositivity):
    """Filter positivity by calculating univariate support
    """

    def __init__(self, quantile_alpha=0.1, continuous_columns=[], categorical_columns=[]):
        """

        Args:
            quantile_alpha (float, optional): Quantile cut-off for continuous
                variable support calculation. If not None, then the support for
                the continuous variables will be calculated using the data at 
                quantile quantile_alpha/2 as the left end and quantile 
                1 - quantile_alpha/2 on the right end. Defaults to 0.1.
            continuous_columns (List[str], optional): Column names to
                treat as Continuous variables. Defaults to None.
            categorical_columns (List[str], optional): Column names to
                treat as categorical variables. Defaults to None.

        """
        self.quantile_alpha = quantile_alpha
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns

    def fit(self, X, a):
        """Fit the propensity filter

        This fits a `Support` object for every column depending on its dtype.
        It also calculates the scales of the original data.

        Args:
            X (pd.DataFrame): covariates DataFrame
            a (pd.Series): treatment assignment Series

        Returns:
            UnivariateBoundingBox: Fitted positivity filter
        """
        self.treatment_support_ = {
            c: self.fit_column(X[a == 1][c]) for c in X.columns}
        self.control_support_ = {c: self.fit_column(
            X[a == 0][c]) for c in X.columns}
        self.joint_support_ = {c: self.treatment_support_[c].intersection(
            self.control_support_[c]) for c in X.columns}
        self.scales_ = self._calc_scales(X, a)
        return self

    def fit_column(self, Xcol):
        """Fit an individual column

        Args:
            Xcol (pd.Series|pd.DataFrame): a single column of data

        Returns:
            Support: a fitted Support object
        """
        if self._is_column_Continuous(Xcol):
            if self.quantile_alpha is None:
                return ContinuousSupport().fit(Xcol)
            else:
                return QuantileContinuousSupport(alpha=self.quantile_alpha).fit(Xcol)
        else:
            return CategoricalSupport().fit(Xcol)


    def predict(self, X, a=None):
        """Predict whether the sample is in the support for all variables

        Note that the treatment assignment vector `a` is not used with this
        method. Every sample must be in every joint support to be considered
        in the overlapped set, regardless of its treatment value.

        Args:
            X (pd.DataFrame): covariates
            a (pd.Series): treatment assignment 

        Returns:
            pd.Series: a binary series of length `X.shape[0]` with True for each
                sample determined to be in the support else False

        Raises:
            NotFittedError: if not fitted
        """
        self.assert_is_fitted()
        in_overlap_for_column = {c: s.predict for c,
                                 s in self.joint_support_.items()}
        return X.transform(in_overlap_for_column).apply(all, axis=1)

    @property
    def supports_table_(self):
        """DataFrame summarizing the fitted support variables.

        Raises:
            NotFittedError: if not fitted
        """
        self.assert_is_fitted()
        d=dict()
        for i,j,k,l in zip(
                self.treatment_support_,
                self.treatment_support_.values(),
                self.control_support_.values(),
                self.joint_support_.values()):
            d[i] = dict(treatment=j, control=k, joint=l)
        return pd.DataFrame(d).T


    @property
    def scaled_supports_table_(self):
        """DataFrame summarizing the fitted support variables in rescaled units.

        Raises:
            NotFittedError: if not fitted
        """
        self.assert_is_fitted()
        d=dict()
        def rescaler(column_name):
            def f(support):
                if isinstance(support.support,list):
                    scaled_support =  [support.support[0]/self.scales_[column_name], support.support[1]/self.scales_[column_name]]
                    return type(support)(support = scaled_support)
                else:
                    return support
            return f
        for i,j,k,l in zip(
                self.treatment_support_,
                self.treatment_support_.values(),
                self.control_support_.values(),
                self.joint_support_.values()):
            rs = rescaler(i)
            d[i] = dict(treatment=rs(j), control=rs(k), joint=rs(l))
        return pd.DataFrame(d).T

    def assert_is_fitted(self):
        """Check if filter is fitted

        Raises:
            NotFittedError: if not fitted
        """
        if not hasattr(self, "joint_support_"):
            raise NotFittedError("You must run `fit` first")

    @staticmethod
    def _calc_scales( X, a):
        n0 = X[a == 0].shape[0] - 1
        n1 = X[a == 0].shape[0] - 1
        return np.sqrt((n0*X[a == 0].var() + n1*X[a == 1].var())/(n0 + n1))

    def _is_column_Continuous(self, Xcol):
        if Xcol.name in self.categorical_columns:
            return False
        if Xcol.name in self.continuous_columns:
            return True
        if Xcol.dtype == float:
            return True
        return False
