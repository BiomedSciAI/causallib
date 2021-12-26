import pandas as pd
from abc import ABC, abstractmethod


class SurvivalBase(ABC):
    """
    Interface class for causal survival analysis with fixed baseline covariates.
    """
    @abstractmethod
    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: pd.Series):
        """
        Fits internal learner(s).

        Args:
            X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            t (pd.Series): Followup duration, size (num_subjects,).
            y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_population_outcome(self,
                                    **kwargs) -> pd.DataFrame:
        """
        Returns population averaged survival curves.

        Returns:
            pd.DataFrame: with time-step index, treatment values as columns and survival as entries
        """
        raise NotImplementedError


class SurvivalTimeVaryingBase(SurvivalBase):
    """
    Interface class for causal survival analysis estimators that support time-varying followup covariates.
    Followup covariates matrix (XF) needs to have a 'time' column, and indexed by subject IDs that correspond to
    the other inputs (X, a, y, t). All columns other than 'time' will be used for time-varying adjustments.

    Example XF format:
    +----+------+------+------+
    | id |   t  | var1 | var2 |
    +----+------+------+------+
    |  1 |    0 |  1.4 |   22 |
    |  1 |    4 |  1.2 |   22 |
    |  1 |    8 |  1.5 |  NaN |
    |  2 |    0 |  1.6 |   10 |
    |  2 |   11 |  1.6 |   11 |
    +----+------+------+------+
    """

    @abstractmethod
    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: pd.Series,
            XF: pd.DataFrame = None) -> None:
        """
        Fits internal survival functions.

        Args:
            X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            t (pd.Series): Followup duration, size (num_subjects,).
            y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).
            XF (pd.DataFrame): Time-varying followup covariate matrix

        Returns:
            A fitted estimator with precalculated survival functions.
        """
