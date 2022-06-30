#!/usr/bin/env python3

from abc import ABC, abstractmethod

from causallib.estimation.base_estimator import IndividualOutcomeEstimator


class TimeVaryingBaseEstimator(ABC, IndividualOutcomeEstimator):
    """
    Interface class for Time Varying analysis with fixed baseline covariates.
    """
    @abstractmethod
    def fit(self, X, a, t, y):
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
    def estimate_population_outcome(self, X, a, t, y, **kwargs):
        """
            Returns population averaged estimated curves.

            Returns:
               pd.DataFrame: with time-step index, co-variates as columns and co-variates' values as entries
        """
        raise NotImplementedError


