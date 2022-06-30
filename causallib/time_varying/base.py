#!/usr/bin/env python3

import abc
import pandas as pd
from abc import abstractmethod
from typing import Optional, Any, OrderedDict, Callable

from causallib.estimation.base_estimator import IndividualOutcomeEstimator


class TimeVaryingBaseEstimator(IndividualOutcomeEstimator):
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


class BaseGMethod(TimeVaryingBaseEstimator):
    """
        GFormula base Estimator
    """
    def __init__(self,
                 outcome_model: None,
                 treatment_model: Any,
                 covariate_models: OrderedDict,
                 refit_models=True):
        """
            outcome_model(IndividualOutcomeEstimator): A causal model that estimate on individuals level
                                                      (e.g. Standardization).
            treatment_model (???):
            covariate_models (OrderedDict): {
                        ‘x1’: Standardization(LogisticRegression()),
                        ‘x2’: Standardization(LinearRegression()),
                            so forth
                        },
            refit_models (bool): if True, re-fit the treatment model and covariate models.

        """
        super().__init__(outcome_model, treatment_model,
                         covariate_models, refit_models)
        super(BaseGMethod, self).__init__(lambda **x: None)
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.covariate_models = covariate_models
        self.refit_models = refit_models


    @abc.abstractmethod
    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: Optional[Any] = None,
            refit_models: bool = True,
            **kwargs
            ):
        """
            Fits parametric models and calculates prediction curves.

            Args:
                X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                a (pd.Series): Treatment assignment of size (num_subjects,).
                t (pd.Series): Followup duration, size (num_subjects,).
                y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).
                refit_models (bool): if True, re-fit a the treatment model and covariate models.
                kwargs (dict): Optional kwargs for fit call of survival model

            Returns:
                self
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_individual_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    treatment_strategy: Callable,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
              Returns individual estimated curves for each subject row in X/a/t

              Args:
                  X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                  a (pd.Series): Treatment assignment of size (num_subjects,).
                                Currently, only supports single treatment.
                                Future work will be to expand single treatment to multiple treatments.
                  t (pd.Series): Followup durations, size (num_subjects,).
                  y: NOT USED (for API compatibility only).
                  treatment_strategy (callable): A function that describes the treatment strategy.
                                                eg.
                  timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                        from 'timeline_start' for all patients.
                                        If None, will predict from first observed event (t.min()).
                  timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                      for all patients.
                                      If None, will predict up to last observed event (t.max()).

              Returns:
                  pd.DataFrame: with time-step index, treatment (a) as columns and treatment values as entries
              """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    treatment_strategy: Callable,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
              Returns Population estimated curves.

              Args:
                  X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                  a (pd.Series): Treatment assignment of size (num_subjects,).
                  t (pd.Series): Followup durations, size (num_subjects,).
                                Currently, only supports single treatment.
                                Future work will be to expand single treatment to multiple treatments.
                  treatment_strategy (callable): A function that describes the treatment strategy.
                                                eg.
                  y: NOT USED (for API compatibility only).
                  timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                        from 'timeline_start' for all patients.
                                        If None, will predict from first observed event (t.min()).
                  timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                      for all patients.
                                      If None, will predict up to last observed event (t.max()).

              Returns:
                  pd.DataFrame: with time-step index, treatment (a) as columns and treatment values as entries
              """
        raise NotImplementedError

    def _prepare_data(self, X, a, t, y):
        return pd.DataFrame()


    @staticmethod
    def _predict_trajectory(self, X, a, t) -> pd.DataFrame:
        """ Predicts the trajectories for all covariates X and treatment a.

            Args:
                  X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                  a (pd.Series): Treatment assignment of size (num_subjects,).
                  t (pd.Series): Followup durations, size (num_subjects,).

            Returns:
                 pd.DataFrame: with time-step index, subject IDs (X.index and a) columns and
                               point values for each column as entries
        """
        raise NotImplementedError





