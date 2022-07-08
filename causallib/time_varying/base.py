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
               pd.DataFrame: with time-step index, treatment (a) as columns and treatment values as entries
        """
        raise NotImplementedError


class GMethodBase(TimeVaryingBaseEstimator):
    """
        GFormula base Estimator
    """
    def __init__(self,
                 treatment_model: Any,
                 covariate_models: OrderedDict,
                 outcome_model: Optional[Any]=None,
                 refit_models=True):
        """
            treatment_model (???):
            covariate_models (OrderedDict): {
                        ‘x1’: Standardization(LogisticRegression()),
                        ‘x2’: Standardization(LinearRegression()),
                            so forth
                        },
            outcome_model(IndividualOutcomeEstimator): A causal model that estimate on individuals level
                                                      (e.g. Standardization).
            refit_models (bool): if True, re-fit the treatment model and covariate models.

        """
        super(GMethodBase, self).__init__(lambda **x: None)
        self.treatment_model = treatment_model
        self.covariate_models = covariate_models
        self.outcome_model = outcome_model
        self.refit_models = refit_models


    @abc.abstractmethod
    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: Optional[pd.Series] = None,
            y: Optional[Any] = None,
            refit_models: bool = True,
            **kwargs
            ):
        """
            Fits parametric models and calculates prediction curves.

            Args:
                X (pd.DataFrame): Baseline covariate matrix of size (num_subjects * number_time_points, num_features).
                a (pd.Series): Treatment assignment of size (num_subjects * number_time_points,).
                              Currently, only supports single treatment.  
                t (pd.Series): Followup duration, size (num_subjects,).
                y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,). When not provided, index of X is used
                refit_models (bool): if True, re-fit the treatment, covariate, and outcome models (if any).
                kwargs (dict): Optional kwargs for fit call of GMethodBase model

            Returns:
                self
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_individual_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: Optional[pd.Series] = None,
                                    y: Optional[Any] = None,
                                    treatment_strategy: Callable = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
            Returns individual estimated curves for each subject row in X/a/t

            Args:
                X (pd.DataFrame): Baseline covariate matrix of size (num_subjects * number_time_points, num_features).
                a (pd.Series): Treatment assignment of size (num_subjects * number_time_points,).
                              Currently, only supports single treatment.  
                t (pd.Series): Followup durations, size (num_subjects,).
                y: NOT USED (for API compatibility only).
                treatment_strategy (Callable): A Callable class that computes the treatment outcome based on
                                            the strategy implemented.
                                            e.g. Treatment_Strategy from causallib.time_varying.treatment_strategy

                timeline_start (int): Common start time-step. If provided, will generate simulations starting
                                      from 'timeline_start' for all patients.
                                      If None, will predict from first observed event (t.min()).
                timeline_end (int): Common end time-step. If provided, will generate simulations up to 'timeline_end'
                                    for all patients.
                                    If None, will predict up to last observed event (t.max()).

            Returns:
                pd.DataFrame: with time-step index, covariate, treatment, and outcome (if any) as columns, and their corresponding values  as entries. Index over the simulation period
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    treatment_strategy: Callable = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
            Returns Population estimated curves.

            Args:
                X (pd.DataFrame): Baseline covariate matrix of size (num_subjects * number_time_points, num_features).
                a (pd.Series): Treatment assignment of size (num_subjects * number_time_points,).
                              Currently, only supports single treatment.  
                t (pd.Series): Followup durations, size (num_subjects,).
                y: NOT USED (for API compatibility only).
                treatment_strategy (Callable): A Callable class that computes the treatment outcome based on
                                            the strategy implemented.
                                            e.g. Treatment_Strategy from causallib.time_varying.treatment_strategy
                timeline_start (int): Common start time-step. If provided, will generate simulations starting
                                      from 'timeline_start' for all patients.
                                      If None, will predict from first observed event (t.min()).
                timeline_end (int): Common end time-step. If provided, will generate simulations up to 'timeline_end'
                                    for all patients.
                                    If None, will predict up to last observed event (t.max()).

            Returns:
                pd.DataFrame: with time-step index, covariate, treatment, and outcome (if any) as columns, and their corresponding values  as entries. Index over the simulation period
        """
        raise NotImplementedError





