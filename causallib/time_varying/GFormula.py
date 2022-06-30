#!/usr/bin/env python3

import abc
import pandas as pd
from typing import Optional, Any, OrderedDict
from .base_GFormula import TimeVaryingBaseEstimator


class GFormulaBase(TimeVaryingBaseEstimator):
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
            covariate_models (dict): {
                        ‘x1’: Standardization(LogisticRegression()),
                        ‘x2’: Standardization(LinearRegression()),
                            so forth
                        },
            refit_models (bool): if True, re-fit a the treatment model and covariate models.

        """
        super(GFormulaBase, self).__init__(lambda **x: None)
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
            refit_models=True,
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
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
              Returns individual estimated curves for each subject row in X/a/t

              Args:
                  X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                  a (pd.Series): Treatment assignment of size (num_subjects,).
                  t (pd.Series): Followup durations, size (num_subjects,).
                  y: NOT USED (for API compatibility only).
                  timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                        from 'timeline_start' for all patients.
                                        If None, will predict from first observed event (t.min()).
                  timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                      for all patients.
                                      If None, will predict up to last observed event (t.max()).

              Returns:
                  pd.DataFrame: with time-step index, subject IDs (X.index) as columns and ??
              """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
              Returns individual estimated curves for each subject row in X/a/t

              Args:
                  X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                  a (pd.Series): Treatment assignment of size (num_subjects,).
                  t (pd.Series): Followup durations, size (num_subjects,).
                  y: NOT USED (for API compatibility only).
                  timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                        from 'timeline_start' for all patients.
                                        If None, will predict from first observed event (t.min()).
                  timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                      for all patients.
                                      If None, will predict up to last observed event (t.max()).

              Returns:
                  pd.DataFrame: with time-step index, subject IDs (X.index) as columns and ??
              """
        raise NotImplementedError

    @abc.abstractmethod
    def apply_noise(self):
        pass

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
        pass


class GFormula(GFormulaBase):
    """
        GFormula class that is based on Monte Carlo Simulation for creating noise. #TODO more on this
    """
    def __init__(self,
                 outcome_model,
                 treatment_model,
                 covariate_models,
                 refit_models=True
                 ):
        super().__init__(outcome_model, treatment_model,
                         covariate_models, refit_models)

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: Optional[Any] = None,
            refit_models=True,
            **kwargs
            ):

        if kwargs is None:
            kwargs = {}

        #TODO preprocess data to fit in the model
        if refit_models:
            self.treatment_model.fit(X, a, y, **kwargs)

            for cov in self.covariate_models:
                self.covariate_models[cov].fit(X, a, y, **kwargs)

        self.outcome_model.fit(X, a, y, **kwargs)
        return self


    def estimate_individual_outcome(self, X: pd.DataFrame, a: pd.Series, t: pd.Series, y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None) -> pd.DataFrame:

        min_time = timeline_start if timeline_start is not None else int(t.min())
        max_time = timeline_end if timeline_end is not None else int(t.max())

        contiguous_times = pd.Series(data=range(min_time, max_time + 1), name=t.name)  # contiguous time steps for inference
        unique_treatment_values = a.unique()
        res = pd.DataFrame()
        # TODO
        # logic to get the prediction curve for individual treatment types

        return res

    def estimate_population_outcome(self, X: pd.DataFrame, a: pd.Series, t: pd.Series, y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None) -> pd.DataFrame:

        unique_treatment_values = a.unique()
        res = {}
        for treatment_value in unique_treatment_values:
            assignment = pd.Series(data=treatment_value, index=X.index, name=a.name)
            individual_survival_curves = self.estimate_individual_outcome(X=X, a=assignment, t=t,
                                                                          timeline_start=timeline_start,
                                                                          timeline_end=timeline_end)
            res[treatment_value] = individual_survival_curves.mean(axis='columns')
        res = pd.DataFrame(res)

        # Setting index/column names
        res.index.name = t.name
        res.columns.name = a.name
        return res

    def apply_noise(self):
        pass



