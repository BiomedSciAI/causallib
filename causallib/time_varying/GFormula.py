#!/usr/bin/env python3

from typing import Optional, Any
import pandas as pd
from causallib.time_varying.base_GFormula import TimeVaryingBaseEstimator


class GFormulaBase(TimeVaryingBaseEstimator):
    """
        GFormula base Estimator
    """
    def __init__(self,
                 outcome_model,
                 treatment_model,
                 covariate_models: dict,
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
        delattr(self, "learner")

        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.covariate_models = covariate_models
        self.refit_models = refit_models

    def fit(self, X, a, t, y, refit_models=True, **kwargs):
        """ TODO --- Referenced from standardized_survival
            Fits parametric models and calculates internal survival functions.

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

    def estimate_population_outcome(self, X: pd.DataFrame, a: pd.Series, t: pd.Series, y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None) -> pd.DataFrame:
        """   TODO --- Referenced from standardized_survival
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
        pass

    def _prepare_data(self, X, a, t):
        """
            placeholder method to clean and process the inputs: (X, a, t)
            Args:
                X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                a (pd.Series): Treatment assignment of size (num_subjects,).
                t (pd.Series): Followup durations, size (num_subjects,).

            Returns:
                #TODO ?
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

        super().__init__(outcome_model, treatment_model, covariate_models, refit_models)

    def fit(self, X, a, t,  y, refit_models=True, **kwargs):
        pass

    def predict(self, X, a, t):
        raise NotImplementedError("Predict is not well defined for doubly robust and thus unimplemented.")



