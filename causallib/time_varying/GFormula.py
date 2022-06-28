#!/usr/bin/env python3

from typing import Optional, Any
import pandas as pd
from causallib.time_varying.base_GFormula import TimeVaryingBaseEstimator


class GFormulaBase(TimeVaryingBaseEstimator):
    def __init__(self,
                 outcome_model,
                 treatment_model,
                 covariate_models: dict,
                 refit_models=True):
        super(GFormulaBase, self).__init__(lambda **x: None)
        delattr(self, "learner")

        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.covariate_models = covariate_models
        self.refit_models = refit_models

    def fit(self, X, a, t, y, refit_weight_model=True, **kwargs):
        raise NotImplementedError

    def estimate_population_outcome(self, X: pd.DataFrame, a: pd.Series, t: pd.Series, y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None) -> pd.DataFrame:
        pass

    def _prepare_data(self, X, a):
        pass


class GFormula(GFormulaBase):
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



