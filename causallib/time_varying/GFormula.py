#!/usr/bin/env python3

import pandas as pd
from typing import Optional, Any, Callable
from causallib.time_varying.base import GMethodBase
from causallib.utils import general_tools as g_tools


class GFormula(GMethodBase):
    """
        GFormula class that is based on Monte Carlo Simulation for creating the noise.
    """

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: Optional[Any] = None,
            refit_models: bool = True,
            **kwargs
            ):

        raise NotImplementedError

        if kwargs is None:
            kwargs = {}

        #TODO More to work on preparing data to be fed into the model
        treatment_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.treatment_model.learner)
        if refit_models or treatment_model_is_not_fitted:
            self.treatment_model.fit(X, a, y, **kwargs)

        for cov in self.covariate_models:
            cov_model = self.covariate_models[cov]
            cov_model_is_not_fitted = not g_tools.check_learner_is_fitted(cov_model.learner)

            if refit_models or cov_model_is_not_fitted:
                cov_model.fit(X, a, y, **kwargs)

        self.outcome_model.fit(X, a, y, **kwargs)
        return


    def estimate_individual_outcome(self, X: pd.DataFrame, a: pd.Series, t: pd.Series, y: Optional[Any] = None,
                                    treatment_strategy: Callable = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None) -> pd.DataFrame:

        raise NotImplementedError

        min_time = timeline_start if timeline_start is not None else int(t.min())
        max_time = timeline_end if timeline_end is not None else int(t.max())

        contiguous_times = pd.Series(data=range(min_time, max_time + 1), name=t.name)  # contiguous time steps for inference
        unique_treatment_values = a.unique()
        res = pd.DataFrame()

        # TODO logic to get the prediction curve for individual treatment types
        return res

    def estimate_population_outcome(self, X: pd.DataFrame, a: pd.Series, t: pd.Series, y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None) -> pd.DataFrame:

        raise NotImplementedError

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

    def _apply_noise(self):
        pass

    def _prepare_data(self, X, a, t, y):
        pass

    @staticmethod
    def _predict_trajectory(self, X, a, t) -> pd.DataFrame:
        pass



