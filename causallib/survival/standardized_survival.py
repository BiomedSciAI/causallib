from .regression_curve_fitter import RegressionCurveFitter
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from typing import Any, Optional
import pandas as pd
from copy import deepcopy
from .survival_utils import canonize_dtypes_and_names, safe_join
from .base_survival import SurvivalBase


class StandardizedSurvival(SurvivalBase):
    def __init__(self,
                 survival_model: Any,
                 stratify: bool = True, **kwargs):
        """
        Standardization survival estimator.
        Computes parametric curve by fitting a time-varying hazards model that includes baseline covariates.
        Args:
            survival_model: Two alternatives:
                1. Scikit-Learn estimator (needs to implement `predict_proba`) - compute parametric curve by fitting a
                    time-varying hazards model that includes baseline covariates. Note that the model is fitted on a
                    person-time table with all covariates, and might be computationally and memory expansive.
                2. lifelines RegressionFitter - use lifelines fitter to compute survival curves from baseline covariates,
                    events and durations
            stratify (bool): if True, fit a separate model per treatment group
        """
        self.stratify = stratify

        if isinstance(survival_model, SKLearnBaseEstimator):
            # Construct default curve fitter, parametric with a scikit-learn estimator
            self.survival_model = RegressionCurveFitter(survival_model)
        else:
            # Initialized lifelines RegressionFitter (or any implementation with a compatible API)
            self.survival_model = survival_model

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: pd.Series,
            w: Optional[pd.Series] = None,
            fit_kwargs: Optional[dict] = None):
        """
        Fits parametric models and calculates internal survival functions.

        Args:
            X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            t (pd.Series): Followup duration, size (num_subjects,).
            y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).
            w (pd.Series): Optional subject weights.
            fit_kwargs (dict): Optional kwargs for fit call of survival model

        Returns:
            self
        """
        a, t, y, w, X = canonize_dtypes_and_names(a=a, t=t, y=y, w=w, X=X)
        if w is not None:
            fit_data, (w_name,) = safe_join(df=X, list_of_series=[w], return_series_names=True)
        else:
            fit_data = X
            w_name = None

        if fit_kwargs is None:
            fit_kwargs = {}

        self.stratified_curve_fitters_ = {}
        if self.stratify:
            fit_data, (t_name, y_name) = safe_join(df=fit_data, list_of_series=[t, y], return_series_names=True)
            unique_treatment_values = a.unique()
            for treatment_value in unique_treatment_values:
                stratum_curve_fitter = deepcopy(self.survival_model)
                stratum_curve_fitter.fit(df=fit_data[a == treatment_value], duration_col=t_name, event_col=y_name,
                                         weights_col=w_name, **fit_kwargs)
                self.stratified_curve_fitters_[treatment_value] = stratum_curve_fitter
        else:
            fit_data, (a_name, t_name, y_name) = safe_join(df=fit_data, list_of_series=[a, t, y],
                                                           return_series_names=True)
            self.survival_model.fit(df=fit_data, duration_col=t_name, event_col=y_name, weights_col=w_name,
                                    **fit_kwargs)

        return self

    def estimate_individual_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
        Returns individual survival curves for each subject row in X/a/t

        Args:
            X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            t (pd.Series): Followup durations, size (num_subjects,).
            y: NOT USED (for API compatibility only).
            timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                  from 'timeline_start' for all patients. If None, will predict from first observed event (t.min()).
            timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                for all patients. If None, will predict up to last observed event (t.max()).

        Returns:
            pd.DataFrame: with time-step index, subject IDs (X.index) as columns and point survival as entries
        """

        min_time = timeline_start if timeline_start is not None else int(t.min())
        max_time = timeline_end if timeline_end is not None else int(t.max())
        contiguous_times = pd.Series(data=range(min_time, max_time + 1),
                                     name=t.name)  # contiguous time steps for inference

        a, _, _, _, X = canonize_dtypes_and_names(a=a, w=None, X=X)
        unique_treatment_values = sorted(a.unique())
        res = {}
        for treatment_value in unique_treatment_values:
            if self.stratify:
                predict_data = X
                model = self.stratified_curve_fitters_[treatment_value]
            else:
                assignment = pd.Series(treatment_value, index=a.index, name=a.name)
                predict_data, a_name = safe_join(
                    df=X, list_of_series=[assignment],
                    return_series_names=True
                )
                model = self.survival_model
            treatment_individual_survival_curves = model.predict_survival_function(
                X=predict_data,
                times=contiguous_times
            )
            res[treatment_value] = treatment_individual_survival_curves
        res = pd.concat(res, axis="columns", names=[a.name])
        return res

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
        Returns population averaged survival curves.

        Args:
            X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            t (pd.Series): Followup durations, size (num_subjects,).
            y: NOT USED (for API compatibility only).
            timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                  from 'timeline_start' for all patients. If None, will predict from first observed event (t.min()).
            timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                for all patients. If None, will predict up to last observed event (t.max()).

        Returns:
            pd.DataFrame: with time-step index, treatment values as columns and survival as entries
        """
        a, t, _, _, X = canonize_dtypes_and_names(a=a, t=t,  X=X)
        individual_survival_curves = self.estimate_individual_outcome(
            X=X, a=a, t=t,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
        )
        res = individual_survival_curves.T.groupby(level=0).mean().T

        # Setting index/column names
        res.index.name = t.name
        res.columns.name = a.name
        return res
