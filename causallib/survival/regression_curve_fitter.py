import pandas as pd
import numpy as np
from typing import Optional, Union, List
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from .survival_utils import get_person_time_df, safe_join, compute_survival_from_single_hazard_curve, \
    get_regression_predict_data
from causallib.estimation.standardization import _add_sample_weight_fit_params


class RegressionCurveFitter:
    def __init__(self, learner: SKLearnBaseEstimator):
        """
        Default implementation of a parametric survival curve fitter with covariates (pooled regression).
        API follows 'lifelines' convention for regression models, see here for example:
        https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.CoxPHFitter.fit

        Args:
            learner: scikit-learn estimator (needs to implement `predict_proba`) - compute parametric curve by fitting a
                    time-varying hazards model that includes baseline covariates. Note that the model is fitted on a
                    person-time table with all covariates, and might be computationally and memory expansive.
        """
        self.learner = learner

    def fit(self,
            df: pd.DataFrame,
            duration_col: str,
            event_col: Optional[str] = None,
            weights_col: Optional[str] = None,
            ):
        """
        Fits a parametric curve with covariates.

        Args:
            df (pd.DataFrame):  DataFrame, must contain a 'duration_col', and optional 'event_col' / 'weights_col'.
                                All other columns are treated as baseline covariates.
            duration_col (str):  Name of column with subjects' lifetimes (time-to-event)
            event_col (Optional[str]):  Name of column with event type (outcome=1, censor=0).
                                        If unspecified, assumes that all events are 'outcome' (no censoring).
            weights_col (Optional[str]):  Name of column with optional subject weights.

        Returns:
            Self
        """
        # Time to event
        durations = df[duration_col]

        # Type of event (outcome=1, censor=0). If unspecified, assumes that all events are 'outcome' (no censoring)
        event_observed = df[event_col] if event_col is not None else pd.Series(data=1, index=df.index, name='y')

        # Optional weights column
        weights = df[weights_col] if weights_col is not None else None

        self.timeline_ = np.sort(np.unique(durations))

        # Get covariates only (exclude durations, observed events and weights columns)
        X = df.drop(columns=[duration_col, event_col, weights_col], errors='ignore')

        # Get person-time data format
        pt_X, pt_w, _, pt_t, pt_y = get_person_time_df(X=X, a=None, t=durations,
                                                       y=event_observed, w=weights,
                                                       return_individual_series=True)

        # Prepare fit data
        fit_data_X, pt_t_name = safe_join(df=pt_X, list_of_series=[pt_t], return_series_names=True)
        fit_data_y = pt_y

        # Fit
        # Comply with both Pipelines and Estimators ('sample_weights' param)
        fit_params = _add_sample_weight_fit_params(estimator=self.learner, sample_weight=pt_w)
        self.learner.fit(X=fit_data_X, y=fit_data_y, **fit_params)

        return self

    def predict_survival_function(
            self,
            X: Optional[Union[pd.Series, pd.DataFrame]] = None,
            times: Optional[Union[List[float], np.ndarray, pd.Series]] = None) -> pd.DataFrame:
        """
        Predicts survival function (table) for individuals, given their covariates.
        Args:
            X (pd.DataFrame / pd.Series):  Subjects covariates
            times (Optional[Iterable]):  An iterable of increasing time points to predict cumulative hazard at.
                                         If unspecified, predict all observed time points in data.

        Returns:
            pd.DataFrame:  Each column contains a survival curve for an individual, indexed by time-steps
        """

        # Prepare prediction data
        if times is None:
            times = self.timeline_
        if not isinstance(times, pd.Series):
            times = pd.Series(times)

        if X is None:
            # Predict using times only (without covariates)
            pred_data_X = pd.DataFrame({'times': times})
            pred_data_X.index = [0] * len(pred_data_X)  # fake single subject ID
            t_name = 'times'
        else:
            # Concatenate time column to covariates
            pred_data_X, t_name = get_regression_predict_data(X, times)

        # Predict
        preds = self.learner.predict_proba(pred_data_X)[:, 1]  # array of length len(X) * len(times)

        # Convert predicted hazards into a DataFrame where columns are subject IDs and index is times
        hazards = pd.DataFrame({t_name: pred_data_X[t_name], 'hazard': preds, 'subject_id': pred_data_X.index})
        individual_hazard_curves = hazards.pivot(index=t_name, columns='subject_id', values='hazard')

        # Compute survival from hazards (per each subject individually)
        individual_survival_curves = individual_hazard_curves.transform(func=compute_survival_from_single_hazard_curve)

        # Restrict to user requested times
        individual_survival_curves = individual_survival_curves.asof(times).squeeze()

        # Set index name
        individual_survival_curves.index.name = 't'

        # Round near-zero values (may occur when all subjects "died" at some point)
        individual_survival_curves[np.abs(individual_survival_curves) < np.finfo(float).resolution] = 0

        return individual_survival_curves
