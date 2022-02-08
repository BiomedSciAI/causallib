import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from .survival_utils import safe_join
from .regression_curve_fitter import RegressionCurveFitter


class UnivariateCurveFitter:
    def __init__(self, learner: Optional[SKLearnBaseEstimator] = None):
        """
        Default implementation of a univariate survival curve fitter.
        Construct a curve fitter, either non-parametric (Kaplan-Meier) or parametric.
        API follows 'lifelines' convention for univariate models, see here for example:
        https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter.fit
        Args:
            learner: optional scikit-learn estimator (needs to implement `predict_proba`). If provided, will
                     compute parametric curve by fitting a time-varying hazards model. if None, will compute
                     non-parametric Kaplan-Meier estimator.
        """
        self.learner = learner

    def fit(self, durations, event_observed=None, weights=None):
        """
        Fits a univariate survival curve (Kaplan-Meier or parametric, if a learner was provided in constructor)

        Args:
            durations (Iterable):  Duration subject was observed
            event_observed (Optional[Iterable]):  Boolean or 0/1 iterable, where True means 'outcome event' and False
                                                  means 'right censoring'. If unspecified, assumes that all events are
                                                  'outcome' (no censoring).
            weights (Optional[Iterable]):  Optional subject weights

        Returns:
            Self
        """
        # If 'event_observed' is unspecified, assumes that all events are 'outcome' (no censoring).
        if event_observed is None:
            event_observed = pd.Series(data=1, index=durations.index)

        if weights is None:
            weights = pd.Series(data=1, index=durations.index, name='weights')
        else:
            weights = pd.Series(data=weights, index=durations.index, name='weights')
        self.timeline_ = np.sort(np.unique(durations))

        # If sklearn classifier is provided, fit parametric curve
        if self.learner is not None:
            self.curve_fitter_ = RegressionCurveFitter(learner=self.learner)
            fit_data, (duration_col_name, event_col_name, weights_col_name) = safe_join(
                df=None, list_of_series=[durations, event_observed, weights], return_series_names=True
            )
            self.curve_fitter_.fit(df=fit_data, duration_col=duration_col_name, event_col=event_col_name,
                                   weights_col=weights_col_name)

        # Else, compute Kaplan Meier estimator non parametrically
        else:
            # Code inspired by lifelines KaplanMeierFitter
            df = pd.DataFrame({
                't': durations,
                'removed': weights.to_numpy(),
                'observed': weights.to_numpy() * (event_observed.to_numpy(dtype=bool))
            })

            death_table = df.groupby("t").sum()
            death_table['censored'] = (death_table['removed'] - death_table['observed']).astype(int)

            births = pd.DataFrame(np.zeros(durations.shape[0]), columns=["t"])
            births['entrance'] = np.asarray(weights)
            births_table = births.groupby("t").sum()
            event_table = death_table.join(births_table, how="outer", sort=True).fillna(
                0)  # http://wesmckinney.com/blog/?p=414
            event_table['at_risk'] = event_table['entrance'].cumsum() - event_table['removed'].cumsum().shift(1).fillna(
                0)
            self.event_table_ = event_table

        return self

    def predict(self, times=None, interpolate=False):
        """
        Compute survival curve for time points given in 'times' param.
        Args:
            times: sequence of time points for prediction
            interpolate: if True, linearly interpolate non-observed times. Otherwise, repeat last observed time point.

        Returns:
            pd.Series: with times index and survival values

        """
        if times is None:
            times = self.timeline_
        else:
            times = sorted(times)

        if self.learner is not None:
            # Predict parametric survival curve
            survival = self.curve_fitter_.predict_survival_function(X=None, times=pd.Series(times))
        else:
            # Compute hazard at each time step
            hazard = self.event_table_['observed'] / self.event_table_['at_risk']
            timeline = hazard.index  # if computed non-parametrically, timeline is all observed data points
            # Compute survival from hazards
            survival = pd.Series(data=np.cumprod(1 - hazard), index=timeline, name='survival')

        if interpolate:
            survival = pd.Series(data=np.interp(times, survival.index.values, survival.values),
                                 index=pd.Index(data=times, name='t'), name='survival')
        else:
            survival = survival.asof(times).squeeze()

        # Round near-zero values (may occur when using weights and all observed subjects "died" at some point)
        survival[np.abs(survival) < np.finfo(float).resolution] = 0

        return survival
