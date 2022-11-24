from causallib.estimation.base_weight import WeightEstimator
from .univariate_curve_fitter import UnivariateCurveFitter
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from typing import Any
import pandas as pd
from copy import deepcopy
from .survival_utils import canonize_dtypes_and_names
from .base_survival import SurvivalBase
from typing import Optional


class WeightedSurvival(SurvivalBase):
    """
    Weighted survival estimator
    """

    def __init__(self,
                 weight_model: WeightEstimator = None,
                 survival_model: Any = None):
        """
        Weighted survival estimator.
        Args:
            weight_model: causallib compatible weight model (e.g., IPW)
            survival_model: Three alternatives:
                1. None - compute non-parametric KaplanMeier survival curve
                2. Scikit-Learn estimator (needs to implement `predict_proba`) - compute parametric curve by fitting a
                    time-varying hazards model
                3. lifelines UnivariateFitter - use lifelines fitter to compute survival curves from events and durations
        """
        self.weight_model = weight_model

        # Construct default curve fitter, non parametric estimation (Kaplan-Meier)
        if survival_model is None:
            self.survival_model = UnivariateCurveFitter()
        # Construct default curve fitter, parametric with a scikit-learn estimator
        elif isinstance(survival_model, SKLearnBaseEstimator):
            self.survival_model = UnivariateCurveFitter(survival_model)
        # Initialized lifelines univariate fitter (or any implementation with a compatible API)
        else:
            self.survival_model = survival_model

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series = None,
            y: pd.Series = None,
            fit_kwargs: Optional[dict] = None):
        """
        Fits internal weight module (e.g. IPW module, adversarial weighting, etc).

        Args:
            X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            t (pd.Series): NOT USED (for compatibility only)
            y (pd.Series): NOT USED (for compatibility only)
            fit_kwargs (dict): Optional kwargs for fit call of survival model (NOT USED, since fit
                               call of survival model occurs in 'estimate_population_outcome' rather than here)

        Returns:
            self
        """
        a, _, y, _, X = canonize_dtypes_and_names(a=a, t=None, y=y, w=None, X=X)
        if self.weight_model is not None:
            self.weight_model.fit(X=X, a=a, y=y)

        return self

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: pd.Series,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
        Returns population averaged survival curves.

        Args:
            X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            t (pd.Series|int): Followup durations, size (num_subjects,).
            y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).
            timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                  from 'timeline_start' for all patients. If None, will predict from first observed event.
            timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                for all patients. If None, will predict up to last observed event.

        Returns:
            pd.DataFrame: with timestep index, treatment values as columns and survival as entries
        """
        self.stratified_curve_fitters_ = {}
        a, t, y, _, X = canonize_dtypes_and_names(a=a, t=t, y=y, w=None, X=X)
        min_time = timeline_start if timeline_start is not None else int(t.min())
        max_time = timeline_end if timeline_end is not None else int(t.max())

        if self.weight_model is not None:
            # Generate inverse propensity for treatment weights (IPTW)
            iptw_weights = self.weight_model.compute_weights(X, a)
            iptw_weights.name = 'w'
        else:
            iptw_weights = None

        # Fit or compute survival curves
        treatment_values = a.unique()
        survival_curves = []
        for treatment_value in treatment_values:
            stratum_indices = a == treatment_value
            stratum_curve_fitter = deepcopy(self.survival_model)

            # Fit curve model
            stratum_curve_fitter.fit(durations=t[stratum_indices], event_observed=y[stratum_indices],
                                     weights=iptw_weights[stratum_indices] if iptw_weights is not None else None)
            self.stratified_curve_fitters_[treatment_value] = stratum_curve_fitter

            # Predict curve model
            curve = stratum_curve_fitter.predict(times=range(min_time, max_time + 1))
            curve.rename(treatment_value, inplace=True)
            survival_curves.append(curve)

        res = pd.concat(survival_curves, axis=1)

        # Setting index/column names
        res.index.name = t.name
        res.columns.name = a.name
        return res
