from .survival_utils import canonize_dtypes_and_names
from .standardized_survival import StandardizedSurvival
from causallib.estimation.base_weight import WeightEstimator
import pandas as pd
from typing import Any, Optional


class WeightedStandardizedSurvival(StandardizedSurvival):
    """
    Combines WeightedSurvival and StandardizedSurvival:
        1. Adjusts for treatment assignment by creating weighted pseudo-population (e.g., inverse propensity weighting).
        2. Computes parametric curve by fitting a time-varying hazards model that includes baseline covariates.
    """

    def __init__(self,
                 weight_model: WeightEstimator,
                 survival_model: Any,
                 stratify: bool = True):
        """
        Combines WeightedSurvival and StandardizedSurvival:
            1. Adjusts for treatment assignment by creating weighted pseudo-population (e.g., inverse propensity weighting).
            2. Computes parametric curve by fitting a time-varying hazards model that includes baseline covariates.
        Args:
            weight_model: causallib compatible weight model (e.g., IPW)
           survival_model: Two alternatives:
                1. Scikit-Learn estimator (needs to implement `predict_proba`) - compute parametric curve by fitting a
                    time-varying hazards model that includes baseline covariates. Note that the model is fitted on a
                    person-time table with all covariates, and might be computationally and memory expansive.
                2. lifelines RegressionFitter - use lifelines fitter to compute survival curves from baseline covariates,
                    events and durations
            stratify (bool): if True, fit a separate model per treatment group
        """
        self.weight_model = weight_model
        super().__init__(survival_model=survival_model, stratify=stratify)

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
            w (pd.Series): NOT USED (for compatibility only) optional subject weights.
            fit_kwargs (dict): Optional kwargs for fit call of survival model

        Returns:
            self
        """
        a, t, y, _, X = canonize_dtypes_and_names(a=a, t=t, y=y, w=None, X=X)
        self.weight_model.fit(X=X, a=a, y=y)
        iptw_weights = self.weight_model.compute_weights(X, a)

        # Call fit from StandardizedSurvival, with added ipt weights
        super().fit(X=X, a=a, t=t, y=y, w=iptw_weights, fit_kwargs=fit_kwargs)
        return self


