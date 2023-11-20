from .survival_utils import canonize_dtypes_and_names
from .standardized_survival import StandardizedSurvival
from causallib.estimation.base_weight import WeightEstimator
import pandas as pd
from typing import Any, Optional


class WeightedStandardizedSurvival(StandardizedSurvival):
    def __init__(
        self,
        weight_model: WeightEstimator,
        survival_model: Any,
        stratify: bool = True,
        outcome_covariates=None,
        weight_covariates=None,
    ):
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
            outcome_covariates (array): Covariates to use for outcome model.
                                        If None - all covariates passed will be used.
                                        Either list of column names or boolean mask.
            weight_covariates (array): Covariates to use for weight model.
                                       If None - all covariates passed will be used.
                                       Either list of column names or boolean mask.
        """
        self.weight_model = weight_model
        super().__init__(survival_model=survival_model, stratify=stratify)
        self.outcome_covariates = outcome_covariates
        self.weight_covariates = weight_covariates

    def _prepare_data(self, X, *args, **kwargs):
        """
        Extract the relevant parts for outcome model and weight model for the entire data matrix

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).

        Returns:
            (pd.DataFrame, pd.DataFrame): X_outcome, X_weight
                Data matrix for outcome model and data matrix weight model
        """
        outcome_covariates = X.columns if self.outcome_covariates is None else self.outcome_covariates
        X_outcome = X[outcome_covariates]
        weight_covariates = X.columns if self.weight_covariates is None else self.weight_covariates
        X_weight = X[weight_covariates]
        return X_outcome, X_weight

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
        X_outcome, X_weight = self._prepare_data(X)

        self.weight_model.fit(X=X_weight, a=a, y=y)
        iptw_weights = self.weight_model.compute_weights(X_weight, a)

        # Call fit from StandardizedSurvival, with added ipt weights
        super().fit(X=X_outcome, a=a, t=t, y=y, w=iptw_weights, fit_kwargs=fit_kwargs)
        return self

    def estimate_individual_outcome(
        self,
        X: pd.DataFrame,
        a: pd.Series,
        t: pd.Series,
        y: Optional[Any] = None,
        timeline_start: Optional[int] = None,
        timeline_end: Optional[int] = None
    ) -> pd.DataFrame:
        X_outcome, _ = self._prepare_data(X)
        potential_outcomes = super().estimate_individual_outcome(
            X_outcome,
            a, t, y,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
        )
        return potential_outcomes
