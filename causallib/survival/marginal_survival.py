from .weighted_survival import WeightedSurvival
from typing import Any


class MarginalSurvival(WeightedSurvival):
    """
    Marginal (un-adjusted) survival estimator.
    Essentially it is a degenerated WeightedSurvival instance without a weight model.
    """
    def __init__(self,
                 survival_model: Any = None):
        """
        Marginal (un-adjusted) survival estimator.
        Args:
            survival_model: Three alternatives:
                    1. None - compute non-parametric KaplanMeier survival curve
                    2. Scikit-Learn estimator (needs to implement `predict_proba`) - compute parametric curve by fitting a time-varying hazards model
                    3. lifelines UnivariateFitter - use lifelines fitter to compute survival curves from events and durations
        """

        super().__init__(weight_model=None, survival_model=survival_model)

