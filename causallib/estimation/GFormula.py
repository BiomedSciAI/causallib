from typing import Optional, Any
import pandas as pd
from .base_GFormula import GFormulaEstimator
from ..utils import general_tools as g_tools


class GFormulaBase(GFormulaEstimator):
    def __init__(self,
                 outcome_model,
                 treatment_model,
                 covariate_models : dict,
                 refit_weight_model=True):
        super(GFormulaBase, self).__init__(lambda **x: None)
        delattr(self, "learner")

        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.covariate_models = covariate_models
        self.refit_weight_model = refit_weight_model

    def fit(self, X, a, t, y, refit_weight_model=True, **kwargs):
        raise NotImplementedError

    def estimate_population_outcome(self, X: pd.DataFrame, a: pd.Series, t: pd.Series, y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None) -> pd.DataFrame:
        pass

    def _prepare_data(self, X, a):
        pass

    def __repr__(self):
        repr_string = g_tools.create_repr_string(self)
        # Make a new line between outcome_model and weight_model
        repr_string = repr_string.replace(", weight_model",
                                          ",\n{spaces}weight_model".format(spaces=" " * (len(self.__class__.__name__)
                                                                                         + 1)))
        return repr_string


class StandardGMethod(GFormulaBase):
    def __init__(self,
                 outcome_model,
                 treatment_model,
                 covariate_models,
                 refit_weight_model=True
                 ):

        super().__init__(outcome_model, treatment_model, covariate_models, refit_weight_model,)

    def fit(self, X, a, y, refit_weight_model=True, **kwargs):
        pass

    def predict(self, X, a):
        raise NotImplementedError("Predict is not well defined for doubly robust and thus unimplemented.")

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        pass

    def _estimate_corrected_individual_outcome(self, X, a, y, treatment_values=None, predict_proba=None):
        pass

    def estimate_population_outcome(self, X, a, y=None, treatment_values=None, predict_proba=None, agg_func="mean"):
        pass

    def estimate_effect(self, outcome1, outcome2, agg="population", effect_types="diff"):
        pass


