from abc import ABC, abstractmethod
from causallib.estimation.base_estimator import EffectEstimator
import pandas as pd
from typing import Any, Optional


class TimeVariantBaseEstimator(ABC):

    @abstractmethod
    def fit(self, X, a, t, y):
        raise NotImplementedError

    @abstractmethod
    def estimate_population_outcome(self, X, a, t, y, **kwargs):
        raise NotImplementedError


class GFormulaEstimator(TimeVariantBaseEstimator, EffectEstimator):
    @abstractmethod
    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: pd.Series,
            XF: pd.DataFrame = None) -> None:
        pass

    @staticmethod
    def _aggregate_population_outcome(y, agg_func="mean"):
        if agg_func == "mean":
            return y.mean()
        elif agg_func == "median":
            return y.median()
        # TODO: consider adding max and min aggregation
        else:
            raise LookupError("Not supported aggregation function ({})".format(agg_func))

    def estimate_effect(self, outcome1, outcome2, agg="population", effect_types="diff"):
        if agg == "population":
            outcome1 = self._aggregate_population_outcome(outcome1)
            outcome2 = self._aggregate_population_outcome(outcome2)
        effect = super(GFormulaEstimator, self).estimate_effect(outcome1, outcome2, effect_types)
        return effect

