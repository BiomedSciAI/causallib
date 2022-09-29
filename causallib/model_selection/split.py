from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target


class TreatmentOutcomeStratifiedKFold(StratifiedKFold):
    """Creates stratified folds based on both the treatment assignment 
    and the outcome. 
    That is, every fold preserves both the treatment prevalence and
    outcome prevalence within each treatment. 
    
    For non-class outcomes, stratification is done only based on treatment. 
    
    """
    __doc__ += StratifiedKFold.__doc__

    @staticmethod
    def _combine_treatment_outcome_labels(a, y):
        """combines every `a` x `y` values as a unique label"""
        # Assuming n_a < 10, n_y < 10: labels = a*10+y. Implements a generic version.
        a_unique = np.unique(a)
        y_unique = np.unique(y)
        combinations = product(a_unique, y_unique)
        combinations_mapping = {c: i for i, c in enumerate(combinations)}
        combined_labels = [combinations_mapping[(ai, yi)] for ai, yi in zip(a, y)]
        combined_labels = pd.Series(combined_labels, index=a.index)
        return combined_labels

    def _get_labels_for_split(self, a, y):
        target_type = type_of_target(y)
        if target_type not in ("binary", "multiclass"):
            # `y` is incompatible with stratification
            raise ValueError(
                f"Outcome type should either be 'binary' or 'multiclass'."
                f"Received {target_type} instead."
            )
        labels = self._combine_treatment_outcome_labels(a, y)
        return labels

    def split(self, joinedXa, y, groups=None):
        X = joinedXa.iloc[:, :-1]
        a = joinedXa.iloc[:, -1]
        splits = self._split(X, a, y, groups=groups)
        # labels = self._get_labels_for_split(a, y)
        # splits = super().split(X, labels, groups=groups)
        return splits

    def _split(self, X, a, y, groups=None):
        """A causallib-like `X, a, y` interface for split"""
        labels = self._get_labels_for_split(a, y)
        splits = super().split(X, labels, groups=groups)
        return splits


class TreatmentStratifiedKFold(StratifiedKFold):
    """Creates stratified folds based on the treatment assignment.
    That is, every fold preserves the treatment prevalence.
    """
    __doc__ += StratifiedKFold.__doc__

    def split(self, joinedXa, y=None, groups=None):
        X = joinedXa.iloc[:, :-1]
        a = joinedXa.iloc[:, -1]
        splits = self._split(X, a, y, groups=groups)
        return splits

    def _split(self, X, a, y=None, groups=None):
        """A causallib-like `X, a, y` interface for split"""
        splits = super().split(X, a, groups=groups)
        return splits
