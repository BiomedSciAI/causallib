"""Predictor classes.

Predictors generate sets of predictions for a single fold with no cross-validation
or train-test logic.
"""

import abc
from copy import deepcopy
from typing import Union

import pandas as pd

from ..estimation.base_estimator import IndividualOutcomeEstimator
from ..estimation.base_weight import PropensityEstimator, WeightEstimator
from ..utils.stat_utils import robust_lookup

from .predictions import PropensityPredictions, WeightPredictions, OutcomePredictions


def predict_cv(estimator, X, a, y, cv, refit=True, phases=("train", "valid")):
    """Obtain predictions on the provided data in cross-validation

    Args:
        X (pd.DataFrame): Covariates.
        a (pd.Series): Treatment assignment.
        y (pd.Series): Outcome.
        cv (list[tuples]): list the number of folds containing tuples of indices
            (train_idx, validation_idx)
        refit (bool): Whether to refit the model on each fold.
        phases (list[str]): {["train", "valid"], ["train"], ["valid"]}.
            Phases names to evaluate on - train ("train"), validation ("valid") or both.
            'train' corresponds to cv[i][0] and 'valid' to  cv[i][1]
    Returns:
        (dict[str, list], list): A two-tuple containing:

            * predictions: dictionary with keys being the phases provided and values are
                list the size of the number of folds in cv and containing the output of
                the estimator on that corresponding fold.
                For example, predictions["valid"][3] contains the prediction of the estimator on
                untrained data of the third fold (i.e. validation set of the third fold)
            * models: list the size of the number of folds in cv containing the fitted estimator
                on the training data of that fold.
    """
    predictor = BasePredictor.from_estimator(estimator)(estimator)
    predictions = {phase: [] for phase in phases}
    models = []
    for train_idx, valid_idx in cv:
        data = {
            "train": {
                "X": X.iloc[train_idx],
                "a": a.iloc[train_idx],
                "y": y.iloc[train_idx],
            },
            "valid": {
                "X": X.iloc[valid_idx],
                "a": a.iloc[valid_idx],
                "y": y.iloc[valid_idx],
            },
        }
        # TODO: use dict-comprehension to map between phases[0] to cv[0]
        # instead of writing "train" explicitly

        if refit:
            predictor.fit(
                X=data["train"]["X"], a=data["train"]["a"], y=data["train"]["y"]
            )

        for phase in phases:
            fold_prediction = predictor.predict(X=data[phase]["X"], a=data[phase]["a"])
            predictions[phase].append(fold_prediction)

        models.append(deepcopy(predictor.estimator))
    return predictions, models


class BasePredictor:
    """Generate predictions from estimator for evaluation (base class)."""

    @staticmethod
    def from_estimator(
        estimator: Union[
            IndividualOutcomeEstimator, PropensityEstimator, WeightEstimator
        ]
    ):
        """Select subclass based on estimator.

        Args:
            estimator (Union[IndividualOutcomeEstimator, PropensityEstimator, WeightEstimator]):
                Estimator to generate evaluation predictions from.

        Returns:
            Union[PropensityPredictor, WeightPredictor, OutcomePredictor]: the correct predictor for
                the supplied estimator
        """
        # import outside toplevel is the price you pay for having a factory method
        # of the base class

        if isinstance(estimator, PropensityEstimator):
            return PropensityPredictor
        if isinstance(estimator, WeightEstimator):
            return WeightPredictor
        if isinstance(estimator, IndividualOutcomeEstimator):
            return OutcomePredictor
        raise ValueError(f"Received unsupported estimator type {type(estimator)}")

    def __init__(self, estimator):
        self.estimator = estimator

    @abc.abstractmethod
    def fit(self, X, a, y):
        """Fit an estimator."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X, a):
        """Predict (weights, outcomes, etc. depending on the model).
        The output can be as flexible as desired, but score_estimation should know to handle it."""
        raise NotImplementedError


class OutcomePredictor(BasePredictor):
    """Generate evaluation predictions for IndividualOutcomeEstimator models."""

    def __init__(self, estimator):
        """
        Args:
            estimator (IndividualOutcomeEstimator):
        """
        if not isinstance(estimator, IndividualOutcomeEstimator):
            raise TypeError(
                f"OutcomePredictor must be initialized with IndividualOutcomeEstimator. "
                f"Received ({ type(estimator)}) instead."
            )
        super().__init__(estimator)

    def fit(self, X, a, y):
        """Fit estimator."""
        self.estimator.fit(X=X, a=a, y=y)

    def predict(self, X, a):
        """Predict on data."""
        prediction = self.estimator.estimate_individual_outcome(
            X, a, predict_proba=False
        )
        # Use predict_probability if possible since it is needed for most evaluations:
        prediction_event_prob = self.estimator.estimate_individual_outcome(
            X, a, predict_proba=True
        )
        fold_prediction = OutcomePredictions(prediction, prediction_event_prob)
        return fold_prediction


class WeightPredictor(BasePredictor):
    """Generate evaluation predictions for WeightEstimator models."""

    def __init__(self, estimator):
        """
        Args:
            estimator (WeightEstimator):
        """
        if not isinstance(estimator, WeightEstimator):
            raise TypeError(
                "WeightPredictor must be initialized with WeightEstimator."
                f"Received got ({type(estimator)}) instead."
            )
        super().__init__(estimator)

    def fit(self, X, a, y=None):
        """Fit estimator. `y` is ignored."""
        self.estimator.fit(X=X, a=a)

    def predict(self, X, a):
        """Predict on data.

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment

        Returns:
            WeightEvaluatorPredictions
        """
        weight_by_treatment_assignment = self.estimator.compute_weights(
            X, a, treatment_values=None, use_stabilized=False
        )
        weight_for_being_treated = self.estimator.compute_weights(
            X, a, treatment_values=a.max(), use_stabilized=False
        )
        treatment_assignment_pred = self.estimator.learner.predict(
            X
        )  # TODO: maybe add predict_label to interface instead
        treatment_assignment_pred = pd.Series(treatment_assignment_pred, index=X.index)

        prediction = WeightPredictions(
            weight_by_treatment_assignment,
            weight_for_being_treated,
            treatment_assignment_pred,
        )
        return prediction


class PropensityPredictor(WeightPredictor):
    """Generate evaluation predictions for PropensityEstimator models."""

    def __init__(self, estimator):
        """
        Args:
            estimator (PropensityEstimator):
        """
        if not isinstance(estimator, PropensityEstimator):
            raise TypeError(
                "PropensityPredictor must be initialized with PropensityEstimator. "
                f"Received ({type(estimator)}) instead."
            )
        super().__init__(estimator)

    def predict(self, X, a):
        """Predict on data.

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment

        Returns:
            PropensityEvaluatorPredictions
        """
        propensity = self.estimator.compute_propensity(X, a, treatment_values=a.max())
        propensity_matrix = self.estimator.compute_propensity_matrix(X)
        propensity_by_treatment_assignment = robust_lookup(propensity_matrix, a)

        weight_prediction = super().predict(X, a)
        # Do not force stabilize=False as in WeightEvaluator:
        weight_by_treatment_assignment = self.estimator.compute_weights(X, a)
        prediction = PropensityPredictions(
            weight_by_treatment_assignment,
            weight_prediction.weight_for_being_treated,
            weight_prediction.treatment_assignment_pred,
            propensity,
            propensity_by_treatment_assignment,
        )
        return prediction
