
import abc

from copy import deepcopy
from ..estimation.base_weight import PropensityEstimator, WeightEstimator
from ..estimation.base_estimator import IndividualOutcomeEstimator

class BasePredictor:
    @staticmethod
    def from_estimator(estimator):
        if isinstance(estimator, PropensityEstimator):
            from .weight_predictor import PropensityPredictor

            return PropensityPredictor
        if isinstance(estimator, WeightEstimator):
            from .weight_predictor import WeightPredictor

            return WeightPredictor
        if isinstance(estimator, IndividualOutcomeEstimator):
            from .outcome_predictor import OutcomePredictor

            return OutcomePredictor

    def __init__(self, estimator):
        self.estimator = estimator

    def predict_cv(self, X, a, y, cv, refit=True, phases=("train", "valid")):
        """Obtain predictions on the provided data in cross-validation

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            cv (list[tuples]): list the number of folds containing tuples of indices (train_idx, validation_idx)
            refit (bool): Whether to refit the model on each fold.
            phases (list[str]): {["train", "valid"], ["train"], ["valid"]}.
                                Phases names to evaluate on - train ("train"), validation ("valid") or both.
                                'train' corresponds to cv[i][0] and 'valid' to  cv[i][1]
        Returns:
            (dict[str, list], list): A two-tuple containing:

                * predictions: dictionary with keys being the phases provided and values are list the size of the number
                               of folds in cv and containing the output of the estimator on that corresponding fold.
                               For example, predictions["valid"][3] contains the prediction of the estimator on
                               untrained data of the third fold (i.e. validation set of the third fold)
                * models: list the size of the number of folds in cv containing of fitted estimator on the training data
                          of that fold.
        """

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
            # TODO: use dict-comprehension to map between phases[0] to cv[0] instead writing "train" explicitly

            if refit:
                self._estimator_fit(
                    X=data["train"]["X"], a=data["train"]["a"], y=data["train"]["y"]
                )

            for phase in phases:
                fold_prediction = self._estimator_predict(
                    X=data[phase]["X"], a=data[phase]["a"]
                )
                predictions[phase].append(fold_prediction)

            models.append(deepcopy(self.estimator))
        return predictions, models

    @abc.abstractmethod
    def _estimator_fit(self, X, a, y):
        """Fit an estimator."""
        raise NotImplementedError

    @abc.abstractmethod
    def _estimator_predict(self, X, a):
        """Predict (weights, outcomes, etc. depending on the model).
        The output can be as flexible as desired, but score_estimation should know to handle it."""
        raise NotImplementedError
