"""
(C) Copyright 2019 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Dec 25, 2018

"""

import abc
from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Dict
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ..estimation.base_weight import PropensityEstimator, WeightEstimator
from ..estimation.base_estimator import IndividualOutcomeEstimator
from .metrics import Scorer
from .plots import Plotter

# TODO: How doubly robust fits in to show both weight and outcome model (at least show the plots on the same figure?)


@dataclass
class EvaluationResults:
    """Data structure to hold evaluation results.
        Args:
            evaluated_metrics (pd.DataFrame or WeightEvaluatorScores):
            models (list[WeightEstimator or IndividualOutcomeEstimator]): Models trained during evaluation
            predictions (dict[str, List[Predictions]])
            cv (list[tuple[list[int], list[int]]])
        """
    evaluated_metrics: Union[pd.DataFrame, Any] #really Any is WeightEvaluatorScores
    models: List[Union[WeightEstimator, IndividualOutcomeEstimator]]
    predictions: Dict[str, List[Any]] #really Any is one of the Predictions objects and key is "train" or "valid"
    cv: List[Tuple[List[int], List[int]]]

    def get_data_for_plot(self, plot_name, X, a, y, phase="train"):
        # Retrieve all indices of the different folds in the phase [idx_fold_1, idx_folds_2, ...]
        cv_idx_folds = [
                fold_idx[0] if phase == "train" else fold_idx[1] for fold_idx in self.cv
            ]
        predictions_folds = self.predictions[phase]


            

class Predictor:
    @staticmethod
    def from_estimator(estimator):
        if isinstance(estimator, PropensityEstimator):
            from .weight_evaluator import PropensityPredictor
            return PropensityPredictor
        if isinstance(estimator, WeightEstimator):
            from .weight_evaluator import WeightPredictor
            return WeightPredictor
        if isinstance(estimator, IndividualOutcomeEstimator):
            from .outcome_evaluator import OutcomePredictor
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
            data = {"train": {"X": X.iloc[train_idx], "a": a.iloc[train_idx], "y": y.iloc[train_idx]},
                    "valid": {"X": X.iloc[valid_idx], "a": a.iloc[valid_idx], "y": y.iloc[valid_idx]}}
            # TODO: use dict-comprehension to map between phases[0] to cv[0] instead writing "train" explicitly

            if refit:
                self._estimator_fit(X=data["train"]["X"], a=data["train"]["a"], y=data["train"]["y"])

            for phase in phases:
                fold_prediction = self._estimator_predict(X=data[phase]["X"], a=data[phase]["a"])
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



class Evaluator:


    def __init__(self, estimator):
        """

        Args:
            estimator (causallib.estimation.base_weight.WeightEstimator | causallib.estimation.base_estimator.IndividualOutcomeEstimator):
        """
        self.predictor = Predictor.from_estimator(estimator)(estimator)
        self.scorer = Scorer()
        self.plotter = Plotter.from_estimator(estimator)()



    def evaluate_simple(self, X, a, y, metrics_to_evaluate=None, plots=None):
        """Evaluate model on the provided data

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.
            plots (list[str] | None): list of plots to make. If None, none are generated.

        Returns:
            EvaluationResults
        """
        # simple evaluation without cross validation on the provided data
        # (can be to test the model on its train data or on new data

        phases = ["train"]  # dummy phase
        cv = pd.RangeIndex(start=0, stop=X.shape[0])  # All DataFrame rows when using iloc
        cv = [(cv, cv)]  # wrap in a tuple format compatible with sklearn's cv output
        results = self.evaluate_cv(X, a, y, cv=cv, refit=False, phases=phases,
                                   metrics_to_evaluate=metrics_to_evaluate, plots=plots)

        # Remove redundant information accumulated due to the use of cross-validation process
        results.models = results.models[0]
        evaluation_metrics = [results.evaluated_metrics] if isinstance(results.evaluated_metrics, pd.DataFrame) else results.evaluated_metrics
        for metric in evaluation_metrics:
            metric.reset_index(level=["phase", "fold"], drop=True, inplace=True)

        return results

    def evaluate_bootstrap(self, X, a, y, n_bootstrap, n_samples=None, replace=True,
                           refit=False, metrics_to_evaluate=None):
        """Evaluate model on a bootstrap sample of the provided data

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            n_bootstrap (int): Number of bootstrap sample to create.
            n_samples (int | None): Number of samples to sample in each bootstrap sampling.
                                    If None - will use the number samples (first dimension) of the data.
            replace (bool): Whether to use sampling with replacements.
                            If False - n_samples (if provided) should be smaller than X.shape[0])
            refit (bool): Whether to refit the estimator on each bootstrap sample.
                          Can be computational intensive if n_bootstrap is large.
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            EvaluationResults
        """
        n_samples = n_samples or X.shape[0]
        # Evaluation using bootstrap
        phases = ["train"]  # dummy phase

        # Generate bootstrap sample:
        cv = []
        X_ilocs = pd.RangeIndex(start=0, stop=X.shape[0])  # All DataFrame rows when using iloc
        for i in range(n_bootstrap):
            # Get iloc positions of a bootstrap sample (sample the size of X with replacement):
            # idx = X.sample(n=X.shape[0], replace=True).index
            # idx = np.random.random_integers(low=0, high=X.shape[0], size=X.shape[0])
            idx = np.random.choice(X_ilocs, size=n_samples, replace=replace)
            cv.append((idx, idx))  # wrap in a tuple format compatible with sklearn's cv output

        results = self.evaluate_cv(X, a, y, cv=cv, refit=refit, phases=phases,
                                   metrics_to_evaluate=metrics_to_evaluate, plots=None)

        # Remove redundant information accumulated due to the use of cross-validation process:
        results.models = results.models[0] if len(results.models) == 1 else results.models
        evaluation_metrics = [results.evaluated_metrics] if isinstance(results.evaluated_metrics, pd.DataFrame) else results.evaluated_metrics
        for metric in evaluation_metrics:
            metric.reset_index(level=["phase"], drop=True, inplace=True)
            metric.index.rename("sample", "fold", inplace=True)
        return results

    def evaluate_cv(self, X, a, y, cv=None, kfold=None, refit=True, phases=("train", "valid"),
                    metrics_to_evaluate=None, plots=None):
        """Evaluate model in cross-validation of the provided data

        Args:
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            cv (list[tuples] | generator[tuples]): list the number of folds containing tuples of indices
                                                   (train_idx, validation_idx) in an iloc manner (row number).
            kfold(sklearn.model_selection.BaseCrossValidator): Initialized fold object (e.g. KFold).
                                                               defaults to StratifiedKFold of 5 splits on treatment.
            refit (bool): Whether to refit the model on each fold.
            phases (list[str]): {["train", "valid"], ["train"], ["valid"]}.
                                Phases names to evaluate on - train ("train"), validation ("valid") or both.
                                'train' corresponds to cv[i][0] and 'valid' to  cv[i][1]
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.
            plots (list[str] | None): list of plots to make. If None, none are generated.

        Returns:
            EvaluationResults
        """
        # There's a need to have consistent splits for predicting, scoring and plotting.
        # If cv is a generator, it would be lost after after first use. if kfold has shuffle=True, it would be
        # inconsistent. In order to keep consistent reproducible folds across the process, we save them as a list.
        if cv is not None:
            cv = list(cv)  # if cv is generator it would listify it, if cv is already a list this is idempotent
        else:
            kfold = kfold or StratifiedKFold(n_splits=5)
            cv = list(kfold.split(X=X, y=a))

        predictions, models = self.predictor.predict_cv(X, a, y, cv, refit, phases)

        evaluation_metrics = self.scorer.score_cv(predictions, X, a, y, cv, metrics_to_evaluate)
        return_values = EvaluationResults(evaluated_metrics=evaluation_metrics,
                                          predictions=predictions,
                                          cv=cv,
                                          models=models if refit is True else [self.predictor.estimator])

        if plots is not None:
           self.plotter.plot_cv(predictions, X, a, y, cv, plots)

        return return_values





