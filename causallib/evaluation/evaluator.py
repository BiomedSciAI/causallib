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
from .metrics import Scorer, WeightEvaluatorScores, calculate_covariate_balance
from sklearn import metrics


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

    evaluated_metrics: Union[pd.DataFrame, Any]
    models: Union[List[WeightEstimator], List[IndividualOutcomeEstimator]]
    predictions: Dict[
        str, List[Any]
    ]  # really Any is one of the Predictions objects and key is "train" or "valid"
    cv: List[Tuple[List[int], List[int]]]


class WeightEvaluationResults(EvaluationResults):
    from .weight_evaluator import WeightEvaluatorPredictions

    evaluated_metrics: WeightEvaluatorScores
    models: List[WeightEstimator]
    predictions: Dict[str, List[WeightEvaluatorPredictions]]

    def _get_data_for_plot(self, plot_name, X, a, y, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots.py module.

        Args:
            plot_name (str): Plot name.
            folds_predictions (list[WeightEvaluatorPredictions]): Predictions for each fold.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Target variable - treatment assignment
            y: *IGNORED*
            cv list[np.ndarray]: Indices (in iloc positions) of each fold.

        Returns:
            tuple: Plot data
        """
        from .plots.plotters import _calculate_roc_curve_data, _calculate_pr_curve_data

        cv_idx_folds = [
            fold_idx[0] if phase == "train" else fold_idx[1] for fold_idx in self.cv
        ]
        folds_predictions = self.predictions[phase]
        if plot_name in {"weight_distribution"}:
            folds_predictions = [
                prediction.weight_for_being_treated for prediction in folds_predictions
            ]
            return folds_predictions, a
        elif plot_name in {"roc_curve"}:
            curve_data = self._calculate_curve_data(
                folds_predictions, a, metrics.roc_curve, metrics.roc_auc_score
            )
            roc_curve_data = _calculate_roc_curve_data(curve_data)
            return (roc_curve_data,)
        elif plot_name in {"pr_curve"}:
            curve_data = self._calculate_curve_data(
                folds_predictions,
                a,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
            )
            pr_curve_data = _calculate_pr_curve_data(curve_data, a)
            return (pr_curve_data,)
        elif plot_name in {"covariate_balance_love", "covariate_balance_slope"}:
            distribution_distances = []
            for fold_prediction in folds_predictions:
                fold_w = fold_prediction.weight_by_treatment_assignment
                fold_X = X.loc[fold_w.index]
                fold_a = a.loc[fold_w.index]
                dist_dist = calculate_covariate_balance(fold_X, fold_a, fold_w)
                distribution_distances.append(dist_dist)
            return (distribution_distances,)
        else:
            return None

    def _calculate_curve_data(
        self,
        folds_predictions: List[WeightEvaluatorPredictions],
        targets,
        curve_metric,
        area_metric,
        **kwargs
    ):
        """Calculate different performance (ROC or PR) curves

        Args:
            folds_predictions (list[WeightEvaluatorPredictions]): Predictions for each fold.
            targets (pd.Series): True labels
            curve_metric (callable): Performance metric returning 3 output vectors - metric1, metric2 and thresholds.
                                    Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
            area_metric (callable): Performance metric of the area under the curve.
            **kwargs:

        Returns:
            dict[str, dict[str, list[np.ndarray]]]: Evaluation of the metric for each fold and for each curve.
                2 curves:
                    * "unweighted" (regular)
                    * "weighted" (weighted by weights of each sample (according to their assignment))
                On general: {curve_name: {metric1: [evaluation_fold_1, ...]}}.
                For example: {"weighted": {"FPR": [FPR_fold_1, FPR_fold_2, FPR_fold3]}}
        """
        from .plots.plotters import _calculate_performance_curve_data_on_folds

        folds_treatment_weight = [p.weight_for_being_treated for p in folds_predictions]
        folds_targets = []
        for fold_predictions in folds_treatment_weight:
            # Since this is weight estimator, which takes the inverse of a class prediction
            fold_targets = targets.loc[fold_predictions.index]
            min_target, max_target = fold_targets.min(), fold_targets.max()
            fold_targets = fold_targets.replace(
                {
                    min_target: max_target,
                    max_target: min_target,
                }
            )
            folds_targets.append(fold_targets)

        folds_sample_weights = {
            "unweighted": [None for _ in folds_predictions],
            "weighted": [p.weight_by_treatment_assignment for p in folds_predictions],
        }
        curve_data = {}
        for curve_name, sample_weights in folds_sample_weights.items():
            (
                area,
                first_ret_value,
                second_ret_value,
                threshold_folds,
            ) = _calculate_performance_curve_data_on_folds(
                folds_treatment_weight,
                folds_targets,
                sample_weights,
                area_metric,
                curve_metric,
            )

            curve_data[curve_name] = {
                "first_ret_value": first_ret_value,
                "second_ret_value": second_ret_value,
                "Thresholds": threshold_folds,
                "area": area,
            }

        # Rename keys (as will be presented as curve labels in legend)
        curve_data["Weights"] = curve_data.pop("unweighted")
        curve_data["Weighted"] = curve_data.pop("weighted")
        return curve_data


class OutcomeEvaluationResults(EvaluationResults):
    from .outcome_evaluator import OutcomeEvaluatorPredictions

    evaluated_metrics: pd.DataFrame
    models: List[IndividualOutcomeEstimator]
    predictions: Dict[str, List[OutcomeEvaluatorPredictions]]


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
        cv = pd.RangeIndex(
            start=0, stop=X.shape[0]
        )  # All DataFrame rows when using iloc
        cv = [(cv, cv)]  # wrap in a tuple format compatible with sklearn's cv output
        results = self.evaluate_cv(
            X,
            a,
            y,
            cv=cv,
            refit=False,
            phases=phases,
            metrics_to_evaluate=metrics_to_evaluate,
            plots=plots,
        )

        # Remove redundant information accumulated due to the use of cross-validation process
        results.models = results.models[0]
        evaluation_metrics = (
            [results.evaluated_metrics]
            if isinstance(results.evaluated_metrics, pd.DataFrame)
            else results.evaluated_metrics
        )
        for metric in evaluation_metrics:
            metric.reset_index(level=["phase", "fold"], drop=True, inplace=True)

        return results

    def evaluate_bootstrap(
        self,
        X,
        a,
        y,
        n_bootstrap,
        n_samples=None,
        replace=True,
        refit=False,
        metrics_to_evaluate=None,
    ):
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
        X_ilocs = pd.RangeIndex(
            start=0, stop=X.shape[0]
        )  # All DataFrame rows when using iloc
        for i in range(n_bootstrap):
            # Get iloc positions of a bootstrap sample (sample the size of X with replacement):
            # idx = X.sample(n=X.shape[0], replace=True).index
            # idx = np.random.random_integers(low=0, high=X.shape[0], size=X.shape[0])
            idx = np.random.choice(X_ilocs, size=n_samples, replace=replace)
            cv.append(
                (idx, idx)
            )  # wrap in a tuple format compatible with sklearn's cv output

        results = self.evaluate_cv(
            X,
            a,
            y,
            cv=cv,
            refit=refit,
            phases=phases,
            metrics_to_evaluate=metrics_to_evaluate,
            plots=None,
        )

        # Remove redundant information accumulated due to the use of cross-validation process:
        results.models = (
            results.models[0] if len(results.models) == 1 else results.models
        )
        evaluation_metrics = (
            [results.evaluated_metrics]
            if isinstance(results.evaluated_metrics, pd.DataFrame)
            else results.evaluated_metrics
        )
        for metric in evaluation_metrics:
            metric.reset_index(level=["phase"], drop=True, inplace=True)
            metric.index.rename("sample", "fold", inplace=True)
        return results

    def evaluate_cv(
        self,
        X,
        a,
        y,
        cv=None,
        kfold=None,
        refit=True,
        phases=("train", "valid"),
        metrics_to_evaluate=None,
        plots=None,
    ):
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
            cv = list(
                cv
            )  # if cv is generator it would listify it, if cv is already a list this is idempotent
        else:
            kfold = kfold or StratifiedKFold(n_splits=5)
            cv = list(kfold.split(X=X, y=a))

        predictions, models = self.predictor.predict_cv(X, a, y, cv, refit, phases)

        evaluation_metrics = self.scorer.score_cv(
            predictions, X, a, y, cv, metrics_to_evaluate
        )
        return_values = EvaluationResults(
            evaluated_metrics=evaluation_metrics,
            predictions=predictions,
            cv=cv,
            models=models if refit is True else [self.predictor.estimator],
        )

        if plots is not None:
            self.plotter.plot_cv(predictions, X, a, y, cv, plots)

        return return_values
