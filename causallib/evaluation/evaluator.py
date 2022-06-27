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
from typing import Any, List, Literal, Tuple, Union, Dict
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from .plots import lookup_name, get_subplots
from ..estimation.base_weight import WeightEstimator
from ..estimation.base_estimator import IndividualOutcomeEstimator
# TODO: How doubly robust fits in to show both weight and outcome model (at least show the plots on the same figure?)


@dataclass
class EvaluationResults:
    """Data structure to hold evaluation results.
        Args:
            scores (pd.DataFrame or WeightEvaluatorScores):
            plots (dict): plot names to plot axes mapping.
                          Can be further nested by phase ("train"/"valid")
            models (list[WeightEstimator or IndividualOutcomeEstimator]): Models trained during evaluation
        """
    evaluation_metrics: Union[pd.DataFrame, Any] #really Any is WeightEvaluatorScores
    models: List[Union[WeightEstimator, IndividualOutcomeEstimator]]
    predictions: Dict[Union[Literal["train"], Literal["valid"]], List[Any]] #really Any is one of the Predictions objects
    cv: List[Tuple[List[int], List[int]]]


class BaseEvaluator:
    _numerical_classification_metrics = {"accuracy": metrics.accuracy_score,
                                         "precision": metrics.precision_score,
                                         "recall": metrics.recall_score,
                                         "f1": metrics.f1_score,
                                         "roc_auc": metrics.roc_auc_score,
                                         "avg_precision": metrics.average_precision_score,
                                         "hinge": metrics.hinge_loss,
                                         "matthews": metrics.matthews_corrcoef,
                                         "0_1": metrics.zero_one_loss,
                                         "brier": metrics.brier_score_loss}
    _nonnumerical_classification_metrics = {"confusion_matrix": metrics.confusion_matrix,
                                            "roc_curve": metrics.roc_curve,
                                            "pr_curve": metrics.precision_recall_curve}
    _classification_metrics = {**_numerical_classification_metrics, **_nonnumerical_classification_metrics}

    _regression_metrics = {"expvar": metrics.explained_variance_score,
                           "mae": metrics.mean_absolute_error,
                           "mse": metrics.mean_squared_error,
                           "msle": metrics.mean_squared_log_error,
                           # Allow mdae receive sample_weight argument but ignore it. This unifies the interface:
                           "mdae": lambda y_true, y_pred, **kwargs: metrics.median_absolute_error(y_true, y_pred),
                           "r2": metrics.r2_score}

    def __init__(self, estimator):
        """

        Args:
            estimator (causallib.estimation.base_weight.WeightEstimator | causallib.estimation.base_estimator.IndividualOutcomeEstimator):
        """
        self.estimator = estimator


    def score_binary_prediction(self, y_true, y_pred_proba=None, y_pred=None, sample_weight=None,
                                metrics_to_evaluate=None, only_numeric_metric=True):
        """Evaluates a binary prediction against true labels.

        Args:
            y_true (pd.Series): True labels
            y_pred_proba (pd.Series): continuous output of predictor (as in `predict_proba` or `decision_function`).
            y_pred (pd.Series): label (i.e., categories, decisions) predictions.
            sample_weight (pd.Series | None): weight of each sample.
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.
            only_numeric_metric (bool): If metrics_to_evaluate not provided and default is used, whether to use only
                                        numerical metrics (non-numerical are for example roc_curve, that returns vectors
                                        and not scalars).
                                        Ignored if metrics_to_evaluate is provided

        Returns:
            pd.Series: name of metric as index and the evaluated score as value.
        """
        if metrics_to_evaluate is None:
            metrics_to_evaluate = self._numerical_classification_metrics if only_numeric_metric \
                else self._classification_metrics
        scores = {}
        for metric_name, metric_func in metrics_to_evaluate.items():
            if metric_name in {"hinge", "brier", "roc_curve", "roc_auc", "pr_curve", "avg_precision"}:
                prediction = y_pred_proba
            else:
                prediction = y_pred

            if prediction is None:
                continue

            try:
                scores[metric_name] = metric_func(y_true, prediction, sample_weight=sample_weight)
            except ValueError as v:  # if y_true has single value
                warnings.warn('metric {} could not be evaluated'.format(metric_name))
                warnings.warn(str(v))
                scores[metric_name] = np.nan

        dtype = float if all([np.isscalar(score) for score in scores.values()]) else np.dtype(object)
        return pd.Series(scores, dtype=dtype)

    def score_regression_prediction(self, y_true, y_pred, sample_weight=None, metrics_to_evaluate=None):
        """Evaluates continuous prediction against true labels

        Args:
            y_true (pd.Series): True label.
            y_pred (pd.Series): Predictions.
            sample_weight (pd.Series | None): weight for each sample.
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            pd.Series: name of metric as index and the evaluated score as value.
        """
        metrics_to_evaluate = metrics_to_evaluate or self._regression_metrics
        metrics = {}
        for metric_name, metric_func in metrics_to_evaluate.items():
            try:
                metrics[metric_name] = metric_func(y_true, y_pred, sample_weight=sample_weight)
            except ValueError as v:
                metrics[metric_name] = np.nan
                warnings.warn('While evaluating ' + metric_name + ': ' + str(v))
        return pd.Series(metrics)

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
        evaluation_metrics = [results.evaluation_metrics] if isinstance(results.evaluation_metrics, pd.DataFrame) else results.evaluation_metrics
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
        evaluation_metrics = [results.evaluation_metrics] if isinstance(results.evaluation_metrics, pd.DataFrame) else results.evaluation_metrics
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

        predictions, models = self.predict_cv(X, a, y, cv, refit, phases)

        evaluation_metrics = self.score_cv(predictions, X, a, y, cv, metrics_to_evaluate)
        return_values = EvaluationResults(evaluation_metrics=evaluation_metrics,
                                          predictions=predictions,
                                          cv=cv,
                                          models=models if refit is True else [self.estimator])

        if plots is not None:
           self.plot_cv(predictions, X, a, y, cv, plots)

        return return_values

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

    def score_cv(self, predictions, X, a, y, cv, metrics_to_evaluate=None):
        """Evaluate the prediction against the true data using evaluation score metrics.

        Args:
            predictions (dict[str, list]): the output of predict_cv.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            cv (list[tuples]): list the number of folds containing tuples of indices (train_idx, validation_idx)
            metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives true labels, prediction
                                               and sample_weights (the latter is allowed to be ignored).
                                               If not provided, default are used.

        Returns:
            pd.DataFrame | WeightEvaluatorScores:
                DataFrame whose columns are different metrics and each row is a product of phase x fold x strata.
                WeightEvaluatorScores also has a covariate-balance result in a DataFrame.
        """
        phases = predictions.keys()
        scores = {phase: [] for phase in phases}
        for i, (train_idx, valid_idx) in enumerate(cv):
            data = {"train": {"X": X.iloc[train_idx], "a": a.iloc[train_idx], "y": y.iloc[train_idx]},
                    "valid": {"X": X.iloc[valid_idx], "a": a.iloc[valid_idx], "y": y.iloc[valid_idx]}}
            # TODO: use dict-comprehension to map between phases[0] to cv[0] instead writing "train" explicitly

            for phase in phases:
                X_fold, a_fold, y_fold = data[phase]["X"], data[phase]["a"], data[phase]["y"]
                prediction = predictions[phase][i]

                fold_scores = self.score_estimation(prediction, X_fold, a_fold, y_fold, metrics_to_evaluate)
                scores[phase].append(fold_scores)

        scores = self._combine_fold_scores(scores)
        return scores

    def plot_cv(self, predictions, X, a, y, cv, plots):
        """Plots prediction performance across different folds.

        Args:
            predictions (dict[str, list]): the output of predict_cv.
            X (pd.DataFrame): Covariates.
            a (pd.Series): Treatment assignment.
            y (pd.Series): Outcome.
            cv (list[tuples]): list the number of folds containing tuples of indices (train_idx, validation_idx)
            plots (list[str]): list of plots to make

        Returns:
            dict[str, dict[str, plt.Axes]]:
                {phase_name: {plot_name: plot_axes}}.
                 For example: {"train": {"roc_curve": plt.Axes, "calibration": plt.Axes}}
        """
        phases = predictions.keys()
        all_axes = {phase: {} for phase in phases}

        for phase in phases:
            phase_fig, phase_axes = get_subplots(len(plots))
            phase_axes = phase_axes.ravel()  # squeeze a vector out of the matrix-like structure of the returned fig.

            # Retrieve all indices of the different folds in the phase [idx_fold_1, idx_folds_2, ...]
            cv_idx_folds = [fold_idx[0] if phase == "train" else fold_idx[1] for fold_idx in cv]
            predictions_folds = predictions[phase]

            for i, plot_name in enumerate(plots):
                plot_data = self._get_data_for_plot(plot_name, predictions_folds, X, a, y, cv_idx_folds)
                # TODO: ^ consider _get_data_for_plot returning args (tuple) and kwargs (dictionary) which will be
                #       expanded when calling plot_func: plot_func(*plot_args, **plot_kwargs).
                #       This will allow more flexible specification of parameters by the caller
                #       (For example, Propensity Distribution with kde=True and Weight Distribution with kde=False)
                plot_func = lookup_name(plot_name)
                if plot_func is None or plot_data is None:
                    plot_ax = None
                else:
                    plot_ax = plot_func(*plot_data, cv=cv_idx_folds, ax=phase_axes[i])
                all_axes[phase][plot_name] = plot_ax
            phase_fig.suptitle("Evaluation on {} phase".format(phase))
        return all_axes

    @abc.abstractmethod
    def _estimator_fit(self, X, a, y):
        """Fit an estimator."""
        raise NotImplementedError

    @abc.abstractmethod
    def _estimator_predict(self, X, a):
        """Predict (weights, outcomes, etc. depending on the model).
        The output can be as flexible as desired, but score_estimation should know to handle it."""
        raise NotImplementedError

    @abc.abstractmethod
    def score_estimation(self, prediction, X, a_true, y_true, metrics_to_evaluate=None):
        """Should know how to handle the _estimator_predict output provided in `prediction`.
        Can utilize any of the true values provided (covariates `X`, treatment assignment `a` or outcome `y`)."""
        raise NotImplementedError

    @staticmethod
    def _combine_fold_scores(scores):
        """
        Combines scores of each phase and fold into a single object (DataFrame) of scores.

        Args:
            scores (dict[str, list[pd.DataFrame]]):
                scores of each fold of each phase. The structure is {phase_name: [fold_1_score, fold_2_score...]}.
                Where phase_name is usually "train" or "valid", and each fold_i_score is a DataFrame which columns are
                evaluation metrics and rows are results of that metrics in that fold.

        Returns:
            pd.DataFrame: Row-concatenated DataFrame with MultiIndex accounting for the concatenated folds and phases.
        """
        # Concatenate the scores from list of folds to DataFrame with rows as folds, keeping it by different phases:
        scores = {phase: pd.concat(scores_fold, axis="index", keys=range(len(scores_fold)), names=["fold"])
                  for phase, scores_fold in scores.items()}
        # Concatenate the train/validation DataFrame scores into DataFrame with rows as phases:
        scores = pd.concat(scores, axis="index", names=["phase"])
        return scores

    @abc.abstractmethod
    def _get_data_for_plot(self, plot_name, folds_predictions, X, a, y, cv):
        """Return a tuple containing the relevant data needed for the specific plot provided in `plot_name`"""
        raise NotImplementedError

    # Calculating ROC/PR curves:
    def _calculate_roc_curve_data(self, folds_predictions, targets, stratify_by=None):
        """Calculates ROC curve on the folds

        Args:
            folds_predictions (list[WeightEvaluatorPredictions | OutcomeEvaluatorPredictions]):
                list of the predictions, each entry correspond to a fold.
            targets (pd.Series): True labels.
            stratify_by (pd.Series): A vector (mostly, treatment assignment) to perform groupby with.

        Returns:
            dict[str, list[np.ndarray]]: Keys being "FPR", "TPR" and "AUC" (ROC metrics) and values are a list the size
                                         of number of folds with the evaluation of each fold.
        """
        curve_data = self._calculate_curve_data(folds_predictions, targets,
                                                metrics.roc_curve, metrics.roc_auc_score,
                                                stratify_by=stratify_by)
        for curve_name in curve_data.keys():
            curve_data[curve_name]["FPR"] = curve_data[curve_name].pop("first_ret_value")
            curve_data[curve_name]["TPR"] = curve_data[curve_name].pop("second_ret_value")
            curve_data[curve_name]["AUC"] = curve_data[curve_name].pop("area")
        return curve_data

    def _calculate_pr_curve_data(self, folds_predictions, targets, stratify_by=None):
        """Calculates precision-recall curve on the folds

        Args:
            folds_predictions (list[WeightEvaluatorPredictions | OutcomeEvaluatorPredictions]):
                list of the predictions, each entry correspond to a fold.
            targets (pd.Series): True labels.
            stratify_by (pd.Series): A vector (mostly, treatment assignment) to perform groupby with.

        Returns:
            dict[str, list[np.ndarray]]: Keys being "Precision", "Recall" and "AP" (PR metrics) and values are a list
                                         the size of number of folds with the evaluation of each fold.
                                         Additional "prevalence" key, with positive-label prevalence is added (to be
                                         used by the chance curve).
        """
        curve_data = self._calculate_curve_data(folds_predictions, targets,
                                                metrics.precision_recall_curve,
                                                metrics.average_precision_score,
                                                stratify_by=stratify_by)
        for curve_name in curve_data.keys():
            curve_data[curve_name]["Precision"] = curve_data[curve_name].pop("first_ret_value")
            curve_data[curve_name]["Recall"] = curve_data[curve_name].pop("second_ret_value")
            curve_data[curve_name]["AP"] = curve_data[curve_name].pop("area")
        curve_data["prevalence"] = targets.value_counts(normalize=True).loc[targets.max()]
        return curve_data

    @abc.abstractmethod
    def _calculate_curve_data(self, folds_predictions, targets, curve_metric, area_metric, stratify_by=None):
        """Given a list of predictions (the output of _estimator_predict by folds)
        and a vector of targets. extract (if needed)the relevant parts in each fold prediction
        and apply the curve_metric and area_metric.
        The output is nested dict: first level is the curve name and inner level dict has
        `first_ret_value` `first_ret_value`, `Thresholds` and `area` of that curve` in list
        with each entry corresponding to each fold."""
        raise NotImplementedError

    @staticmethod
    def _calculate_performance_curve_data_on_folds(folds_predictions, folds_targets, sample_weights=None,
                                                   area_metric=metrics.roc_auc_score, curve_metric=metrics.roc_curve,
                                                   pos_label=None):
        """Calculates performance curves (either ROC or precision-recall) of the predictions across folds.

        Args:
            folds_predictions (list[pd.Series]): Score prediction (as in continuous output of classifier,
                                                 `predict_proba` or `decision_function`) for every fold.
            folds_targets (list[pd.Series]): True labels for every fold.
            sample_weights (list[pd.Series] | None): weight for each sample for every fold.
            area_metric (callable): Performance metric of the area under the curve.
            curve_metric (callable): Performance metric returning 3 output vectors - metric1, metric2 and thresholds.
                                    Where metric1 and metric2 depict the curve when plotted on x-axis and y-axis.
            pos_label: What label in `targets` is considered the positive label.

        Returns:
            (list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]):
             For every fold, the calculated metric1 and metric2 (the curves), the thresholds and the area calculations.
        """
        sample_weights = [None] * len(folds_predictions) if sample_weights is None else sample_weights
        # Scikit-learn precision_recall_curve and roc_curve do not return values in a consistent way.
        # Namely, roc_curve returns `fpr`, `tpr`, which correspond to x_axis, y_axis,
        # whereas precision_recall_curve returns `precision`, `recall`, which correspond to y_axis, x_axis.
        # That's why this function will return the values the same order as the Scikit's curves, and leave it up to the
        # caller to put labels on what those return values actually are (specifically, whether they're x_axis or y-axis)
        first_ret_folds, second_ret_folds, threshold_folds, area_folds = [], [], [], []
        for fold_prediction, fold_target, fold_weights in zip(folds_predictions, folds_targets, sample_weights):
            first_ret_fold, second_ret_fold, threshold_fold = curve_metric(fold_target, fold_prediction,
                                                                           pos_label=pos_label,
                                                                           sample_weight=fold_weights)
            try:
                area_fold = area_metric(fold_target, fold_prediction, sample_weight=fold_weights)
            except ValueError as v:  # AUC cannot be evaluated if targets are constant
                warnings.warn('metric {} could not be evaluated'.format(area_metric.__name__))
                warnings.warn(str(v))
                area_fold = np.nan

            first_ret_folds.append(first_ret_fold)
            second_ret_folds.append(second_ret_fold)
            threshold_folds.append(threshold_fold)
            area_folds.append(area_fold)
        return area_folds, first_ret_folds, second_ret_folds, threshold_folds
