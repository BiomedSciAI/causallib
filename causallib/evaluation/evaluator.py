"""
Evaluator object for evaluating causal inference models.

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


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .plots.helpers import plot_evaluation_results
from .predictor import predict_cv
from .results import EvaluationResults
from .scoring import score_cv


def _make_dummy_cv(N):
    phases = ["train"]  # dummy phase
    # All DataFrame rows when using iloc
    cv = pd.RangeIndex(start=0, stop=N)
    cv = [(cv, cv)]  # wrap in a tuple format compatible with sklearn's cv output
    return phases, cv


def _make_bootstrap_cv(N, n_bootstrap, n_samples, replace):
    # Evaluation using bootstrap
    phases = ["train"]  # dummy phase

    # Generate bootstrap sample:
    cv = []
    # All DataFrame rows when using iloc
    X_ilocs = pd.RangeIndex(start=0, stop=N)
    for _ in range(n_bootstrap):
        # Get iloc positions of a bootstrap sample (sample the size of X with replacement):
        # idx = X.sample(n=X.shape[0], replace=True).index
        # idx = np.random.random_integers(low=0, high=X.shape[0], size=X.shape[0])
        idx = np.random.choice(X_ilocs, size=n_samples, replace=replace)
        # wrap in a tuple format compatible with sklearn's cv output
        cv.append((idx, idx))
    return phases, cv


def evaluate(
    estimator,
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
    if cv is None:
        return evaluate_simple(estimator, X, a, y, metrics_to_evaluate, plots)
    # when evaluate_cv gets cv=None it makes an auto cv so turn "auto" to None
    if cv == "auto":
        cv = None

    return evaluate_cv(
        estimator=estimator,
        X=X,
        a=a,
        y=y,
        cv=cv,
        kfold=kfold,
        refit=refit,
        phases=phases,
        metrics_to_evaluate=metrics_to_evaluate,
        plots=plots,
    )


def evaluate_simple(estimator, X, a, y, metrics_to_evaluate=None, plots=None):
    """Evaluate model on the provided data without cross-validation or bootstrap.

    Simple evaluation without cross validation on the provided data can be to test
    the model on its train data or on new data.

    Args:
        X (pd.DataFrame): Covariates.
        a (pd.Series): Treatment assignment.
        y (pd.Series): Outcome.
        metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
            If not provided, defaults from causallib.evaluation.metrics are used.
        plots (list[str] | None): list of plots to make. If None, none are generated.

    Returns:
        causallib.evaluation.results.EvaluationResults
    """

    phases, cv = _make_dummy_cv(X.shape[0])
    results = evaluate_cv(
        estimator,
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


def evaluate_cv(
    estimator,
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
        cv (list[tuples] | generator[tuples]): list the number of folds containing tuples of
            indices (train_idx, validation_idx) in an iloc manner (row number).
        kfold(sklearn.model_selection.BaseCrossValidator): Initialized fold object (e.g. KFold).
            defaults to StratifiedKFold of 5 splits on treatment.
        refit (bool): Whether to refit the model on each fold.
        phases (list[str]): {["train", "valid"], ["train"], ["valid"]}.
            Phases names to evaluate on - train ("train"), validation ("valid") or both.
            'train' corresponds to cv[i][0] and 'valid' to  cv[i][1]
        metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives
            true labels, prediction, and sample_weights (the latter is allowed to be ignored).
            If not provided, defaults from `causallib.evaluation.metrics` are used.
        plots (list[str] | None): list of plots to make. If None, none are generated.

    Returns:
        EvaluationResults
    """
    # We need consistent splits for predicting, scoring and plotting.
    # If cv is a generator, it would be lost after after first use.
    # If kfold has shuffle=True, it would be inconsistent.
    # To keep consistent reproducible folds, we save them as a list.
    if cv is None:

        kfold = kfold or StratifiedKFold(n_splits=5)
        cv = kfold.split(X=X, y=a)

    # if cv is generator it would listify it, if cv is already a list this is idempotent
    cv = list(cv)

    predictions, models = predict_cv(estimator, X, a, y, cv, refit, phases)
    evaluation_metrics = score_cv(predictions, X, a, y, cv, metrics_to_evaluate)
    evaluation_results = EvaluationResults(
        evaluated_metrics=evaluation_metrics,
        predictions=predictions,
        cv=cv,
        models=models if refit is True else [estimator],
    )

    if plots is not None:
        plot_evaluation_results(evaluation_results, X, a, y, plots)

    return evaluation_results


def evaluate_bootstrap(
    estimator,
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
        metrics_to_evaluate (dict | None): key: metric's name, value: callable that receives
            true labels, prediction and sample_weights (the latter is allowed to be ignored).
            If not provided, default from causallib.evaluation.metrics are used.

    Returns:
        EvaluationResults
    """
    if n_samples is None:
        n_samples = X.shape[0]

    phases, cv = _make_bootstrap_cv(X.shape[0], n_bootstrap, n_samples, replace)

    results = evaluate_cv(
        estimator,
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
    results.models = results.models[0] if len(results.models) == 1 else results.models
    evaluation_metrics = (
        [results.evaluated_metrics]
        if isinstance(results.evaluated_metrics, pd.DataFrame)
        else results.evaluated_metrics
    )
    for metric in evaluation_metrics:
        metric.reset_index(level=["phase"], drop=True, inplace=True)
        metric.index.rename("sample", "fold", inplace=True)
    return results
