"""
Methods for evaluating causal inference models.

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

from .predictions import PropensityEvaluatorScores
from .predictor import predict_cv
from .results import EvaluationResults
from .scoring import score_cv

def _make_dummy_cv(n_samples):
    phases = ["train"]  # dummy phase
    # All DataFrame rows when using iloc
    cv = pd.RangeIndex(start=0, stop=n_samples)
    cv = [(cv, cv)]  # wrap in a tuple format compatible with sklearn's cv output
    return phases, cv


def _make_bootstrap_cv(n_samples_total, n_bootstrap, n_samples_bootstrap, replace):
    # Evaluation using bootstrap
    phases = ["train"]  # dummy phase

    # Generate bootstrap sample:
    cv = []
    # All DataFrame rows when using iloc
    X_ilocs = pd.RangeIndex(start=0, stop=n_samples_total)
    for _ in range(n_bootstrap):
        # Get iloc positions of a bootstrap sample (sample the size of X with replacement):
        # idx = X.sample(n=X.shape[0], replace=True).index
        # idx = np.random.random_integers(low=0, high=X.shape[0], size=X.shape[0])
        idx = np.random.choice(X_ilocs, size=n_samples_bootstrap, replace=replace)
        # wrap in a tuple format compatible with sklearn's cv output
        cv.append((idx, idx))
    return phases, cv


def evaluate(
    estimator,
    X,
    a,
    y,
    cv=None,
    metrics_to_evaluate="defaults",
    plots=False,
):
    """Evaluate model in cross-validation of the provided data

    Args:
        estimator (causallib.estimation.base_estimator.IndividualOutcomeEstimator |
            causallib.estimation.base_weight.WeightEstimator |
            causallib.estimation.base_weight.PropensityEstimator) : an estimator. If using cv, it
            will be refit, otherwise it should already be fit.
        X (pd.DataFrame): Covariates.
        a (pd.Series): Treatment assignment.
        y (pd.Series): Outcome.
        cv (list[tuples] | generator[tuples] | None): list the number of folds containing tuples of
            indices (train_idx, validation_idx) in an iloc manner (row number).
            If None, there will be no cross-validation. If `cv="auto"`, a stratified Kfold with
            5 folds will be created and used for cross-validation.
        metrics_to_evaluate (dict | "defaults" | None): key: metric's name, value: callable that
            receives true labels, prediction, and sample_weights (the latter may be ignored).
            If `"defaults"`, default metrics are selected. If `None`, no metrics are evaluated.
        plots (bool): whether to generate plots

    Returns:
        EvaluationResults
    """

    if cv is None:
        return _evaluate_simple(estimator, X, a, y, metrics_to_evaluate, plots)
    # when evaluate_cv gets cv=None it makes an auto cv so turn "auto" to None
    if cv == "auto":
        cv = None

    return _evaluate_cv(
        estimator=estimator,
        X=X,
        a=a,
        y=y,
        cv=cv,
        refit=True,
        phases=("train", "valid"),
        metrics_to_evaluate=metrics_to_evaluate,
        plots=plots,
    )


def _evaluate_simple(estimator, X, a, y, metrics_to_evaluate=None, plots=False):
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
    results = _evaluate_cv(
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

    results.remove_spurious_cv()

    return results


def _evaluate_cv(
    estimator,
    X,
    a,
    y,
    cv=None,
    refit=True,
    phases=("train", "valid"),
    metrics_to_evaluate=None,
    plots=False,
):
    """Evaluate model in cross-validation of the provided data

    Args:
        X (pd.DataFrame): Covariates.
        a (pd.Series): Treatment assignment.
        y (pd.Series): Outcome.
        cv (list[tuples] | generator[tuples]): list the number of folds containing tuples of
            indices (train_idx, validation_idx) in an iloc manner (row number).
        refit (bool): Whether to refit the model on each fold.
        phases (list[str]): {["train", "valid"], ["train"], ["valid"]}.
            Phases names to evaluate on - train ("train"), validation ("valid") or both.
            'train' corresponds to cv[i][0] and 'valid' to  cv[i][1]
        metrics_to_evaluate (dict | None): key: metric's name, value: callable that
            receives true labels, prediction, and sample_weights (the latter may be ignored).
        plots (bool): whether to generate plots

    Returns:
        EvaluationResults
    """
    # We need consistent splits for predicting, scoring and plotting.
    # If cv is a generator, it would be lost after after first use.
    # If kfold has shuffle=True, it would be inconsistent.
    # To keep consistent reproducible folds, we save them as a list.
    if cv is None:
        kfold = StratifiedKFold(n_splits=5)
        cv = kfold.split(X=X, y=a)

    # if cv is generator it would listify it, if cv is already a list this is idempotent
    cv = list(cv)

    predictions, models = predict_cv(estimator, X, a, y, cv, refit, phases)
    if metrics_to_evaluate is not None:
        evaluated_metrics = score_cv(predictions, X, a, y, cv, metrics_to_evaluate)
    else:
        evaluated_metrics = None
    evaluation_results = EvaluationResults.make(
        evaluated_metrics=evaluated_metrics,
        predictions=predictions,
        cv=cv,
        models=models if refit is True else [estimator],
        X=X,
        a=a,
        y=y,
    )

    if plots:
        evaluation_results.plot_all()
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
    n_samples_total = X.shape[0]

    if n_samples is None:
        n_samples = n_samples_total

    phases, cv = _make_bootstrap_cv(n_samples_total, n_bootstrap, n_samples, replace)

    results = _evaluate_cv(
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

    
    results.remove_spurious_cv()
    if results.evaluated_metrics is not None:
        if isinstance(results.evaluated_metrics, PropensityEvaluatorScores):
            results.evaluated_metrics.covariate_balance.index.rename("sample", "fold", inplace=True)
            results.evaluated_metrics.prediction_scores.index.rename("sample", "fold", inplace=True)
        else:
            results.evaluated_metrics.index.rename("sample", "fold", inplace=True)
    
    return results
