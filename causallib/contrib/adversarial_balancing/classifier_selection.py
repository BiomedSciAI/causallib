# (C) Copyright 2019 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created on Oct 30, 2019

import sklearn
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict, ParameterGrid
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import numpy as np


def select_classifier(model, X, A, n_splits=5, loss_type='01', seed=None):
    """Utility for selecting best classifier using cross-validation.

    Args:
        model: Either one of: scikit-learn classifier, scikit-learn SearchCV model (GridSearchCV, RandomizedSearchCV),
                              list of classifiers.
        X (np.ndarray): Covariate matrix size (num_samples, num_features)
        A (np.ndarray): binary labels indicating the source and target populations (num_samples,)
        n_splits (int): number of splits in cross-validation. relevant only if list of classifiers is passed.
        loss_type (str): name of loss metric to select classifier by. Either '01' for zero-one loss, otherwise
                         cross-entropy is used (and classifiers must implement predict_proba).
                         relevant only if list of classifiers is passed.
        seed (int): random seed for cross-validation split. relevant only if list of classifiers is passed.

    Returns:
        classifier: best performing classifier on validation set.
    """
    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        selected_model = _select_classifier_from_sk_search(model, X, A)
    elif isinstance(model, list):
        selected_model = _select_classifier_from_list(candidates=model, X=X, A=A, n_splits=n_splits, seed=seed,
                                                      loss_type=loss_type)
    elif isinstance(model, dict):
        selected_model = _select_classifier_from_grid(X=X, A=A, n_splits=n_splits, seed=seed, **model,
                                                      loss_type=loss_type)
    else:  # A regular classifier was passed
        selected_model = model
    return selected_model


def _select_classifier_from_sk_search(estimator, X, A):
    """Return best model from a scikit-learn Search-estimator model.

    Args:
        estimator (GridSearchCV | RandomizedSearchCV): An initialized sklearn SearchCV classifier.
        X (np.ndarray): Covariate matrix size (num_samples, num_features)
        A (np.ndarray): binary labels indicating the source and target populations (num_samples,)

    Returns:
        classifier: model.best_estimator_ - best-performing classifier.
                    See scikit-learn's GridSearchCV and RandomizedSearchCV documentation for details on their return
                    values.
    """
    estimator.fit(X, A)
    best_estimator = clone(estimator.best_estimator_)
    return best_estimator


def _select_classifier_from_grid(estimator, X, A, param_grid, n_splits=5, seed=1, loss_type='01'):
    candidates = []
    for params in ParameterGrid(param_grid):
        estimator2 = clone(estimator)
        for key, value in params.items():
            setattr(estimator2, key, value)
        candidates.append(estimator2)

    return _select_classifier_from_list(candidates, X, A, n_splits=n_splits, seed=seed, loss_type=loss_type)


def _select_classifier_from_list(candidates, X, A, n_splits=5, seed=None, loss_type='01'):
    accuracies = np.zeros(len(candidates))

    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(A), y=A)[LabelEncoder().fit_transform(A)]

    if n_splits >= 2:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for model_idx, m in enumerate(candidates):
            if sklearn.__version__ >= "1.4":
                # TODO: at time of writing scikit-learn 1.4.0 is <1 year old.
                #       Once matured, you may erase the deprecated `fit_params` and just use `params`.
                if loss_type == '01':
                    pred = cross_val_predict(
                        m, X=X, y=A,
                        cv=cv, params={'sample_weight': class_weight}
                    ).reshape(-1)
                else:
                    ps = cross_val_predict(
                        m, X=X, y=A,
                        cv=cv, params={'sample_weight': class_weight},
                        method='predict_proba'
                    )
                    pred = ps[:, 1]
            else:
                if loss_type == '01':
                    pred = cross_val_predict(m, X=X, y=A, cv=cv, fit_params={'sample_weight': class_weight}).reshape(-1)
                else:
                    ps = cross_val_predict(m, X=X, y=A, cv=cv, fit_params={'sample_weight': class_weight},
                                           method='predict_proba')
                    pred = ps[:, 1]
    else:
        for model_idx, m in enumerate(candidates):
            m.fit(X, A, sample_weight=class_weight)
            if loss_type == '01':
                pred = m.predict(X=X)
            else:
                pred = m.predict_proba(X=X)[:, 1]

    if loss_type == '01':
        accuracies[model_idx] = np.sum(class_weight[pred == A]) / np.sum(class_weight)
    else:
        logl = np.zeros(A.shape)
        logl[A == -1] = np.log(1.0 - pred[A == -1])
        logl[A == 1] = np.log(pred[A == 1])
        accuracies[model_idx] = np.sum(class_weight * logl) / np.sum(class_weight)

    i_best = np.argmax(accuracies)
    # print('accuracies =', accuracies, "accuracies-sorted", sorted(accuracies))
    # print('Selected model {} {}'.format(i_best, candidates[i_best]))
    return candidates[i_best]
