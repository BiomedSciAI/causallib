from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.metaestimators import _safe_split
import pandas as pd
from sklearn.base import clone


def cross_fitting(estimator, X, y, n_splits=5, predict_proba=False,
                  return_estimator=True):
    """

    Args:
        estimator(object): sklearn object
        X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
        y (pd.Series): Observed outcome of size (num_subjects,).
        n_splits (int): number of folds
        predict_proba (bool): If True, the treatment model is a classifier
                                and use 'predict_proba',
                              If False, use 'predict'.
        return_estimator (bool): If true return fitted estimators of each fold

    Returns:
        array of held-out prediction,
        if return estimator:
            a tuple of estimators on held-out-data
    """

    cv = StratifiedKFold(n_splits=n_splits) if predict_proba else KFold(
        n_splits=n_splits)
    ret = [_fit_and_predict(clone(estimator), X, y, train, test,
                            predict_proba=predict_proba)
           for train, test in cv.split(X, y)]
    zipped_ret = list(zip(*ret))
    if return_estimator:
        return pd.concat(zipped_ret[0]), zipped_ret[1]
    else:
        return pd.concat(zipped_ret[0])


def _fit_and_predict(estimator, X, y, train, test, predict_proba):
    """
    fit the estimator with the train samples and make prediction with the test data
    Args:
        estimator(object): sklearn object
        X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
        y (pd.Series): Observed outcome of size (num_subjects,).
        train:
        test:
        predict_proba (bool): If True, the treatment model is a classifier
                                and use 'predict_proba',
                              If False, use 'predict'.

    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)
    estimator.fit(X_train, y_train)
    if predict_proba:
        pred = estimator.predict_proba(X_test)[:, 1]
    else:
        pred = estimator.predict(X_test)

    return pd.Series(pred, index=X_test.index), estimator
