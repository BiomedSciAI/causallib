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

Created on Apr 25, 2018

"""
from typing import Mapping
import inspect

import pandas as pd
from numpy import isscalar
from sklearn.base import clone as sk_clone
from sklearn.preprocessing import OneHotEncoder

from .base_estimator import IndividualOutcomeEstimator
from ..utils import general_tools as g_tools


def _standardization_predict(estimator, X, predict_proba):
    """
    Single prediction call.

    Args:
        estimator: Fitted scikit estimator to use for prediction.
        X (pd.DataFrame): Data to predict on.
        predict_proba (bool): If applicable in the estimator (classification model) and if True - predict a continuous
                              value utilizing `predict_proba` or `decision_function`, rather than classifying with
                              `predict`.

    Returns:
        pd.DataFrame | pd.Series: If regression model or predict_proba=False then it returns a vector-like array (with
                                  prediction for each sample). If classification and predict_proba=True then returns
                                  a matrix-like array of n_samples by n_classes.
    """
    # Predict continuous values for classification if desired (predict_proba=True) and applicable (model has
    # `predict_proba` or `decision_function` methods to utilize:
    if predict_proba and hasattr(estimator, "predict_proba"):
        prediction = estimator.predict_proba(X)
    elif predict_proba and hasattr(estimator, "decision_function"):
        prediction = estimator.decision_function(X)
    else:
        prediction = estimator.predict(X)

    # Wrap results in pandas indexed array:
    if len(prediction.shape) == 1:  # A vector:
        prediction = pd.Series(prediction, index=X.index, name="y")
    else:  # A matrix:
        prediction = pd.DataFrame(prediction, index=X.index, columns=estimator.classes_)
        prediction.columns.names = ["y"]
    return prediction


def _add_sample_weight_fit_params(estimator, sample_weight):
    """Return fit params according to whether estimator is a simple estimator or a pipeline"""
    is_pipeline = hasattr(estimator, "steps")
    if is_pipeline:
        # Attribute the provided sample_weights to the final estimator in the pipeline.
        # Attribution is done by step name followed by dunder, see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        estimator_name, estimator = estimator.steps[-1]
        fit_params = {"{}__sample_weight".format(estimator_name): sample_weight}
    else:
        fit_params = dict(sample_weight=sample_weight)

    if "sample_weight" not in inspect.signature(estimator.fit).parameters and sample_weight is None:
        # Estimator does not support "sample_weight" parameter and sample_weight is not provided
        fit_params = {}

    return fit_params


class StratifiedStandardization(IndividualOutcomeEstimator):
    """
    Standardization model that learns a model for each treatment group (i.e. subgroup of subjects with the same
    treatment assignment).
    """

    def __init__(self, learner, treatment_values=None, predict_proba=False):
        """

        Args:
            learner: Initialized sklearn model or a mapping (dict) between treatment value and initialized model,
                      For example: {0: Ridge(alpha=5), 1: Ridge(alpha=0.1)},
                      or even different models all over: {0: Ridge(), 1: RandomForestRegressor}
                      Make sure these treatment_values keys represent all treatment values found in later use.
            treatment_values (list): list of unique values of treatment (can be a single value as well).
                                     If known beforehand (on initialization time), can be passed now to init, otherwise
                                     would be inferred during fit (where treatment assignment must be supplied).
                                     Make sure these treatment_values represent all treatment values found in later use.
            predict_proba (bool): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications.
        """
        super(StratifiedStandardization, self).__init__(learner, predict_proba=predict_proba)
        if isinstance(learner, Mapping):
            self.learner = learner
        elif treatment_values is not None:  # overwrite native `learner` with dictionary based one
            self.learner = self._clone_learner(treatment_values)
        self.treatment_values = treatment_values

    def _clone_learner(self, treatment_values):
        """
        Create a copy of underlining learner object for each of the treatment values.
        Args:
            treatment_values: lLst of unique values of treatment (can be a single value and not a list as well).

        Returns:
            dict[Any, learner]: Dictionary that holds for each treatment value (key) a learner object (value) that
                                was passed during initialization.
        """
        treatment_values = [treatment_values] if isscalar(treatment_values) else treatment_values
        learners = {treatment_value: sk_clone(self.learner) for treatment_value in treatment_values}
        return learners

    def _predict(self, X, treatment_value, predict_proba=None):
        """

        Args:
            X (pd.DataFrame): Data to predict on.
            treatment_value: What model to use. Each treatment value has its own trained model. `treatment_value` is
                             used to retrieve the appropriate model.
            predict_proba (bool): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications.

        Returns:
            pd.DataFrame | pd.Series: If regression model or predict_proba=False then it returns a vector-like array
                                      (with prediction for each sample). If classification and predict_proba=True then
                                      returns a matrix-like array of n_samples by n_classes.
        """
        predict_proba = self.predict_proba if predict_proba is None else predict_proba
        prediction = _standardization_predict(estimator=self.learner[treatment_value], X=X,
                                              predict_proba=predict_proba)
        return prediction

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        treatment_values = g_tools.get_iterable_treatment_values(treatment_values, a)

        res = {}
        for treatment_value in treatment_values:
            prediction = self._predict(X=X, treatment_value=treatment_value, predict_proba=predict_proba)
            res[treatment_value] = prediction
        # TODO: should combine the results by the observed treatment into additional vector?
        res = pd.concat(res, axis="columns", names=[a.name or "a"])
        return res

    def fit(self, X, a, y, sample_weight=None):
        self.treatment_values_ = g_tools.get_iterable_treatment_values(None, a)
        if not isinstance(self.learner, dict):  # sk-learner was not cloned yet to have one copy for each stratum
            self.learner = self._clone_learner(self.treatment_values_)

        for cur_X, cur_y, cur_sw, treatment_value in self._prepare_data(X, a, y, sample_weight):
            fit_params = _add_sample_weight_fit_params(self.learner[treatment_value], cur_sw)
            self.learner[treatment_value] = self.learner[treatment_value].fit(cur_X, cur_y, **fit_params)
        return self

    def __repr__(self):
        # Since the learner is the same one only duplicated, it is redundant to repeat printing it.
        # Therefore, repeat it only once. And since there are no other attributes to consider, it is easy.
        repr_string = "{cls_name}(learner={params})".format(cls_name=self.__class__.__name__,
                                                            params=next(iter(self.learner.values())))
        return repr_string

    # def predict(self, X, a, treatment_values=None):
    #     res = self.estimate_individual_outcome(X, a, treatment_values)
    #     return res

    @staticmethod
    def _prepare_data(X, a, y=None, w=None):
        """
        Manipulating the data to fit the model specifications.
        This methods iterates of different treatment values, slices out the subgroups that were assigned to the
        specific treatment and yields the relevant dataset.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series | None): Observed outcome of size (num_subjects,).
            w (pd.Series | None): sample_weights

        Yields:
            (pd.DataFrame, pd.Series, Any): A three-tuple containing:

             * the covariates for individual under specific treatment,
             * the observed outcomes for these individuals (if y was passed and is not None),
             * the current treatment value.
        """
        treatment_values = g_tools.get_iterable_treatment_values(None, a)
        for treatment_value in treatment_values:
            treated = a == treatment_value
            cur_X = X.loc[treated, :]
            cur_y = y[treated] if y is not None else None
            cur_w = w[treated] if w is not None else None
            yield cur_X, cur_y, cur_w, treatment_value


class Standardization(IndividualOutcomeEstimator):
    """
    Standard standardization model for causal inference.
    Learns a model that takes into account the treatment assignment, and later, this value can be intervened, changing
    the predicted outcome.
    """

    def __init__(self, learner, encode_treatment=False, predict_proba=False):
        """

        Args:
            learner: Initialized sklearn model.
            encode_treatment (bool): Whether to encode the treatment as one-hot matrix.
                                     Usually good if n_treatment > 2.
            predict_proba (bool): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications.
        """
        super(Standardization, self).__init__(learner, predict_proba=predict_proba)
        self.encode_treatment = encode_treatment

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        treatment_values = g_tools.get_iterable_treatment_values(treatment_values, a)
        res = {}
        for treatment_value in treatment_values:
            treatment_assignment = pd.Series(
                treatment_value, index=X.index, name=a.name
            )  # a vector of a single-valued treatment assignment
            prediction = self._predict(X, treatment_assignment, predict_proba=predict_proba)  # predict
            res[treatment_value] = prediction  # Save prediction
        res = pd.concat(res, axis="columns", names=[a.name or "a"])
        return res

    # def _initialize_encoder(self, treatment_values):
    #     self.treatment_encoder_ = OneHotEncoder(n_values=len(treatment_values), sparse=False)

    def fit(self, X, a, y, sample_weight=None):
        if self.encode_treatment:
            # setattr(self, "treatment_encoder_", OneHotEncoder(sparse=False))
            self.treatment_encoder_ = OneHotEncoder(categories="auto")
            self.treatment_encoder_.fit(a.to_frame())
        X = self._prepare_data(X, a)
        fit_params = _add_sample_weight_fit_params(self.learner, sample_weight)
        self.learner.fit(X, y, **fit_params)
        return self

    def _predict(self, X, a, predict_proba=None):
        """

        Args:
            X (pd.DataFrame): Covariates to predict on.
            a (pd.Series): Corresponding treatment assignment to utilize for prediction.
            predict_proba (bool): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications.

        Returns:
            pd.DataFrame | pd.Series: If regression model or predict_proba=False then it returns a vector-like array
                                      (with prediction for each sample). If classification and predict_proba=True then
                                      returns a matrix-like array of n_samples by n_classes.
        """
        predict_proba = self.predict_proba if predict_proba is None else predict_proba

        cur_X = self._prepare_data(X, a)  # concatenate treatment assignment to data

        prediction = _standardization_predict(estimator=self.learner, X=cur_X, predict_proba=predict_proba)
        return prediction

    def _prepare_data(self, X, a):
        """
        Manipulating the data to fit the model specifications.
        This method concatenates the treatment assignment (either a vector or a One-Hot matrix representing that vector)
        to the covariates and then learns a model on that augmented design matrix.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).

        Returns:
            pd.DataFrame: concatenation of treatment column/s to the provided covariate matrix (A | X).
        """
        a_name = a.name
        if self.encode_treatment:
            a_transformed = self.treatment_encoder_.transform(a.to_frame())
            a_transformed = a_transformed.toarray()
            a = pd.DataFrame(a_transformed, index=a.index, columns=self.treatment_encoder_.categories_[0])
        cur_X = g_tools.column_name_type_safe_join(X, a, a_name=a_name)
        return cur_X


