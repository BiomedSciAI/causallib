"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Created on March 2, 2021
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple
from sklearn.base import BaseEstimator


class BasePositivity(ABC, BaseEstimator):

    @abstractmethod
    def fit(self,
            X: pd.DataFrame, a: pd.Series) -> BasePositivity:
        """Fit positivity checker.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame, a: pd.Series) -> pd.Series:
        """Predict whether a sample is in the overlap of treatments.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features)
            a (pd.Series): Treatment assignment of size (num_subjects,).

        Returns:
            pd.Series: a Series of length `X.shape[0]` with the same index as
               `X` and only boolean values
        """
        raise NotImplementedError

    def transform(self,
                  X: pd.DataFrame, a: pd.Series, *args: pd.Series
                  ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Transform the input data to remove positivity violations.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            *args (pd.Series): Zero or more pd.Series objects corresponding to 
               outcomes. Each argument must be indexed the same as the other 
               arguments and have size (num_subjects,).

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: Subsets of `X`, `a` and
            the output series objects of `args` corresponding to the samples
            which do not violate the positivity assumption.
        """
        indices_to_keep = self.predict(X, a)
        return_list = [X.loc[indices_to_keep], a.loc[indices_to_keep]]
        for output in args:
            return_list.append(output.loc[indices_to_keep])
        return return_list

    def fit_predict(self, X: pd.DataFrame, a: pd.Series) -> pd.Series:
        """Fit positivity checker and predict overlap membership.

        This is a convenience function that calls `fit` and `predict`.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features)
            a (pd.Series): Treatment assignment of size (num_subjects,).

        Returns:
            pd.Series: a Series of length `X.shape[0]` with the same index as
               `X` and only boolean values
        """
        self.fit(X, a)
        return self.predict(X, a)

    def fit_transform(self,
                      X: pd.DataFrame, a: pd.Series, *args: pd.Series
                      ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Fit and transform data by removing positivity violations.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            *args (pd.Series): Zero or more pd.Series objects corresponding to 
               outcomes. Each argument must be indexed the same as the other 
               arguments and have size (num_subjects,).

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: Subsets of `X`, `a` and
            the output series objects of `args` corresponding to the samples
            which do not violate the positivity assumption.
        """
        self.fit(X, a)
        return self.transform(X, a, *args)

    def score(self,
              X: pd.DataFrame, a: pd.Series,
              **kwargs):
        """Score the positivity violation
        This is a generic function, but right now it receives
        only one kind of scorer - cross_covaraince_score
        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            **kwargs : kwargs that are corresponding to the scoring metric.

        Returns:
            float: a non-negative score that quantifies the violation
            of positivity
        """
        from .metrics.metrics import cross_covariance_score
        X_trans, a_trans = self.transform(X, a)
        return cross_covariance_score(X_trans, a_trans, **kwargs)

