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

Created on Sep 9, 2021
"""

import warnings
from .base_estimator import IndividualOutcomeEstimator
import pandas as pd
from copy import deepcopy
from sklearn.dummy import DummyClassifier


class XLearner(IndividualOutcomeEstimator):
    """
    An X-learner model for causal inference (künzel et al. 2018. pnas, https://www.pnas.org/content/116/10/4156).
    Uses two outcome estimators. The first is used to calculate the response while the second is used invertly to
    calculate the treatment which is averaged according to the propensity of the treatment assignment.
    """

    def __init__(self, outcome_model, effect_model=None, treatment_model=None, predict_proba=True, effect_types='diff'):
        """
        Args:
            outcome_model (IndividualOutcomeEstimator): Initialized causallib estimator that will be used to predict the
                                                        outcome of each treatment  given a case and a certain. To adhere
                                                         to the XLearner algorithm a StratifiedStandardization object
                                                         should be used for both outcome and cate model initialized
                                                         with comparable sklearn learners. Xlearner algorithm is suitable
                                                         for a binary outcome, if a non binary outcome will be used the
                                                         class will view the last outcome versus the rest as the binary
                                                         outcome.
            effect_model (IndividualOutcomeEstimator | None): Initialized causallib estimator that will be used to predict
                                                            the treatment effect of each case. The treatment effect
                                                            is estimated on the observed set using the outcome model
                                                            if the treatment effect is continuous use a regression model.
                                                            The default estimator is cloned from the outcome model.
                                                            The cloning is done after the outcome model is fitted to
                                                            enable warm start of the cate model by the outcome model
                                                            if outcome_model has its warm_start attribute on.
            treatment_model: Initialized sklearn prediction model that will predict the probability of each treatment.
                             Xlearner algorithm is suitable for binary treatment.
            predict_proba (bool) : In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications. Xlearner effect estimation (in the case of binary effect)
                                requires the outcome estimator to predict probabilities of classification
                                (predict_proba=True)
            effect_types (str): string from the set of EffectEstimator.CALCULATE_EFFECT keys
                        if none the sklearn DummyClassifier with prior strategy will be used.
        """
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        if treatment_model is None:
            treatment_model = DummyClassifier(strategy="prior")
        self.treatment_model = treatment_model
        self.effect_types = effect_types
        learner = {'outcome_model': self.outcome_model,
                   'cate_model': self.effect_model,
                   'treatment_model': self.treatment_model}
        super(XLearner, self).__init__(learner, predict_proba=predict_proba)

    def estimate_effect(self, X, a, agg="population", predict_proba=None, effect_types=None):
        """Estimates the causal effect between treatment groups.

        Args:
            X (pd.DataFrame): Covariates to predict on.
            a (pd.Series): Corresponding treatment assignment to utilize for prediction.
                           Assumes treated group is coded as 1, and control group as 0.
            agg (str): Either "population" or "individual" - whether to calculate individual effect or population
                       effect.
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications. If None, will use the object's initialized predict_proba value
            effect_types (None): IGNORED

        Returns:
             pd.Series: the estimated causal effect
        """
        if effect_types is not None:
            warnings.warn("Effect type is determined during fit")
        weighted_cat = self._estimate_weighted_effect(X, a, predict_proba=predict_proba)
        effect = super(XLearner, self).estimate_effect(
            weighted_cat[1], (-1) * weighted_cat[0], agg=agg, effect_types='diff'
        )
        # The effect type here is used for the weighted sum according to the treatment probability
        # the effect type is determined during fit so we need to modify the outputted data frame
        # if agg is population we get a single values series and we can use rename if agg is
        # individual we get a data frame that we squeeze into a series and rename. If we will
        # squeeze the population series we will get a scalar not a series so we need the condition..
        if agg == "population":
            effect.rename({'diff': self.effect_types})
        else:
            effect = effect.squeeze().rename(self.effect_types)
        return effect

    def _estimate_weighted_effect(self, X, a, predict_proba=None):
        """

        Args:
            X (pd.DataFrame): Covariates to predict on.
            a (pd.Series): Corresponding treatment assignment to utilize for prediction.
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications. If None, will use the object's initialized predict_proba value

        Returns (pd.Series): The propensity weighted treatment effect
        """
        predict_proba = self.predict_proba if predict_proba is None else predict_proba
        outcomes = self.effect_model.estimate_individual_outcome(X, a, predict_proba=predict_proba)
        propensity = self.treatment_model.predict_proba(X)
        return outcomes * propensity[:, ::-1]

    def estimate_individual_outcome(self, X, a, treatment_values=None, predict_proba=None):
        warnings.warn("If you're trying to estimate the individual outcome effect from individual outcome "
                      "prediction directly.\n You should use stratified or standarized estimator \n "
                      "XLearner effect estimation is obtained using the estimate_effect method")

        return self.outcome_model.estimate_individual_outcome(
            X, a, treatment_values=treatment_values, predict_proba=predict_proba)

    def fit(self, X, a, y, sample_weight=None, predict_proba=None):
        """
        Trains a causal model from observed data.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            sample_weight: To be passed to the underlining outcome model fit method.
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications. If None, will use the object's initialized predict_proba value

        Returns:
            IndividualOutcomeEstimator: A causal model with an inner models fitted.
        """

        self.outcome_model.fit(X, a, y,  sample_weight=sample_weight)
        imp_te = self._estimate_imputed_treatment_effect(X, a, y, predict_proba=predict_proba)
        # Copying the outcome model post fit enables the user to use warm start that may benefit the cate model
        if self.effect_model is None:
            self.effect_model = deepcopy(self.outcome_model)
        self.effect_model.fit(X, a, imp_te, sample_weight=sample_weight)
        self.treatment_model.fit(X, a, sample_weight=sample_weight)
        return self

    def _estimate_imputed_treatment_effect(self, X, a, y, predict_proba=None):
        """
        Calculates the imputed treatment effect

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications. If None, will use the object's initialized predict_proba value

        Returns:
             pd.Series: The imputed treatment effect for each observation
        """
        ind_outcomes = self._obtain_ind_outcomes(X, a, y, predict_proba=predict_proba)
        return self._obtain_imputed_treatment_effect(ind_outcomes, a, y)

    def _obtain_ind_outcomes(self, X, a, y, predict_proba=None):
        """
        This method returns the outcomes model individual outcomes as a two column matrix
        for classification sklearn models predict_proba=True returns a matrix of probability for each y value
        for complying with the XLearner algorithm we extract only the probabilities for the maximal value of y
        (which is usually y==1 in binary tasks).

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).
            predict_proba (bool | None): In case the outcome task is classification and in case `learner` supports the
                                  operation, if True - prediction will utilize learner's `predict_proba` or
                                  `decision_function` which returns a continuous matrix of size (n_samples, n_classes).
                                  If False - `predict` will be used and return value will be based on a vector of class
                                  classifications. If None, will use the object's initialized predict_proba value

        Returns:
            pd.DataFrame: DataFrame which columns are treatment values and rows are individuals: each column is a vector
              size (num_samples,) that contains the estimated outcome for each individual under the
              treatment value in the corresponding key.
        """
        predict_proba = self.predict_proba if predict_proba is None else predict_proba
        ind_outcomes = self.outcome_model.estimate_individual_outcome(X, a, predict_proba=predict_proba)
        if isinstance(ind_outcomes.columns, pd.MultiIndex):
            # predict_proba is True, columns are treatment-values over outcome-values
            ind_outcomes = ind_outcomes.xs(y.max(), axis="columns", level=-1, drop_level=True)
        return ind_outcomes

    def _obtain_imputed_treatment_effect(self, ind_outcomes, a, y):
        """
        Imputes outcome predictions with observed values for the factual outcome
        and calculates the effect.

        Args:
            ind_outcomes (pd.DataFrame): DataFrame which columns are treatment values and rows are individuals: each
            column is a vector size (num_samples,) that contains the estimated outcome for each individual under the
            treatment value in the corresponding key.
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y (pd.Series): Observed outcome of size (num_subjects,).

        Returns:
             pd.Series: The imputed treatment effect for each observation calculated according to the treatment
        """
        ind_outcomes.loc[a == 0, 0] = y.loc[a == 0]
        ind_outcomes.loc[a == 1, 1] = y.loc[a == 1]
        effect_func = self.CALCULATE_EFFECT[self.effect_types]
        ind_effect = effect_func(ind_outcomes[1], ind_outcomes[0])
        return ind_effect
