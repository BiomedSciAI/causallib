"""(C) Copyright 2019 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Dec 5, 2021
"""
import copy
from itertools import combinations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from causallib.positivity import BasePositivity


class MultipleTreatmentPositivity(BasePositivity, ABC):
    """
    Abstract class defining the methods require for a multiple treatment positivity scheme
    """

    def __init__(self, base_positivity_estimator, verbose=False):
        """

        Args:
            base_positivity_estimator: (MLHLS positivity model): The positivity model to be used consecutively for the
            positivity calculations.
            verbose: If true a progress bar will appear while fitting the single estimators
        """
        self.base_positivity_estimator = base_positivity_estimator
        self.positivity_estimators_ = dict()
        self.verbose = verbose

    def predict(self, X, a=None):
        """
            Predicts which rows in X are positive and which are not
        Args:
            X (pd.DataFrame): Covariates to predict on, each row is a instance and column is a feature
            a (pd.Series): The treatment assignment vector
        Returns:
            (pd.Series): A boolean series that indicate which instances are overlapping (True value) and which is
            not overlapping (False value)
        """
        return self.positivity_profile(X, a).min(axis=1).astype(bool)

    def _calculate_single_positivity(self, rows: pd.Series, treatment: str, X: pd.DataFrame, a=None) -> pd.Series:
        """
            Calculates the positivity indication on the covariates in the specified rows
        Args:
            rows: The rows to predict the covariates on
            treatment: The name of the single treatment for positivity prediction, (must correspond to a key in the
                        positivity_estimators)
            X (pd.DataFrame): Covariates to predict on, each row is a instance and column is a feature
            a: The treatment of the corresponding covariates, transferred to the base positivity estimator (for example
                matching requires the treatment)

        Returns:
            (pd.Series) : Boolean series of rows length indicating whether subject is within positivity overlap
        """
        a_tmp = None if a is None else a[rows]
        return (
            self.positivity_estimators_[treatment]
                .predict(X.loc[rows, :], a=a_tmp)
                .values
        )

    @abstractmethod
    def _create_positivity_profile_columns(self):
        """
        Calculates the columns of the positivity profile
        Returns:
            columns for the positivity profile
        """

    def positivity_profile(
            self, X: pd.DataFrame, a=None, estimation_population='Both'
    ) -> pd.DataFrame:
        """
            Creates the positivity profile for the covariates in X for certain populations
        Args:
            X (pd.DataFrame): Covariates to predict on, each row is a instance and column is a feature.
            a (pd.Series): The treatments given to the patients.
            estimation_population (str): Can be "treated" (treated group only), "control" (control group only), "both"
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.

        Returns:

        """
        positivity_profile = pd.DataFrame(
            columns=self._create_positivity_profile_columns(), index=X.index,
            dtype="boolean",
        )
        for treatment in positivity_profile.columns:
            rows = self._get_treated_rows_indicator(X, a, estimation_population, treatment=treatment)
            positivity_profile.loc[rows, treatment] = self._calculate_single_positivity(X=X, a=a, rows=rows,
                                                                                        treatment=treatment)
        return positivity_profile

    @abstractmethod
    def _treatment_positivity_summary_table(
            self, X: pd.DataFrame, a=None, estimation_population='Both'
    ) -> pd.DataFrame:
        """
        Returns the positivity treatment profile, the rate of positivity per treatment
        Args:
            X (pd.DataFrame): Covariates to predict positivity on
            a (pd.Series): Treatment assignment series
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.

        Returns:
            pd.DataFrame: A table of the rate of treatments.
        """

    @abstractmethod
    def _treatment_positivity_summary_contingency(
            self, X: pd.DataFrame, a=None, estimation_population='Both'
    ) -> pd.DataFrame:
        """
        The method returns the positivity profile in a form of contingency matrix where every treatment
        appears once in the rows and once in the columns of the profile. This form might be easier for viewing
        in the one versus another case and can be used for graph based algorithms. The values in the mat are rate of
        overlapping treatments by both single estimations.
        Args:
            X (pd.DataFrame): Covariates to predict on
            a (pd.Series) : The treatment assignment to be passed to the base positivity if required and
            corresponding single positivity estimator or all of the patients'
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.
        Returns:
            pd.DataFrame: A contingency table of the rate of treatments.

        """

    def treatment_positivity_summary(
            self, X: pd.DataFrame, a=None, estimation_population='Both', as_contingency_table=False
    ) -> pd.DataFrame:
        """
            The method summarized the positivity rate per treatment and reference group
        Args:
            X (pd.DataFrame): Covariates to predict on
            a (pd.Series) : The treatment assignment to be passed to the base positivity if required and
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.
            as_contingency_table (bool): If true returns a symmetric with every treatment as row and columns.
            otherwise, return a matrox where the rows correspond to the first treatment of the pair and the columns
            to the second treatment in each treatment pair. This form might be easier for viewing
            in the one versus another case and can be used for graph based algorithms.
        Returns:
            pd.DataFrame|pd.Series : a data frame describing the positivity rate of each single positivity estimation
        """
        if as_contingency_table:
            return self._treatment_positivity_summary_contingency(X=X, a=a, estimation_population=estimation_population)
        else:
            return self._treatment_positivity_summary_table(X=X, a=a, estimation_population=estimation_population)

    @abstractmethod
    def _get_treated_rows_indicator(self, X, a, estimation_population='Both', treatment=None):
        """
        Returns indicator variable stating which row in x is to be used for the positivity prediction.
        If the treatment vector is not supplied every coordinate will be used for positivity estimation
        Args:
            X (pd.DataFrame): Covariates
            a (pd.Series) : Treatment assignments of the treated
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.

        Returns:

        """


class OneVersusRestPositivity(MultipleTreatmentPositivity):
    """
    This is class converts the multiple treatment positivity calculation to
    a multiple positivity calculations The class receives a multi-treatment
    data and a base positivity estimator and creates multiple estimators one
    per treatment where the one treatment is the treated and the rest are the control.
    """

    def fit(self, X, a):
        """
            Fits a one versus the rest positivity in it the experimental class is the treated and every other treatment
            class is the control jointly
        Args:
            X (pd.DataFrame): covariates to predict on, each row is a instance and column is a feature
            a (pd.Series): A series of length equal to X with the treatment assignment

        Returns:
            one_versus_all_positivity: A causal model with an inner models fitted.
        """
        clf = self.base_positivity_estimator
        if self.verbose:
            print("Fitting {} treatment".format(len(a.unique())))
        for treatment in a.unique():
            clf = copy.deepcopy(clf)
            clf.fit(X, a == treatment)
            self.positivity_estimators_[treatment] = clf
        return self

    def _create_positivity_profile_columns(self):
        """
        Create a list with the names of the treatment as the columns
        Returns:
            a set containing the names of the treatments to be used as columns
        """
        kys = list(self.positivity_estimators_.keys())
        kys.sort()
        return kys

    def _treatment_positivity_summary_table(
            self, X: pd.DataFrame, a=None, estimation_population='Both'
    ) -> pd.DataFrame:
        """
        Returns the positivity treatment profile, the rate of positivity per treatment
        Args:
            X (pd.DataFrame): Covariates to predict positivity on
            a (pd.Series): Treatment assignment series
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.

        Returns:
            pd.DataFrame: A table of the rate of treatments.
        """
        return self.positivity_profile(X=X, a=a, estimation_population=estimation_population).mean(
            axis=0)

    def _treatment_positivity_summary_contingency(
            self, X: pd.DataFrame, a=None, estimation_population='Both'
    ) -> pd.DataFrame:
        """
        The method returns the positivity profile in a form of contingency matrix where every treatment
        appears once in the rows and once in the columns of the profile. This form might be easier for viewing
        in the one versus another case and can be used for graph based algorithms. The values in the mat are rate of
        overlapping treatments by both single estimations.
        Args:
            X (pd.DataFrame): Covariates to predict on
            a (pd.Series) : The treatment assignment to be passed to the base positivity if required and
            corresponding single positivity estimator or all of the patients'
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.
        Returns:
            pd.DataFrame: A contingency table of the rate of treatments.

        """
        treatment_profile = self.positivity_profile(X=X, a=a, estimation_population=estimation_population)
        treats = treatment_profile.columns.sort_values().values
        contingency_frame = pd.DataFrame(index=treats, columns=treats)
        for a1, a2 in combinations(treats, 2):
            contingency_frame.loc[a1, a2] = np.mean(
                np.logical_and(
                    treatment_profile[a1].fillna(False),
                    treatment_profile[a2].fillna(False)
                )
            )
        return contingency_frame.astype(float)

    def _get_treated_rows_indicator(self, X, a, estimation_population='Both', treatment=None):
        """
        Returns indicator variable stating which row in x is to be used for the positivity prediction.
        If the treatment vector is not supplied every coordinate will be used for positivity estimation
        Args:
            X (pd.DataFrame): Covariates
            a (pd.Series) : Treatment assignments of the treated
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.

        Returns:

        """
        if a is None or estimation_population.lower() in ['every', 'both']:
            return pd.Series(np.repeat(True, X.shape[0]), index=X.index)
        elif estimation_population.lower() == 'control':
            return a != treatment
        elif estimation_population.lower() == 'treated':
            return a == treatment


class OneVersusAnotherPositivity(MultipleTreatmentPositivity):
    """
    This is class converts the multiple treatment positivity calculation to
    multiple one vs. one positivity calculations. The class receives a multi-treatment
    data and a base positivity estimator and creates multiple estimators one
    per treatment pair where the one treatment is the treated and the second is the control.
    """

    def __init__(
            self,
            base_positivity_estimator,
            verbose=False,
            treatment_pairs_list=None,
            treatment_a_name="treatmentA",
            treatment_b_name="treatmentB",
    ):
        """

        Args:
            base_positivity_estimator: (MLHLS positivity model): The positivity model to be used consecutively for the
            positivity calculations.
            verbose: If true a progress bar will appear while fitting the single estimator
            treatment_pairs_list (None| tuple list|int|float|bool|str): A list of tuples each containing a pair
            of treatments to estimate the positivity between them. The first element is considered the control and the
            second is treated. If None a list of  all pairwise of treatments will be created.
            If the value is a string, bool or numeric it will be used the control treatment and a list  containing all
            pairs of treatment versus it will be created.
            treatment_a_name: name for the first enry of the data frame report multi index
            treatment_b_name: name for the second enry of the data frame report multi index
        """
        self.treatment_pairs_list = treatment_pairs_list
        self.treatment_a_name = treatment_a_name
        self.treatment_b_name = treatment_b_name
        super().__init__(
            base_positivity_estimator=base_positivity_estimator, verbose=verbose
        )

    def fit(self, X: pd.DataFrame, a: pd.Series) -> BasePositivity:
        """
            Fits a one versus the another positivity in it the user defines the pairs of treatment where one will be
            control and the other the experimental.
        Args:
            X (pd.DataFrame): covariates to predict on, each row is a instance and column is a feature
            a (pd.Series): A series of length equal to X with the treatment assignment
        Returns:
            one_versus_another_positivity: A MultipleTreatmentPositivity model with an inner models fitted.
        """
        if self.treatment_pairs_list is None:
            self.treatment_pairs_list = list(combinations(a.unique(), 2))
        elif isinstance(self.treatment_pairs_list, (int, float, bool, str)):
            self.treatment_pairs_list = [(v, self.treatment_pairs_list) for v in a.unique()
                                         if v != self.treatment_pairs_list]

        clf = self.base_positivity_estimator
        if self.verbose:
            print("Fitting treatment {}".format(len(self.treatment_pairs_list)))
        for a1, a2 in self.treatment_pairs_list:
            clf = copy.deepcopy(clf)
            clf.fit(X.loc[a.isin([a1, a2])], a[a.isin([a1, a2])] == a1)
            self.positivity_estimators_[(a1, a2)] = clf
        return self

    def _create_positivity_profile_columns(self):
        """
        Create a list with the names of the treatment as multi index columns
        Returns:
            a multi index column name
        """
        return pd.MultiIndex.from_tuples(
            self.treatment_pairs_list, names=[self.treatment_a_name, self.treatment_b_name]
        )

    def _treatment_positivity_summary_table(
            self, X: pd.DataFrame, a=None, estimation_population='Both'
    ) -> pd.DataFrame:
        """
            The method summarized the positivity rate per treatment and reference group
        Args:
            X (pd.DataFrame): Covariates to predict on
            a (pd.Series) : The treatment assignment to be passed to the base positivity if required and
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment). If a is not supplied
            then the mode every is selected automatically.
        Returns:
            pd.DataFrame|pd.Series : a data frame describing the positivity rate of each single positivity estimation

        """
        treatment_profile = self.positivity_profile(X=X, a=a,
                                                    estimation_population=estimation_population).transpose().mean(
            axis=1)
        treatment_profile_df = treatment_profile.reset_index()
        return (
            treatment_profile_df.groupby(
                [self.treatment_a_name, self.treatment_b_name]
            )
                .mean()
                .unstack(-1)
                .droplevel(0, axis=1)
        )

    def _treatment_positivity_summary_contingency(
            self, X: pd.DataFrame, a=None, estimation_population='Both'
    ) -> pd.DataFrame:
        """
        The method returns the positivity profile in a form of contingency matrix where every treatment (be it the one
        or another) appears once in the rows and once in the columns of the profile. This form might be easier for
        viewing in the one versus another case and can be used for graph based algorithms.
        Args:
            X (pd.DataFrame): Covariates to predict on
            a (pd.Series) : The treatment assignment to be passed to the base positivity if required and
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.

        Returns:
            pd.DataFrame: A contingency table of the rate of treatments.

        """
        treatment_profile = self.positivity_profile(X=X, a=a,
                                                    estimation_population=estimation_population).transpose().mean(
            axis=1)
        treats = list(
            {item for sublist in treatment_profile.index for item in sublist}
        )
        treats.sort()
        contingency_frame = pd.DataFrame(index=treats, columns=treats)
        for a1, a2 in combinations(treats, 2):
            if (a1, a2) in treatment_profile.index:
                contingency_frame.loc[a1, a2] = treatment_profile.loc[(a1, a2)]
            else:
                contingency_frame.loc[a1, a2] = treatment_profile.loc[(a2, a1)]
        return contingency_frame.astype(float)

    def _get_treated_rows_indicator(self, X, a, estimation_population='Both', treatment=None):
        """
        Returns indicator variable stating which row in x is to be used for the positivity prediction.
        If the treatment vector is not supplied every coordinate will be used for positivity estimation
        Args:
            X (pd.DataFrame): Covariates
            a (pd.Series) : Treatment assignments of the treated
            estimation_population (str): Can be treated (treated group only), control (control group only) both
            (either treated or control group) or Every (every patient that was given a treatment) . If a is not supplied
            then the mode every is selected automatically.

        Returns:

        """
        if a is None or estimation_population.lower() == 'every':
            return pd.Series(np.repeat(True, X.shape[0]), index=X.index)
        elif estimation_population.lower() == 'both':
            return a.isin(treatment)
        elif estimation_population.lower() == 'control':
            return a == treatment[0]
        elif estimation_population.lower() == 'treated':
            return a == treatment[1]
