# (C) Copyright 2020 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created on Nov 12, 2020

import unittest
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from causallib.estimation import Matching, PropensityMatching
import pickle

from causallib.preprocessing.transformers import PropensityTransformer, MatchingTransformer


class TestMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 100
        cls.k = 10
        cls.random_seed = 5432
        np.random.seed(cls.random_seed)
        cls.data_serial_x = cls._generate_serial_covariates_data(cls.n)
        cls.data_serial_unbalanced_x = cls._generate_serial_covariates_unbalanced_treatment_data(
            cls.n, cls.k)
        cls.data_3feature_linear_effect = cls._generate_3feature_linear_effect_data(
            cls.random_seed)

        cls.test_data = [cls.data_serial_x, cls.data_serial_unbalanced_x,
                         cls.data_3feature_linear_effect]

    @classmethod
    def _generate_serial_covariates_data(cls, n):
        '''
        The simplest dataset. Covariates are 1d integers 0 to n-1. Treatment is
        0 for one set of 0 to n-1, 1 for another set of 0 to n-1. Outcome is 
        random uniform for untreated and 2x random uniform for treated.
        '''
        X = pd.DataFrame(np.hstack([np.arange(n), np.arange(n)]))
        X.columns = [f"x{i}" for i in range(len(X.columns))]
        a = pd.to_numeric(
            pd.Series(np.hstack([np.zeros(n), np.ones(n)]), name="treatment"),
            downcast="integer")

        y = pd.Series(
            np.hstack([np.random.rand(n), 2 * np.random.rand(n)]), name="outcome")
        return X, a, y

    @classmethod
    def _generate_serial_covariates_unbalanced_treatment_data(cls, n, k):
        '''
        The same as above but imbalanced. Covariates are 1d integers 0 to n-1. 
        Treatment is 0 for one set of 0 to n-1, 1 for another set of 0 to k-1. 
        Outcome is equal to covariates for untreated and 2x covariates for treated.
        '''
        X = pd.DataFrame(np.hstack([np.arange(n), np.arange(k)]))
        X.columns = [f"x{i}" for i in range(len(X.columns))]
        a = pd.to_numeric(
            pd.Series(np.hstack([np.zeros(n), np.ones(k)]), name="treatment"),
            downcast="integer")
        y = pd.Series(
            np.hstack([np.arange(n), 2 * np.arange(k)]), name="outcome")

        return X, a, y

    @classmethod
    def _generate_3feature_linear_effect_data(cls, random_state):
        '''
        Two stages of data creation. A generic classification set of 500 samples
        to predict treatment (so that the propensity model should find something).
        Outcome is theta*X + 5*a so there is a linear effect of strength 5 plus a 
        covariate dependency via the vector theta which is hardcoded to (0.1, 0.2, 0.3)
        '''
        X, a = make_classification(
            n_features=3,
            n_samples=500,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.0,
            class_sep=1.0,
            random_state=random_state,
        )
        X = (
            pd.DataFrame(X)
            .assign(treatment=a)
            .assign(outcome=X @ (0.1, 0.2, 0.3) + 5 * a)
        )
        a = X.pop("treatment")
        y = X.pop("outcome")
        X.columns = [f"x{i}" for i in range(len(X.columns))]

        return X, a, y

    def setUp(self):
        self.transformer = MatchingTransformer()
        self.matching = Matching()

    def test_that_serial_integer_covariates_match_exactly_with_replacement(self):
        X, a, y = self.data_serial_x

        self.matching.fit(X, a, y)
        self._check_index_matching_for_serial_indices_(X, a)

    def test_that_serial_integer_covariates_match_exactly_with_no_replacement(self):
        X, a, y = self.data_serial_x
        self.matching.with_replacement = False

        self.matching.fit(X, a, y)
        self._check_index_matching_for_serial_indices_(X, a)

    def test_that_serial_integer_covariates_match_exactly_when_unbalanced_with_replacement(
        self,
    ):
        X, a, y = self.data_serial_unbalanced_x

        self.matching.fit(X, a, y)
        self._check_index_matching_for_serial_indices_(X, a)

    def test_that_serial_integer_covariates_match_exactly_when_unbalanced_with_no_replacement(
        self,
    ):
        X, a, y = self.data_serial_unbalanced_x
        self.matching.with_replacement = False

        self.matching.fit(X, a, y)
        self._check_index_matching_for_serial_indices_(X, a)

    def test_transform_for_serial_integer_covariates_when_unbalanced_with_no_replacement(
        self,
    ):
        X, a, y = self.data_serial_unbalanced_x

        self.transformer.set_params(with_replacement=False)
        Xm, am, ym = self.transformer.fit_transform(X, a, y)
        np.testing.assert_array_equal(Xm, X[X <= (X[a == 1].max())].dropna())

    def test_transform_for_serial_integer_covariates_when_unbalanced_with_replacement(
        self,
    ):
        X, a, y = self.data_serial_unbalanced_x
        self.transformer.set_params(with_replacement=True, caliper=0.5)
        Xm, am, ym = self.transformer.fit_transform(X, a, y)
        np.testing.assert_array_equal(Xm, X[X <= (X[a == 1].max())].dropna())

    def test_transform_for_serial_integer_covariates_when_unbalanced_with_replacement_partial_match(
        self,
    ):
        X, a, y = self.data_serial_unbalanced_x
        self.transformer.set_params(
            with_replacement=True, matching_mode="treatment_to_control"
        )
        Xm, am, ym = self.transformer.fit_transform(X, a, y)
        np.testing.assert_array_equal(Xm, X[X <= (X[a == 1].max())].dropna())

        self.transformer.set_params(matching_mode="control_to_treatment")
        Xm, am, ym = self.transformer.fit_transform(X, a, y)

        np.testing.assert_array_equal(Xm, X)

    def test_unsupported_matching_mode_error(self):
        X, a, y = self.data_serial_unbalanced_x
        self.transformer.set_params(matching_mode="i'm not supported")

        with self.assertRaises(NotImplementedError):
            self.transformer.fit_transform(X, a, y)

    def test_effect_of_treatment_on_treated_serial_covariates_when_unbalanced_with_no_replacement(
        self,
    ):
        X, a, y = self.data_serial_unbalanced_x

        self.matching.with_replacement = False
        self.matching.fit(X, a, y)
        Ypop = self.matching.estimate_population_outcome(X, a)
        self.assertAlmostEqual(Ypop[1] / Ypop[0], 2)
        Yind = self.matching.estimate_population_outcome(X, a)
        self.assertAlmostEqual(2 * Yind[0], Yind[1])

    def test_effect_of_treatment_on_treated_serial_covariates_when_unbalanced_with_replacement(
        self,
    ):
        X, a, y = self.data_serial_unbalanced_x

        self.matching.with_replacement = True
        self.matching.caliper = 0.5
        self.matching.fit(X, a, y)

        Ypop = self.matching.estimate_population_outcome(X, a)
        self.assertAlmostEqual(Ypop[1] / Ypop[0], 2)

    def test_propensity_matching_object(self):

        self.matching = PropensityMatching(
            learner=LogisticRegression(solver="liblinear"), with_replacement=True
        )
        X, a, y = self.data_3feature_linear_effect
        # pull naive effect estimate
        naive_ate = y[a == 1].mean() - y[a == 0].mean()

        # now do the no caliper distance and effect estimates
        self.matching.propensity_transform = PropensityTransformer(
            learner=LogisticRegression(solver="liblinear"),
            include_covariates=False
        )
        self.matching.fit(X, a, y)

        self.assertEqual(
            1, self.matching.propensity_transform.transform(X).shape[1])
        self.assertTrue(
            "propensity" in self.matching.propensity_transform.transform(X))
        # verify that values of `a` are higher when propensity is larger
        covariates = (
            self.matching.propensity_transform.transform(X)
            .join(a)
            .sort_values("propensity")
        )
        self.assertLess(
            covariates.iloc[:covariates.shape[0] // 2].treatment.mean(),
            covariates.iloc[covariates.shape[0] // 2:].treatment.mean(),
        )

    def test_match_on_multiple_covariates_with_replacement_and_caliper(self):
        X, a, y = self.data_3feature_linear_effect

        # pull naive effect estimate
        naive_ate = y[a == 1].mean() - y[a == 0].mean()

        # now do the no caliper distance and effect estimates
        self.matching.fit(X, a, y)
        self.matching.with_replacement = True
        self.matching.caliper = None
        self.matching.match(X, a)
        match_distances_no_caliper = np.hstack(
            [self._get_nearest_match_distances_for_treatment_value(t,a) for t in [0, 1]])
        no_caliper_ate = self.matching.estimate_population_outcome(X, a).diff()[
            1]

        # with no caliper we should get exactly as many valid matches as samples
        self.assertEqual(match_distances_no_caliper.shape[0], len(X))
        # and the ate estimate should be closer to the ground truth effect of 5
        self.assertLess(abs(5 - no_caliper_ate), abs(5 - naive_ate))

        self.matching.caliper = np.quantile(match_distances_no_caliper, 0.01)

        # now do the caliper constrained distance and effect estimate
        self.matching.with_replacement = True
        self.matching.match(X, a, use_cached_result=False)
        match_distances_with_caliper = np.hstack(
            [self._get_nearest_match_distances_for_treatment_value(t,a) for t in [0, 1]])
        caliper_ate = self.matching.estimate_population_outcome(X, a).diff()[1]

        # with a caliper there will be fewer matches
        self.assertLess(
            match_distances_with_caliper.shape[0], match_distances_no_caliper.shape[0]
        )
        # check that caliper works by checking the maximal distance is 
        # lower than the caliper
        self.assertLess(match_distances_with_caliper.max(), self.matching.caliper)
        # and the ate estimate should be EVEN closer to the ground truth effect of 5
        self.assertLess(abs(5 - caliper_ate), abs(5 - no_caliper_ate))

    def test_match_on_multiple_covariates_with_no_replacement_and_caliper(self):
        X, a, y = self.data_3feature_linear_effect

        # pull naive effect estimate
        naive_ate = y[a == 1].mean() - y[a == 0].mean()

        # now do the no caliper distance and effect estimates
        self.matching.fit(X, a, y)
        self.matching.with_replacement = False
        self.matching.caliper = None
        self.matching.match(X, a)

        match_distances_no_caliper = np.hstack(
            [self._get_nearest_match_distances_for_treatment_value(t,a) for t in [0, 1]])
        no_caliper_ate = self.matching.estimate_population_outcome(X, a).diff()[
            1]

        # with no caliper we should get exactly as many valid matches as samples
        self.assertEqual(match_distances_no_caliper.shape[0], len(X))
        # and the ate estimate should be exactly as good as the naive estimate because there is no caliper
        # and the groups are balanced
        self.assertAlmostEqual(abs(5 - no_caliper_ate), abs(5 - naive_ate))

        self.matching.caliper = np.quantile(match_distances_no_caliper, 0.01)

        # now do the caliper constrained distance and effect estimate
        self.matching.with_replacement = False
        self.matching.match(X, a, use_cached_result=False)


        match_distances_with_caliper = np.hstack(
            [self._get_nearest_match_distances_for_treatment_value(t,a) for t in [0, 1]])
        caliper_ate = self.matching.estimate_population_outcome(X, a).diff()[1]

        # with a caliper there will be fewer matches
        self.assertLess(
            match_distances_with_caliper.shape[0], match_distances_no_caliper.shape[0]
        )
        # and the ate estimate should be EVEN closer to the ground truth effect of 5
        self.assertLess(abs(5 - caliper_ate), abs(5 - no_caliper_ate))

    def test_match_on_multiple_covariates_with_propensity_added(self):
        X, a, y = self.data_3feature_linear_effect
        # pull naive effect estimate
        naive_ate = y[a == 1].mean() - y[a == 0].mean()

        # now do the no caliper distance and effect estimates
        self.matching.propensity_transform = PropensityTransformer(
            learner=LogisticRegression(solver="liblinear"),
            include_covariates=True
        )
        self.matching.fit(X, a, y)

        self.assertTrue(
            "propensity" in self.matching.propensity_transform.transform(
                X).columns
        )
        self.assertEqual(
            X.shape[1] + 1,
            self.matching.propensity_transform.transform(X).shape[1],
        )
        # verify that values of `a` are higher when propensity is larger
        covariates = (
            self.matching.propensity_transform.transform(X)
            .join(a)
            .sort_values("propensity")
        )
        # after sorting by propensity, there should be more a=1 in the upper
        # half of the list than the lower half, else the propensity model is
        # broken
        self.assertLess(
            covariates.iloc[:covariates.shape[0] // 2].treatment.mean(),
            covariates.iloc[covariates.shape[0] // 2:].treatment.mean(),
        )

        # now do the no caliper distance and effect estimates
        self.matching.caliper = None

        no_caliper_ate = self.matching.estimate_population_outcome(X, a).diff()[
            1]
        # and the ate estimate should be closer to the ground truth effect of 5
        self.assertLess(abs(5 - no_caliper_ate), abs(5 - naive_ate))

    def test_match_on_multiple_covariates_with_propensity_replacing_covariates_for_matching(
        self,
    ):
        X, a, y = self.data_3feature_linear_effect
        # pull naive effect estimate
        naive_ate = y[a == 1].mean() - y[a == 0].mean()

        # now do the no caliper distance and effect estimates
        self.matching.propensity_transform = PropensityTransformer(
            learner=LogisticRegression(solver="liblinear"),
            include_covariates=False
        )
        self.matching.fit(X, a, y)

        self.assertEqual(
            1, self.matching.propensity_transform.transform(X).shape[1])
        self.assertTrue(
            "propensity" in self.matching.propensity_transform.transform(
                X).columns
        )
        # verify that values of `a` are higher when propensity is larger
        covariates = (
            self.matching.propensity_transform.transform(X)
            .join(a)
            .sort_values("propensity")
        )
        # after sorting by propensity, there should be more a=1 in the upper
        # half of the list than the lower half, else the propensity model is
        # broken
        self.assertLess(
            covariates.iloc[:covariates.shape[0] // 2].treatment.mean(),
            covariates.iloc[covariates.shape[0] // 2:].treatment.mean(),
        )

        # now do the caliper distance and effect estimates

        caliper_ate = self.matching.estimate_population_outcome(X, a).diff()[1]
        # and the ate estimate should be closer to the ground truth effect of 5
        self.assertLess(abs(5 - caliper_ate), abs(5 - naive_ate))

    def test_weights_with_no_replacement(self):
        X, a, y = self.data_serial_unbalanced_x

        self.matching.with_replacement = False
        self.matching.fit(X, a, y)
        self.matching.match(X, a)
        weights = self.matching.matches_to_weights()
        np.testing.assert_array_equal(
            weights["control_to_treatment"], weights["treatment_to_control"]
        )

    def test_weights_with_multiple_neighbors(self):
        X, a, y = self.data_serial_unbalanced_x

        self.matching.with_replacement = True
        self.matching.n_neighbors = 1
        self.matching.fit(X, a, y)
        self.matching.match(X, a)
        weights = self.matching.matches_to_weights()
        self.assertEqual(weights.iloc[-1, 0], self.n - self.k + 1)
        self.assertEqual(
            weights["treatment_to_control"]
            .iloc[self.k + (self.matching.n_neighbors + 1) // 2: self.n]
            .sum(),
            0,
        )

        self.matching.n_neighbors = 5
        self.matching.match(X, a)
        weights = self.matching.matches_to_weights()
        self.assertEqual(
            weights["treatment_to_control"]
            .iloc[self.k + (self.matching.n_neighbors + 1) // 2: self.n]
            .sum(),
            0,
        )

    def test_compute_weights(self):
        X, a, y = self.data_serial_unbalanced_x

        matching_mode = "control_to_treatment"
        self.matching.matching_mode = matching_mode
        self.matching.with_replacement = False
        self.matching.fit(X, a, y)

        w = self.matching.compute_weights(X, a)

        with self.subTest("Compare to `matches_to_weights` base function"):
            full_weights = self.matching.matches_to_weights()
            pd.testing.assert_series_equal(w, full_weights[matching_mode])

        with self.subTest("Fails for `matching_mode='both'`"):
            with self.assertRaises(ValueError):
                self.matching.matching_mode = "both"
                self.matching.compute_weights(X, a)

    def test_compute_weight_matrix(self):
        X, a, y = self.data_serial_unbalanced_x
        with self.assertRaises(NotImplementedError):
            self.matching.compute_weight_matrix(X, a)

    def test_exception_for_no_replacement_and_n_neighbors_gt_1(self):
        X, a, y = self.data_serial_x
        self.matching.n_neighbors = 2
        self.matching.with_replacement = False
        self.matching.fit(X, a, y)

        with self.assertRaises(NotImplementedError):
            self.matching.match(X, a)

    def test_exception_for_outcome_estimation_following_failed_matching(self):
        X, a, y = self.data_3feature_linear_effect
        self.matching.caliper = 0
        self.matching.with_replacement = False
        self.matching.fit(X, a, y)
        with self.assertWarns(Warning):
            self.matching.match(X, a)
        with self.assertRaises(ValueError):
            self.matching.estimate_population_outcome(X, a)

    def test_covariate_matching(self):
        from sklearn.exceptions import NotFittedError

        X, a, y = self.data_serial_unbalanced_x
        self.matching.fit(X, a, y)
        self.matching.with_replacement = True
        self.matching.caliper = None

        with self.assertRaises(NotFittedError):
            self.matching.get_covariates_of_matches(0, 1, X)
        self.matching.match(X, a)
        covariates_0_1 = self.matching.get_covariates_of_matches(0, 1, X)
        covariates_1_0 = self.matching.get_covariates_of_matches(1, 0, X)
        self.assertTrue(all(covariates_0_1[["delta"]].values[:, 0] == 0))
        np.testing.assert_array_equal(
            covariates_1_0[["delta"]].values[self.k:, 0],
            np.arange(1, self.n + 1 - self.k),
        )
        np.testing.assert_array_equal(
            covariates_1_0[["delta"]].values[: self.k, 0], np.zeros((self.k,))
        )

    def test_that_out_of_sample_outcome_prediction_finds_fitted_data(self):
        from sklearn.model_selection import train_test_split

        for data in self.test_data:
            X, a, y = data
            Xt, Xv, at, av, yt, yv = train_test_split(
                X, a, y, random_state=self.random_seed)
            self.matching.fit(Xt, at, yt)
            # estimate outcome for validation should pull data from train
            ypredv = self.matching.estimate_individual_outcome(Xv, av, yv)
            ypredv_melt = ypredv.melt(var_name=at.name, value_name=yt.name)
            trainval_match = (
                pd.concat(
                    {"train": pd.concat([yt, at], axis=1),
                     "val": ypredv_melt},
                    sort=True,
                )
                .reset_index()
                .rename(columns={"level_0": "fold", "level_1": "sample_id"})
            )
            # to check that we are finding a training data for every matched validation
            # we create a dataframe combining outcomes for both folds and then
            # group by the predicted outcome and check that every group that has a
            # validation sample also has a training sample, in other words every time
            # a validation sample matches to something, it is matching to a valid
            # sample within the training data
            for gval, gidx in trainval_match.groupby(yt.name).groups.items():
                gx = trainval_match.loc[gidx]
                # if there is a validation with this outcome
                if len(gx[gx.fold == "val"]) > 1:
                    # there must also be a training sample
                    self.assertGreater(len(gx[gx.fold == "train"]), 0)

    def test_classify_task(self):

        X, a, y = self.data_3feature_linear_effect
        y = y.astype(int)
        self.matching.fit(X, a, y)
        self.matching.caliper = None
        ypred = self.matching.estimate_individual_outcome(
            X, a, predict_proba=False)
        # if not all integers will raise errors
        if pd.__version__ > "2.1":
            # TODO: at time of writing pandas 2.1.0 is <1 year old.
            #       Once matured, you may erase the deprecated `applymap` and just use `map`.
            ypred.map(str).map(int)
            with self.assertRaises(ValueError):
                self.matching.estimate_individual_outcome(
                    X, a, predict_proba=True
                ).map(str).map(int)
        else:
            ypred.applymap(str).applymap(int)
            with self.assertRaises(ValueError):
                self.matching.estimate_individual_outcome(
                    X, a, predict_proba=True
                ).applymap(str).applymap(int)

    def test_classify_task_with_wrong_inputs_warning_check(self):
        X, a, y = self.data_3feature_linear_effect
        # y = y.astype(int)
        self.matching.fit(X, a, y)
        self.matching.caliper = None
        with self.assertWarns(Warning):
            self.matching.estimate_individual_outcome(
                X, a, predict_proba=False)

    def test_treatment_to_control_control_to_treatment_match(self):
        X, a, y = self.data_serial_unbalanced_x
        self.matching.fit(X, a, y)
        self.matching.estimate_observed_outcome = False

        self.matching.matching_mode = "treatment_to_control"
        ypred = self.matching.estimate_individual_outcome(X, a)
        np.testing.assert_array_equal(ypred[1].values, y[a == 1].values)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(ypred[0].values, y[a == 0].values)

        self.matching.matching_mode = "control_to_treatment"
        ypred = self.matching.estimate_individual_outcome(X, a)
        np.testing.assert_array_equal(ypred[0].values, y[a == 0].values)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(ypred[1].values, y[a == 1].values)

        self.matching.matching_mode = "i'm a mistake"
        with self.assertRaises(NotImplementedError):
            self.matching.estimate_individual_outcome(X, a)

    def _check_index_matching_for_serial_indices_(self, X, a):

        match_df = self.matching.match(X, a)

        matched_indices_1_to_0 = (
            match_df.loc[0][a == 1]
            .matches.apply(lambda x: x[0] if x else np.nan)
            .dropna()
            .values
        )
        matched_indices_0_to_1 = (
            match_df.loc[1][a == 0]
            .matches.apply(lambda x: x[0] if x else np.nan)
            .dropna()
            .values
        )

        zero_indices = X[a == 0].index.values
        one_indices = X[a == 1].index.values

        self.assertEqual(
            sum(np.diff(list(zip(matched_indices_0_to_1, one_indices)))), 0
        )
        self.assertEqual(
            sum(np.diff(list(zip(matched_indices_1_to_0, zero_indices)))), 0
        )

    def test_all_backends_agree(self):
        try:
            import faiss
        except ImportError:
            self.skipTest("Skipping test of faiss backend, faiss not found.")
        self.matching.n_neighbors = 3
        from causallib.contrib import faissknn
        def faiss_ivfflat(metric): return faissknn.FaissNearestNeighbors(index_type="ivfflat",metric=metric)
        def faiss_flatl2(metric): return faissknn.FaissNearestNeighbors(index_type="flatl2",metric=metric)
        for data in self.test_data:
            X, a, y = data

            def get_estimate_for_backend(backend, metric):
                self.matching.knn_backend = backend(metric) if callable(backend) else backend
                self.matching.metric = metric
                self.matching.fit(X, a, y)
                
                with warnings.catch_warnings():  # for some models this will throw a UserWarning
                    warnings.simplefilter(action="ignore", category=UserWarning)
                    Y = self.matching.estimate_population_outcome(X, a, y)
                return Y.values

            euclidean_results, mahalanobis_results = [[], []]
            for backend in ["sklearn", faiss_ivfflat, faiss_flatl2]:
                euclidean_results.append(
                    get_estimate_for_backend(backend, "euclidean"))
                mahalanobis_results.append(
                    get_estimate_for_backend(backend, "mahalanobis")
                )
            euclidean_results = np.array(euclidean_results)
            mahalanobis_results = np.array(mahalanobis_results)
            np.testing.assert_allclose(
                euclidean_results - euclidean_results[0], 0)
            np.testing.assert_allclose(
                mahalanobis_results - mahalanobis_results[0], 0)

    def _get_nearest_match_distances_for_treatment_value(self, treatment_value, a):

        return (self.matching.match_df_.loc[treatment_value][a != treatment_value]
                .distances.apply(lambda x: x[0] if x else np.nan)
                .dropna()
                .values)


    def test_is_pickleable(self):
        X, a, y = self.data_serial_unbalanced_x
        self.matching.fit(X,a,y)
        self.matching.match(X,a,)
        prepickle_covariates = self.matching.get_covariates_of_matches(0, 1, X)
        prepickle_estimate = self.matching.estimate_population_outcome(X, a).values
        ms = pickle.dumps(self.matching)

        m2 = pickle.loads(ms)
        postpickle_estimate = m2.estimate_population_outcome(X, a).values
        postpickle_covariates = m2.get_covariates_of_matches(0, 1, X)

        np.testing.assert_array_equal(
            prepickle_estimate, postpickle_estimate)
        np.testing.assert_array_equal(
            prepickle_covariates, postpickle_covariates)

    def test_matching_one_way_works_even_when_other_is_undefined(self):
        X, a, y = self.data_serial_unbalanced_x
        for n_neighbors in [5, 20, 50]:
            self.check_matching_too_few_neighbors_adapts_matches(n_neighbors, X, a, y)

    def check_matching_too_few_neighbors_adapts_matches(self, n_neighbors, X, a, y):
        matching = Matching(n_neighbors=n_neighbors, matching_mode="treatment_to_control")
        matching.fit(X,a,y)
        match_df = matching.match(X, a)
        n_matches_actual = match_df.distances.apply(len).groupby(level=0).max()
        self.assertEqual(n_matches_actual[0], min(n_neighbors, self.n))
        self.assertEqual(n_matches_actual[1], min(n_neighbors, self.k))
        