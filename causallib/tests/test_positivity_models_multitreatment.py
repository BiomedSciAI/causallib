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
# pylint:disable=protected-access,missing-function-docstring,missing-module-docstring,missing-class-docstring

import unittest

from scipy.special import comb
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from causallib.positivity.multiple_treatment_positivity import (
    OneVersusRestPositivity,
    OneVersusAnotherPositivity,
    )

from causallib.positivity import Trimming, UnivariateBoundingBox


class MultipleTreatmentPositivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_samples=500,
            n_features=3,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.01,
            class_sep=1.0,
            random_state=899,
        )
        X, a = X[:, :-1], pd.Series(
            pd.qcut(X[:, -1], 5, labels=[1, 2, 3, 4, 5], retbins=True)[0].to_numpy()
        )
        ab = a % 2
        cls.data = {
            "X": pd.DataFrame(X),
            "a": pd.Series(a),
            "ab": pd.Series(ab),
            "y": pd.Series(y),
        }

    def _get_data_set(self, bin_treat=False):
        if bin_treat:
            return self.data["X"], self.data["ab"], self.data["y"]
        else:
            return self.data["X"], self.data["a"], self.data["y"]

    def test_is_the_one_versus_all_fits(self):
        overlap_calculator = Trimming(LogisticRegression())
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        self.assertTrue(
            len(onvsall.positivity_estimators_.keys()) == len(self.data["a"].unique())
        )

    def test_is_the_one_versus_all_fits_transform(self):
        overlap_calculator = Trimming(LogisticRegression())
        onvsall = OneVersusRestPositivity(overlap_calculator)
        a = onvsall.fit_transform(self.data["X"], self.data["a"])
        self.assertEqual(
            len(onvsall.positivity_estimators_.keys()), len(self.data["a"].unique())
        )

    def test_is_the_one_versus_all_predicts(self):
        overlap_calculator = UnivariateBoundingBox(quantile_alpha=0.05)
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.predict(self.data["X"])
        self.assertTrue(pred.mean() != 1)

    def test_is_the_one_versus_all_positivity_profile(self):
        overlap_calculator = UnivariateBoundingBox(quantile_alpha=0.05)
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.positivity_profile(self.data["X"])
        svec = [self.data["X"].shape[0], len(self.data["a"].unique())]
        self.assertSequenceEqual(pred.shape, svec)

    def test_is_the_one_versus_all_positivity_profile(self):
        overlap_calculator = UnivariateBoundingBox(quantile_alpha=0.05)
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall._treatment_positivity_summary_contingency(self.data["X"])
        self.assertEqual(pred.shape[0], pred.shape[1])

    def test_is_the_one_versus_all_pred_length(self):
        overlap_calculator = UnivariateBoundingBox(quantile_alpha=0.05)
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.predict(self.data["X"])
        self.assertTrue(len(pred) == len(self.data["a"]))

    def test_is_the_one_versus_all_pred_values(self):
        overlap_calculator = UnivariateBoundingBox(quantile_alpha=0.2)
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.predict(self.data["X"])
        np.testing.assert_array_equal(np.sort(np.unique(pred)), [False, True])

    def test_is_the_one_versus_all_pred_values_trimming(self):
        overlap_calculator = Trimming(LogisticRegression())
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.predict(self.data["X"])
        np.testing.assert_array_equal(np.sort(np.unique(pred)), [False, True])

    def test_is_the_one_versus_all_profile(self):
        overlap_calculator = UnivariateBoundingBox(quantile_alpha=0.2)
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.positivity_profile(self.data["X"])
        np.testing.assert_array_equal(np.sort(np.unique(pred)[::]), [False, True])

    def test_is_the_one_versus_all_profile_different(self):
        overlap_calculator = UnivariateBoundingBox(quantile_alpha=0.2)
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.positivity_profile(self.data["X"])
        eq_val = np.sum(pred, axis=1).mean()
        self.assertTrue((eq_val != 0) and (eq_val != len(self.data["a"].unique())))

    def test_is_the_one_versus_all_profile_different_trimming(self):
        overlap_calculator = Trimming(LogisticRegression())
        onvsall = OneVersusRestPositivity(overlap_calculator)
        onvsall.fit(self.data["X"], self.data["a"])
        pred = onvsall.positivity_profile(self.data["X"])
        eq_val = np.sum(pred, axis=1).mean()
        self.assertTrue((eq_val != 0) and (eq_val != len(self.data["a"].unique())))

    def test_logis_one_versus_all(self):
        positivity_estimator = OneVersusRestPositivity(Trimming(LogisticRegression()))
        positivity_estimator.fit(self.data["X"], self.data["a"])
        pred = positivity_estimator.positivity_profile(self.data["X"])
        eq_val = np.sum(pred, axis=1).mean()
        self.assertTrue(eq_val != 0)

    def test_logis_one_versus_all_fit_transform(self):
        positivity_estimator = OneVersusRestPositivity(Trimming(LogisticRegression()))
        pred = positivity_estimator.fit_transform(self.data["X"], self.data["a"])
        pr = positivity_estimator.predict(self.data["X"], self.data["a"])
        self.assertEqual(pred[0].shape[0] ,np.sum(pr))

    def test_logis_one_versus_all_predict_transform(self):
        positivity_estimator = OneVersusRestPositivity(Trimming(LogisticRegression()))
        positivity_estimator.fit(self.data["X"], self.data["a"])
        pred = positivity_estimator.transform(self.data["X"], self.data["a"])
        pr = positivity_estimator.predict(self.data["X"], self.data["a"])
        self.assertEqual(pred[0].shape[0], np.sum(pr))

    def test_one_vs_another_fits(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression())
        )
        positivity_estimator.fit(self.data["X"], self.data["a"])
        profile = positivity_estimator.positivity_profile(self.data["X"])
        self.assertEqual(len(profile.columns), comb(len(self.data["a"].unique()), 2))

    def test_one_vs_another_transform(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression())
        )
        positivity_estimator.fit(self.data["X"], self.data["a"])
        pos_X, pos_a = positivity_estimator.transform(self.data["X"], self.data["a"])
        self.assertLess(pos_X.shape[0], self.data["X"].shape[0])

    def test_one_vs_another_fit_transform(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression())
        )
        res = positivity_estimator.fit_transform(
            self.data["X"], self.data["a"]
        )
        self.assertLess(res[0].shape[0], self.data["X"].shape[0])

    def test_one_vs_another_treat_report(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression())
        )
        positivity_estimator.fit(self.data["X"], self.data["a"])
        treatment_rep = positivity_estimator._treatment_positivity_summary_contingency(
            self.data["X"])
        self.assertEqual(len(treatment_rep.columns), len(self.data["a"].unique()))
        self.assertEqual(len(treatment_rep.index), len(self.data["a"].unique()))

    def test_one_vs_another_treat_report_direct(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression())
        )
        positivity_estimator.fit(self.data["X"], self.data["a"])
        treatment_rep = positivity_estimator.treatment_positivity_summary(
            self.data["X"], as_contingency_table=True)
        self.assertEqual(len(treatment_rep.columns), len(self.data["a"].unique()))
        self.assertEqual(len(treatment_rep.index), len(self.data["a"].unique()))


    def test_one_vs_another_treat_report_control(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression()), treatment_pairs_list=5
        )
        positivity_estimator.fit(self.data["X"], self.data["a"])
        treatment_rep = positivity_estimator.treatment_positivity_summary(
            self.data["X"])
        self.assertEqual(len(treatment_rep.index), len(self.data["a"].unique())-1)


    def test_treatment_population_selection(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression()), treatment_pairs_list=5
        )
        positivity_estimator.fit(self.data["X"], self.data["a"])
        profile = positivity_estimator.positivity_profile(self.data["X"])
        self.assertEqual(profile.isna().sum().sum(), 0)

    def test_treatment_population_selection_both(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression()))
        positivity_estimator.fit(self.data["X"], self.data["a"])
        profile = positivity_estimator.positivity_profile(self.data["X"], self.data["a"], estimation_population='both')
        for cl in profile.columns:
            self.assertEqual(profile[cl].notna().sum(),  self.data["a"].isin(cl).sum())

    def test_treatment_population_selection_treatment(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression()))
        positivity_estimator.fit(self.data["X"], self.data["a"])
        profile = positivity_estimator.positivity_profile(self.data["X"], self.data["a"], estimation_population='treated')
        for cl in profile.columns:
            self.assertEqual(profile[cl].notna().sum(),  (self.data["a"]==cl[0]).sum())


    def test_treatment_population_selection_control(self):
        positivity_estimator = OneVersusAnotherPositivity(
            Trimming(LogisticRegression()))
        positivity_estimator.fit(self.data["X"], self.data["a"])
        profile = positivity_estimator.positivity_profile(self.data["X"], self.data["a"], estimation_population='control')
        for cl in profile.columns:
            self.assertEqual(profile[cl].notna().sum(),  (self.data["a"]==cl[1]).sum())
