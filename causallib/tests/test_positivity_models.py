import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

from causallib.positivity import (
    Trimming, UnivariateBoundingBox, Matching
)
from causallib.positivity.univariate_bbox import (
    QuantileContinuousSupport, Support, ContinuousSupport, CategoricalSupport
)
from causallib.positivity.datasets.positivity_data_simulator import (
    make_1d_overlap_data, make_1d_normal_distribution_overlap_data, make_random_y_like
)


class TestTrimming(unittest.TestCase):

    # TODO: check the minimization criterion - following Crump's paper

    @classmethod
    def setUpClass(cls):
        # Propensity model without regularization
        cls.trimming = Trimming(
            learner=LogisticRegression(solver="sag", penalty=None),
            threshold="auto"
        )

    def test_is_fitted(self):
        X, a = make_1d_normal_distribution_overlap_data()
        self.trimming.fit(X, a)
        self.assertTrue(hasattr(self.trimming.learner, "coef_"))
        self.assertTrue(isinstance(self.trimming.threshold_, float))

    def test_not_trimming_1d_same_distribution(self):
        # Treated and control are drawn from the same distribution with
        # probability to be treated = 0.5
        X, a = make_1d_normal_distribution_overlap_data()
        [X_trimmed, a_trimmed] = self.trimming.fit_transform(X, a)
        self.assertEqual(a_trimmed.shape[0], a.shape[0])

        # Treated and control are drawn from the same distribution with
        # probability to be treated = 0.2
        X, a = make_1d_normal_distribution_overlap_data(probability_treated=0.2)
        [X_trimmed, a_trimmed] = self.trimming.fit_transform(X, a)
        self.assertEqual(a_trimmed.shape[0], a.shape[0])

    def test_trimming_different_distribution(self):
        X, a = make_1d_normal_distribution_overlap_data(
            treatment_params=(-2, 2),
            control_params=(2, 1),
            probability_treated=0.4,
            n_samples=1000
        )
        [X_trimmed_crump, a_trimmed_crump] = self.trimming.fit_transform(X, a)
        n_untrimmed_crump = a_trimmed_crump.shape[0]
        with self.subTest("Trimmed samples with crump"):
            self.assertTrue(n_untrimmed_crump < a.shape[0])

        n_untrimmed_big_threshold = \
            self.trimming.predict(X, a, threshold=0.25).sum()
        with self.subTest("Trimmed samples with large threshold"):
            self.assertTrue(n_untrimmed_big_threshold < a.shape[0])

        with self.subTest("Test crump has more sample than large threshold"):
            self.assertTrue(n_untrimmed_big_threshold < n_untrimmed_crump)

    def test_pipeline(self):
        X, a = make_1d_normal_distribution_overlap_data()

        with self.subTest("Test fit with propensity model"):
            self.trimming.fit(X, a)
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test predict of the filtered data"):
            self.trimming.predict(X, a)
            self.assertTrue(True)  # Dummy assert for not thrown exception

        with self.subTest("Test transform"):
            self.trimming.transform(X, a)
            self.assertTrue(True)  # Dummy assert for not thrown exception

    def test_hardcoded_threshold_initialization_chooses_no_method(self):
        self.trimming = Trimming(threshold=0.1)
        self.assertFalse(hasattr(self.trimming, 'threshold'))

    def test_default_initialization_chooses_crump(self):
        self.trimming = Trimming()
        self.assertEqual(self.trimming.threshold, "crump")



def inbounds(bounds):
    return lambda x: x >= bounds[0] and x <= bounds[1]


class TestUnivariateBBox(unittest.TestCase):
    def assertHasAttr(self, obj, intendedAttr):
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(testBool)

    @classmethod
    def setUpClass(cls):
        cls.bbox = UnivariateBoundingBox()
        cls.treatment_bounds_1d = (0, 75)
        cls.control_bounds_1d = (0, 75)
        cls.overlap_data_discrete_integer = make_1d_overlap_data(
            treatment_bounds=cls.treatment_bounds_1d,
            control_bounds=cls.control_bounds_1d)

        cls.treatment_params = (0, 1)
        cls.control_params = (.5, 1)
        cls.overlap_data_normal = make_1d_normal_distribution_overlap_data(
            treatment_params=cls.treatment_params, control_params=cls.control_params)

    def test_predict_is_series(self):
        X, a = self.overlap_data_discrete_integer
        self.bbox.fit(X, a)
        pred = self.bbox.predict(X, a)
        self.assertIsInstance(pred, pd.Series)

    def test_fit_makes_support_(self):
        X, a = self.overlap_data_discrete_integer
        self.bbox.fit(X, a)
        self.assertHasAttr(self.bbox, "joint_support_")

    def test_fit_predict_trues_are_correct_integers(self):
        X, a = self.overlap_data_discrete_integer
        pred = self.bbox.fit_predict(X, a)
        self.assertGreaterEqual(
            X[pred].squeeze().min(),
            max(self.treatment_bounds_1d[0], self.control_bounds_1d[0]))
        self.assertLessEqual(
            X[pred].squeeze().max(),
            min(self.treatment_bounds_1d[1], self.control_bounds_1d[1]))

    def test_fit_predict_falses_are_correct_integers(self):
        X, a = self.overlap_data_discrete_integer
        pred = self.bbox.fit_predict(X, a)
        lr1 = self.control_bounds_1d
        lr2 = self.treatment_bounds_1d

        Xinbounds = X[~pred].squeeze().apply(
            lambda x: inbounds(lr1)(x) and inbounds(lr2)(x))
        self.assertFalse(any(Xinbounds))

    def test_fit_predict_trues_are_correct_floats(self):
        X, a = self.overlap_data_normal
        pred, in_both_bounds = self._check_in_bounds(X, a)
        self.assertTrue(all(X[pred].squeeze().apply(in_both_bounds)))

    def test_fit_predict_falses_are_correct_floats(self):
        X, a = self.overlap_data_normal
        pred, in_both_bounds = self._check_in_bounds(X, a)
        self.assertFalse(any(X[~pred].squeeze().apply(in_both_bounds)))

    def _check_in_bounds(self, X, a):
        pred = self.bbox.fit_predict(X, a)
        X = X.squeeze()
        lr1 = self.bbox.fit_column(X[a == 0]).support
        lr2 = self.bbox.fit_column(X[a == 1]).support
        def in_both_bounds(x): return inbounds(lr1)(x) and inbounds(lr2)(x)
        return pred, in_both_bounds

    def test_continuous_support(self):
        X, a = self.overlap_data_normal
        self.bbox.quantile_alpha = None
        X = X.squeeze()

        self._column_matches_continuous_support(X[a == 0])
        self._column_matches_continuous_support(X[a == 1])

    def _column_matches_continuous_support(self, Xt_col):
        Xtsupp = self.bbox.fit_column(Xt_col)
        self.assertListEqual([min(Xt_col), max(Xt_col)], Xtsupp.support)

    def test_continuous_fit(self):
        X, a = self.overlap_data_normal
        self.bbox.quantile_alpha = None
        self.bbox.fit(X, a)
        self._check_manual_minmax_to_bbox_support(X, a, 0)
        self._check_manual_minmax_to_bbox_support(X, a, 1)

    def _check_manual_minmax_to_bbox_support(self, X, a, treatment_value):
        c = X.columns[0]
        minmax = [X[a == treatment_value]
                  [c].min(), X[a == treatment_value][c].max()]
        bbox_support = self.bbox.control_support_[
            c] if treatment_value == 0 else self.bbox.treatment_support_[c]
        self.assertListEqual(minmax, bbox_support.support)

    def test_continuous_support_quantile(self):
        self.bbox.quantile_alpha = 0.1

        X, a = self.overlap_data_normal
        X = X.squeeze()

        q = [self.bbox.quantile_alpha/2, 1 - (self.bbox.quantile_alpha/2)]

        self._fit_column_and_support_matches_quantile(X[a == 1], q)
        self._fit_column_and_support_matches_quantile(X[a == 0], q)

    def _fit_column_and_support_matches_quantile(self, Xt_col, q):
        Xtsupp = self.bbox.fit_column(Xt_col)
        self.assertListEqual(list(Xt_col.quantile(q)), Xtsupp.support)

    def test_continuous_fit_quantile(self):
        X, a = self.overlap_data_normal
        self.bbox.quantile_alpha = 0.1
        self.bbox.fit(X, a)
        self._check_manual_quantile_to_bbox_support(X, a, 0)
        self._check_manual_quantile_to_bbox_support(X, a, 1)

    def _check_manual_quantile_to_bbox_support(self, X, a, treatment_value):
        c = X.columns[0]
        X = X.squeeze()
        minmax = [
            X[a == treatment_value].quantile(self.bbox.quantile_alpha/2),
            X[a == treatment_value].quantile(1 - (self.bbox.quantile_alpha/2))]
        bbox_support = self.bbox.control_support_[
            c] if treatment_value == 0 else self.bbox.treatment_support_[c]
        self.assertListEqual(minmax, bbox_support.support)

    def test_intersecting_nonmatching_supports_raises_value_error(self):
        Xn, an = self.overlap_data_normal
        Xi, ai = self.overlap_data_discrete_integer
        cts_support = self.bbox.fit_column(Xn.squeeze())
        int_support = self.bbox.fit_column(Xi.squeeze())
        with self.assertRaises(ValueError):
            cts_support.intersection(int_support)
        with self.assertRaises(ValueError):
            int_support.intersection(cts_support)

    def test_intersecting_continuous_and_quantile_supports(self):
        Xn, an = self.overlap_data_normal
        Xn = Xn.squeeze()
        cts_support = ContinuousSupport().fit(Xn)
        quantile_support = QuantileContinuousSupport(alpha=0.1).fit(Xn)
        joint_support = cts_support.intersection(quantile_support)
        joint_support2 = quantile_support.intersection(cts_support)
        self.assertEqual(joint_support.support, quantile_support.support)
        self.assertEqual(joint_support2.support, quantile_support.support)

    def test_supports_table(self):
        Xn, an = self.overlap_data_normal
        self.bbox.fit(Xn, an)
        self.assertListEqual(
            sorted(list(self.bbox.supports_table_.columns)),
            sorted(["control", "joint", "treatment", ]))
        self.assertListEqual(sorted(list(self.bbox.supports_table_.index)),
                             sorted(Xn.columns))
        if pd.__version__ > "2.1":
            # TODO: keep some support for pandas <2.1.0 and python 3.8. remove in the future.
            self.bbox.supports_table_.map(lambda x: isinstance(x, Support))
        else:
            self.bbox.supports_table_.applymap(lambda x: isinstance(x, Support))

    def test_subtract_continuous_support_when_equal(self):
        csupport1 = ContinuousSupport(support=[1, 2])
        csupport2 = ContinuousSupport(support=[1, 2])
        self.assertListEqual(csupport1 - csupport2, [0, 0])

    def test_substract_continuous_support_when_unequal(self):
        csupport1 = ContinuousSupport(support=[1, 2])
        csupport2 = ContinuousSupport(support=[2, 5])
        self.assertListEqual(csupport1 - csupport2, [-1, -3])

    def test_subtract_discrete_support_when_equal(self):
        dsupport1 = CategoricalSupport(support=[1, 2, 3])
        dsupport2 = CategoricalSupport(support=[2, 3, 1])
        self.assertFalse(dsupport1 - dsupport2)

    def test_subtract_discrete_support_when_unequal(self):
        dsupport1 = CategoricalSupport(support=[2])
        dsupport2 = CategoricalSupport(support=[2, 3, 1])
        self.assertSetEqual(dsupport1 - dsupport2, {3, 1})

    def test_is_fitted_check(self):
        self.bbox = UnivariateBoundingBox()
        with self.assertRaises(NotFittedError):
            self.bbox.assert_is_fitted()

    def test_str_representation_for_continuous_support(self):
        self.assertEqual(str(ContinuousSupport([0, 1])), "support: [0, 1]")
        self.assertEqual(str(ContinuousSupport()), "support: no support")

    def test_str_representation_for_discrete_support(self):
        self.assertEqual(str(CategoricalSupport([0, 1])), "support: {0, 1}")
        self.assertEqual(str(CategoricalSupport()), "support: no support")

    def test_can_select_continuous_columns(self):
        X, a = self.overlap_data_discrete_integer
        self.bbox = UnivariateBoundingBox(continuous_columns=["X1"])
        self.bbox.fit(X, a)
        self.assertIsInstance(self.bbox.joint_support_[
                              "X1"], ContinuousSupport)

    def test_can_select_categorical_columns(self):
        X, a = self.overlap_data_discrete_integer
        self.bbox = UnivariateBoundingBox(categorical_columns=["X1"])
        self.bbox.fit(X, a)
        self.assertIsInstance(self.bbox.joint_support_[
                              "X1"], CategoricalSupport)

    def test_non_overlapping_intervals_are_none(self):
        s1 = ContinuousSupport().fit([1, 2, 3])
        s2 = ContinuousSupport().fit([4, 5, 6])
        s3 = s1.intersection(s2)
        self.assertIsNone(s3.support)
        s2 = ContinuousSupport().fit([1, 2, 3])
        s1 = ContinuousSupport().fit([4, 5, 6])
        s3 = s1.intersection(s2)
        self.assertIsNone(s3.support)

    def test_overlapping_with_none_is_none(self):
        s1 = ContinuousSupport(support=None)
        s2 = ContinuousSupport().fit([1,2,3])
        s3 = s1.intersection(s2)
        self.assertIsNone(s3.support)

    def test_scaled_support_table(self):
        Xn, an = self.overlap_data_normal
        self.bbox.fit(Xn, an)
        supports_table = self.bbox.supports_table_["treatment"]
        scaled_supports_table = self.bbox.scaled_supports_table_["treatment"]
        something_asserted = False
        for col in Xn.columns:
            if isinstance(supports_table[col].support,list):
                s0,s1 = supports_table[col].support
                ss0,ss1 = scaled_supports_table[col].support
                s = self.bbox.scales_[col]
                self.assertAlmostEqual(s0/s,ss0)
                self.assertAlmostEqual(s1/s,ss1)
                something_asserted = True
        self.assertTrue(something_asserted, "No asserts made, check data")


class TestMatching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matching = Matching()

    def test_1d_euclidean_caliper_1_match(self):
        X, a= make_1d_overlap_data(treatment_bounds=(0, 75),
                                      control_bounds=(25, 100))
        self.matching.matching_transformer.set_params(metric="euclidean", caliper=1.0)
        predict = self.matching.fit_predict(X, a)
        Xp, ap= self.matching.transform(X, a,)
        self.assertEqual(sum(predict), 100)
        self.assertEqual(sum(predict), Xp.shape[0])
        self.assertEqual(Xp.values.min(), 25)
        self.assertEqual(Xp.values.max(), 74)

    def test_transform_multiple_outputs(self):
        X, a= make_1d_overlap_data(treatment_bounds=(0, 75),
                                      control_bounds=(25, 100))
        y = make_random_y_like(a)
        t = make_random_y_like(a)

        self.matching.matching_transformer.set_params(metric="euclidean", caliper=1.01)
        self.matching.fit(X, a)

        Xm, am = self.matching.transform(X, a)
        unique_sample_counts = {i.shape[0] for i in (Xm, am)}
        self.assertEqual(len(unique_sample_counts), 1)

        Xm, am, ym= self.matching.transform(X, a, y)
        unique_sample_counts = {i.shape[0] for i in (Xm, am, ym)}
        self.assertEqual(len(unique_sample_counts), 1)

        Xm, am, ym, tm = self.matching.transform(X, a, y, t)
        unique_sample_counts = {i.shape[0] for i in (Xm, am, ym, tm)}
        self.assertEqual(len(unique_sample_counts), 1)
