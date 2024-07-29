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

Created on Feb 21, 2019

"""

import unittest
import warnings

import pandas as pd

from causallib.utils import general_tools
from causallib.utils.stat_utils import robust_lookup
from causallib.utils.exceptions import ColumnNameChangeWarning


class TestUtils(unittest.TestCase):
    def ensure_learner_is_fitted(self, unfitted_learner, X, y):
        model_name = str(unfitted_learner.__class__).split(".")[-1]
        with self.subTest("Check is_fitted of {}".format(model_name)):
            self.assertFalse(general_tools.check_learner_is_fitted(unfitted_learner))
            unfitted_learner.fit(X, y)
            self.assertTrue(general_tools.check_learner_is_fitted(unfitted_learner))

    def test_check_classification_learner_is_fitted(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from sklearn.datasets import make_classification
        X, y = make_classification()
        for clf in [LogisticRegression(solver='lbfgs'), DecisionTreeClassifier(),
                    RandomForestClassifier(), LinearSVC()]:
            self.ensure_learner_is_fitted(clf, X, y)

    def test_check_regression_learner_is_fitted(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.datasets import make_regression
        X, y = make_regression()
        for regr in [LinearRegression(), ExtraTreeRegressor(),
                     GradientBoostingRegressor(), SVR()]:
            self.ensure_learner_is_fitted(regr, X, y)

    def test_robust_lookup(self):
        import pandas as pd
        with self.subTest("zero-one columns, range index"):
            X = pd.DataFrame({
                0: [10, 20, 30, 40],
                1: [100, 200, 300, 400]
            })
            a = pd.Series([0, 1, 0, 1])
            extracted = robust_lookup(X, a)
            expected = pd.Series([10, 200, 30, 400])
            pd.testing.assert_series_equal(expected, extracted)

        with self.subTest("integer columns, range index"):
            X = pd.DataFrame({
                3: [10, 20, 30, 40],
                4: [100, 200, 300, 400]
            })
            a = pd.Series([3, 4, 3, 4])
            extracted = robust_lookup(X, a)
            expected = pd.Series([10, 200, 30, 400])
            pd.testing.assert_series_equal(expected, extracted)

        with self.subTest("text columns, range index"):
            X = pd.DataFrame({
                "a": [10, 20, 30, 40],
                "b": [100, 200, 300, 400]
            })
            a = pd.Series(["a", "b", "a", "b"])
            extracted = robust_lookup(X, a)
            expected = pd.Series([10, 200, 30, 400])
            pd.testing.assert_series_equal(expected, extracted)

        with self.subTest("text columns, text index"):
            X = pd.DataFrame({
                "a": [10, 20, 30, 40],
                "b": [100, 200, 300, 400]
            }, index=["w", "x", "y", "z"])
            a = pd.Series(["a", "b", "a", "b"], index=["w", "x", "y", "z"])
            extracted = robust_lookup(X, a)
            expected = pd.Series([10, 200, 30, 400], index=["w", "x", "y", "z"])
            pd.testing.assert_series_equal(expected, extracted)

        with self.subTest("index mismatch"):
            X = pd.DataFrame({
                0: [10, 20, 30, 40],
                1: [100, 200, 300, 400]
            }, index=[0, 1, 2, 3])
            a = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 4])
            extracted = robust_lookup(X, a)
            expected = pd.Series([10, 200, 30, None], index=[0, 1, 2, 4])
            pd.testing.assert_series_equal(expected, extracted)

        with self.subTest("columns mismatch"):
            X = pd.DataFrame({
                0: [10, 20, 30, 40],
                1: [100, 200, 300, 400]
            })
            a = pd.Series([0, 1, 0, 2])
            extracted = robust_lookup(X, a)
            expected = pd.Series([10, 200, 30, None])
            pd.testing.assert_series_equal(expected, extracted)


class TestColumnNameSafeJoin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = pd.DataFrame([[1, 10], [2, 20], [3, 30], [4, 40]])  # 4x2 dataframe
        cls.a = pd.Series([0, 0, 1, 1])
        cls.A = cls.X.copy()
        assert cls.X.shape[0] == cls.a.shape[0]

    # def setUp(self):
    #     # `ColumnNameChangeWarning` to act as an exception to be able to catch
    #     warnings.simplefilter("error", ColumnNameChangeWarning)

    def join_and_ensure_a_single_type(self, X, a):
        Xa = pd.concat([a, X], axis="columns")
        types = {type(col) for col in Xa.columns}
        self.assertEqual(1, len(types))

    def test_renaming_X_string_a_string(self):
        # `ColumnNameChangeWarning` to act as an exception to be able to catch
        warnings.simplefilter("error", ColumnNameChangeWarning)

        X = self.X.rename(columns={0: "x0", 1: "x1"})
        a = self.a.rename("a")
        # No `ColumnNameChangeWarning` should be raised
        reX, rea = general_tools.align_column_name_types_for_join(X, a)
        pd.testing.assert_frame_equal(X, reX)
        pd.testing.assert_series_equal(a, rea)
        self.join_and_ensure_a_single_type(reX, rea)

    def test_renaming_X_numeric_a_numeric(self):
        # `ColumnNameChangeWarning` to act as an exception to be able to catch
        warnings.simplefilter("error", ColumnNameChangeWarning)

        X = self.X.rename(columns={0: 0, 1: 1})
        a = self.a.rename(2)
        # No `ColumnNameChangeWarning` should be raised
        reX, rea = general_tools.align_column_name_types_for_join(X, a)
        pd.testing.assert_frame_equal(X, reX)
        pd.testing.assert_series_equal(a, rea)
        self.join_and_ensure_a_single_type(reX, rea)

    def test_renaming_X_string_a_numeric(self):
        X = self.X.rename(columns={0: "x0", 1: "x1"})
        a = self.a.rename(0)
        with self.assertWarns(ColumnNameChangeWarning):
            reX, rea = general_tools.align_column_name_types_for_join(X, a)
        pd.testing.assert_frame_equal(X, reX)
        self.assertEqual("0", rea.name)  # Stringify its name
        self.join_and_ensure_a_single_type(reX, rea)

    def test_renaming_X_numeric_a_string(self):
        X = self.X.rename(columns={0: 0, 1: 1})
        a = self.a.rename("a")
        with self.assertWarns(ColumnNameChangeWarning):
            reX, rea = general_tools.align_column_name_types_for_join(X, a)
        self.assertListEqual(["0", "1"], reX.columns.tolist())
        pd.testing.assert_series_equal(a, rea)
        self.join_and_ensure_a_single_type(reX, rea)

    def test_renaming_overwriting_a_name(self):
        X = self.X.rename(columns={0: "x0", 1: "x1"})
        a = self.a.rename("a")
        # with self.assertWarns(ColumnNameChangeWarning):
        reX, rea = general_tools.align_column_name_types_for_join(X, a, a_name="w")
        pd.testing.assert_frame_equal(X, reX)
        self.assertEqual("w", rea.name)  # Stringify its name
        self.join_and_ensure_a_single_type(reX, rea)

    def test_renaming_no_a_name(self):
        X = self.X.rename(columns={0: "x0", 1: "x1"})
        a = self.a.rename(None)
        with self.assertWarns(ColumnNameChangeWarning):
            reX, rea = general_tools.align_column_name_types_for_join(X, a)
        pd.testing.assert_frame_equal(X, reX)
        self.assertEqual("a", rea.name)  # Default value
        self.join_and_ensure_a_single_type(reX, rea)

    def test_renaming_X_string_A_numeric(self):
        X = self.X.rename(columns={0: "x0", 1: "x1"})
        A = self.A.copy()
        with self.assertWarns(ColumnNameChangeWarning):
            reX, reA = general_tools.align_column_name_types_for_join(X, A)
        pd.testing.assert_frame_equal(X, reX)
        self.assertEqual(["a_0", "a_1"], reA.columns.tolist())
        self.join_and_ensure_a_single_type(reX, reA)

    def test_renaming_X_numeric_A_string(self):
        X = self.X.rename(columns={0: 0, 1: 1})
        A = self.A.rename(columns={0: "a0", 1: "a1"})
        with self.assertWarns(ColumnNameChangeWarning):
            reX, reA = general_tools.align_column_name_types_for_join(X, A)
        self.assertListEqual(["0", "1"], reX.columns.tolist())
        self.assertListEqual(["a_a0", "a_a1"], reA.columns.tolist())
        self.join_and_ensure_a_single_type(reX, reA)

    def test_renaming_X_mixed_a_string(self):
        X = self.X.rename(columns={0: "x0", 1: 1})
        a = self.a.rename("a")
        with self.assertWarns(ColumnNameChangeWarning):
            reX, rea = general_tools.align_column_name_types_for_join(X, a)
        self.assertListEqual(["x0", "1"], reX.columns.tolist())
        pd.testing.assert_series_equal(a, rea)
        self.join_and_ensure_a_single_type(reX, rea)

    def test_renaming_X_mixed_a_numeric(self):
        X = self.X.rename(columns={0: "x0", 1: 1})
        a = self.a.rename(2)
        with self.assertWarns(ColumnNameChangeWarning):
            reX, rea = general_tools.align_column_name_types_for_join(X, a)
        self.assertListEqual(["x0", "1"], reX.columns.tolist())
        self.assertEqual("2", rea.name)  # Stringify its name
        self.join_and_ensure_a_single_type(reX, rea)

    # # ##Scikit-learn doesn't seem to care about duplicated names in `X` ## # #
    # def test_renaming_X_numeric_a_numeric_duplicated(self):
    #     X = self.X.rename(columns={0: 0, 1: 1})
    #     a = self.a.rename(0)
    #     # with self.assertWarns(ColumnNameChangeWarning):
    #     reX, rea = general_tools.align_column_name_types_for_join(X, a)
    #     pd.testing.assert_frame_equal(X, reX)
    #     self.assertEqual(2, rea.name)  # max(cols)=1 + 1 = 2
    #     self.join_and_ensure_a_single_type(reX, rea)

    def test_join_X_full_a_full(self):
        X = self.X.rename(columns={0: "x0", 1: "x1"})
        a = self.a.rename(None)
        Xa = general_tools.column_name_type_safe_join(X, a)

        with self.subTest("Column types"):
            types = {type(col) for col in Xa.columns}
            self.assertSetEqual({str}, types)

        with self.subTest("Content"):
            self.assertListEqual(a.tolist(), Xa.iloc[:, 0].tolist())
            pd.testing.assert_frame_equal(X, Xa.iloc[:, 1:])

    def test_join_X_empty_a_full(self):
        X = pd.DataFrame(index=self.a.index)
        a = self.a.rename("a")
        Xa = general_tools.column_name_type_safe_join(X, a)

        with self.subTest("Column types"):
            types = {type(col) for col in Xa.columns}
            self.assertSetEqual({str}, types)

        with self.subTest("Content"):
            self.assertEqual(1, Xa.shape[1])
            self.assertListEqual(a.tolist(), Xa.squeeze().tolist())

    def test_join_X_full_a_empty(self):
        X = self.X.copy()
        a = pd.Series(dtype=object)
        Xa = general_tools.column_name_type_safe_join(X, a)

        with self.subTest("Column types"):
            types = {type(col) for col in Xa.columns}
            self.assertSetEqual({int}, types)

        with self.subTest("Content"):
            self.assertEqual(2, Xa.shape[1])
            pd.testing.assert_frame_equal(X, Xa)

    def test_join_X_full_A_full(self):
        X = self.X.rename(columns={0: "x0", 1: "x1"})
        A = self.A.copy()
        Xa = general_tools.column_name_type_safe_join(X, A)

        with self.subTest("Column types"):
            types = {type(col) for col in Xa.columns}
            self.assertSetEqual({str}, types)

        with self.subTest("Content"):
            A.columns = ["_0", "_1"]
            pd.testing.assert_frame_equal(A, Xa.iloc[:, :2])
            pd.testing.assert_frame_equal(X, Xa.iloc[:, 2:])

    def test_join_X_empty_A_full(self):
        X = pd.DataFrame()
        A = self.A.copy()
        Xa = general_tools.column_name_type_safe_join(X, A)

        with self.subTest("Column types"):
            types = {type(col) for col in Xa.columns}
            self.assertSetEqual({int}, types)

        with self.subTest("Content"):
            self.assertEqual(2, Xa.shape[1])
            pd.testing.assert_frame_equal(A, Xa)

    def test_join_X_full_A_empty(self):
        X = self.X.rename(columns={0: "x0", 1: "x1"})
        A = pd.DataFrame()
        Xa = general_tools.column_name_type_safe_join(X, A)

        with self.subTest("Column types"):
            types = {type(col) for col in Xa.columns}
            self.assertSetEqual({str}, types)

        with self.subTest("Content"):
            self.assertEqual(2, Xa.shape[1])
            pd.testing.assert_frame_equal(X, Xa)
