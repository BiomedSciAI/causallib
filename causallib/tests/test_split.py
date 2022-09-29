import unittest
import pandas as pd

from causallib.model_selection import TreatmentOutcomeStratifiedKFold
from causallib.model_selection import TreatmentStratifiedKFold


class TestTreatmentOutcomeStratifiedKFold(unittest.TestCase):
    def test_binary_splits(self):
        X = pd.DataFrame({0: [10] * 8})  # Dummy corresponding X
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        kfold = TreatmentOutcomeStratifiedKFold(n_splits=2)
        folds = list(kfold._split(X, a, y))[0]  # 2 folds, so first element suffices

        self.assertListEqual(
            list(folds[0]), [1, 3, 5, 7],
        )
        self.assertListEqual(
            list(folds[1]), [0, 2, 4, 6],
        )

        labels = kfold._combine_treatment_outcome_labels(a, y)
        self.assertListEqual(
            list(labels),
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def ensure_multiclass_splits(self, X, a, y):
        kfold = TreatmentOutcomeStratifiedKFold(n_splits=2)
        folds = list(kfold._split(X, a, y))[0]  # 2 folds, so first element suffices

        self.assertListEqual(
            list(folds[0]),  # Train indices
            [1, 3, 5, 7, 9, 11]
        )
        self.assertListEqual(
            list(folds[1]),  # Train indices
            [0, 2, 4, 6, 8, 10]
        )

    def test_multi_outcome_splits(self):
        X = pd.DataFrame({0: [10] * 12})  # Dummy corresponding X
        a = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y = pd.Series([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
        self.ensure_multiclass_splits(X, a, y)

        labels = TreatmentOutcomeStratifiedKFold._combine_treatment_outcome_labels(a, y)
        self.assertListEqual(
            list(labels),
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        )

    def test_multi_treatment_splits(self):
        X = pd.DataFrame({0: [10] * 12})  # Dummy corresponding X
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        self.ensure_multiclass_splits(X, a, y)

        labels = TreatmentOutcomeStratifiedKFold._combine_treatment_outcome_labels(a, y)
        self.assertListEqual(
            list(labels),
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        )

    def test_continuous_outcome(self):
        X = pd.DataFrame({0: [10] * 4})  # Dummy corresponding X
        a = pd.Series([0, 0, 1, 1])
        y = pd.Series([42.0, 42.1, 42.2, 42.3])
        kfold = TreatmentOutcomeStratifiedKFold(n_splits=2, shuffle=False)
        with self.assertRaises(ValueError):
            kfold._split(X, a, y)

    def test_sklearn_search_compatibility(self):
        X = pd.DataFrame({0: [10] * 8})  # Dummy corresponding X
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        Xa = X.join(a.to_frame("a"))
        kfold = TreatmentOutcomeStratifiedKFold(n_splits=2)
        folds = list(kfold.split(Xa, y))[0]  # 2 folds, so first element suffices

        self.assertListEqual(
            list(folds[0]), [1, 3, 5, 7],
        )
        self.assertListEqual(
            list(folds[1]), [0, 2, 4, 6],
        )


class TestTreatmentStratifiedKFold(unittest.TestCase):
    def test_binary_splits(self):
        X = pd.DataFrame({0: [10] * 8})  # Dummy corresponding X
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])  # should be ignored
        kfold = TreatmentStratifiedKFold(n_splits=2)
        folds = list(kfold._split(X, a, y))[1]  # 2 folds, so first element suffices

        self.assertListEqual(
            list(folds[0]), [0, 1, 4, 5],
        )
        self.assertListEqual(
            list(folds[1]), [2, 3, 6, 7],
        )

    def test_with_no_outcome(self):
        X = pd.DataFrame({0: [10] * 8})  # Dummy corresponding X
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        Xa = X.join(a.to_frame("a"))
        kfold = TreatmentStratifiedKFold(n_splits=2)
        folds = list(kfold.split(Xa))[1]

        self.assertListEqual(list(folds[0]), [0, 1, 4, 5])
        self.assertListEqual(list(folds[1]), [2, 3, 6, 7])

    def test_multiclass_splits(self):
        X = pd.DataFrame({0: [10] * 12})  # Dummy corresponding X
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        kfold = TreatmentStratifiedKFold(n_splits=2)
        folds = list(kfold._split(X, a, y))[1]  # 2 folds, so first element suffices

        self.assertListEqual(
            list(folds[0]), [0, 1, 4, 5, 8, 9],
        )
        self.assertListEqual(
            list(folds[1]), [2, 3, 6, 7, 10, 11],
        )

    def test_against_sklearn_stratified_kfold(self):
        from sklearn.model_selection import StratifiedKFold

        X = pd.DataFrame({0: [10] * 4})
        a = pd.Series([0, 0, 1, 1])
        y = pd.Series([42.0, 42.1, 42.2, 42.3])

        kfold = TreatmentStratifiedKFold(n_splits=2, shuffle=False)
        folds = list(kfold._split(X, a, y))[0]  # 2 folds, so first element suffices

        sklearn_kfold = StratifiedKFold(n_splits=2, shuffle=False)
        sklearn_folds = list(sklearn_kfold.split(X, a))[0]  # 2 folds, so first element suffices
        self.assertListEqual(
            folds[0].tolist(), sklearn_folds[0].tolist()
        )
        self.assertListEqual(
            folds[1].tolist(), sklearn_folds[1].tolist()
        )

    def test_sklearn_search_compatibility(self):
        X = pd.DataFrame({0: [10] * 8})
        a = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        Xa = X.join(a.to_frame("a"))
        kfold = TreatmentStratifiedKFold(n_splits=2)
        folds = list(kfold.split(Xa, y))[1]

        self.assertListEqual(
            list(folds[0]), [0, 1, 4, 5],
        )
        self.assertListEqual(
            list(folds[1]), [2, 3, 6, 7],
        )
