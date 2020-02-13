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

Created on Aug 29, 2018

"""
import abc
import unittest

import pandas as pd
import numpy as np

from causallib.preprocessing.transformers import StandardScaler, MinMaxScaler


class TestTransformers:
    @abc.abstractmethod
    def test_fit(self):
        pass

    @abc.abstractmethod
    def test_fit_technical(self):
        pass

    @abc.abstractmethod
    def test_transform(self):
        pass

    @abc.abstractmethod
    def test_inverse_transform(self):
        pass


class TestStandardScaler(TestTransformers, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({"binary": [4, 5, 5, np.nan],
                                 "continuous": [0, 2, 4, np.nan]})
        cls.transformer = StandardScaler(with_mean=True, with_std=True, ignore_nans=True)
        cls.transformer.fit(cls.data)

    def test_fit_technical(self):
        with self.subTest("Has mean_ attribute"):
            self.assertTrue(hasattr(self.transformer, "mean_"))

        with self.subTest("Has scale_ attribute"):
            self.assertTrue(hasattr(self.transformer, "scale_"))

        with self.subTest("Applied on the right amount of columns"):
            # Should only be applied on "continuous" column
            self.assertEqual(1, len(self.transformer.mean_))
            self.assertEqual(1, len(self.transformer.scale_))

    def test_fit(self):
        with self.subTest("Test means are right"):
            self.assertEqual(2.0, self.transformer.mean_["continuous"])

        with self.subTest("Test scale is correct"):
            self.assertEqual(2.0, self.transformer.scale_["continuous"])

    def test_transform(self):
        transformed = self.transformer.transform(self.data)

        with self.subTest("Was not applied on binary column"):
            pd.testing.assert_series_equal(self.data["binary"], transformed["binary"])

        with self.subTest("Result is right on the transformed column"):
            pd.testing.assert_series_equal(transformed["continuous"], pd.Series([-1.0, 0.0, 1.0, np.nan]),
                                           check_names=False)

    def test_inverse_transform(self):
        untransformed = self.transformer.inverse_transform(self.transformer.transform(self.data))
        pd.testing.assert_frame_equal(self.data, untransformed)


class TestMinMaxScaler(TestTransformers, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({"binary": [4, 5, 5, np.nan],
                                 "continuous": [0, 2, 4, np.nan]})
        cls.transformer = MinMaxScaler(only_binary_features=True, ignore_nans=True)
        cls.transformer.fit(cls.data)

    def test_fit_technical(self):
        with self.subTest("Has min_ attribute"):
            self.assertTrue(hasattr(self.transformer, "min_"))

        with self.subTest("Has max_ attribute"):
            self.assertTrue(hasattr(self.transformer, "max_"))

        with self.subTest("Has scale_ attribute"):
            self.assertTrue(hasattr(self.transformer, "scale_"))

        with self.subTest("Applied on the right amount of columns"):
            # Should only be applied on "continuous" column
            self.assertEqual(1, len(self.transformer.min_))
            self.assertEqual(1, len(self.transformer.max_))
            self.assertEqual(1, len(self.transformer.scale_))

    def test_fit(self):
        with self.subTest("Test min is right"):
            self.assertEqual(4.0, self.transformer.min_["binary"])

        with self.subTest("Test max is right"):
            self.assertEqual(5.0, self.transformer.max_["binary"])

        with self.subTest("Test scale is correct"):
            self.assertEqual(1.0, self.transformer.scale_["binary"])

    def test_transform(self):
        transformed = self.transformer.transform(self.data)

        with self.subTest("Was not applied on binary column"):
            pd.testing.assert_series_equal(self.data["continuous"], transformed["continuous"])

        with self.subTest("Result is right on the transformed column"):
            pd.testing.assert_series_equal(transformed["binary"], pd.Series([0.0, 1.0, 1.0, np.nan]),
                                           check_names=False)

    def test_inverse_transform(self):
        untransformed = self.transformer.inverse_transform(self.transformer.transform(self.data))
        pd.testing.assert_frame_equal(self.data, untransformed)
