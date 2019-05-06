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

Created on Feb 26, 2019

"""
import unittest
from time import sleep

from pandas import DataFrame, Series

# from causallib.datasets import load_smoking_weight, load_acic16
# from causallib.datasets import load_smoking_weight
from causallib.datasets import fetch_smoking_weight


class BaseTestDatasets(unittest.TestCase):
    def ensure_return_Xay_parameter(self, loader):
        with self.subTest("Test return_Xay=True"):
            data = loader(return_Xay=True)
            self.assertTrue(isinstance(data, tuple))
            self.assertEqual(len(data), 3)  # 3 = X, a, y
            self.assertTrue(isinstance(data[0], DataFrame))
            self.assertTrue(isinstance(data[1], Series))
            self.assertTrue(isinstance(data[2], Series))

        with self.subTest("Test return_Xay=False"):
            data = loader(return_Xay=False)
            self.assertTrue(isinstance(data, dict))
            self.assertTrue(hasattr(data, "X"))
            self.assertTrue(hasattr(data, "a"))
            self.assertTrue(hasattr(data, "y"))
            self.assertTrue(hasattr(data, "descriptors"))
            self.assertTrue(isinstance(data.X, DataFrame))
            self.assertTrue(isinstance(data.a, Series))
            self.assertTrue(isinstance(data.y, Series))

    def ensure_dimensions_agree(self, data):
        self.assertEqual(data.X.shape[0], data.a.shape[0])
        self.assertEqual(data.X.shape[0], data.y.shape[0])


class TestSmokingWeight(BaseTestDatasets):
    def setUp(self):
        # To avoid fast requests to the Harvard server.
        # This is to try and remind that when `fetch` is replaced with `load`,
        # this entire `SetUp` can be removed completely.
        self.assertEqual(fetch_smoking_weight.__name__, "fetch_smoking_weight")
        sleep(5)

    def test_return_Xay_parameter(self):
        self.ensure_return_Xay_parameter(fetch_smoking_weight)

    def test_dimensions_agree(self):
        with self.subTest("Test restrict=True"):
            data = fetch_smoking_weight(restrict=True)
            self.ensure_dimensions_agree(data)

        with self.subTest("Test restrict=False"):
            data = fetch_smoking_weight(restrict=False)
            self.ensure_dimensions_agree(data)

    def test_raw_parameter(self):
        with self.subTest("Test raw=True"):
            data = fetch_smoking_weight(raw=True)
            self.assertTrue(isinstance(data, tuple))
            self.assertEqual(len(data), 2)  # 2 = data and descriptors
            self.assertTrue(isinstance(data[0], DataFrame))
            if data[1] is not None:  # unable to load descriptor Excel file
                self.assertTrue(isinstance(data[1], Series))

        with self.subTest("Test raw=False"):
            # Already asserted in test_return_Xay_parameter, return_Xay=True
            self.assertTrue(True)

    def test_restrict_parameter(self):
        with self.subTest("Test restrict=True"):
            data = fetch_smoking_weight(restrict=True)
            self.assertFalse(data.y.isnull().any())

        with self.subTest("Test restrict=False"):
            data = fetch_smoking_weight(restrict=False)
            self.assertTrue(data.y.isnull().any())

# class TestACIC16(BaseTestDatasets):
#     def test_return_Xay_parameter(self):
#         self.ensure_return_Xay_parameter(load_acic16)
#
#     def test_dimensions_agree(self):
#         for i in range(1, 21):
#             with self.subTest("Test dimension for instance {}".format(i)):
#                 data = load_acic16(i)
#                 self.ensure_dimensions_agree(data)
#
#     def test_non_dummy_loading(self):
#         X_dummy = load_acic16(raw=False).X
#         X_factor = load_acic16(raw=True).X
#         self.assertEqual(X_dummy.shape[0], X_factor.shape[0])
#         self.assertGreater(X_dummy.shape[1], X_factor.shape[1])     # Dummies has more columns
