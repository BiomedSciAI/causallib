# (C) Copyright 2019 IBM Corp.
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
#
# Created on Oct 24, 2019

import unittest

from pandas import DataFrame, Series

from causallib.datasets import load_nhefs, load_acic16


class BaseTestDatasets(unittest.TestCase):
    def ensure_return_types(self, loader):
        data = loader()
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
    def test_return_types(self):
        self.ensure_return_types(load_nhefs)

    def test_dimensions_agree(self):
        with self.subTest("Test restrict=True"):
            data = load_nhefs(restrict=True)
            self.ensure_dimensions_agree(data)

        with self.subTest("Test restrict=False"):
            data = load_nhefs(restrict=False)
            self.ensure_dimensions_agree(data)

    def test_raw_parameter(self):
        with self.subTest("Test raw=True"):
            data = load_nhefs(raw=True)
            self.assertTrue(isinstance(data, tuple))
            self.assertEqual(len(data), 2)  # 2 = data and descriptors
            self.assertTrue(isinstance(data[0], DataFrame))
            self.assertTrue(isinstance(data[1], Series))

        with self.subTest("Test raw=False"):
            # Already asserted in test_return_Xay_parameter, return_Xay=True
            self.assertTrue(True)

    def test_restrict_parameter(self):
        with self.subTest("Test restrict=True"):
            data = load_nhefs(restrict=True)
            self.assertFalse(data.y.isnull().any())

        with self.subTest("Test restrict=False"):
            data = load_nhefs(restrict=False)
            self.assertTrue(data.y.isnull().any())


class TestACIC16(BaseTestDatasets):
    def test_return_types(self):
        self.ensure_return_types(load_acic16)
        data = load_acic16()
        self.assertTrue(hasattr(data, "po"))
        self.assertTrue(isinstance(data.po, DataFrame))

    def test_dimensions_agree(self):
        for i in range(1, 11):
            with self.subTest("Test dimension for instance {}".format(i)):
                data = load_acic16(i)
                self.ensure_dimensions_agree(data)
                self.assertEqual(data.X.shape[0], data.po.shape[0])
                self.assertEqual(data.po.shape[1], 2)

    def test_non_dummy_loading(self):
        X_dummy = load_acic16(raw=False).X
        X_factor, zymu = load_acic16(raw=True)
        self.assertEqual(X_factor.shape[0], X_dummy.shape[0])
        self.assertEqual(X_factor.shape[0], zymu.shape[0])
        self.assertEqual(5, zymu.shape[1])
        self.assertGreater(X_dummy.shape[1], X_factor.shape[1])     # Dummies has more columns
