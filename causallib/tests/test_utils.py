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

from causallib.utils import general_tools


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
