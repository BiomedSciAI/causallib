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

from sklearn.linear_model import LogisticRegression, LinearRegression
from causallib.evaluation import Evaluator, plot_evaluation_results
from causallib.estimation import AIPW, IPW, StratifiedStandardization
from causallib.datasets import load_nhefs
import unittest

import matplotlib

matplotlib.use("Agg")


class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data = load_nhefs()
        ipw = IPW(LogisticRegression(solver="liblinear"), clip_min=0.05, clip_max=0.95)
        std = StratifiedStandardization(LinearRegression())
        self.dr = AIPW(std, ipw)
        self.dr.fit(self.data.X, self.data.a, self.data.y)
        self.prp_evaluator = Evaluator(self.dr.weight_model)
        self.out_evaluator = Evaluator(self.dr.outcome_model)

    def propensity_plot_by_name(self, test_names, alternate_a=None):
        a = self.data.a if alternate_a is None else alternate_a
        nhefs_results = self.prp_evaluator.evaluate_simple(
            self.data.X,
            a,
            self.data.y,
        )
        nhefs_plots = plot_evaluation_results(
            nhefs_results,
            X=self.data.X,
            a=a,
            y=self.data.y,
            plot_names=test_names,
        )
        [self.assertIsNotNone(x) for x in nhefs_plots.values()]
        return True

    def outcome_plot_by_name(self, test_names):
        nhefs_results = self.out_evaluator.evaluate_simple(
            self.data.X, self.data.a, self.data.y
        )
        nhefs_plots = plot_evaluation_results(
            nhefs_results,
            X=self.data.X,
            a=self.data.a,
            y=self.data.y,
            plot_names=test_names,
        )

        [self.assertIsNotNone(x) for x in nhefs_plots.values()]
        return True

    def propensity_plot_multiple_a(self, test_names):
        self.assertTrue(
            self.propensity_plot_by_name(
                test_names, alternate_a=self.data.a.astype(int)
            )
        )
        self.assertTrue(
            self.propensity_plot_by_name(
                test_names, alternate_a=self.data.a.astype(float)
            )
        )
        # self.assertTrue(self.propensity_plot_by_name(test_names, alternate_a=self.data.a.astype(str).factorize()))

    def test_weight_distribution_plot(self):
        self.propensity_plot_multiple_a(["weight_distribution"])

    def test_propensity_roc_plots(self):
        self.propensity_plot_multiple_a(["roc_curve"])

    def test_precision_plots(self):
        self.propensity_plot_multiple_a(["pr_curve"])

    def test_covariate_balance_plots(self):
        self.propensity_plot_multiple_a(["covariate_balance_love"])

    def test_propensity_multiple_plots(self):
        self.propensity_plot_multiple_a(["roc_curve", "covariate_balance_love"])

    def test_accuracy_plot(self):
        self.assertTrue(
            self.outcome_plot_by_name(["common_support", "continuous_accuracy"])
        )


# todo: add more tests (including ones that raise exceptions). No point in doing this right now since a major refactoring for the plots is ongoing
