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

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import pandas as pd


from causallib.evaluation import evaluate
from causallib.estimation import AIPW, IPW, StratifiedStandardization
from causallib.datasets import load_nhefs


matplotlib.use("Agg")


def binarize(cts_output: pd.Series) -> pd.Series:
    """Turn continuous outcome into binary by applying sigmoid.

    Args:
        cts_output (pd.Series): outcomes as continuous variables

    Returns:
        pd.Series: outcomes as binary variables
    """

    y = 1 / (1 + np.exp(-cts_output))
    y = np.random.binomial(1, y)
    y = pd.Series(y, index=cts_output.index)
    return y


class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        data = load_nhefs()
        self.X, self.a, self.y = data.X, data.a, data.y
        ipw = IPW(LogisticRegression(solver="liblinear"), clip_min=0.05, clip_max=0.95)
        std = StratifiedStandardization(LinearRegression())
        self.dr = AIPW(std, ipw)
        self.dr.fit(self.X, self.a, self.y)
        self.cts_outcome_evaluation = evaluate(
            self.dr.outcome_model, self.X, self.a, self.y
        )
        self.propensity_evaluation = evaluate(self.dr.weight_model, self.X, self.a, self.y)
        self.y_bin = binarize(data.y)
        self.std_bin = StratifiedStandardization(LogisticRegression(solver="liblinear"))
        self.std_bin.fit(self.X, self.a, self.y_bin)
        self.bin_outcome_evaluation = evaluate(self.std_bin, self.X, self.a, self.y_bin)

    def test_propensity_plots_only_exist_for_propensity_model(self):
        self.propensity_evaluation.plot_covariate_balance
        self.propensity_evaluation.plot_weight_distribution
        self.propensity_evaluation.plot_pr_curve
        self.propensity_evaluation.plot_roc_curve
        self.propensity_evaluation.plot_calibration_curve

        with self.assertRaises(AttributeError):
            self.cts_outcome_evaluation.plot_covariate_balance
            self.cts_outcome_evaluation.plot_weight_distribution
            self.cts_outcome_evaluation.plot_pr_curve
            self.cts_outcome_evaluation.plot_roc_curve
            self.cts_outcome_evaluation.plot_calibration_curve

    def test_outcome_plots_only_exist_for_outcome_model(self):
        self.cts_outcome_evaluation.plot_continuous_accuracy
        self.cts_outcome_evaluation.plot_residuals
        self.cts_outcome_evaluation.plot_common_support

        with self.assertRaises(AttributeError):
            self.propensity_evaluation.plot_continuous_accuracy
            self.propensity_evaluation.plot_residuals
            self.propensity_evaluation.plot_common_support

    def test_weight_distribution_reflect_has_negative_yaxis(self):
        f, ax = plt.subplots()
        axis = self.propensity_evaluation.plot_weight_distribution(reflect=True, ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        minx, maxx, miny, maxy = axis.axis()
        self.assertLess(miny, 0)
        self.assertLess(maxx, 1)
        self.assertGreater(minx, 0)
        self.assertGreater(maxy, 0)
        plt.close()

    def test_weight_distribution_noreflect_has_nonegative_yaxis(self):
        f, ax = plt.subplots()
        axis = self.propensity_evaluation.plot_weight_distribution(reflect=False, ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        minx, maxx, miny, maxy = axis.axis()
        self.assertEqual(miny, 0)
        self.assertLess(maxx, 1)
        self.assertGreater(minx, 0)
        self.assertGreater(maxy, 0)
        plt.close()

    def test_plot_covariate_balance_love_draws_thresh(self):
        thresh = 0.1
        f, ax = plt.subplots()
        axis = self.propensity_evaluation.plot_covariate_balance(
            kind="love", thresh=thresh, ax=ax
        )
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        self.assertEqual(thresh, axis.get_lines()[0].get_xdata()[0])
        plt.close()

    def test_plot_covariate_balance_scatter_draws_thresh(self):
        thresh = 0.1
        f, ax = plt.subplots()
        axis = self.propensity_evaluation.plot_covariate_balance(
            kind="scatter", thresh=thresh, ax=ax
        )
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        self.assertEqual(thresh, axis.get_lines()[0].get_xdata()[0])
        plt.close()

    def test_plot_covariate_balance_slope_labeled_correctly(self):
        f, ax = plt.subplots()
        axis = self.propensity_evaluation.plot_covariate_balance(kind="slope", ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        self.assertEqual([x.get_xdata() for x in axis.get_lines()][1][0], "unweighted")
        plt.close()

    def test_plot_covariate_balance_types_exchangeable_kwargs(self):
        f, ax = plt.subplots(1, 3)
        for i, kind in enumerate(["love", "slope", "scatter"]):
            self.propensity_evaluation.plot_covariate_balance(
                kind=kind, ax=ax[i],
                plot_semi_grid=True,  # A "love"-only kwarg
                label_imbalanced=True,  # A "slope" and "scatter" only kwarg
                thresh=0.1,  # So that there are imbalanced variables plotted
            )
        plt.close(f)

    def test_roc_curve_has_dashed_diag(self):
        self.ensure_roc_curve_has_dashed_diag(self.propensity_evaluation)
        self.ensure_roc_curve_has_dashed_diag(self.bin_outcome_evaluation)

    def test_calibration_curve_has_dashed_diag(self):
        self.ensure_calibration_curve_has_dashed_diag(self.propensity_evaluation)
        self.ensure_calibration_curve_has_dashed_diag(self.bin_outcome_evaluation)

    def test_pr_curve_has_flat_dashed_chance_line(self):
        self.ensure_pr_curve_has_flat_dashed_chance_line(
            self.propensity_evaluation, chance=self.a.mean()
        )
        self.ensure_pr_curve_has_flat_dashed_chance_line(
            self.bin_outcome_evaluation, chance=self.y_bin.mean()
        )

    def test_accuracy_plot_has_dashed_diag(self):
        f, ax = plt.subplots()
        axis = self.cts_outcome_evaluation.plot_continuous_accuracy(ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        diag = [x for x in axis.lines if len(x.get_xdata()) == 2][0]
        self.assertEqual(diag.get_linestyle(), "--")
        self.assertTrue(all(diag.get_xdata() == diag.get_ydata()))
        plt.close()

    def test_common_support_plot_has_dashed_diag(self):
        f, ax = plt.subplots()
        axis = self.cts_outcome_evaluation.plot_common_support(ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        diag = [x for x in axis.lines if len(x.get_xdata()) == 2][0]
        self.assertEqual(diag.get_linestyle(), "--")
        self.assertTrue(all(diag.get_xdata() == diag.get_ydata()))
        plt.close()
        plt.close()

    def test_plot_all_generates_correct_plot_names(self):
        self.ensure_plot_all_generates_all_plot_names(self.propensity_evaluation)
        self.ensure_plot_all_generates_all_plot_names(self.cts_outcome_evaluation)
        self.ensure_plot_all_generates_all_plot_names(self.bin_outcome_evaluation)



    def test_residuals_plot_has_dashed_zero_line(self):
        f, ax = plt.subplots()
        axis = self.cts_outcome_evaluation.plot_residuals(ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        zero_line = [x for x in axis.lines if len(x.get_xdata()) == 2][0]
        self.assertEqual(zero_line.get_linestyle(), "--")
        self.assertEqual(zero_line.get_ydata()[0], 0)
        self.assertEqual(zero_line.get_ydata()[1], 0)
        plt.close()

    def ensure_roc_curve_has_dashed_diag(self, results_object):
        f, ax = plt.subplots()
        axis = results_object.plot_roc_curve(ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        diag = [x for x in axis.lines if len(x.get_xdata()) == 2][0]
        self.assertEqual(diag.get_linestyle(), "--")
        self.assertTrue(all(diag.get_xdata() == [0, 1]))
        self.assertTrue(all(diag.get_ydata() == [0, 1]))
        plt.close()

    def ensure_calibration_curve_has_dashed_diag(self, results_object):
        f, ax = plt.subplots()
        axis = results_object.plot_calibration_curve(ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        diag = [x for x in axis.lines if len(x.get_xdata()) == 2][0]
        self.assertEqual(diag.get_linestyle(), "--")
        self.assertTrue(all(diag.get_xdata() == diag.get_ydata()))
        plt.close()

    def ensure_pr_curve_has_flat_dashed_chance_line(self, results_object, chance):
        f, ax = plt.subplots()
        axis = results_object.plot_pr_curve(ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        chance_line = [x for x in axis.lines if len(x.get_xdata()) == 2][0]
        self.assertEqual(chance_line.get_label(), "Chance")
        self.assertEqual(chance_line.get_linestyle(), "--")
        self.assertAlmostEqual(chance_line.get_ydata()[0], chance)
        self.assertAlmostEqual(chance_line.get_ydata()[1], chance)
        plt.close()

    def ensure_plot_all_generates_all_plot_names(self, evaluation_results):
        all_plots = evaluation_results.plot_all()
        all_names = evaluation_results.all_plot_names
        self.assertEqual(set(all_plots["train"].keys()), all_names)