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
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression

from causallib.evaluation import evaluate
from causallib.evaluation.evaluator import evaluate_bootstrap
from causallib.evaluation.metrics import (
    get_default_binary_metrics,
    get_default_regression_metrics,
)
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


class TestEvaluations(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        data = load_nhefs()
        self.X, self.a, self.y = data.X, data.a, data.y
        self.y_bin = binarize(data.y)
        ipw = IPW(LogisticRegression(solver="liblinear"), clip_min=0.05, clip_max=0.95)
        std = StratifiedStandardization(LinearRegression())
        self.dr = AIPW(std, ipw)
        self.dr.fit(self.X, self.a, self.y)
        self.std_bin = StratifiedStandardization(LogisticRegression(solver="liblinear"))
        self.std_bin.fit(self.X, self.a, self.y_bin)

    def test_evaluate_bootstrap_with_refit_works(self):
        ipw = IPW(LogisticRegression(solver="liblinear"), clip_min=0.05, clip_max=0.95)
        evaluate_bootstrap(ipw, self.X, self.a, self.y, n_bootstrap=5, refit=True)

    def test_evaluate_cv_works_with_unfit_models(self):
        ipw = IPW(LogisticRegression(solver="liblinear"), clip_min=0.05, clip_max=0.95)
        evaluate(ipw, self.X, self.a, self.y, cv="auto")

    def test_metrics_to_evaluate_is_none_means_no_metrics_evaluated(self):
        for model in (self.dr.outcome_model, self.dr.weight_model):
            self.ensure_metrics_are_none(model)

    def ensure_metrics_are_none(self, model):
        results = evaluate(model, self.X, self.a, self.y, metrics_to_evaluate=None)
        self.assertIsNone(results.evaluated_metrics)

    def test_default_evaluation_metrics_weights(self):
        model = self.dr.weight_model
        results = evaluate(model, self.X, self.a, self.y)
        self.assertEqual(
            set(results.evaluated_metrics.prediction_scores.columns),
            set(get_default_binary_metrics().keys()),
        )

    def test_default_evaluation_metrics_continuous_outcome(self):
        model = self.dr.outcome_model
        results = evaluate(model, self.X, self.a, self.y)
        self.assertEqual(
            set(results.evaluated_metrics.columns),
            set(get_default_regression_metrics().keys()),
        )

    def test_default_evaluation_metrics_binary_outcome(self):
        model = self.std_bin
        results = evaluate(model, self.X, self.a, self.y_bin)
        self.assertEqual(
            set(results.evaluated_metrics.columns),
            set(get_default_binary_metrics().keys()),
        )
