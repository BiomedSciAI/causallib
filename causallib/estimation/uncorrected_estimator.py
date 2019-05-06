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

Created on Apr 25, 2018

"""

import numpy as np

from .base_estimator import PopulationOutcomeEstimator
from ..utils import general_tools as g_tools


class UncorrectedEstimator(PopulationOutcomeEstimator):
    def estimate_population_outcome(self, X, a, y, treatment_values=None):
        treatment_values = g_tools.get_iterable_treatment_values(treatment_values, a)
        res = {}
        for treatment_value in treatment_values:
            y_stratified = y[a == treatment_value]
            average_outcome = np.nanmean(y_stratified)
            res["EY_{}".format(treatment_value)] = average_outcome
        return res
