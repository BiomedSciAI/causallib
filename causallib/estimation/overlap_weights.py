"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Jun 09, 2021

"""

import warnings

import pandas as pd

from causallib.estimation.base_estimator import PopulationOutcomeEstimator
from causallib.estimation.base_weight import PropensityEstimator
from causallib.estimation.ipw import IPW

# TODO: Move fit and _predict methods to PropensityEstimator, instead of rely on it from the ipw module.


class OverlapWeights(IPW):

    def __init__(self, learner, use_stabilized=False):
        """
        Implementation of overlap (propensity score) weighting:

        https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1260466

        A method to balance observed covariates between treatment groups in observational studies.
        Down-weigh observations with extreme propensity and weigh up
        Put less importance to observations with extreme propensity scores,
        and put more emphasis on observations with a central tendency towards
        (i.e. overlapping propensity scores).

        Each unit’s weight is proportional to the probability of that unit being
        assigned to the opposite group:
        w_i = 1 - Pr[A=a_i|Xi]

        This method assumes only two treatment groups exist.

        Args:
            learner: Initialized sklearn model.
            use_stabilized (bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                   See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title
        """
        super(OverlapWeights, self).__init__(learner, use_stabilized)

    def compute_weight_matrix(self, X, a, clip_min=None, clip_max=None, use_stabilized=None):
        """
        Computes individual weight across all possible treatment values.
        w_ij = 1 - Pr[A=a_j | X_i]  for all individual i and treatment j.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            clip_min (None|float): Lower bound for propensity scores. Better be left `None`.
            clip_max (None|float): Upper bound for propensity scores. Better be left `None`.
            use_stabilized (None|bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                        This overrides the use_stabilized parameter provided at initialization.
                                        If True provided, but the model was initialized with use_stabilized=False, then
                                        prevalence is calculated from data at hand, rather than the prevalence from the
                                        training data.
                                        See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title

        Returns:
            pd.DataFrame: A matrix of size (num_subjects, num_treatments) with weight for every individual and every
                          treatment.
        """
        use_stabilized = self.use_stabilized if use_stabilized is None else use_stabilized
        # Check that number of unique classes is 2
        self.__check_number_of_classes_is_legal(a)
        # Truncation is generally bad, check and warn:
        self.__check_truncation_value_is_none(clip_min, clip_max)

        # COmpute propensity scores
        probabilities = self.compute_propensity_matrix(X, a, clip_min, clip_max)
        # weight matrix: 1-P[a_i=1|x]
        # Reverse probabilities to opposite classes:
        probabilities.columns = probabilities.columns[::-1]  # Flip name-based indexing
        # reorder weights_matrix
        weight_matrix = probabilities.iloc[:, ::-1]  # Flip integer (location)-based indexing
        weight_matrix = self.stabilize_weights(a, weight_matrix, use_stabilized)

        return weight_matrix

    def stabilize_weights(self, a, weight_matrix, use_stabilized=False):
        # TODO: Move this function to IPW
        """
             Adjust sample weights according to class prevalence:
             Pr[A=a_i] * w_i

             Args:
                 weight_matrix (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
                 use_stabilized (None|bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                             This overrides the use_stabilized parameter provided at initialization.
                                             If True provided, but the model was initialized with use_stabilized=False, then
                                             prevalence is calculated from data at hand, rather than the prevalence from the
                                             training data.
                                             See Also: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4351790/#S6title

             Returns:
                 pd.DataFrame: A matrix of size (num_subjects, num_treatments) with stabilized (if True)
                                weight for every individual and every treatment.
             """
        if use_stabilized:
            if self.use_stabilized:
                prevalence = self.treatment_prevalence_
            else:
                warnings.warn("Stabilized is asked, however, the model was not trained using stabilization, and "
                              "therefore, stabilized weights are taken from the provided treatment assignment.",
                              RuntimeWarning)
                prevalence = a.value_counts(normalize=True, sort=False)
            prevalence_per_subject = a.replace(prevalence)  # map tx-assign to prevalence
            # pointwise multiplication of each column in weights:
            weight_matrix = weight_matrix.multiply(prevalence_per_subject, axis="index")

        return weight_matrix

    @staticmethod
    def __check_number_of_classes_is_legal(x):
        count_classes = x.nunique()
        if count_classes != 2:
            raise AssertionError("Number of unique classes should be equal 2")

    @staticmethod
    def __check_truncation_value_is_none(clip_min, clip_max):
        if clip_min is not None or clip_max is not None:
            warnings.warn(
                "Trimming observations with Overlap Weighting may be redundant, "
                "as extreme observations can receive greater importance than they should.",
                RuntimeWarning
            )
