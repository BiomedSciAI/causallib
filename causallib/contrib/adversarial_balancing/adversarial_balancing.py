# (C) Copyright 2019 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created on Oct 30, 2019

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .classifier_selection import select_classifier
from ...estimation.base_weight import WeightEstimator, PropensityEstimator
from ...estimation.base_estimator import PopulationOutcomeEstimator

import numpy as np
import pandas as pd
import copy


class AdversarialBalancing(WeightEstimator, PopulationOutcomeEstimator):
    def __init__(self, learner, iterations=20, lr=0.5, decay=1, loss_type='01', use_stabilized=True,
                 verbose=False, *args, **kwargs):
        """
        Adversarial Balancing finds sample weights such that the weighted population under any treatment A
        looks similar (distribution-wise) to the true population.
        Borrowing from GANs, the main idea is that, for each treatment A, the algorithm find weights such that
        a specified classifier cannot distinguish between the entire population and the weighted population under
        treatment `a`.

        At each step we update the weights using the gradient of the exponential loss function, and re-train the
        classifier.
        For a given classifier family, an optimal solution are weights that maximize the minimal error of classifiers
        in this family.

        For more details about the algorithm see:
         *Adversarial Balancing for Causal Inference* by Ozery-Flato and Thodoroff et al.
         https://arxiv.org/abs/1810.07406

        Args:
            learner: An initialized classifier object implementing fit and predict (scikit-learn compatible)
                     Will be used to discriminate between the population under treatment a and the entire
                     global population.
                     A selection for each treatment value can be performed to choose the best classifier for that
                     treatment group. It can be done by providing a scikit-learn initialized SearchCV model (either
                     GridSearchCV or RandomizedSearchCV), or by providing a list of classifiers.
                     If providing a list of classifiers, a selection will be done for each treatment value using
                     cross-validation that will use the best-performing classifier among the list.
                     see select_classifier module.
            iterations (int): The number of iterations to adjust the weights of each sample
            lr (float): Learning rate used to update the weights
            decay (float):  Parameter to decay the learning rate through the iterations
            loss_type (str): Use '01' for zero-one loss, otherwise cross-entropy is used (and provided `learner` should
                             also implement `predict_proba` methods).
            use_stabilized (bool): Whether to re-weigh the learned weights with the prevalence of the treatment.
                                   Note: Adversarial balancing already has inherent component weighting treatment
                                    prevalence.
                                    Setting to False will "de-stabilize" the weights after they are calculated.
            verbose (bool): Whether to print out statistics to console during training.

        Attributes:
            iterative_models_: np.ndarray of size(n_treatment_values, iterations) holding all the models created
                               during training process.
            iterative_normalizing_consts_: np.ndarray of size(n_treatment_values, iterations) holding all the
                                           normalizing constants calculated during training process.
            discriminator_loss_: np.ndarray of size(n_treatment_values, iterations) holding the loss of the learner
                                 throughout the training process.
            treatments_frequency_: if use_stabilized=True, the proportions of the treatment values.
        """
        super().__init__(learner, use_stabilized, *args, **kwargs)  # Crashes when provided a list of classifiers

        self.iterations = iterations
        self.loss_type = loss_type
        self.lr = lr
        self.decay = decay
        self.verbose = verbose
        self.use_stabilized = use_stabilized

    def fit(self, X, a, y=None, w_init=None, **select_kwargs):
        """
        Trains an Adversarial Balancing model.

        Args:
            X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a (pd.Series): Treatment assignment of size (num_subjects,).
            y: IGNORED.
            w_init (pd.Series): Initial sample weights. If not provided, assumes uniform.
            select_kwargs: keywords argument to past into select_classifier.
                           relevant only if model was initialized with list of classifiers in `learner`.

        Returns:
            AdversarialBalancing
        """
        X = _to_ndarray(X)
        a = _to_ndarray(a)
        self._run(X, a, w_init=w_init, is_train=True, **select_kwargs)
        return self

    def compute_weights(self, X, a, treatment_values=None, use_stabilized=None, **kwargs):
        if treatment_values is not None:
            # Create synthetic treatment assignment of treatment_values instead of actual assignment `a`
            a = pd.Series(data=treatment_values, index=X.index)
        idx = X.index
        X, a = _to_ndarray(X), _to_ndarray(a)
        w = self._run(X, a, w_init=None, use_stabilized=use_stabilized, is_train=False)
        w = pd.Series(w, index=idx)
        return w

    def estimate_population_outcome(self, X, a, y, w=None, treatment_values=None):
        if w is None:
            w = self.compute_weights(X, a)
        res = self._compute_stratified_weighted_aggregate(y, sample_weight=w, stratify_by=a,
                                                          treatment_values=treatment_values)
        return res

    def _run(self, X, A, w_init=None, is_train=True, use_stabilized=None, **select_kwargs):
        if use_stabilized is None:
            use_stabilized = self.use_stabilized

        if w_init is None:
            w_init = np.ones((X.shape[0]))

        w = w_init.copy()  # Weights to be returned

        unique_treatments = np.sort(np.unique(A).astype(int))
        n_treatments = unique_treatments.shape[0]

        if is_train:
            if not np.all(unique_treatments == np.arange(n_treatments)):
                raise AssertionError("Treatment values in `a` must be indexed 0, 1, 2, ...")
            self.iterative_models_ = np.empty((n_treatments, self.iterations), dtype=object)
            self.iterative_normalizing_consts_ = np.full((n_treatments, self.iterations), np.NaN)

            self.discriminator_loss_ = np.zeros((n_treatments, self.iterations))
            self.treatments_frequency_ = _compute_treatments_frequency(A)

        for a in unique_treatments:
            # Create an artificial classification problem where the samples with label 1 are the original entire
            # population ("source population"),
            # and the samples with label -1 are the population under treatment a ("target population").
            # Labels 1 and -1 (rather than 0) are used because of the later exponential loss function
            X_augm = np.row_stack((X, X[A == a]))  # create the augmented dataset
            y = np.ones((X_augm.shape[0]))
            y[X.shape[0]:] *= -1  # subpopulation of current treatment (a) has y== -1
            target_pop_mask = y == -1

            # To simplify the task (learning weights) we ensure both target and source populations have the same
            # importance by reweighting classes by their frequency
            y_0_1 = LabelEncoder().fit_transform(y)  # Encode -1 ==> 0  and  1 ==>1
            class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)[y_0_1]

            sample_weight = np.ones((X_augm.shape[0]))
            sample_weight[target_pop_mask] = w[A == a]  # Weights from initialization

            if is_train:
                selected_model = select_classifier(self.learner, X_augm, y,
                                                   loss_type=self.loss_type, **select_kwargs)

            for i in range(self.iterations):
                lr = self.lr * (1.0 / (1 + (self.decay * i)))  # decay the learning rate

                # fit model
                if is_train:
                    train_weight = sample_weight * class_weight
                    selected_model.fit(X_augm, y, sample_weight=train_weight)
                    self.iterative_models_[a, i] = copy.deepcopy(selected_model)
                else:
                    selected_model = self.iterative_models_[a, i]

                # get predictions and their errors
                pred, pred_loss = self.__get_predictions(selected_model, X_augm, y)

                if is_train:
                    total_error = np.sum(train_weight * pred_loss) / np.sum(train_weight)
                    # when using 0-1 loss:
                    # note that often np.sum(train_weight * pred_loss) != np.sum(train_weight[pred_loss.astype(bool)])
                    # and the difference is slight, probably due to numerical issues
                    self.discriminator_loss_[a, i] = total_error
                    if self.verbose:
                        print("Iteration {}/{} - loss: {}".format(i, self.iterations, total_error))

                # Update the weights to minimize the loss of the generator and "fool" the classifier
                if self.loss_type == '01':
                    sample_weight[target_pop_mask] *= np.exp(-lr * y[target_pop_mask] * pred[target_pop_mask])
                else:
                    sample_weight[target_pop_mask] *= np.exp(lr * pred_loss[target_pop_mask])

                if is_train:
                    tx_group_size = np.sum(target_pop_mask)
                    self.iterative_normalizing_consts_[a, i] = tx_group_size / np.sum(sample_weight[target_pop_mask])

                # Normalize the weights with mean 1 (so that weights.sum() == original group size)
                sample_weight[target_pop_mask] *= self.iterative_normalizing_consts_[a, i]

            w[A == a] = sample_weight[target_pop_mask]
            if not use_stabilized:  # Undo scaling by group size
                w[A == a] *= 1 / self.treatments_frequency_[a]

        return w

    def __get_predictions(self, selected_model, X_augm, y):
        if self.loss_type == '01':
            pred = selected_model.predict(X_augm).reshape((-1))
            pred_loss = (pred != y).astype(float)

        else:
            pred = selected_model.predict_proba(X_augm)[:, 1]
            pred_loss = np.log(pred[y == -1])
            pred_loss[y == 1] = np.log(1.0 - pred[y == 1])

        return pred, pred_loss

    def compute_weight_matrix(self, X, a, use_stabilized=None, **kwargs):
        res = {}
        for tx_val in np.sort(np.unique(a)):
            cur_assignment = pd.Series(data=tx_val, index=X.index)
            res[tx_val] = self.compute_weights(X, cur_assignment, use_stabilized=use_stabilized)
        res = pd.DataFrame(res)
        return res

    # def compute_propensity(self, X, a, treatment_values=None, **kwargs):
    #     """Computes probability for being treated
    #     (or probability of any treatment value provided in `treatment_values`).
    #
    #     Note that Adversarial Balance does not optimize for probabilities directly and
    #     this probabilities are calculated post-hoc from the weights.
    #
    #     Args:
    #         X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
    #         a (pd.Series): Treatment assignment of size (num_subjects,).
    #         treatment_values (Any | None): A desired value/s to extract propensity to (i.e. probabilities to what
    #                                       treatment value should be calculated).
    #                                       If not specified, then the maximal treatment value is chosen. This is since
    #                                       the usual case is of treatment (A=1) control (A=0) setting.
    #
    #     Returns:
    #         pd.Series: A vector size (num_subjects,) containing the propensity of each individual to be treatment
    #                    (or get the treatment value provided in `treatment_values`).
    #     """
    #     treatment_values = a.max() if treatment_values is None else treatment_values
    #     res = self.compute_propensity_matrix(X, a, use_stabilized=False)
    #     res = res[treatment_values]
    #     return res
    #
    # def compute_propensity_matrix(self, X, a=None, **kwargs):
    #     """Computes probability for being assigned to any treatment groups.
    #
    #     Note that Adversarial Balance does not optimize for probabilities directly and
    #     this probabilities are calculated post-hoc from the weights.
    #     Therefore, there's no guarantee all probabilities of a given sample will sum to 1.
    #
    #     Args:
    #         X (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
    #         a (pd.Series | None): Treatment assignment of size (num_subjects,),
    #                               used to obtain the unique treatment values.
    #                               If not provided, uses all the treatment values seen during training.
    #
    #     Returns:
    #         pd.DataFrame: A matrix of size (num_subjects, num_treatments) with probability for every individual and e
    #                       very treatment.
    #     """
    #     if a is None:
    #         a = list(self.treatments_frequency_.keys())  # unique treatment values
    #     res = self.compute_weight_matrix(X, a, use_stabilized=False)
    #     res = 1 / res  # type: pd.DataFrame
    #     return res


def _compute_treatments_frequency(A):
    unique_treatments, unique_treatment_counts = np.unique(A, return_counts=True)
    unique_treatments_freq = unique_treatment_counts / (1.0*np.sum(unique_treatment_counts))
    treatments_frequency = dict(zip(unique_treatments, unique_treatments_freq))
    return treatments_frequency


def _to_ndarray(x):
    if not isinstance(x, np.ndarray):
        return np.array(x)
    else:
        return x
