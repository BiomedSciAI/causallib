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
# Created on Sept 25, 2019

import numpy as np
import pandas as pd
import torch

from causallib.estimation.base_estimator import IndividualOutcomeEstimator
from causallib.contrib.hemm.hemm import HEMM as HEMMTorch


class HEMM(IndividualOutcomeEstimator):
    """Causal model implementing an heterogeneous effect mixture model (HEMM). 

    For further details, see `Interpretable Subgroup Discovery in Treatment Effect 
    Estimation with Application to Opioid Prescribing Guidelines <https://arxiv.org/pdf/1905.03297.pdf>`_ paper.

    Notes:
        **Requires PyTorch** >= 1.2.0 (tested on 1.2.0)
    """

    def __init__(self,
                 D_in, K, homo=True, mu=None, std=None, bc=0, lamb=1e-4, spread=0.1, outcome_model='linear',
                 sep_heads=True, epochs=100, batch_size=10, learning_rate=1e-3, weight_decay=0.1, elbo_loss=True,
                 metric='AP', response='bin', use_p_correction=True, imb_fun='mmd2_lin', p_alpha=1e-4, random_seed=0):
        """Instantiate estimator.

        Args:
            D_in (int): Size of the features of the data
            K (int): Number of components to discover. (specifcy K-1: eg. For 2 components use K=1)
            homo (bool): Flag to specify if the final outcome model is same for each discovered subgroup.
                        Default is True ie. same outcome model is used for each subgroup.  
            mu (float): Initialize the components with means of the training data.
            std (float): Initialize the components with std dev of the training data.
            bc (int): The first feature in `x` being a bernoulli variables.
                    Columns 0 up to `bc` should be continuous (gaussian)
            lamb (float): Strength of the beta(0.5, 0.5) prior on the bernoulli variables.
            spread (float): How far should the components be initailized from there means.
            outcome_model (str): 'linear' to specify a linear outcome function.
                                 Or pass another Torch.model as the outcome model.
                                 Several outcome models are available in the `outcome_models.py` module.
            epochs (int): Max number of epochs.
            batch_size (int): Batch size for optimizer.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            elbo_loss (bool): Wether to use evidence lower bound (ELBO) as the loss function.
            metric (str): 'AP' mean average precision, 'AuROC': area under roc curve, 'MSE':mean squared error.
                          If not specified it uses the optimizer cost as the metric.
                          The specified metric is computed for the training set (or dev set if it is specified in `fit`)
                          and used to perform early stopping to prevent overfitting if dev set is provided.
            response (str): Specify if y is continuous ('cont') or binary ('bin').
            use_p_correction (bool): Whether to use population size p(treated) in imbalance penalty (IPM).
            imb_fun (str): Which imbalance penalty to use ('mmd2_lin', 'wass').
            p_alpha (float): Imbalance regularization parameter.
            random_seed (int): Random Seed to initialize model parameters.
        """
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._elbo_loss = elbo_loss
        self._metric = metric
        self._response = response
        self._use_p_correction = use_p_correction
        self._imb_fun = imb_fun
        self._p_alpha = p_alpha
        self._sep_heads = sep_heads
        self._homo = homo
        self._random_seed = random_seed
        
        torch.manual_seed(random_seed)

        model = HEMMTorch(D_in, K, homo, mu, std, bc, lamb, spread=spread, outcomeModel=outcome_model, sep_heads=sep_heads)
        model = model.double()
        super(HEMM, self).__init__(learner=model)

    def estimate_individual_outcome(self, X, a, treatment_values=None, soft=True):
        """
        Estimates individual outcome under different treatment values (interventions)

        Args:
            X (torch.Tensor | pd.DataFrame | np.ndarray): Covariate matrix of size (num_subjects, num_features).
            a (torch.Tensor | pd.Series | np.ndarray): Treatment assignment of size (num_subjects,).
            treatment_values: subset of values to get potential outcome for.
                              If None return for all unique treatment_values found in `a`.
            soft (bool):

        Returns:
            pd.DataFrame: DataFrame which columns are treatment values and rows are individuals: each column is a vector
                          size (num_samples,) that contains the estimated outcome for each individual under the
                          treatment value in the corresponding key.
        """
        ys = self.learner.estimate_individual_outcomes(self._as_tensor(X), self._as_tensor(a),
                                                       response=self._response, soft=soft)
        ys = pd.DataFrame(zip(*[y.cpu().data.numpy() for y in ys]))
        if treatment_values is not None:
            ys = ys[treatment_values]
        return ys

    def get_groups(self, X):
        """Return hard assignment of groups for each sample

        Args:
            X (torch.Tensor | pd.DataFrame | np.ndarray): Covariate matrix of size (num_subjects, num_features).

        Returns:
            groups (pd.Series): Most probable group assignment of each sample, size = (num_samples,)
        """
        groups = self.learner.get_groups(self._as_tensor(X))
        groups = pd.Series(groups)
        return groups

    def get_groups_proba(self, X, log=False):
        """Return soft assignment of probability of each sample to be part of each group.

        Args:
            X (torch.Tensor): Covariate matrix of size (num_subjects, num_features).
            log (bool): If True returns log probabilities

        Returns:
            z_pred (np.array): probability of group membership given X P(Z|X),
                               size = (num_covariates, num_components+1).
        """
        X = self._as_tensor(X)
        groups = self.learner.get_groups_proba(X, log=log)
        groups = pd.DataFrame(groups)
        return groups

    def get_groups_effect(self, X=None, a=None):
        """Returns effect (\\gamma_k in the paper) in each group as estimated during training"""
        group_effect = self.learner.get_groups_effect()
        return group_effect

    def group_sizes(self, X):
        """Return underlying treatment groups sizes.
        Args:
            X (torch.Tensor | pd.DataFrame | np.ndarray): Covariate matrix of size (num_subjects, num_features).

        Returns: collection.Counter giving size for each group
        """
        return self.learner.group_sizes(self._as_tensor(X))

    def fit(self, X, a, y, sample_weight=None, validation_data=None):
        """Trains a causal model from observed data.

        Note: If all dev-data is provided, early stopping criteria will be applied using this data and the metric
              specified in self._metric.

        Args:
            X (torch.Tensor | pd.DataFrame | np.ndarray): Covariate matrix of size (num_subjects, num_features).
            a (torch.Tensor | pd.Series | np.ndarray): Treatment assignment of size (num_subjects,).
            y (torch.Tensor | pd.Series | np.ndarray): Observed outcome of size (num_subjects,).
            sample_weight: *IGNORED*
            validation_data: tuple of validation set: (X_val, a_val, y_val) corresponding to `X, a, y` above.

        Returns:
            IndividualOutcomeEstimator: A causal weight model with an inner learner fitted.
        """
        ltype = 'log' if self._elbo_loss else None
        if validation_data is not None:
            validation_data = (self._as_tensor(d) for d in validation_data)
        else:
            validation_data = None

        return self.learner.fit(
            train=(self._as_tensor(X), self._as_tensor(a), self._as_tensor(y)),
            epochs=self._epochs,
            batch_size=self._batch_size,
            lr=self._learning_rate,
            wd=self._weight_decay,
            ltype=ltype,
            dev=validation_data,
            metric=self._metric,
            response=self._response,
            use_p_correction=self._use_p_correction,
            imb_fun=self._imb_fun,
            p_alpha=self._p_alpha
        )

    @staticmethod
    def _as_tensor(X):
        """Convert the given argument to a Torch tensor.
        
            Args:
                X (pd.DataFrame|pd.Series|np.ndarray|torch.Tensor): argument to convert
            Returns:
                torch.Tensor with the same shape as the given argument
            Raises:
                TypeError: if X is not a DataFrame/Series, Numpy array, or Torch tensor
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return torch.from_numpy(X.values.astype('float64'))
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype('float64'))
        if isinstance(X, torch.Tensor):
            return X
        raise TypeError('Argument must be Numpy array, Panda DataFrame/Series, or PyTorch tensor')
