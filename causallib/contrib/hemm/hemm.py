# coding: utf-8

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

import torch
import torch.nn as nn
import numpy as np

from collections import Counter
from torch import logsumexp as logsumexp  # pylint: disable=E0611
from sklearn.metrics import roc_auc_score, average_precision_score

from .hemm_metrics import wass, mmd2_lin, mmd2_rbf
from copy import deepcopy


class HEMM(torch.nn.Module):
    """This is the model defintion. The model has two parts:

      1. the subgroup discovery component 
      2. the outcome prediction from the subgroup assignment and the interaction with confounders through an MLP. 
    
    Args:
        D_in: the size of the features of the data
        K: number of components to discover.
        homo (bool): Flag to specify if the final outcome model is same for each discovered subgroup.
                    Default is True ie. same outcome model is used for each subgroup.          
        mu: initialize the components with means of the training data.
        std: initialize the components with std dev of the training data.
        bc: the first bc components are considered bernoulli variables
        lamb: strength of the beta(0.5, 0.5) prior on the bernoulli variables
        spread: how far should the components be initialized from there means.
        outcomeModel: 'linear' to specify a linear outcome function. Or pass another Torch.model as the outcome model.
        sep_heads: Setting false will force the adjustment of Confounding to be same independent of treatment assignment.
    """

    def __init__(self, D_in, K, homo=True, mu=None, std=None, bc=0, lamb=0.0001, spread=0.1, outcomeModel='linear', sep_heads=True):

        super(HEMM, self).__init__()
        self.bc = bc
        self.lamb = lamb
        self.homo = homo
        self.K = K 
        self.sep_heads = sep_heads
        
        self.tc = None
        
        lindim = D_in

        if outcomeModel == 'linear':
            outcomeModel = nn.Linear(lindim, 2)

        if mu is not None:
            p = torch.from_numpy(np.repeat(mu[:, bc:], self.K, axis=0))
            mu = torch.from_numpy(np.repeat(mu[:, :bc], self.K, axis=0))
        else:
            p = 0.5 * torch.ones(self.K, D_in - bc)
            mu = torch.zeros(self.K, bc)

        if std is not None:
            std = torch.from_numpy(np.repeat(std[:, :bc], self.K, axis=0))
        else:
            std = torch.ones(self.K, bc)

        for i in range(self.K):
            mu[i] = mu[i] + spread * (torch.rand_like(mu[i]) - 0.5)
            std[i] = std[i] + spread * (torch.rand_like(std[i]) - 0.5)
            p[i] = p[i] + spread * (torch.rand_like(p[i]) - 0.5)

        self.mu = nn.ParameterList(nn.Parameter(mu[i]) for i in range(self.K))
        self.std = nn.ParameterList(nn.Parameter(std[i]) for i in range(self.K))
        self.p = nn.ParameterList(nn.Parameter(p[i]) for i in range(self.K))
        self.alph = nn.Parameter(torch.ones(self.K))

        treat = torch.abs(torch.rand(self.K))
        self.treat = nn.Parameter(treat.double())
        if homo:
            expert = [deepcopy(outcomeModel) for i in range(1)]
        else:
            expert = [deepcopy(outcomeModel) for i in range(self.K)]

        self.expert = nn.ModuleList(expert)

    @staticmethod
    def gaussian_pdf(x, mu, std):
        mu = mu.unsqueeze(0)
        std = std.unsqueeze(0)

        gauss_ = -torch.log(std) - torch.div((x - mu.expand(x.shape)) ** 2, 2 * (std ** 2))
        gauss_ = torch.sum(gauss_, dim=1) - 0.9189 * x.shape[1]  # TODO: what's with the 0.9189?

        return gauss_

    @staticmethod
    def bernoulli_pdf(x, mu):
        loss = nn.BCELoss(reduction='none')

        mu = mu.unsqueeze(0)
        mu = torch.clamp(mu, min=1e-3, max=1 - 1e-3)

        bern_ = -loss(mu.expand(x.shape), x)

        return torch.sum(bern_, dim=1)

    def regularization(self, alpha, beta):
        lamb = self.lamb
        p = []
        for i in range(self.K):
            p.append(torch.clamp(self.p[i], min=1e-3, max=1 - 1e-3))

        p = torch.stack(p, 1)

        cost = ((alpha - 1) * torch.log(p)) + ((beta - 1) * torch.log(1 - p))
        cost = cost.sum()

        return -lamb * cost
    
    def forward(self, x, t, soft=True, infer=True, response='bin'):
        
        if self.sep_heads:
            selector = t.cpu().data.numpy().astype('int').tolist()
        else:
            selector = torch.zeros_like(t).cpu().data.numpy().astype('int').tolist()
            
        selector2 = range(len(selector))

        # TODO: Consider a slightly more general binary-variable mask vector indicated which columns are binary,
        #       rather than force certain structure on the input.
        xg, xb = x[:, :self.bc], x[:, self.bc:]

        # compute q(Z)
        gate_output = []
        for i in range(self.K):
            z_i = self.gaussian_pdf(xg, self.mu[i], self.std[i])
            z_i += self.bernoulli_pdf(xb, self.p[i])
            gate_output.append(z_i)

        gate_output = torch.stack(gate_output, 1)
        gate_output = gate_output + self.alph.expand(gate_output.shape)
        gate_output_ = gate_output
        gate_sum = logsumexp(gate_output, dim=1, keepdim=True)

        lgate_output = gate_output - gate_sum
        gate_output = torch.exp(lgate_output)

        rs = x.shape[0]
        cs = self.treat.shape[0]  # size of K  # TODO: why not use self.K instead?
        treat = self.treat.expand((rs, cs))

        # TODO: Consider renaming `infer` to a more informative variable name.

        expert_output = []
        for i in range(self.K):
            # calculate teffect (main effect due to treatment and coefficient) and add to output of outcomeModel
            teffect = torch.mul(t, treat[:, i])
            if self.homo:
                cur_expert = self.expert[0]
            else:
                cur_expert = self.expert[i]
            cur_expert_output = cur_expert(x)[selector2, selector] + teffect
            if infer:
                if response == 'bin':
                    cur_expert_output = nn.Sigmoid()(cur_expert_output)

            expert_output.append(cur_expert_output)
        expert_output = torch.stack(expert_output, 1)
        
        if not infer:
            return gate_output_, lgate_output, expert_output
        
        else:
            if soft:
                # TODO: consider vectorized math below: `gate_output.mul(expert_output).sum(dim=1)`
                output = torch.zeros_like(x[:, 0])
                for i in range(self.K):
                    output += torch.mul(gate_output[:, i], expert_output[:, i])
                return output
            else:
                ridx = torch.tensor(range(x.shape[0]))
                cidx = torch.argmax(gate_output, 1)
                output = expert_output[ridx, cidx]
                return output

    def fit(self, train, epochs=100, batch_size=10, lr=1e-3, wd=0, ltype='log', dev=None, metric='AP', response='bin',
            use_p_correction=True, imb_fun='mmd2_lin', p_alpha=1e-4):
        """This method uses ELBO to perform parameter updates using a first order optimizer routing.

        Args:
            train: (x_, t_, Y_):= Tuple o torch tensors of the input features, treatment and outcome.
            epochs: Max number of epochs.
            batch_size: Batch size for optimizer.
            lr: Learning rate for the optimizer.
            wd (float): Weight decay
            ltype: 'log' for the ELBO.
            dev: Tuple of validation dataset i.e.: torch tensors (x_, t_, Y_).
                If provided, early-stopping criteria using `metric` will be applied.
            metric (str): 'AP' mean average precision, 'AuROC': area under roc curve, 'MSE':mean squared error.
                          If not specified it uses the optimizer cost as the metric.
                          The specified this metric is computed for the training set (or dev set if it is specified)
                          and used to perform early stopping to prevent overfitting if dev set is provided.
            response: Specify if y is continuous ('cont') or binary ('bin').
            use_p_correction (bool): Whether to use population size p(treated) in imbalance penalty (IPM).
            imb_fun (str): Which imbalance penalty to use ('mmd_lin', 'wass').
            p_alpha (float): Imbalance regularization parameter.
        """
        # TODO: Consider rename `dev` to `val` to correspond with Keras `validation_data` parameters `x_val, y_val`.
        #       https://keras.io/models/model/#fit
        x, t, y = train

        if dev is not None:
            xdev, tdev, ydev = dev
            devcost = -float('inf')
        else:
            xdev, tdev, ydev = x, t, y

        optparam = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(optparam, lr=lr, weight_decay=wd)

        losses = []
        greaters = 0

        for epoch in range(epochs):
            tcost = 0
            for step in range(int(x.shape[0] / batch_size)):
                x_ = x[step * batch_size:(step + 1) * batch_size]
                y_, t_ = y[step * batch_size:(step + 1) * batch_size], t[step * batch_size:(step + 1) * batch_size]
                optimizer.zero_grad()
                cost = self.lcost(x_, t_, y_, ltype=ltype, response=response, use_p_correction=use_p_correction,
                                  imb_fun=imb_fun, p_alpha=p_alpha)
                tcost += cost.cpu().data.numpy()
                cost.backward()
                optimizer.step()

            losses.append(float(tcost) / x.shape[0])

            if dev is not None:
                output = self.forward(xdev, tdev, response=response)
            else:
                output = self.forward(x, t, response=response)

            if response == 'bin':
                if metric == 'AuROC':
                    newdevcost = roc_auc_score(ydev.cpu().data.numpy(), output.cpu().data.numpy())
                elif metric == 'AP':
                    newdevcost = average_precision_score(ydev.cpu().data.numpy(), output.cpu().data.numpy())
                else:
                    newdevcost = -self.lcost(xdev, tdev, ydev, ltype=ltype, response=response,
                                             use_p_correction=use_p_correction, imb_fun=imb_fun, p_alpha=p_alpha)
            elif response == 'cont':
                if metric == 'MSE':
                    newdevcost = -((ydev.cpu().data.numpy() - output.cpu().data.numpy()) ** 2).sum()
                else:
                    newdevcost = -self.lcost(xdev, tdev, ydev, ltype=ltype, response=response,
                                             use_p_correction=use_p_correction, imb_fun=imb_fun, p_alpha=p_alpha)

            if dev is not None:
                if newdevcost < devcost:
                    greaters += 1

                if greaters > 3:
                    break

            devcost = newdevcost
        return losses

    def lcost(self, x_, t_, y_, ltype="log", response='bin', use_p_correction=True, imb_fun=None, p_alpha=1e-4,
              rbf_sigma=0.1, wass_its=20, wass_lambda=10.0):
        """Implements ELBO as the objective function (eq 12 in paper).

        Args:
            x_ (torch.Tensor): Covariate matrix of size (num_subjects, num_features).
            t_ (torch.Tensor): Treatment assignment of size (num_subjects,).
            y_ (torch.Tensor): Outcome of size (num_subjects,).
            ltype: 'log' specifies ELBO
            response: specifies whether outcome `y` is binary ('bin') or continuous ('cont').
            use_p_correction (bool): whether to use population size p(treated) in imbalance penalty (IPM)
            imb_fun (str): which imbalance penalty to use ('mmd_lin', 'mmd_rbf', 'wass', 'wass2')
            p_alpha (float): imbalance regularization parameter
            rbf_sigma (float): RBF MMD sigma
            wass_its (int): Number of iterations in Wasserstein computation.
            wass_lambda (float): Wasserstein lambda.
        """
        if response == 'bin':
            loss = nn.BCEWithLogitsLoss(reduction='none')
        elif response == 'cont':
            loss = nn.MSELoss(reduction='none')
        else:
            raise ValueError("supported response values are 'bin' and 'cont', got '{}' instead".format(response))

        if ltype == 'log':
            # COMPUTE ELBO
            lgate_out_, lgate_out, exp_out = self.forward(x_, t_, infer=False)
            cost = torch.zeros_like(x_[:, 0])

            lexp_out = []
            for i in range(self.K):
                lexp_out.append(-loss(exp_out[:, i], y_))

            lexp_out = torch.stack(lexp_out, 1)
            gate_out = torch.exp(lgate_out)

            cost = torch.sum(torch.mul(gate_out, lexp_out), dim=1)  # eq 12 in paper
            cost = -torch.sum(cost) + self.regularization(0.5, 0.5)

            # IMBALANCE PENALTY (IPM)
            # compute treatment probability
            if use_p_correction:
                p_t = torch.mean(t_)
            else:
                p_t = 0.5

            if imb_fun == 'wass':
                imb_dist, imb_mat = wass(x_, t_, p_t, sq=False, its=wass_its, lam=wass_lambda)
                imb_error = p_alpha * imb_dist
            elif imb_fun == 'wass2':
                imb_dist, imb_mat = wass(x_, t_, p_t, sq=True, its=wass_its, lam=wass_lambda)
                imb_error = p_alpha * imb_dist
            elif imb_fun == 'mmd2_lin':
                imb_dist = mmd2_lin(x_, t_, p_t)
                imb_error = p_alpha * imb_dist
            elif imb_fun == 'mmd2_rbf':
                imb_dist = mmd2_rbf(x_, t_, p_t, rbf_sigma)
                imb_error = p_alpha * imb_dist
            elif imb_fun is None:
                imb_error = 0
            else:
                raise ValueError("supported imb_fun values are ['wass', 'wass2', 'mmd2_lin', 'mmd2_rbf'], "
                                 "got '{}' instead".format(imb_fun))

            cost += imb_error
        else:
            output = self.forward(x_, t_)
            cost = loss(output, y_)
            cost = cost.sum()

        return cost

    def group_sizes(self, X):
        """Returns the number of data points assigned to each subgroup.
        
        Args:
            X (torch.Tensor): Covariate matrix of size (num_subjects, num_features).

        Returns: 
            collections.Counter: giving size for each group
        """
        group = self.get_groups(X)
        counter = Counter(group)
        return counter

    def estimate_individual_outcomes(self, X, T, response='bin', soft=True):
        """Return individual treatment outcomes.
        
        Args:
            X (torch.Tensor): Covariate matrix of size (num_subjects, num_features).
            T (torch.Tensor): Treatment assignment of size (num_subjects,).
            response (str):
            soft (bool):

        Returns:
            list of torch.Tensor, one for each unique value in tensor T
        """
        output = []
        for t in torch.unique(T, sorted=True):
            o = self.forward(X, torch.full_like(T, t), response=response, soft=soft)
            output.append(o)
        return output
        # return [self.forward(X, torch.full_like(T, t), response=response, soft=soft)
        #         for t in torch.unique(T, sorted=True)]

    def get_groups(self, X):
        """Return hard assignment of groups for each sample

        Args:
            X (torch.Tensor): Covariate matrix of size (num_subjects, num_features).

        Returns:
            groups (np.ndarray): Most probable group assignment of each sample, size = (num_samples,)
        """
        # groups = self.forward(X, T, infer=False)[0].data.numpy()
        # groups = np.exp(groups) / np.exp(groups).sum(axis=1).reshape(-1, 1)
        groups = self.get_groups_proba(X)
        groups = np.argmax(groups, axis=1)
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
        a = torch.ones(X.shape[0], dtype=torch.float64)
        z_pred = self.forward(X, a, infer=False)[1].cpu().data.numpy()
        if not log:
            z_pred = np.exp(z_pred)
        return z_pred

    def get_groups_effect(self):
        group_effect = self.treat.cpu().data.numpy()
        return group_effect
