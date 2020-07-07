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

"""
Synthetic data set generation.

This module provides data suitable for testing the HEMM estimator and for 
writing example notebooks.
"""

import numpy as np


def gen_outcomes(X, T, seed=0):
    """Synthetically generate outcome data from provided covariates and treatment assignment.

    Args:
        X: input covariates
        T: treatment assignment
        seed (int): seed to fix the random number generator
    
    Returns:
        y: observed outcomes
        mu1: true mean of treated group
        mu0: true mean of control group
        Z: indicator whether each sample belongs to positive enhanced or negative enhanced subgroup
    """
    np.random.seed(seed)
    n = X.shape[0]
    
    treated = np.where(T==1)[0]
    control = np.where(T==0)[0]

    pos = np.where(1-X[:, 1]<X[:, 0])[0]
    neg = np.where(X[:, 0]<=1-X[:, 1])[0]
    
    endist = (X[:, 0]-0.5 )**2 + (X[:, 1]-0.5 )**2
    
    posen = np.where(endist<0.1)[0]
    negen = np.where(endist>=0.1)[0]
    
    y   = np.zeros(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    
    treatposten  = list(set(treated)&set(pos)&set(posen))
    treatpostnen = list(set(treated)&set(pos)&set(negen))
    treatnegten  = list(set(treated)&set(neg)&set(posen))
    treatnegtnen = list(set(treated)&set(neg)&set(negen))
    controlpos   = list(set(control)&set(pos))
    controlneg   = list(set(control)&set(neg))
    
    y[treatposten]  = np.random.binomial(1, 0.8, len(treatposten))
    y[treatpostnen] = np.random.binomial(1, 0.6, len(treatpostnen))
    y[treatnegten]  = np.random.binomial(1, 0.6, len(treatnegten))
    y[treatnegtnen] = np.random.binomial(1, 0.4, len(treatnegtnen))
    y[controlpos]   = np.random.binomial(1, 0.4, len(controlpos))
    y[controlneg]   = np.random.binomial(1, 0.2, len(controlneg))

    mu0[pos]  = [0.4]*len(pos)
    mu0[neg]  = [0.2]*len(neg)

    posten  = list(set(pos)&set(posen))
    postnen = list(set(pos)&set(negen))
    negten  = list(set(neg)&set(posen))
    negtnen = list(set(neg)&set(negen))
    
    mu1[posten]  = [0.8]*len(posten)
    mu1[postnen] = [0.6]*len(postnen)
    mu1[negten]  = [0.6]*len(negten)
    mu1[negtnen] = [0.4]*len(negtnen)

    z = np.zeros(n)
    
    z[posen] = 1
    z[negen] = 0
    
    return y, mu1, mu0, z


def gen_treat(X, seed):
    """Generate treatment assignments for the given covariates.

    Args:
        X: input covariates
        seed: PRNG seed

    Returns:
        Individual treatment assignment.
    """
    np.random.seed(seed)

    less = np.where(X[:, 0]< 0.5)[0]
    more = np.where(X[:, 0]>=0.5)[0]
    
    treated = np.zeros(len(X))
    treated[more] = np.random.binomial(1, 0.6, len(more))
    treated[less] = np.random.binomial(1, 0.4, len(less))
    
    return treated


def gen_data(n, d, seed=0):
    """Generate outcome, treatment assignment and (d+1)-dimensional covariates for n samples.

    Args:
        n: desired number of samples.
        d: number of covariates.
        seed: PRNG seed

    Returns:
        X: generated covariates array, size = n x d+1
        T: generated treatment assignment
        Y: generated observed outcomes
        Z: generated indicator whether each sample belongs to positive enhanced or negative enhanced subgroup
        mu1: true mean counterfactual outcome under treatment
        mu0: true mean counterfactual outcome under control
    """
    np.random.seed(seed)
    
    treatment_group_size = n//2
    Xt = np.random.rand(treatment_group_size, d)
    Xc = np.random.rand(treatment_group_size, d)
    X = np.vstack([Xt, Xc])
    ones = np.array([np.ones(len(X))]).T
    X = np.hstack([X, ones])
    
    T = gen_treat(X, seed)
    
    Y, mu1, mu0, Z = gen_outcomes(X, T, seed)
    
    return X, T, Y, Z, mu1, mu0


def gen_montecarlo(n, d, num_repeats=100):
    """Generate synthetic train and test data (80:20 proportions).

    Args:
        n: desired number of samples.
        d: dimensionality of each samples (number of covariates) - 1
        num_repeats: number of subgroups to generate, each repeat will produce n covariates with dimension d + 1.

    Returns:
        Dictionary holding test and training data. Each data slot has the form:
            x: generated covariates array, size = n x d+1 x num_repeats
            t: generated treatment assignment for covariates, size = n x num_repeats
            yf: generated observed outcomes for each covariate, size = n x num_repeats
            Z: generated indicator whether each sample belongs to positive enhanced or negative enhanced subgroup.
               size = n x num_repeats
            mu1: true mean counterfactual outcome under treatment, size = n x num_repeats
            mu0: true mean counterfactual outcome under control, size = n x num_repeats
    """
    Xtr, Ttr, Ytr, Ztr, mu1tr, mu0tr = [], [], [], [], [], []
    Xte, Tte, Yte, Zte, mu1te, mu0te = [], [], [], [], [], []

    for repeat in range(num_repeats):
        X_, T_, Y_, Z_, mu1_, mu0_ = gen_data(n, d, repeat)
        
        # Of the n data points to generate, use 80% for training and 20% for testing.
        r = (4*n)//5
        
        X_tr, T_tr, Y_tr, Z_tr, mu1_tr, mu0_tr = X_[:r], T_[:r], Y_[:r], Z_[:r], mu1_[:r], mu0_[:r]
        X_te, T_te, Y_te, Z_te, mu1_te, mu0_te = X_[r:], T_[r:], Y_[r:], Z_[r:], mu1_[r:], mu0_[r:]

        Xtr.append(X_tr.T)
        Ttr.append(T_tr)
        Ytr.append(Y_tr)
        Ztr.append(Z_tr)
        mu1tr.append(mu1_tr)
        mu0tr.append(mu0_tr)
        
        Xte.append(X_te.T)
        Tte.append(T_te)
        Yte.append(Y_te)
        Zte.append(Z_te)
        mu1te.append(mu1_te)
        mu0te.append(mu0_te)   
    
    data = {}
    data['TRAIN'] = {}
    data['TEST'] = {}

    data['TRAIN']['x'] = np.array(Xtr).T
    data['TRAIN']['t'] = np.array(Ttr).T
    data['TRAIN']['yf'] = np.array(Ytr).T
    data['TRAIN']['z'] = np.array(Ztr).T
    data['TRAIN']['mu1'] = np.array(mu1tr).T
    data['TRAIN']['mu0'] = np.array(mu0tr).T
    
    data['TEST']['x'] = np.array(Xte).T
    data['TEST']['t'] = np.array(Tte).T
    data['TEST']['yf'] = np.array(Yte).T
    data['TEST']['z'] = np.array(Zte).T
    data['TEST']['mu1'] = np.array(mu1te).T
    data['TEST']['mu0'] = np.array(mu0te).T
    
    return data
