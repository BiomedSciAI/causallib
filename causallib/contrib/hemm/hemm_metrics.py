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
# Big thanks to Akanksha Atrey <aatrey@cs.umass.edu> for original 
# implementation of this module. 

import torch
import numpy as np


def pdist2sq(X,Y):
    """ 
    Computes the squared Euclidean distance between all pairs x in X, y in Y.
    """
    C = -2*torch.matmul(X,torch.transpose(Y,0,1))
    nx = torch.sum(torch.pow(X,2),dim=1,keepdim=True)
    ny = torch.sum(torch.pow(Y,2),dim=1,keepdim=True)
    D = (C + torch.transpose(ny,0,1)) + nx

    return D


def mmd2_lin(X, t, p):
    """
    Computes linear maximum mean discrepancy (MMD) metric. 
    """
    it = np.where(t==1)[0]
    ic = np.where(t==0)[0]

    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.mean(Xc)
    mean_treated = torch.mean(Xt)

    mmd = torch.sum(torch.pow(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control, 2))

    return mmd


def mmd2_rbf(X, t, p, sig=0.1):
    """
    Computes the l2-RBF maximum mean discrepancy (MMD) for X given t.
    http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf -- Eq3
    """
    it = np.where(t==1)[0]
    ic = np.where(t==0)[0]

    Xc = X[ic]
    Xt = X[it]

    if list(Xc.shape)[0] == 0.0 or list(Xt.shape)[0] == 0.0:
        return torch.tensor(float('nan')) # pylint: disable=E1102

    Kcc = torch.exp(-pdist2sq(Xc,Xc)/np.square(sig))
    Kct = torch.exp(-pdist2sq(Xc,Xt)/np.square(sig))
    Ktt = torch.exp(-pdist2sq(Xt,Xt)/np.square(sig))

    m = float(list(Xc.shape)[0])
    n = float(list(Xt.shape)[0])

    mmd = np.square(1.0-p)/(m*(m-1.0))*(torch.sum(Kcc)-m)
    mmd = mmd + np.square(p)/(n*(n-1.0))*(torch.sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*torch.sum(Kct)
    mmd = 4.0*mmd

    return mmd


def wass(X,t,p,lam=10.0,its=20,sq=False,backpropT=False):
    """
    Computes the Wasserstein metric.

    Algorithm 3 from "Fast Computation of Wasserstein Barycenters", Cuturi and Doucet (2014) (https://arxiv.org/pdf/1310.4375.pdf).
    See supplement B.1 from Shalit et al. (2017) for more details (https://arxiv.org/abs/1606.03976).
    """
    it = np.where(t==1)[0]
    ic = np.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    nc = float(list(Xc.shape)[0])
    nt = float(list(Xt.shape)[0])

    if list(Xc.shape)[0] == 0.0 or list(Xt.shape)[0] == 0.0:
        return torch.tensor(float('nan')), torch.tensor(float('nan')) # pylint: disable=E1102

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = torch.sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = torch.mean(M)
    M_drop = torch.nn.Dropout(1/(nc*nt))(M)
    delta = (torch.max(M)).detach()
    eff_lam = (lam/M_mean).detach()

    ''' Compute new distance matrix '''
    Mt = M
    row = (delta*torch.ones((M[0:1,:]).shape)).type(torch.float64)
    col = torch.cat((delta*torch.ones((M[:,0:1]).shape),torch.zeros((1,1))), 0).type(torch.float64)
    Mt = torch.cat((M,row), 0)
    Mt = torch.cat((Mt,col), 1)

    ''' Compute marginal vectors '''
    a = torch.cat((p*torch.ones((np.where(t>0)[0].reshape(-1,1)).shape)/nt, (1-p)*torch.ones((1,1))), 0).type(torch.float64)
    b = torch.cat(((1-p)*torch.ones((np.where(t<1)[0].reshape(-1,1)).shape)/nc, p*torch.ones((1,1))), 0).type(torch.float64)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = torch.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(torch.matmul(ainvK,(b/torch.transpose(torch.matmul(torch.transpose(u,0,1),K),0,1))))
    v = b/(torch.transpose(torch.matmul(torch.transpose(u,0,1),K),0,1))

    T = u*(torch.transpose(v,0,1)*K)

    if not backpropT:
        T = T.detach()

    E = T*Mt
    D = 2*torch.sum(E)

    return D, Mlam
