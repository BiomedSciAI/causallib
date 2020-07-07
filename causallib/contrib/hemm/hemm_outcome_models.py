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

import torch.nn.functional as F
import torch.nn as nn
import torch


class BalancedNet(nn.Module):
    """A torch.model used as a component of the HEMM module to determine the outcome as a function of confounders. 
    The balanced net consists of two different neural networks for the outcome and counteractual.
    """

    def __init__(self, D_in, H, D_out):
        """Instantiate two nn.Linear modules and assign them as member variables.

        Args:
            D_in: input dimension
            H: dimension of hidden layer
            D_out: output dimension
        """

        super(BalancedNet, self).__init__()
        
        self.f1 = nn.Linear(D_in, H) 
        self.f2 = nn.Linear(H, D_out)
        
        self.cf1 = nn.Linear(D_in, H) 
        self.cf2 = nn.Linear(H, D_out)
  
    def forward(self, x):
        """Accept a Variable of input data and return a Variable of output data.

        We can use Modules defined in the constructor as well as arbitrary operators on Variables.
        """
        h_relu = F.elu(self.f1(x))
        f = self.f2(h_relu)
        
        h_relu = F.elu(self.cf1(x))
        cf = self.cf2(h_relu)
        
        out = torch.cat((f, cf), dim=1)
        
        return out


def genMLPModule(D_in, H, out=1):
    """Fit an MLP with an ELU activation.

    This allows for a single neural network to have two heads for the outcome and counterfactual.
    """
    if type(H) is int:
        model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ELU(), torch.nn.Linear(H, out))
        return model.double()


def genLinearModule(D_in,  out=1):
    """Two separate linear functions of the input covariates."""
    model = torch.nn.Sequential(torch.nn.Linear(D_in, out),)   
    return model.double()
