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
IHDP Data Downloading, Unzipping and Loading

This module provides data suitable for testing the HEMM estimator and for 
writing example notebooks.
"""

import numpy as np
import os
def loadIHDPData():
    #Taken From Fredrik Johansson's personal webiste: http://fredrikjo.com 
    if os.path.exists('IHDP'):
        
        print ("IHDP Data exists")

    else:
        print ("IHDP Does not exist, Downloading... ")
        
        os.system("mkdir IHDP")
        os.system("wget http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip")
        os.system("wget http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip") 
        os.system("mv ihdp_npci_1-1000.train.npz.zip IHDP/")
        os.system("mv ihdp_npci_1-1000.test.npz.zip  IHDP/") 
        os.system("unzip IHDP/ihdp_npci_1-1000.train.npz.zip -d IHDP")
        os.system("unzip IHDP/ihdp_npci_1-1000.test.npz.zip -d IHDP")

       
    
    import numpy as np
    dat = {}
    
    dat['TRAIN'] = dict(np.load("IHDP/ihdp_npci_1-1000.train.npz"))
    dat['TEST']  = dict(np.load("IHDP/ihdp_npci_1-1000.test.npz"))
    
    return dat