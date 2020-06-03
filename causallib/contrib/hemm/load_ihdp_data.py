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
import urllib.request
import zipfile


def __download_data(url_path, local_path, verbose=0):
    if not os.path.exists(local_path):
        if verbose:
            print(f"Downloading data from {url_path} to {local_path}")
        req = urllib.request.urlretrieve(url=url_path, filename=local_path)
        return req[0]
    return local_path


def loadIHDPData(cache_dir=None, verbose=0, delete_extracted=True):
    """Downloads and loads IHDP-1000 dataset.
    Taken From Fredrik Johansson's website: http://www.fredjo.com/

    Args:
        cache_dir (str): Directory to which files will be downloaded
            If None: files will be downloaded to ~/causallib-data/.
        verbose (int): Controls the verbosity: the higher, the more messages.
        delete_extracted (bool): Delete extracted files from disk once loaded

    Returns:
        dict[str, dict[str, np.ndarray]]: "TRAIN" and "TEST" sets as keys.
            Values are dictionaries with `'x', 't', 'yf', 'ycf', 'mu0', 'mu1'` keys standing for
            covariates, treatment, factual outcome, counterfactual outcome, and noiseless potential outcomes

    Notes:
        Requires internet connection in case local data files do not already exist.
        Will save a local copy of the download

    """
    base_remote_url = "http://www.fredjo.com/files/"
    file_name = "ihdp_npci_1-1000.{phase}.npz.zip"

    # Set local download location:
    if cache_dir is None:
        cache_dir = os.path.join("~", 'causallib-data')
        cache_dir = os.path.expanduser(cache_dir)  # Expand ~ component to full path
        cache_dir = os.path.join(cache_dir, "IHDP")
    # cache_dir = cache_dir.replace("/", os.sep)
    os.makedirs(cache_dir, exist_ok=True)

    data = {}
    for phase in ["train", "test"]:
        # Obtain local copy of the data:
        phase_file_name = file_name.format(phase=phase)
        file_path = __download_data(
            url_path=base_remote_url + phase_file_name,
            local_path=os.path.join(cache_dir, phase_file_name),
            verbose=verbose
        )

        # Extract zipped data:
        npz_file_path = file_path.rsplit(".", maxsplit=1)[0]  # Remove ".zip" extension
        if not os.path.exists(npz_file_path):
            with zipfile.ZipFile(file_path) as zf:
                if verbose:
                    print(f"Extracting file into {npz_file_path}")
                zf.extractall(path=cache_dir)

        # Load data:
        phase_data = np.load(npz_file_path)
        phase_data = dict(phase_data)  # Load into memory, avoid lazy-loading
        data[phase.upper()] = phase_data

        # # In-memory extraction, works only in python>=3.7 https://github.com/python/cpython/pull/4966
        # with zipfile.ZipFile(file_path) as zf:
        #     internal_file_name = phase_file_name.rsplit(".", maxsplit=1)[0]  # Remove ".zip" extension
        #     with zf.open(internal_file_name, 'r') as npz_file:
        #         data[phase.upper()] = dict(np.load(npz_file))

        if delete_extracted:
            if verbose:
                print(f"Deleting extracted file {npz_file_path}")
            os.remove(npz_file_path)

    return data
