# (C) Copyright 2019 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created on Oct 24, 2019

from causallib.utils.stat_utils import robust_lookup
import os
import pandas as pd
from sklearn.utils import Bunch

DATA_DIR_NAME = "data"


def load_data_file(file_name, data_dir_name, sep=","):
    module_path = os.path.dirname(__file__)
    file_path = os.path.join(module_path, data_dir_name, file_name)
    data = pd.read_csv(file_path, sep=sep)
    return data


def load_nhefs(raw=False, restrict=True, augment=True, onehot=True):
    """Loads the NHEFS smoking-cessation and weight-loss dataset.

    Data was gathered during an observational study conducted by the NHANS
    during the 1970's and 1980'. It follows a cohort a people whom some
    decided to quite smoking and some decided to persist, and record the
    gain in weight for each individual to try estimate the causal contribution
    of smoking cessation on weight gain.

    This dataset is used throughout Hernán and Robins' Causal Inference Book.
     https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
    If used for academic purposes, please consider citing the book:
     Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.

    Args:
        raw (bool): Whether to return the entire DataFrame and descriptors or not.
                    If False, only confounders are used for the data.
                    If True, returns a (pd.DataFrame, pd.Series) tuple (data and description).
        restrict (bool): Whether to apply exclusion criteria on missing data or not.
                         Note: if False - data will have censored (NaN) outcomes.
        augment (bool): Whether to add augmented (squared) features
                    If False, only original data returned.
                    If True, squares continuous valued columns ['age', 'wt71', 'smokeintensity', 'smokeyrs']
                    and joins to data frame with suffix '^2'
        onehot (bool): Whether to one-hot encode categorical data.
                    If False, categorical data ["active", "education", "exercise"], will be returned
                    in individual columns with categorical values.
                    If True, extra columns with the categorical value one-hot encoded.

    Returns:
        Bunch: dictionary-like object
               attributes are: `X` (covariates), `a` (treatment assignment) `y` (outcome),
                               `descriptors` (feature description)
    """
    dir_name = os.path.join(DATA_DIR_NAME, "nhefs")
    data = load_data_file("NHEFS.csv", dir_name)
    descriptors = load_data_file("NHEFS_codebook.csv", dir_name)

    descriptors = descriptors.set_index("Variable name")["Description"]

    if raw:
        return data, descriptors

    confounders = ["active", "age", "education", "exercise", "race",
                   "sex", "smokeintensity", "smokeyrs", "wt71"]

    if restrict:
        restrictions = ["wt82"]
        missing = data[restrictions].isnull().any(axis="columns")
        data = data.loc[~missing]

    a = data.pop("qsmk")
    y = data.pop("wt82_71")
    X = data[confounders]
    descriptors = descriptors[confounders + ["qsmk", "wt82_71"]]
    if onehot:
        X = pd.get_dummies(
            X, columns=["active", "education", "exercise"], drop_first=True)
    if augment:
        X = X.join(X[['age', 'wt71', 'smokeintensity', 'smokeyrs']]
                   ** 2, rsuffix="^2")

    data = Bunch(X=X, a=a, y=y, descriptors=descriptors)
    return data


def load_nhefs_survival(augment=True, onehot=True):
    """
    Loads and pre-processes the NHEFS smoking-cessation dataset.

    Data was gathered in an observational study conducted by the NHANS
    during the 1970's and 1980'. It follows a cohort a people whom some
    decided to quite smoking and some decided to persist, and record the
    death events within 10 years of follow-up.

    This dataset is used throughout Hernán and Robins' Causal Inference Book.
    https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
    If used for academic purposes, please consider citing the book:
    Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.

    Args:
        augment (bool): Whether to add augmented (squared) features
                    If False, only original data returned.
                    If True, squares continuous valued columns ['age', 'wt71', 'smokeintensity', 'smokeyrs']
                    and joins to data frame with suffix '^2'
        onehot (bool): Whether to one-hot encode categorical data.
                    If False, categorical data ["active", "education", "exercise"], will be returned
                    in individual columns with categorical values.
                    If True, extra columns with the categorical value one-hot encoded.

    Returns:
        X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
        a (pd.Series): Treatment assignment of size (num_subjects,). Quit smoking vs. non-quit.
        t (pd.Series): Followup duration, size (num_subjects,).
        y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).
    """
    nhefs_all = load_nhefs(raw=True)[0]
    t = (nhefs_all["yrdth"] - 83) * 12 + nhefs_all["modth"]
    t = t.fillna(120)
    t = t.rename("longevity")
    y = nhefs_all["death"]

    nhefs = load_nhefs(augment=augment, onehot=onehot, restrict=False)
    a = nhefs.a
    X = nhefs.X

    data = Bunch(X=X, a=a, t=t, y=y, descriptors=nhefs.descriptors)
    return data


def load_acic16(instance=1, raw=False):
    """ Loads single dataset from the 2016 Atlantic Causal Inference Conference data challenge.

    The dataset is based on real covariates but synthetically simulates the treatment assignment
    and potential outcomes. It therefore also contains sufficient ground truth to evaluate
    the effect estimation of causal models.
    The competition introduced 7700 simulated files (100 instances for each of the 77
    data-generating-processes). We provide a smaller sample of one instance from 10
    DGPs. For the full dataset, see the link below to the competition site.

    If used for academic purposes, please consider citing the competition organizers:
     Vincent Dorie, Jennifer Hill, Uri Shalit, Marc Scott, and Dan Cervone. "Automated versus do-it-yourself methods
     for causal inference: Lessons learned from a data analysis competition."
     Statistical Science 34, no. 1 (2019): 43-68.

    Args:
        instance (int): number between 1-10 (inclusive), dataset to load.
        raw (bool): Whether to apply contrast ("dummify") on non-numeric columns
                    If True, returns a (pd.DataFrame, pd.DataFrame) tuple (one for covariates and the second with
                    treatment assignment, noisy potential outcomes and true potential outcomes).

    Returns:
        Bunch: dictionary-like object
               attributes are: `X` (covariates), `a` (treatment assignment), `y` (outcome),
                               `po` (ground truth potential outcomes: `po[0]` potential outcome for controls and
                                `po[1]` potential outcome for treated),
                               `descriptors` (feature description).


    See Also:
        * `Publication <https://projecteuclid.org/euclid.ss/1555056030>`_
        * `Official competition site <http://jenniferhill7.wixsite.com/acic-2016/competition>`_
        * `Official github with data generating code <https://github.com/vdorie/aciccomp/tree/master/2016>`_
    """
    dir_name = os.path.join(DATA_DIR_NAME, "acic_challenge_2016")

    X = load_data_file("x.csv", dir_name)
    zymu = load_data_file("zymu_{}.csv".format(instance), dir_name)

    if raw:
        return X, zymu

    non_numeric_cols = X.select_dtypes(include=[object]).columns
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)

    a = zymu["z"].rename("a")
    # # Extract observed outcome:
    y = zymu[["y0", "y1"]]
    y = y.rename(columns=lambda x: int(x.strip("y")))  # remove 'y' prefix to allow lookup
    y = robust_lookup(y, a)
    # # Potential outcomes:
    po = zymu[["mu0", "mu1"]]
    po = po.rename(columns=lambda x: x.strip("mu"))

    descriptors = pd.Series(data="No true meaning", index=X.columns)
    data = Bunch(X=X, a=a, y=y, po=po, descriptors=descriptors)
    return data
