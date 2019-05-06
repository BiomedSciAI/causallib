"""
(C) Copyright 2019 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import os
import warnings

import pandas as pd
from sklearn.utils import Bunch

DATA_DIR_NAME = "data"


def load_data_file(file_name, data_dir_name=DATA_DIR_NAME, sep=","):
    module_path = os.path.dirname(__file__)
    file_path = os.path.join(module_path, data_dir_name, file_name)
    data = pd.read_csv(file_path, sep=sep)
    return data


# todo: contact authors for approval to redistribute the smoking data, then use load instead of fetch
# def load_smoking_weight(raw=False, restrict=True, return_Xay=False):
#     """Fetch and return the smoking-cessation-effect-on-wight-gain dataset.
#
#     This dataset is used throughout Hernan and Robins' Causal Inference Book.
#      https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
#
#     Args:
#         raw (bool): Whether to return the entire DataFrame and descriptors or not.
#                     If False, only confounders are used for the data.
#         restrict (bool): Whether to apply exclusion criteria on the data or not.
#                          Note: if False - data will have censored (NaN) outcomes.
#         return_Xay (bool): Whether to return (data, treatment_assignment, outcome)
#                            or a data-structure with X (data), a (treatment assignment),
#                            and y (outcome) attributes; (and including feature descriptors)
#
#     Returns:
#         Bunch: dictionary-like object
#                attributes are: ‘X’ (covariates), ‘a’ (treatment assignment) `y` (outcome),
#                                ‘descriptors’ (feature description)
#
#                (pd.DataFrame, pd.Series, pd.Series): if return_Xay is True
#     """
#     data = load_data_file("NHEFS.csv")
#     descriptors = load_data_file("NHEFS_codebook.csv").set_index("Variable name")["Description"]
#
#     if raw:
#         return data, descriptors
#
#     return _process_smoking_weight(data, descriptors, restrict, return_Xay)


def fetch_smoking_weight(raw=False, restrict=True, return_Xay=False):
    """Fetch and return the smoking-cessation-effect-on-wight-gain dataset.

    This dataset is used throughout Hernan and Robins' Causal Inference Book.
     https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

    Args:
        raw (bool): Whether to return the entire DataFrame and descriptors or not.
                    If False, only confounders are used for the data.
        restrict (bool): Whether to apply exclusion criteria on the data or not.
                         Note: if False - data will have censored (NaN) outcomes.
        return_Xay (bool): Whether to return (data, treatment_assignment, outcome)
                           or a data-structure with X (data), a (treatment assignment),
                           and y (outcome) attributes; (and including feature descriptors)

    Returns:
        Bunch: dictionary-like object
               attributes are: ‘X’ (covariates), ‘a’ (treatment assignment) `y` (outcome),
                               ‘descriptors’ (feature description)

               (pd.DataFrame, pd.Series, pd.Series): if return_Xay is True
    """
    data = pd.read_csv("https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv")
    try:
        descriptors = pd.read_excel(
            "https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/2012/10/NHEFS_Codebook.xls")
        descriptors = descriptors.set_index("Variable name")["Description"]
    except (ModuleNotFoundError, ImportError) as e:
        warnings.warn('Could not load variable-descriptors of NHEFS data.')
        warnings.warn(str(e))
        descriptors = None

    if raw:
        return data, descriptors

    return _process_smoking_weight(data, descriptors, restrict, return_Xay)


def _process_smoking_weight(data, descriptors=None, restrict=True, return_Xay=False):
    confounders = ["active", "age", "education", "exercise", "race",
                   "sex", "smokeintensity", "smokeyrs", "wt71"]
    if restrict:
        restrictions = ["alcoholpy", "age", "ht", "race", "school",
                        "sex", "smokeintensity", "wt82"]
        missing = data[restrictions].isnull().any(axis="columns")
        data = data.loc[~missing]

    a = data.pop("qsmk")
    y = data.pop("wt82_71")
    X = data[confounders]

    if descriptors is not None:
        descriptors = descriptors[confounders + ["qsmk", "wt82_71"]]

    X = pd.get_dummies(X, columns=["active", "education", "exercise"], drop_first=True)
    X = X.join(X[['age', 'wt71', 'smokeintensity', 'smokeyrs']] ** 2, rsuffix="^2")

    if return_Xay:
        return X, a, y

    data = Bunch(X=X, a=a, y=y, descriptors=descriptors)
    return data

# def load_acic16(instance=1, raw=False, return_Xay=False):
#     """
#     Loads single dataset from the 2016 Atlantic Causal Inference Conference data challenge.
#
#     If used for academic purposes, please consider citing the competition organizers:
#     ```
#     @article{dorie2017automated,
#       title={Automated versus do-it-yourself methods for causal inference:
#              Lessons learned from a data analysis competition},
#       author={Dorie, Vincent and Hill, Jennifer and Shalit, Uri and Scott, Marc and Cervone, Dan},
#       journal={arXiv preprint arXiv:1707.02641},
#       year={2017}
#     }
#     ```
#
#
#     See Also:
#         [Official competition site](http://jenniferhill7.wixsite.com/acic-2016/competition)
#         [Official github with data generating code](https://github.com/vdorie/aciccomp/tree/master/2016)
#         [Paper pre-print](https://arxiv.org/abs/1707.02641)
#
#     Args:
#         instance (int): number between 1-20 (including), dataset to load.
#         raw (bool): Whether to apply contrast ("dummify") on non-numeric columns
#         return_Xay (bool): Whether to return a tuple of (data, treatment_assignment, outcome)
#                            or a data-structure with X (data), a (treatment assignment),
#                            and y (outcome) attributes; (and including feature descriptors).
#
#     Returns:
#         Bunch: dictionary-like object
#                attributes are: ‘X’ (covariates), ‘a’ (treatment assignment) `y` (outcome),
#                                ‘descriptors’ (feature description)
#
#                (pd.DataFrame, pd.Series, pd.Series): if return_Xay is True
#     """
#     dir_name = os.path.join(DATA_DIR_NAME, "acic_challenge_2016")
#
#     X = load_data_file("x.csv", dir_name)
#     descriptors = pd.Series(["No true meaning"] * X.shape[1], index=X.columns)
#     if not raw:
#         # non_numeric_cols = X.columns[X.dtypes == object]
#         non_numeric_cols = X.select_dtypes(include=[object]).columns
#         X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
#     # TODO: redo loading with the counterfactual files and load counterfactual outcomes as well.
#     zy = load_data_file("zy_{}.csv".format(instance), dir_name)
#     a = zy["z"].rename("a")
#     y = zy["y"]
#
#     if return_Xay:
#         return X, a, y
#
#     data = Bunch(X=X, a=a, y=y, descriptors=descriptors)
#     return data
