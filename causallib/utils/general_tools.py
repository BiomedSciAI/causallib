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

Created on Jun 27, 2018

General (i.e. non-scientific) utils used throughout the package.
"""
import warnings

import pandas as pd
from numpy import isscalar as np_is_scalar
from pandas import Series

from .exceptions import ColumnNameChangeWarning


def get_iterable_treatment_values(treatment_values, treatment_assignment, sort=True):
    """
    Convert an optionally provided specification of unique treatment values to an iterable of the unique treatment
    options.
    Since user can provide treatment values as either an iterable or a single value, this conversion to an iterable
    allows a generic approach of going over all provided treatment values.

    Args:
        treatment_values (None|Any|list[Any]): Unique values of possible treatment values.
                                               Can be either one value (scalar) or list of values (any iterable).
                                               Can be None, if None - treatment values are inferred from treatment
                                               assignment.
        treatment_assignment (Series): The observed treatment assignment, used to infer a list of unique treatment
                                       values in case no treatment values are provided (None is passed to
                                       treatment_values).
        sort (bool): Whether to sort the treatment values

    Returns:
        list[Any]: list of unique treatment values.
    """
    treatment_values = treatment_assignment.unique() if treatment_values is None else treatment_values
    treatment_values = [treatment_values] if np_is_scalar(treatment_values) else treatment_values
    if sort:
        treatment_values = sorted(treatment_values)
    return treatment_values


def create_repr_string(o):
    """

    Args:
        o (object): any core object

    Returns:
        str: repr string based on internal attributes
    """
    # Filter peripheral unimportant attribute names:
    params = [
        attr for attr in dir(o) if not attr.startswith('__')  # Data-model dunder methods
                                   and not callable(getattr(o, attr, None))  # Remove other methods, keep only fields
                                   and not attr.startswith("_abc")  # Remove abstract-related attributes
                                   and not attr.endswith("_")  # Remove attributes stated after initialization
                                   and not attr == "CALCULATE_EFFECT"  # Remove the EffectEstimator attribute
    ]

    # Special treatment for scikit-learn's learner object (the "learner" attribute) - place last in a new line:
    learner_included = False
    if "learner" in params:
        params.remove("learner")
        # params.append("learner")    # move learner to be last parameter
        learner_included = True

    # Couple attribute name with attribute value
    params = [(attr, getattr(o, attr)) for attr in params]
    params_string = ", ".join("{}={}".format(*param) for param in params)

    if learner_included:
        # Place "learner" attribute last in a different line:
        params_string += ",\n{spaces}learner={learner}".format(spaces=" " * (len(o.__class__.__name__) + 1),
                                                               learner=getattr(o, "learner"))
    repr_string = "{cls_name}({params})".format(cls_name=o.__class__.__name__,
                                                params=params_string)
    return repr_string


def check_learner_is_fitted(learner):
    """Return True if fitted and False otherwise"""
    # Following scikit-learn's convention,
    # fitted models have additional attributes ending with underscores.
    # See: https://scikit-learn.org/dev/glossary.html#term-fitted
    # Hence checking whether these exist is sufficient:
    after_init_attr = [attr for attr in learner.__dict__.keys() if attr.endswith("_")]
    is_fitted = len(after_init_attr) > 0
    return is_fitted


def align_column_name_types_for_join(X, a, a_name=None):
    """Align columns/name types in `X` and `a` to match so that joining them
    creates homogeneous column names type and sklearn>=1.2 don't break."""
    if a_name is None:
        warnings.warn("`a.name` is None. Renaming to 'a'.", ColumnNameChangeWarning)
        a_name = "a"

    column_names_types = {type(c) for c in X.columns}
    if len(column_names_types) > 1:
        X.columns = X.columns.astype(str)
        warnings.warn(
            f"Column names of `X` contain mixed types "
            f"({ {t.__name__ for t in column_names_types} }), "
            f"which sklearn>1.2 will raise for. "
            f"Therefore `X.columns` were all converted to string.",
            ColumnNameChangeWarning,
        )
    column_names_type = column_names_types.pop()

    if hasattr(a, "columns"):  # a DataFrame
        a_name_type = {type(c) for c in a.columns}.pop()
    elif hasattr(a, "name"):  # a Series
        a_name_type = type(a_name)
    else:
        raise RuntimeError(
            f"Variable `a` doesn't seem to be neither a `DataFrame` nor a `Series`, "
            f"but rather a {type(a).__qualname__}.",
        )

    if a_name_type == str:
        X.columns = X.columns.astype(str)
        warnings.warn(
            "Converting `X.columns` to strings to match `a.name` type.",
            ColumnNameChangeWarning,
        )
        if hasattr(a, "columns"):  # a DataFrame
            a = a.add_prefix(f"{a_name}_")
        elif hasattr(a, "name"):  # a Series
            a.name = a_name

    if a_name_type == int and column_names_type == str:
        X.columns = list(range(X.shape[1]))

    return X, a


def column_name_type_safe_join(X, a, join="outer"):
    """Joins the columns of 2 pandas Dataframe/Series
    in a way that respects scikit-learn's  demand for a single-type
    column name (e.g., either all ints or all strings).

    Args:
        X (pd.DataFrame | pd.Series):
        a (pd.DataFrame | pd.Series):
        join (str): {"outer", "inner", "left" right"}. Compatible with `pd.concat`

    Returns:
        pd.DataFrame
    """
    if hasattr(u, "columns"):
        columns = u.columns
    elif hasattr(u, "name"):
        columns = [u.name]
    else:
        columns = []

    

    res = pd.concat([u, w], join=join, axis="columns")
    return res

