import pandas as pd
import numpy as np
from typing import List, Optional, Union
import uuid
from collections import Counter


def get_person_time_df(t: pd.Series, y: pd.Series, a: Optional[pd.Series] = None, w: Optional[pd.Series] = None,
                       X: Optional[pd.DataFrame] = None, return_individual_series: bool = False) -> pd.DataFrame:
    """
    Converts standard input format into an expanded person-time format.
    Input series need to be indexed by subject IDs and have non-null names (including index name).

    Args:
        t (pd.Series): Followup duration, size (num_subjects,).
        y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).
        a (pd.Series): Treatment assignment of size (num_subjects,).
        w (pd.Series): Optional subject weights
        X (pd.DataFrame): Optional baseline covariate matrix of size (num_subjects, num_features).
        return_individual_series (bool): If True, returns a tuple of Series/DataFrames instead of a single DataFrame

    Returns:
        pd.DataFrame: Expanded person-time format with columns from X and expanded 'a', 'y', 't' columns

    Examples:
        This example standard input:

            age  height  a  y  t
        id
        1    22     170  0  1  2
        2    40     180  1  0  1
        3    30     165  1  0  2


        Will be expanded to:

            age  height  a  y  t
        id
        1    22     170  0  0  0
        1    22     170  0  0  1
        1    22     170  0  1  2
        2    40     180  1  0  0
        2    40     180  1  0  1
        3    30     165  1  0  0
        3    30     165  1  0  1
        3    30     165  1  0  2

    """
    a, t, y, w, X = canonize_dtypes_and_names(a, t, y, w, X)
    data = pd.concat([X, w, a], axis=1)  # X and/or w may be None - in which case they're ignored by pandas

    result_col_order = list(data.columns) + [y.name, t.name]

    # Create discrete-increment time vector according to the time of each id (index):
    t_expand = t.reindex(t.index.repeat(t + 1))  # Replicate rows by subject time (+1 since it's a closed-interval)
    t_expand = t_expand.groupby(t_expand.index).cumcount()  # filling each row with time from 0 to t_id
    t_expand.name = t.name

    # prepare `y` to merge on `id` and time
    yt = pd.concat([y, t], axis=1)

    # merge outcome with unfolded-time, so that the outcome only matches the existing (last) event time,
    yt_expand = pd.merge(left=t_expand.reset_index(), right=yt.reset_index(), how="left", on=[t.index.name, t.name])

    # wherever `y` didn't match for the `id, t` combo, it was set to NaN - so impute with zero
    yt_expand[y.name] = yt_expand[y.name].fillna(0).astype(int)

    # merge covariates-treatment on `id` will simply duplicate the covariates/treatment to fit the expanded format
    res = pd.merge(left=data, right=yt_expand, how='right', left_index=True, right_on=data.index.name)
    res.set_index(t.index.name, inplace=True)  # restore index by subject id

    if return_individual_series:
        # split into X, w, a, y and t (all in person-time format)
        res = (
            res[X.columns] if X is not None else None,
            res[w.name] if w is not None else None,
            res[a.name] if a is not None else None,
            res[t.name],
            res[y.name],
        )
    else:
        res = res[result_col_order]  # canonize columns order: covariate columns, w, a, y, t
    return res


def get_regression_predict_data(X: pd.DataFrame, times: pd.Series):
    """
    Generates prediction data for a regression fitter: repeats patient covariates per time point in 'times'.
    Example:

        X
        ===================
                age  height
            id
            1    22     170
            2    40     180

        times
        =====================
            0
            1
            2

        Result - pred_data_X
        =====================
                age  height  t
            id
            1    22     170  0
            1    22     170  1
            1    22     170  2
            2    40     180  0
            2    40     180  1
            2    40     180  2

    Args:
        X (pd.DataFrame):  Covariates DataFrame
        times (pd.Series):  A Series of time points to predict

    Returns:
        pred_data_X (pd.DataFrame):  DataFrame with repeated covariates per time point.
                                     Index is subject ID with repeats, columns are X + a time column, which is a repeat
                                     of 'times' per subject.
        t_name (str):  Name of time column in pred_data_X. Default is 't', but since we concatenate a column to a
                       covariates frame, we might need to add a random suffix to it.

    """
    X_repeated = X.loc[X.index.repeat(len(times))]
    times_repeated = pd.concat([times] * X.shape[0], axis=0)
    times_repeated.index = X_repeated.index
    times_repeated.name = times.name if times.name is not None else 't'
    times_repeated = times_repeated.astype('int64')

    # Pred_data_X is a cartesian product of times and features (X rows are repeated per time point)
    pred_data_X, new_col_names = safe_join(df=X_repeated, list_of_series=[times_repeated], return_series_names=True)
    t_name = new_col_names[0]

    return pred_data_X, t_name


def compute_survival_from_single_hazard_curve(hazard: List, logspace: bool = False) -> List:
    """
    Computes survival curve from an array of point hazards.
    Note that trailing NaN are supported
    Args:
        hazard (list):  list/array of point hazards
        logspace (bool):  whether to compute in logspace, for numerical stability

    Returns:
        list: survival at each time-step
    """
    survival = 1 - hazard
    if logspace:
        log_survival = np.log(survival)
        log_survival = np.cumsum(log_survival)
        survival = np.exp(log_survival)
    else:
        survival = np.cumprod(survival)

    return survival


def canonize_dtypes_and_names(a=None, t=None, y=None, w=None, X=None):
    """
    Housekeeping method that assign names for unnamed series and canonizes their data types.

    Args:
        a (pd.Series|None): Treatment assignment of size (num_subjects,).
        t (pd.Series|None): Followup duration, size (num_subjects,).
        y (pd.Series|None): Observed outcome (1) or right censoring event (0), size (num_subjects,).
        w (pd.Series|None): Optional subject weights
        X (pd.DataFrame|None): Baseline covariate matrix of size (num_subjects, num_features).

    Returns:
        a, y, t, w, X
    """
    if a is not None:
        a = a.astype(int)
        a.name = 'a' if a.name is None else a.name

    if t is not None:
        t = t.astype(int)
        t.name = 't' if t.name is None else t.name

    if y is not None:
        y = y.astype(int)
        y.name = 'y' if y.name is None else y.name

    if w is not None:
        w.name = 'w' if w.name is None else w.name

    # Set index name
    for obj in [a, y, t, w, X]:
        if obj is not None and obj.index.name is None:
            obj.index.name = 'id'

    return a, t, y, w, X


def add_random_suffix(name, suffix_length=4):
    """
    Adds a random suffix to string, by computing uuid64.hex.

    Args:
        name: input string
        suffix_length: length of desired added suffix.

    Returns:
        string with suffix
    """
    if suffix_length:
        name += "_" + uuid.uuid4().hex[:suffix_length]
    return name


def safe_join(df: Optional[pd.DataFrame] = None, list_of_series: List[pd.Series] = None, return_series_names=False):
    """
    Safely joins (concatenates on axis 1) a collection of Series (or one DataFrame and multiple Series),
    while renaming Series that have a duplicate name (a name that already exists in DataFrame or another Series).
    * Note that DataFrame columns are never changed (only Series names are).

    Args:
        df (pd.DataFrame):  optional DataFrame. If provided, will join Series to DataFrame
        list_of_series (List[pd.Series]): list of Series for safe-join
        return_series_names (bool):  if True, returns a list of (potentially renamed) Series names

    Returns:
        1. single concatenated DataFrame
        2. list of (potentially renamed) Series names
    """

    list_of_frames_or_series: List[Union[pd.DataFrame, pd.Series]] = []
    if df is not None:
        list_of_frames_or_series.append(df)
    list_of_frames_or_series.extend(list_of_series)

    # Get all column names
    all_col_names = []
    for item in list_of_frames_or_series:
        if isinstance(item, pd.Series):
            all_col_names.append(item.name)
        elif isinstance(item, pd.DataFrame):
            all_col_names.extend(item.columns)

    duplicate_col_names = [col_name for col_name, count in Counter(all_col_names).items() if count > 1]

    # If no duplicates, concat without renaming
    if len(duplicate_col_names) == 0:
        concat_res = pd.concat(list_of_frames_or_series, axis=1)
        new_names = [s.name for s in list_of_series]

    # Rename Series with duplicate name
    else:
        list_of_renamed_series = []
        new_names = []
        for series in list_of_series:
            if series.name in duplicate_col_names or series.name is None:
                orig_name = str(series.name)
                new_name = add_random_suffix(orig_name)
                while new_name in all_col_names:
                    new_name = add_random_suffix(orig_name)
                series_copy = series.copy()
                series_copy.name = new_name

                list_of_renamed_series.append(series_copy)
                new_names.append(new_name)
            else:
                list_of_renamed_series.append(series)
                new_names.append(series.name)

        concat_res = pd.concat([df] + list_of_renamed_series, axis=1)

    if return_series_names:
        return concat_res, new_names
    else:
        return concat_res
