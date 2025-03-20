from __future__ import annotations
import pandas as pd
from causallib.preprocessing.transformers import MatchingTransformer
from causallib.positivity import BasePositivity
from typing import Union, Optional
import sklearn.neighbors
import sklearn.base


class Matching(BasePositivity):

    def __init__(
        self,
        propensity_transform: Optional[sklearn.base.TransformerMixin] = None,
        caliper: Optional[float] = None,
        with_replacement: bool = True,
        n_neighbors: int = 1,
        matching_mode: str = "both",
        metric: str = "mahalanobis",
        knn_backend: Union[str,
                           sklearn.neighbors.NearestNeighbors] = "sklearn",
    ):
        """Fix positivity by matching.

        Args:
            propensity_transform (sklearn.TransformerMixin): an object for data
                preprocessing which implements `fit` and `transform` 
                (default: None)
            caliper (float) : maximal distance for a match to be accepted. If
                not defined, all matches will be accepted. If defined, some
                samples may not be matched and their outcomes will not be
                estimated. (default: None)
            with_replacement (bool): whether samples can be used multiple times
                for matching. If set to False, the matching process will optimize
                the linear sum of distances between pairs of treatment and
                control samples and only `min(N_treatment, N_control)` samples
                will be estimated. Matching with no replacement does not make
                use of the `fit` data and is therefore not implemented for
                out-of-sample data (default: True)
            n_neighbors (int) : number of nearest neighbors to include in match.
                Must be 1 if `with_replacement` is `False.` If larger than 1, the
                estimate is calculated using the `regress_agg_function` or 
                `classify_agg_function` across the `n_neighbors`. Note that when
                the `caliper` variable is set, some samples will have fewer than
                `n_neighbors` matches. (default: 1).
            matching_mode (str) : Direction of matching: `treatment_to_control`,
                `control_to_treatment` or `both` to indicate which set should
                be matched to which. All sets are cross-matched in `match`
                and when `with_replacement` is `False` all matching modes 
                coincide. With replacement there is a difference.
            metric (str) : Distance metric string for calculating distance
                between samples. Note: if an external built `knn_backend`
                object with a different metric is supplied, `metric` needs to
                be changed to reflect that, because `Matching` will set its 
                inverse covariance matrix if "mahalanobis" is set. (default: 
                "mahalanobis", also supported: "euclidean")
            knn_backend (str or callable) : Backend to use for nearest neighbor
                search. Options are "sklearn"  or a callable  which returns an 
                object implementing `fit`, `kneighbors` and `set_params` 
                like the sklearn `NearestNeighbors` object. (default: "sklearn"). 

        """
        self.matching_transformer = MatchingTransformer(
            propensity_transform=propensity_transform,
            caliper=caliper,
            with_replacement=with_replacement,
            n_neighbors=n_neighbors,
            matching_mode=matching_mode,
            metric=metric,
            knn_backend=knn_backend,
        )

    def fit(self, X: pd.DataFrame, a: pd.Series) -> Matching:
        """Fit matching positivity checker.

        Args:
            X (pd.DataFrame): samples
            a (pd.Series): treatment assignment
        """
        self.matching_transformer.fit(X, a, pd.Series())
        return self

    def predict(self, X: pd.DataFrame, a: pd.Series) -> pd.Series:
        """Predict whether or not a sample is in the overlap region.

        Find samples of treatment and control that successfully match and 
        return a boolean indexer which is `True` if they matched and `False` if
        they did not. This function calls the `match` method of the underlying
        `Matching` object.

        Args:
            X (pd.DataFrame): samples
            a (pd.Series): treatment assignment

        Returns:
            pd.Series: a Series of length `X.shape[0]` with the same index as
               `X` and only boolean values
        """
        self.matching_transformer.matching.match(X, a)
        matching_indices = self.matching_transformer.find_indices_of_matched_samples(
            X, a)
        return matching_indices
