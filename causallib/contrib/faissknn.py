# (C) Copyright 2021 IBM Corp.
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

import numpy as np
import faiss


class FaissNearestNeighbors:

    def __init__(self,
                 metric="mahalanobis",
                 index_type="flatl2", n_cells=100, n_probes=10):
        """NearestNeighbors object utilizing the faiss library for speed

        Implements the same API as sklearn but runs 5-10x faster. Utilizes the 
        `faiss` library https://github.com/facebookresearch/faiss . Tested with 
        version 1.7.0. If `faiss-gpu` is installed from pypi, GPU acceleration
        will be used if available. 

        Args:
            metric (str) :  Distance metric for finding nearest neighbors
                (default: "mahalanobis")
            index_type (str) : Index type within faiss to use
                (supported: "flatl2" and "ivfflat")
            n_cells (int) : Number of voronoi cells (only used for "ivfflat",
                default: 100)
            n_probes (int) : Number of voronoi cells to search in
                (only used for "ivfflat", default: 10)
        Attributes (after running `fit`):
            index_ : the faiss index fit from the data. For details about
            faiss indices, see the faiss documentation at 
            https://github.com/facebookresearch/faiss/wiki/Faiss-indexes .
        """
        self.metric = metric
        self.n_cells = n_cells
        self.n_probes = n_probes
        self.index_type = index_type

    def fit(self, X):
        """Create faiss index and train with data.

        Args:
            X (np.array): Array of N samples of shape (NxM)

        Returns:
            self: Fitted object
        """
        X = self._transform_covariates(X)
        if self.index_type == "flatl2":
            self.index_ = faiss.IndexFlatL2(X.shape[1])
            self.index_.add(X)
        elif self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatL2(X.shape[1])
            n_cells = max(1, min(self.n_cells, X.shape[0]//200))
            n_probes = min(self.n_probes, n_cells)
            self.index_ = faiss.IndexIVFFlat(
                quantizer, X.shape[1], n_cells)
            self.index_.train(X)
            self.index_.nprobe = n_probes
            self.index_.add(X)
        else:
            raise NotImplementedError(
                "Index type {} not implemented. Please select"
                "one of [\"flatl2\", \"ivfflat\"]".format(self.index_type))
        return self

    def kneighbors(self, X, n_neighbors=1):
        """Find the k nearest neighbors of each sample in X

        Args:
            X (np.array):  Array of shape (N,M) of samples to search
                for neighbors of. M must be the same as the fit data.
            n_neighbors (int, optional): Number of neighbors to find.
                Defaults to 1.

        Returns:
            (distances, indices): Two np.array objects of shape (N,n_neighbors)
                containing the distances and indices of the closest neighbors.
        """
        X = self._transform_covariates(X)
        distances, indices = self.index_.search(X, n_neighbors)
        # faiss returns euclidean distance squared
        return np.sqrt(distances), indices

    def _transform_covariates(self, X):
        if self.metric == "mahalanobis":
            if not hasattr(self, "VI"):
                raise AttributeError("Set inverse covariance VI first.")
            X = np.dot(X, self.VI.T)
        return np.ascontiguousarray(X).astype("float32")

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == "metric_params":
                self.set_params(**value)
            else:
                self._setattr(parameter, value)
        return self

    def get_params(self, deep=True):
        # `deep` plays no role because there are no sublearners
        params_to_return = ["metric", "n_cells", "n_probes", "index_type"]
        return {i: self.__getattribute__(i) for i in params_to_return}

    def _setattr(self, parameter, value):
        # based on faiss docs https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
        if parameter == "VI":
            value = np.linalg.inv(value)
            chol = np.linalg.cholesky(value)
            cholvi = np.linalg.inv(chol)
            value = cholvi
        setattr(self, parameter, value)
