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

import numpy as np
from collections import Counter


def KFoldStratifiedMultiClass(labels, n, seed=0):
    """Returns a stratified n-fold in a dictionary, trying to distribute classes across folds as well as possible."""
    np.random.seed(seed)
    freqs = np.sum(labels, axis=0)  # class frequencies
    tups = zip(freqs, range(19))
    tups = sorted(tups)
    freq_order = map(lambda i: i[1], tups)  # classes sorted by frequency
    folds = {}

    # folds is a dictionary indexed by integer, which contains
    # the indexes of the samples in each fold
    for i in range(n):
        folds[i] = []

    # visited is a set to keep the samples already sorted into folds
    visited = set()
    for j in freq_order:
        occs = np.where(labels[:, j] == 1)[0]  # find the occurrences of the class
        for i, k in enumerate(occs):
            # for each occurrence check if it's been sorted
            if k not in visited:
                # and in case not append it to a given fold
                folds[i % n].append(k)
        # after sorting each class update visited set
        for i in folds.keys():
            visited = visited.union(set(folds[i]))

    # sort remaining examples into folds, trying to keep all folds
    # equally populated.
    for i in range(labels.shape[0]):
        if i not in visited:
            f = np.argmin(map(lambda i: len(folds[i]), range(n)))
            folds[f].append(i)

    return folds


def getMeanandStd(X):
    """Takes the features and computes the mean and std dev of each column."""
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return np.array([mu]).astype('float64'), np.array([std]).astype('float64')


def genSplits(T, Y, k=5):
    """Returns k folds indices of the input data."""
    star = ((T * 10) + (Y ** 1)).tolist()
    star = np.vstack((T, Y)).T
    starsum = star.sum(axis=1)
    starsum = ((T * 10) + (Y ** 1))

    Counter(starsum)

    splits = KFoldStratifiedMultiClass(star, k)
    # for split in splits:
    # print(split,len(splits[split]), T[splits[split]].sum(), Y[splits[split]].sum(), Counter(starsum[splits[split]]))

    overlap = set(range(len(T)))
    for split in splits:
        overlap &= set(splits[split])
    # print("Total Overlap:", len(overlap))

    return splits


def returnIndices(splits):
    """Takes the splits and return the train and dev splits."""
    test = []
    train = []
    dev = []

    for split in splits.keys():
        if split in [0]:
            dev += splits[split]
        else:
            train += splits[split]

    return train, dev


def returnIndicesTest(splits):
    """Takes the splits and return the train, test and dev splits."""
    test = []
    train = []
    dev = []

    for split in splits.keys():
        if split in [0, 1]:
            test += splits[split]
        if split in [2]:
            dev += splits[split]
        else:
            train += splits[split]

    return train, test, dev
