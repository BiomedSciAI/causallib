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

Created on Jul 03, 2017

"""
from __future__ import division

import unittest
from builtins import range, zip
from collections import namedtuple

import numpy as np
import pandas as pd

import causallib.simulation.CausalSimulator3 as CS3m  # CS3 module    to test helper functions
from causallib.simulation.CausalSimulator3 import CausalSimulator3 as CS3


class CS3TestCase(unittest.TestCase):
    def setUp(self):
        # DEFAULTS:
        Params = namedtuple("Params", ["topology", "var_types", "outcome_types", "link_types", "prob_cat", "snr",
                                       "treatment_importance", "effect_sizes"])

        topology = np.zeros((4, 4), dtype=np.bool_)  # topology[i,j] if node j is a parent of node i
        topology[1, 0] = topology[2, 0] = topology[2, 1] = topology[3, 1] = topology[3, 2] = True
        self.no_X = Params(topology=topology, var_types=["hidden", "covariate", "treatment", "outcome"],
                           outcome_types=["continuous"], link_types=['linear'] * 4,
                           prob_cat=[None, None, [0.5, 0.5], None],
                           snr=0.5, treatment_importance=0.8, effect_sizes=None)
        self.NUM_SAMPLES = 2000
        self.X_NUM_SAMPLES = 2500

        self.X_GIVEN = pd.DataFrame(np.random.RandomState(0).normal(size=(self.X_NUM_SAMPLES, 5)),
                                    columns=["x{}".format(i) for i in range(5)])
        topology = np.zeros((5 + 4, 5 + 4), dtype=np.bool_)  # topology[i,j] if node j is a parent of node i
        topology[1, 0] = topology[2, 0] = topology[2, 1] = topology[3, 1] = topology[3, 2] = True
        topology[1, 4] = topology[2, 4] = topology[2, 6] = topology[3, 6] = topology[3, 7] = True
        var_names = pd.Index(list(range(4))).append(self.X_GIVEN.columns)
        self.with_X = Params(topology=topology,
                             var_types=pd.Series(data=["hidden", "covariate", "treatment", "outcome"] +
                                                      ["hidden"] + ["covariate"] * 4, index=var_names),
                             outcome_types=["continuous"], link_types=pd.Series(data=['linear'] * 9, index=var_names),
                             prob_cat=pd.Series([None, None, [0.5, 0.5], None] + [None] * 5, index=var_names),
                             snr=0.5, treatment_importance=0.8, effect_sizes=None)

    def dependency_from_topology(self, sim):
        """
        Tests to see that the matrix topology is well converted into graph dependencies.
        """
        for i, row in enumerate(sim.topology):
            i = sim.var_names[i]
            parents_mat_idx = set(sim.var_names[np.where(row)[0]])
            parents_grp_idx = set(sim.graph_topology.predecessors(i))
            self.assertEqual(parents_mat_idx, parents_grp_idx,
                             msg="variable {i} had these predecessors in matrix topology: {mat} "
                                 "and these predecessors in graph topology: {grp}".format(i=i, mat=parents_mat_idx,
                                                                                          grp=parents_grp_idx))

    def test_bad_input(self):
        # lengths:
        # with self.assertRaises(ValueError) as assert_checker:
        var_types = ["covariate", "treatment", "outcome"]
        self.assertRaises(ValueError, CS3, topology=self.no_X.topology, var_types=var_types,
                          prob_categories=self.no_X.prob_cat, link_types=self.no_X.link_types,
                          treatment_importances=self.no_X.treatment_importance, snr=self.no_X.snr,
                          outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes)

        # outcome has more than one treatment predecessor:
        var_types = ["covariate", "treatment", "treatment", "outcome"]
        self.assertRaises(ValueError, CS3, topology=self.no_X.topology, var_types=var_types,
                          prob_categories=self.no_X.prob_cat, link_types=self.no_X.link_types,
                          treatment_importances=self.no_X.treatment_importance, snr=self.no_X.snr,
                          outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes)

        # No valid link type:
        self.assertRaises(ValueError, CS3, topology=self.no_X.topology, var_types=self.no_X.var_types,
                          prob_categories=self.no_X.prob_cat,
                          link_types=["linear", "linear", "linear", "leniar"],
                          treatment_importances=self.no_X.treatment_importance, snr=self.no_X.snr,
                          outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes)

        # No valid treatment method:
        self.assertRaises(ValueError, CS3, topology=self.no_X.topology, var_types=self.no_X.var_types,
                          prob_categories=self.no_X.prob_cat, link_types=self.no_X.link_types,
                          treatment_importances=self.no_X.treatment_importance, snr=self.no_X.snr,
                          outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes,
                          treatment_methods="rndom")

        # lengths:
        self.assertRaises(ValueError, CS3, topology=self.no_X.topology, var_types=self.no_X.var_types,
                          prob_categories=self.no_X.prob_cat, link_types=self.no_X.link_types,
                          treatment_importances=self.no_X.treatment_importance, snr=[0, 1],
                          outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes,
                          treatment_methods="gaussian")
        self.assertRaises(ValueError, CS3, topology=self.no_X.topology, var_types=self.no_X.var_types,
                          prob_categories=self.no_X.prob_cat, link_types=self.no_X.link_types,
                          treatment_importances=[0.5, 0.5],
                          snr=self.no_X.snr, treatment_methods="gaussian",
                          outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes)

        # no generation input:
        sim = CS3(topology=self.no_X.topology, var_types=self.no_X.var_types, prob_categories=self.no_X.prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  snr=self.no_X.snr, outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes)
        self.assertRaises(ValueError, sim.generate_data)

        # categorical treatment:
        self.assertRaises(ValueError, sim.generate_treatment_col,
                          X_parents=pd.DataFrame([None]), link_type=None, snr=1, prob_category=None)

        # wrong probabilities:
        prob_cat = [[0.5, -0.5, 1], None, [0.5, 0.5], None]
        sim = CS3(topology=self.no_X.topology, var_types=self.no_X.var_types, prob_categories=prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  snr=self.no_X.snr, outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes)
        self.assertRaises(ValueError, sim.generate_data, num_samples=100)
        prob_cat = [None, None, [0.5, 0.6], None]
        sim = CS3(topology=self.no_X.topology, var_types=self.no_X.var_types, prob_categories=prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  snr=self.no_X.snr, outcome_types=self.no_X.outcome_types, effect_sizes=self.no_X.effect_sizes)
        self.assertRaises(ValueError, sim.generate_data, num_samples=100)

    def test_different_types_of_paramaters(self):
        """
        Tests to see what happens when supplying parameters of different types (lists, dicts, arrays, etc.)
        """
        topology = np.zeros((5, 5), dtype=np.bool_)  # topology[i,j] if node j is a parent of node i
        topology[1, 0] = topology[2, 0] = topology[2, 1] = topology[3, 1] = topology[3, 2] = topology[3, 4] = True
        var_types = ["hidden", "covariate", "treatment", "outcome", "covariate"]
        sim = CS3(topology=topology, var_types=var_types, prob_categories=self.no_X.prob_cat + [None],
                  link_types=None, treatment_importances=pd.Series(data=0.7, index=[2]),
                  outcome_types={3: "continuous"}, snr=0.5,
                  # effect_sizes={2: 0.8},
                  effect_sizes=[0.8],
                  treatment_methods=["gaussian"])
        self.assertTrue(all(sim.link_types == "linear"))
        self.assertEqual(len(sim.link_types), 5)
        self.assertTrue(all([x == 0.7 for x in sim.treatment_importances]))
        self.assertTrue(sim.outcome_types.equals(pd.Series({3: "continuous"})))
        self.assertTrue(all(sim.snr == 0.5))
        self.assertEqual(len(sim.snr), 5)
        self.assertTrue(sim.effect_sizes.equals(pd.Series({3: 0.8})))

    # ### Test to see that structure make sense ### #
    def test_dependency_from_topology(self):
        """
        Tests to see that the matrix topology is well converted into graph dependencies for with and without dataset.
        """
        sim = CS3(topology=self.no_X.topology, var_types=self.no_X.var_types, prob_categories=self.no_X.prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=self.no_X.snr, effect_sizes=self.no_X.effect_sizes)
        self.dependency_from_topology(sim)
        sim = CS3(topology=self.with_X.topology, var_types=self.with_X.var_types,
                  prob_categories=self.with_X.prob_cat, link_types=self.with_X.link_types,
                  treatment_importances=self.with_X.treatment_importance, outcome_types=self.with_X.outcome_types,
                  snr=self.with_X.snr, effect_sizes=self.with_X.effect_sizes)
        self.dependency_from_topology(sim)

    def test_dataset_size(self):
        """
        Tests to see the the size of the generated dataset is ok under several configurations
        """
        # No given X, all non-hidden
        var_types = ["covariate", "covariate", "treatment", "outcome"]
        sim = CS3(topology=self.no_X.topology, var_types=var_types, prob_categories=self.no_X.prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=self.no_X.snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)
        self.assertEqual(X.shape, (self.NUM_SAMPLES, 4),
                         msg="Generated dataset shape is {X} "
                             "but supposed to be {supp}".format(X=X.shape, supp=(self.NUM_SAMPLES, 4)))
        self.assertEqual(prop.shape, (self.NUM_SAMPLES, 2),
                         msg="Generated propensity shape is {X} "
                             "but supposed to be {supp}".format(X=prop.shape, supp=(self.NUM_SAMPLES, 2)))
        self.assertEqual(cf.shape, (self.NUM_SAMPLES, 2),
                         msg="number of generated counterfactuals is {X} "
                             "but supposed to be {supp}".format(X=cf.shape, supp=(self.NUM_SAMPLES, 2)))
        df_obs, df_cf = sim.format_for_training(X, prop, cf)
        self.assertEqual(df_obs.shape, (self.NUM_SAMPLES, 4),
                         msg="Generated dataset shape is {X} "
                             "but supposed to be {supp}".format(X=df_obs.shape, supp=(self.NUM_SAMPLES, 4)))

        # No given X, with hidden
        var_types = ["hidden", "hidden", "treatment", "outcome"]
        sim = CS3(topology=self.no_X.topology, var_types=var_types, prob_categories=self.no_X.prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=self.no_X.snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)
        df_obs, df_cf = sim.format_for_training(X, prop, cf)
        self.assertEqual(df_obs.shape, (self.NUM_SAMPLES, 2),
                         msg="Generated dataset shape is {X} "
                             "but supposed to be {supp}".format(X=df_obs.shape, supp=(self.NUM_SAMPLES, 2)))

        # Given X, with hidden vars
        sim = CS3(topology=self.with_X.topology, var_types=self.with_X.var_types,
                  prob_categories=self.with_X.prob_cat, link_types=self.with_X.link_types,
                  treatment_importances=self.with_X.treatment_importance, outcome_types=self.with_X.outcome_types,
                  snr=self.with_X.snr, effect_sizes=self.with_X.effect_sizes)
        X, prop, cf = sim.generate_data(X_given=self.X_GIVEN)
        self.assertEqual(X.shape, (self.X_NUM_SAMPLES, 9),
                         msg="Generated dataset shape is {X} "
                             "but supposed to be {supp}".format(X=X.shape, supp=(self.X_NUM_SAMPLES, 9)))
        self.assertEqual(prop.shape, (self.X_NUM_SAMPLES, 2),
                         msg="Generated propensity shape is {X} "
                             "but supposed to be {supp}".format(X=prop.shape, supp=(self.X_NUM_SAMPLES, 2)))
        self.assertEqual(cf.shape, (self.X_NUM_SAMPLES, 2),
                         msg="Number of counterfactuals generated is {X} "
                             "but supposed to be {supp}".format(X=cf.shape, supp=(self.X_NUM_SAMPLES, 2)))
        df_obs, df_cf = sim.format_for_training(X, prop, cf)
        self.assertEqual(df_obs.shape, (self.X_NUM_SAMPLES, 7),
                         msg="Generated dataset shape is {X} "
                             "but supposed to be {supp}".format(X=df_obs.shape, supp=(self.X_NUM_SAMPLES, 7)))

    def test_multi_treatment_outcome(self):
        topology = np.zeros((6, 6), dtype=bool)
        topology[2, 0] = topology[3, 0] = topology[2, 1] = topology[3, 1] = topology[4, 2] = topology[5, 3] = True
        var_types = ["covariate", "covariate", "treatment", "treatment", "outcome", "outcome"]
        link_types = ["linear"] * 6
        prob_cat = [None] * 6
        prob_cat[2] = prob_cat[3] = [0.5, 0.5]
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=self.no_X.snr, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=["continuous", "continuous"], effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)
        self.assertEqual(prop.shape, (self.NUM_SAMPLES, 4),
                         msg="Generated propensity shape is {X} "
                             "but supposed to be {supp}".format(X=prop.shape, supp=(self.NUM_SAMPLES, 4)))
        self.assertEqual(cf.shape, (self.NUM_SAMPLES, 4),
                         msg="Number of generated counterfactuals is {X} "
                             "but supposed to be {supp}".format(X=cf.shape, supp=(self.NUM_SAMPLES, 4)))

    def test_multi_categorical_treatment(self):
        t_probs = pd.Series([0.2, 0.2, 0.1, 0.5])
        prob_cat = [None, None, t_probs, None]
        treatment_methods = ["quantile_gauss_fit", "odds_ratio"]
        decimals = [1, 1]
        for treatment_method, decimal in zip(treatment_methods, decimals):
            sim = CS3(topology=self.no_X.topology, var_types=self.no_X.var_types, prob_categories=prob_cat,
                      link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                      outcome_types=self.no_X.outcome_types, snr=self.no_X.snr, effect_sizes=self.no_X.effect_sizes,
                      treatment_methods=treatment_method)
            n = self.NUM_SAMPLES * 50
            X, prop, cf = sim.generate_data(num_samples=n)
            np.testing.assert_array_almost_equal(prop.sum(axis="columns"), np.ones(n),
                                                 err_msg="multi-categorical preopensities of treatment method {method} "
                                                         "does not sum to 1".format(method=treatment_method))
            np.testing.assert_array_almost_equal(np.array(X[2].value_counts(normalize=True) - t_probs), np.zeros(4),
                                                 decimal=decimal,
                                                 err_msg="treatment method {method} does not produce proportions as "
                                                         "required".format(method=treatment_method))

    # ### Test helpers ### #
    def test_random_topology_generation(self):
        # ### without given variables: ### #
        T, var_types = CS3m.generate_random_topology(n_covariates=4, p=0.4, n_treatments=2, n_outcomes=2,
                                                     n_censoring=0, given_vars=[], p_hidden=0)
        # test output structure:
        self.assertEqual(T.shape[0], T.shape[1], msg="Graph has no square shape")
        self.assertEqual(T.shape[0], 8,
                         msg="Number of Graph variables {emp} "
                             "does not match it supposed number {sup}".format(emp=T.shape[0], sup=8))
        self.assertEqual(T.shape[0], var_types.size)
        # test number of variables of each type matches:
        self.assertEqual(sum(var_types == "covariate"), 4)
        self.assertEqual(sum(var_types == "treatment"), 2)
        self.assertEqual(sum(var_types == "outcome"), 2)
        self.assertEqual(sum(var_types == "hidden"), 0)
        self.assertEqual(sum(var_types == "censor"), 0)
        # test that each treatment is coupled with one outcome:
        self.assertEqual(all(T.loc[var_types == "outcome", var_types == "treatment"].sum(axis=1) == np.array([1, 1])),
                         True, msg="each outcome variable does not have exactly one predecessor treatment variable")

        # ### with hidden variables and censor variables: ### #
        T, var_types = CS3m.generate_random_topology(n_covariates=100, p=0.4, n_treatments=2, n_outcomes=2,
                                                     n_censoring=2, given_vars=[], p_hidden=0.4)
        # test output structure:
        self.assertEqual(T.shape[0], T.shape[1], msg="Graph has no square shape")
        self.assertEqual(T.shape[0], 106,
                         msg="Number of Graph variables {t} does not match it supposed number {s}".format(t=T.shape[0],
                                                                                                          s=106))
        self.assertEqual(T.shape[0], var_types.size)
        # test number of variables of each type matches:
        self.assertEqual(sum(var_types == "censor"), 2)
        hist = var_types.value_counts()
        self.assertAlmostEqual(hist["hidden"] / 100.0, 0.4, delta=1e-2)

        # graph = nx.from_numpy_matrix(T.values.transpose(), create_using=nx.DiGraph())
        # ### with given variables: ### #
        X = pd.DataFrame(np.random.RandomState(0).normal(size=(4800, 5)))
        T, var_types = CS3m.generate_random_topology(n_covariates=4, p=0.4, n_treatments=2, n_outcomes=2,
                                                     n_censoring=0, given_vars=X.columns, p_hidden=0)
        self.assertEqual(sum(var_types == "covariate"), 9)
        # test that given variable has no predecessors:
        np.testing.assert_array_equal(T.loc[X.columns, :].sum(axis="columns"), np.zeros(5))

        # Test for DAGness:
        from networkx import DiGraph, from_numpy_array, is_directed_acyclic_graph
        NUM_TESTS = 50
        for test in range(NUM_TESTS):
            n_cov = np.random.randint(low=10, high=100)
            p = np.random.rand()  # type: float
            n_tre_out = np.random.randint(low=1, high=4)
            n_cen = np.random.randint(low=0, high=n_tre_out)
            T, _ = CS3m.generate_random_topology(n_covariates=n_cov, p=p, n_treatments=n_tre_out, n_outcomes=n_tre_out,
                                                 n_censoring=n_cen, given_vars=[], p_hidden=0)
            G = from_numpy_array(T.values.transpose(), create_using=DiGraph())
            res = is_directed_acyclic_graph(G)
            self.assertTrue(res)

    # ### Test to see that values make sense ### #
    def test_categorical_proportions(self):
        probs = np.array([0.25, 0.25, 0.5])
        prob_cat = self.no_X.prob_cat
        prob_cat[1] = probs
        sim = CS3(topology=self.no_X.topology, var_types=self.no_X.var_types, prob_categories=prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=self.no_X.snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES * 10, random_seed=0)
        # hist = np.array(X.loc[:, 1].value_counts(normalize=True))
        hist = X.loc[:, 1].value_counts(normalize=True)
        probs = pd.Series(probs)
        np.testing.assert_array_almost_equal(probs - hist, pd.Series(data=0, index=probs.index), decimal=2,
                                             err_msg="Empirical distribution {emp} of categories "
                                                     "is too far from desired distribution {des}".format(emp=hist,
                                                                                                         des=probs))

    def test_linear_linking(self):
        topology = np.zeros((3, 3), dtype=bool)
        topology[2, 0] = topology[2, 1] = True
        var_types = ["covariate", "treatment", "outcome"]
        snr = 1
        prob_cat = [None, [0.5, 0.5], None]
        treatment_importance = None
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat,
                  link_types="linear", treatment_importances=treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)

        singular_values = np.linalg.svd(X.astype(float).values, compute_uv=False)
        eps = 1e-10
        rank = np.sum(singular_values > eps)
        self.assertEqual(rank, 2,
                         msg="discovered rank of matrix is {emp} instead of {des}."
                             "so the linear linking does not work properly".format(emp=rank, des=2))

    def test_affine_linking(self):
        topology = np.zeros((3, 3), dtype=bool)
        topology[2, 0] = topology[2, 1] = True
        var_types = ["covariate", "treatment", "outcome"]
        snr = 1
        prob_cat = [None, [0.5, 0.5], None]
        treatment_importance = None
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat,
                  link_types="affine", treatment_importances=treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)

        singular_values = np.linalg.svd(X.astype(float).values, compute_uv=False)
        eps = 1e-10
        rank = np.sum(singular_values > eps)
        self.assertEqual(rank, 3,
                         msg="discovered rank of matrix is {emp} instead of {des}."
                             "so the affine linking does not work properly".format(emp=rank, des=3))

    def test_poly_linking(self):
        topology = np.zeros((3, 3), dtype=bool)
        topology[2, 0] = topology[2, 1] = True
        var_types = ["covariate", "treatment", "outcome"]
        snr = 1
        prob_cat = [None, [0.5, 0.5], None]
        treatment_importance = None
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat,
                  link_types="poly", treatment_importances=treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)

        singular_values = np.linalg.svd(X.astype(float).values, compute_uv=False)
        eps = 1e-10
        rank = np.sum(singular_values > eps)
        self.assertEqual(rank, 3,
                         msg="discovered rank of matrix is {emp} instead of {des}."
                             "so the poly linking does not work properly".format(emp=rank, des=3))

    def test_exp_linking(self):
        topology = np.zeros((3, 3), dtype=bool)
        topology[2, 0] = topology[2, 1] = True
        var_types = ["covariate", "treatment", "outcome"]
        snr = 1
        prob_cat = [None, [0.5, 0.5], None]
        treatment_importance = None
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat,
                  link_types="exp", treatment_importances=treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)

        singular_values = np.linalg.svd(X.astype(float).values, compute_uv=False)
        eps = 1e-10
        rank = np.sum(singular_values > eps)
        self.assertEqual(rank, 3,
                         msg="discovered rank of matrix is {emp} instead of {des}."
                             "so the exp linking does not work properly".format(emp=rank, des=3))

    def test_log_linking(self):
        topology = np.zeros((3, 3), dtype=bool)
        topology[2, 0] = topology[2, 1] = True
        var_types = ["covariate", "treatment", "outcome"]
        snr = 1
        prob_cat = [None, [0.5, 0.5], None]
        treatment_importance = None
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat,
                  link_types="log", treatment_importances=treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)

        singular_values = np.linalg.svd(X.astype(float).values, compute_uv=False)
        eps = 1e-10
        rank = np.sum(singular_values > eps)
        self.assertEqual(rank, 3,
                         msg="discovered rank of matrix is {emp} instead of {des}."
                             "so the log linking does not work properly".format(emp=rank, des=3))

    def test_treatment_logistic(self):
        topology = np.zeros((6, 6), dtype=bool)
        topology[2, 0] = topology[3, 0] = topology[2, 1] = topology[3, 1] = topology[4, 2] = topology[5, 3] = True
        var_types = ["covariate", "covariate", "treatment", "treatment", "outcome", "outcome"]
        link_types = ["linear"] * 6
        prob_cat = [None] * 6
        prob_cat[2] = prob_cat[3] = [0.5, 0.5]
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=self.no_X.snr, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=["continuous", "continuous"], effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=["logistic", "logistic"])
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)
        # TODO: how to check logistic? (using logit?)

    def test_treatment_random(self):
        topology = np.zeros((6, 6), dtype=bool)
        topology[2, 0] = topology[3, 0] = topology[2, 1] = topology[3, 1] = topology[4, 2] = topology[5, 3] = True
        var_types = ["covariate", "covariate", "treatment", "treatment", "outcome", "outcome"]
        link_types = ["linear"] * 6
        prob_cat = [None] * 6
        prob_cat[2] = [0.5, 0.5]
        prob_cat[3] = [0.2, 0.8]
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=self.no_X.snr, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=["continuous", "continuous"], effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=["random", "random"])
        num_samples = self.NUM_SAMPLES * 10
        X, prop, cf = sim.generate_data(num_samples=num_samples)
        np.testing.assert_array_equal(prop[2][1], [0.5] * num_samples)
        np.testing.assert_array_equal(prop[3][1], [0.8] * num_samples)
        hist = X[2].value_counts(normalize=True)
        np.testing.assert_almost_equal(hist, [0.5, 0.5], decimal=2)
        hist = X[3].value_counts(normalize=True)
        np.testing.assert_almost_equal(hist.sort_index(), [0.2, 0.8], decimal=2)

    def test_treatment_gaussian(self):
        topology = np.zeros((6, 6), dtype=bool)
        topology[2, 0] = topology[3, 0] = topology[2, 1] = topology[3, 1] = topology[4, 2] = topology[5, 3] = True
        var_types = ["covariate", "covariate", "treatment", "treatment", "outcome", "outcome"]
        link_types = ["linear"] * 6
        prob_cat = [None] * 6
        prob_cat[2] = [0.5, 0.5]
        prob_cat[3] = [0.2, 0.8]
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=self.no_X.snr, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=["continuous", "continuous"], effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=["random", "random"])
        num_samples = self.NUM_SAMPLES * 5
        X, prop, cf = sim.generate_data(num_samples=num_samples)
        np.testing.assert_array_equal(prop[2][1], [0.5] * num_samples)
        np.testing.assert_array_equal(prop[3][1], [0.8] * num_samples)
        hist = X[2].value_counts(normalize=True)
        np.testing.assert_almost_equal(hist, [0.5, 0.5], decimal=2)
        hist = X[3].value_counts(normalize=True)
        np.testing.assert_almost_equal(hist.sort_index(), [0.2, 0.8], decimal=2)

        sim = CS3(topology=self.no_X.topology, var_types=self.no_X.var_types, prob_categories=self.no_X.prob_cat,
                  link_types=self.no_X.link_types, treatment_importances=self.no_X.treatment_importance,
                  outcome_types=self.no_X.outcome_types, snr=self.no_X.snr, effect_sizes=self.no_X.effect_sizes)
        X, prop, cf = sim.generate_data(num_samples=num_samples, random_seed=0)
        hist = X.loc[:, 2].value_counts(normalize=True)  # treatment
        np.testing.assert_almost_equal(hist, [0.5, 0.5], decimal=2)

    def test_survival_outcome(self):
        topology = np.zeros((5, 5), dtype=bool)
        topology[3, 0] = topology[4, 0] = topology[3, 1] = topology[3, 2] = topology[4, 2] = topology[4, 3] = True
        var_types = ["covariate", "covariate", "hidden", "treatment", "outcome"]
        link_types = ["linear"] * 5
        prob_cat = [None] * 5
        prob_cat[3] = [0.2, 0.8]
        outcome_type = "survival"
        snr = 0.95
        treatment_importance = None
        treatment_method = "logistic"
        survival_distribution = "expon"
        survival_baseline = 0.8
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=snr, treatment_importances=treatment_importance,
                  outcome_types=outcome_type, effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=treatment_method,
                  survival_distribution=survival_distribution, survival_baseline=survival_baseline)
        num_samples = self.NUM_SAMPLES
        X, prop, cf = sim.generate_data(num_samples=num_samples)
        # TODO: how to test this?

    def test_censoring(self):
        # survival censor
        topology = np.zeros((5, 5), dtype=bool)
        topology[2, 0] = topology[3, 0] = topology[4, 0] = topology[2, 1] = topology[3, 1] = topology[4, 1] = topology[
            3, 2] = topology[4, 2] = topology[4, 3] = True  # make censor be dependent like the outcome
        var_types = ["covariate", "covariate", "treatment", "censor", "outcome"]
        link_types = ["linear"] * 5
        prob_cat = [None, None, [0.2, 0.8], [0.85, 0.15], None]
        outcome_type = "survival"
        snr = 0.95
        treatment_importance = None
        treatment_method = "logistic"
        survival_distribution = "expon"
        survival_baseline = 0.8
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=snr, treatment_importances=treatment_importance,
                  outcome_types=outcome_type, effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=treatment_method,
                  survival_distribution=survival_distribution, survival_baseline=survival_baseline)
        num_samples = self.NUM_SAMPLES
        X, prop, cf = sim.generate_data(num_samples=num_samples, random_seed=783454)
        self.assertAlmostEqual(np.abs(X[4].le(X[3]).sum() / num_samples), prob_cat[3][0], places=1)
        # df_obs, df_cf = sim.format_for_training(X, prop, cf)

        # binary censor
        topology = np.zeros((5, 5), dtype=bool)
        topology[2, 0] = topology[4, 0] = topology[2, 1] = topology[3, 1] = topology[4, 2] = topology[4, 3] = True
        var_types = ["covariate", "covariate", "treatment", "censor", "outcome"]
        link_types = ["linear"] * 5
        prob_cat = [None, None, [0.2, 0.8], [0.85, 0.15], None]
        outcome_type = "continuous"
        snr = 0.95
        treatment_importance = None
        treatment_method = "logistic"
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=snr, treatment_importances=treatment_importance,
                  outcome_types=outcome_type, effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=treatment_method)
        num_samples = self.NUM_SAMPLES
        X, prop, cf = sim.generate_data(num_samples=num_samples)
        self.assertAlmostEqual(X[3].astype(int).sum() / num_samples, prob_cat[3][1])
        df_obs, df_cf = sim.format_for_training(X, prop, cf)
        self.assertEqual(X[3].astype(int).sum(), df_obs["y_4"].isnull().sum())

        # independent categorical censor
        topology = np.zeros((5, 5), dtype=bool)
        topology[3, 0] = topology[4, 0] = topology[3, 2] = topology[4, 2] = topology[4, 3] = True
        var_types = ["covariate", "covariate", "treatment", "censor", "outcome"]
        link_types = ["linear"] * 5
        prob_cat = [None, None, [0.2, 0.8], [0.85, 0.15], None]
        outcome_type = "continuous"
        snr = 0.95
        treatment_importance = None
        treatment_method = "logistic"
        survival_distribution = "expon"
        survival_baseline = 0.8
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=snr, treatment_importances=treatment_importance,
                  outcome_types=outcome_type, effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=treatment_method,
                  survival_distribution=survival_distribution, survival_baseline=survival_baseline)
        num_samples = self.NUM_SAMPLES
        X, prop, cf = sim.generate_data(num_samples=num_samples)
        self.assertAlmostEqual(X[3].astype(int).sum() / num_samples, prob_cat[3][1])
        # df_obs, df_cf = sim.format_for_training(X, prop, cf)

        # independent survival censor
        topology = np.zeros((5, 5), dtype=bool)
        topology[3, 0] = topology[4, 0] = topology[3, 2] = topology[4, 2] = topology[4, 3] = True
        var_types = ["covariate", "covariate", "treatment", "censor", "outcome"]
        link_types = ["linear"] * 5
        prob_cat = [None, None, [0.2, 0.8], [0.85, 0.15], None]
        outcome_type = "survival"
        snr = 0.95
        treatment_importance = None
        treatment_method = "logistic"
        survival_distribution = "expon"
        survival_baseline = 0.8
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat, link_types=link_types,
                  snr=snr, treatment_importances=treatment_importance,
                  outcome_types=outcome_type, effect_sizes=self.no_X.effect_sizes,
                  treatment_methods=treatment_method,
                  survival_distribution=survival_distribution, survival_baseline=survival_baseline)
        num_samples = 10000
        X, prop, cf = sim.generate_data(num_samples=num_samples)
        # self.assertAlmostEqual(X[4].le(X[3]).sum() / float(num_samples), prob_cat[3][0], places=1)
        # df_obs, df_cf = sim.format_for_training(X, prop, cf)

        # TODO: test different link types
        # TODO: test marginal structural model (both in continuous, dichotomous and probability settings)

    def test_effect_modifier(self):
        topology = np.zeros((4, 4), dtype=bool)
        topology[2, 0] = topology[2, 1] = topology[2, 3] = True
        var_types = ["effect_modifier", "treatment", "outcome", "covariate"]
        snr = 1
        prob_cat = [None, [0.5, 0.5], None, None]
        treatment_importance = None
        sim = CS3(topology=topology, var_types=var_types, prob_categories=prob_cat,
                  link_types=["linear","linear","marginal_structural_model","linear"], treatment_importances=treatment_importance,
                  outcome_types="continuous", snr=snr, effect_sizes=None)
        X, prop, cf = sim.generate_data(num_samples=self.NUM_SAMPLES)
        
        beta = sim.linking_coefs
        self.assertNotEqual(beta[2].loc[0,0], beta[2].loc[0,1],
                         msg="coefficients for potential outcomes are the same: {beta_1} = {beta_0}."
                             "so the effect modifier does not behave properly".format(beta_0=beta[2].loc[0,0], beta_1=beta[2].loc[0,1]))
        self.assertEqual(beta[2].loc[3,0], beta[2].loc[3,1],
                         msg="coefficients for potential outcomes are not the same: {beta_1} != {beta_0}."
                             "so the covariate does not behave properly".format(beta_0=beta[2].loc[0,0], beta_1=beta[2].loc[0,1]))


if __name__ == "__main__":
    unittest.main()
