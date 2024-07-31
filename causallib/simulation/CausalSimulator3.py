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

Created on Jun 21, 2017

"""
from __future__ import division

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats

from ..utils.stat_utils import robust_lookup


# TODO: support categorical (non-numeric) data predecessors.

COVARIATE = "covariate"
HIDDEN = "hidden"
TREATMENT = "treatment"
OUTCOME = "outcome"
CENSOR = "censor"
EFFECT_MODIFIER = "effect_modifier"
VALID_VAR_TYPES = {COVARIATE, HIDDEN, TREATMENT, OUTCOME, CENSOR, EFFECT_MODIFIER}
CATEGORICAL = "categorical"
SURVIVAL = "survival"
CONTINUOUS = "continuous"
PROBABILITY = "probability"
DEFAULT_LINK_TYPE = "linear"

BASELINE_SURVIVAL_PARAM = 1.0


class CausalSimulator3(object):
    TREATMENT_METHODS = {"random": lambda x, p, snr, params: CausalSimulator3._treatment_random(x, p),
                         "odds_ratio": lambda x, p, snr, params: CausalSimulator3._treatment_odds_ratio(x, p, snr),
                         "quantile_gauss_fit": lambda x, p, snr, params: CausalSimulator3._treatment_quantile_gauss_fit(
                             x, p, snr),
                         "logistic": lambda x, p, snr, params: CausalSimulator3._treatment_logistic_dichotomous(x, p,
                                                                                                                params=params),
                         "gaussian": lambda x, p, snr, params: CausalSimulator3._treatment_gaussian_dichotomous(x, p,
                                                                                                                snr)}
    # G for general - applicable to all types of variables
    G_LINKING_METHODS = {"linear": lambda x, beta=None: CausalSimulator3._linear_link(x, beta),
                         "affine": lambda x, beta=None: CausalSimulator3._affine_link(x, beta),
                         "exp": lambda x, beta=None: CausalSimulator3._exp_linking(x, beta),
                         "log": lambda x, beta=None: CausalSimulator3._log_linking(x, beta),
                         "poly": lambda x, beta=None: CausalSimulator3._poly_linking(x, beta)}
    # O for outcome - outcome specific linking
    O_LINKING_METHODS = {
        "marginal_structural_model": lambda x, t, m, beta=None: CausalSimulator3._marginal_structural_model_link(
            x, t, m, beta=beta),
        None: lambda x, beta=None: x
    }

    def __init__(self, topology, var_types, prob_categories, link_types, snr, treatment_importances,
                 treatment_methods="gaussian", outcome_types=CATEGORICAL, effect_sizes=None,
                 survival_distribution="expon", survival_baseline=1, params=None):
        """
        Constructor

        Args:
            topology (np.ndarray): A boolean adjacency matrix for variables (including covariates, treatment and outcome
                                   variables of the model).
                                   Every row is a binary vector for a variable, where v[i, j] = 1 iff j is a parent of i
            var_types (Sequence[str]): Vector the size of variables stating every variable to be "covariate",
                                       "hidden", "outcome", "treatment", "censor".
                                       **Notes**: if type(pd.Series) variable names will be var_types.index, otherwise,
                                       if no-key-vector - var names will be just range(num-of-variables).
            prob_categories (Sequence[float|None]): vector the size of the number of variables.
                                             if prob_categories[i] = None -> than variable i is  considered continuous.
                                             otherwise -> prob_categories[i] should be a list (or any iterable) which
                                             size specifies number of categories variable i has, and it contains
                                             multinomial probabilities for those categories (i.e. list non negative and
                                             sums to 1).
            link_types (str|Sequence[str]): set of string the size or string or specifying the relation between
                                            covariate parents to the covariate itself
            snr (float|Sequence[float]): Signal to noise ratio (use 1.0 to eliminate noise in the system).
                                          May be a vector the size of number of variables for stating different snr
                                          values for different variables.
            treatment_importances (float|Sequence[float]): The effect of treatment on the outcome. A float between 0
                                                            and 1.0 stating how much weight the treatment variable have
                                                            vs. the other parents of an outcome variable.
                                                            *To support multi-treatment* - place a list the size of the
                                                            number of treatment variables (as stated in var_types).
                                                            The matching between treatment variable and its importance
                                                            will be according to the order of the treatment variables
                                                            and the order of the list. If all treatments variables has
                                                            the same importance - pass the float value.
            treatment_methods (str|Sequence[str]): method for creating treatment assignment and propensities, can be
                                                one of {"random", "gaussian", "logistic"}.
                                                *To support multi-treatment* - place a list the size of the number of
                                                treatment variables. The matching between treatment variable and its
                                                creation method will be according to the order of the treatment
                                                variables and the order of the list. If all treatment variables has the
                                                same type - pass the str value.
            outcome_types (str|Sequence[str]): outcome can either be 'survival' or 'binary'.
                                         *To support multi-outcome* - place a list the size of the number of outcome
                                         variables (as stated in var_types). The matching between outcome variable and
                                         its type will be according to the order of the outcome variables and the order
                                         of the list. If all outcome variables has the same type - pass the str value.
            effect_sizes (float|Sequence[float|None]|None): The wanted mean effect size between two counterfactuals.
                                                  If None - The mean effect size will not be adjusted, but will be
                                                  whatever generated.
                                                  If float - The mean effect size will be adjusted to be approximately
                                                  the given number (considering the noise)
                                                  *To support multi-outcome* - a list the size the number of the outcome
                                                  variables (as stated in var_types). The matching between outcome
                                                  variable and its effect size will be according to the order of the
                                                  outcome variables and the order of the list.
            survival_distribution (Sequence[str] or str): The distribution family from which to generate the outcome
                                                values of outcome variables that their corresponding outcome_types is
                                                "survival".
                                                Default value is exponent distribution.
                                                The same survival distribution will be used for the corresponding
                                                censoring variable as well.
                                                *To support multi-outcome* - place a list the size of the number of
                                                outcome variables of type "survival" (as stated in outcome_types). The
                                                matching between survival outcome variable and its survival distribution
                                                will be according to the order of the outcome variables and the order of
                                                the list. If all outcome variables has the same survival distribution -
                                                pass the str value (if present).
                                                *Ignore if no outcome variable is of type survival*
            survival_baseline (Sequence[float] or float): The survival baseline from the CoxPH model that will be the
                                                basics for the parameters of the corresponding survival_distribution.
                                                The same survival baseline will be used for the corresponding censoring
                                                variable as well (if present).
                                                Default value is 1 (no multiplicative meaning for baseline value).
                                                *To support multi-outcome* - place a list the size of the number of
                                                outcome variables of type "survival" (as stated in outcome_types). The
                                                matching between survival outcome variable and its survival distribution
                                                will be according to the order of the outcome variables and the order of
                                                the list. If all outcome variables has the same survival distribution -
                                                pass the str value.
                                                *Ignore if no outcome variable is of type survival*
            params (dict | None): Various parameters related to the generation process (e.g. the slope for
                                        sigmoid-based functions etc.).
                                        The form of: {var_name: {param_name: param_value, ...}, ...}
        """
        # Find the indices of each type of variable:
        var_types = pd.Series(var_types)
        self.var_names = var_types.index.to_series().reset_index(drop=True)
        self.var_types = var_types
        self.treatment_indices = var_types[var_types == TREATMENT].index
        self.outcome_indices = var_types[var_types == OUTCOME].index
        self.covariate_indices = var_types[(var_types == COVARIATE) | (var_types == HIDDEN)].index
        self.hidden_indices = var_types[var_types == HIDDEN].index
        self.censor_indices = var_types[var_types == CENSOR].index
        self.effmod_indices = var_types[var_types == EFFECT_MODIFIER].index

        self.linking_coefs = {}  # will accumulate the generated coefficients. {var: Series(coef, predecessors)}

        # COMPLETE topology INTO A SQUARE ADJACENCY MATRIX:
        # # let M be number of total variables, H number of variables to generate and L=M-H number of variables in a
        # # given baseline dataset (that generated variables can be based on). Given Topology matrix can have either a
        # # shape of MxM or HxM - in the latter case the matrix is completed into MxM by adding zero rows (since L
        # # given variables would not be re-genreated anyway, they will be consider independent variables).
        # if topology.shape[0] != topology.shape[1]:
        #     rows, cols = topology.shape
        #     if cols > rows:
        #         null_submatrix = np.zeros((cols - rows, cols), dtype=bool)
        #         topology = np.row_stack((topology, null_submatrix))
        #     else:
        #         raise ValueError("Topology matrix has {rows} rows and {cols} columns. This is not supported since"
        #                          "T[i,j] = 1 iff j is parent of i. ")

        if topology.shape[0] != len(var_types):
            raise ValueError("Number of variables in topology graph do not correspond to the number of variables states"
                             " in the variable types")
        self.m = len(var_types)  # number of variables

        # Create a graph out of matrix topology:
        self.topology = topology
        self.graph_topology = nx.from_numpy_array(topology.transpose(), create_using=nx.DiGraph())  # type:nx.DiGraph
        self.graph_topology = nx.relabel_nodes(self.graph_topology,
                                               dict(list(zip(list(range(self.m)), self.var_names))))

        # check that outcome variable is not dependant on more than 1 treatment variable
        for i in self.outcome_indices:
            predecessors = list(self.graph_topology.predecessors(i))
            treatment_predecessors = self.treatment_indices.intersection(predecessors)
            if len(treatment_predecessors) > 1:  # outcome variable is dependent on more than one treatment
                raise ValueError(
                    "Outcome {outcome} should have only one treatment affecting it. The current topology has outcome"
                    " variable dependant on {n_parent_treat} treatment parents which are: "
                    "{treatment_parents}".format(outcome=i, n_parent_treat=len(treatment_predecessors),
                                                 treatment_parents=treatment_predecessors))
            elif len(treatment_predecessors) == 0:  # outcome variable is dependent on exactly one treatment
                warnings.warn("Outcome variable {} has no treatment effecting it".format(i), UserWarning)

        # check that outcome variable is dependant on most 1 censor variable
        for i in self.outcome_indices:
            predecessors = list(self.graph_topology.predecessors(i))
            censor_predecessors = self.censor_indices.intersection(predecessors)
            if len(censor_predecessors) > 1:  # outcome variable is dependent on more than one treatment
                raise ValueError(
                    "Outcome {outcome} should have at most one censor variable affecting it. The current topology has "
                    "outcome variable dependant on {n_parent_cens} treatment parents which are: "
                    "{cens_parents}".format(outcome=i, n_parent_cens=len(censor_predecessors),
                                            cens_parents=censor_predecessors))

        # check that effect modifier is independent on treatment and affects only the outcome:
        for i in self.effmod_indices:
            successors = list(self.graph_topology.successors(i))
            if len(successors) == 0 or self.outcome_indices.intersection(successors).size < 1:
                raise ValueError("Effect modifier variable {name} must affect an outcome variable".format(name=i))
            ancestors = nx.ancestors(self.graph_topology, i)
            if self.treatment_indices.intersection(ancestors).size > 0:
                raise ValueError("Effect modifier variable {name} must not be affected by "
                                 "treatment variable (which is one of {ances})".format(name=i, ances=ancestors))

        # convert scalars to vectors if necessary.
        self.prob_categories = self._convert_scalars_to_vectors(x=prob_categories, default_value=None,
                                                                x_type="prob_categories")
        self.prob_categories = self.prob_categories.map(lambda x: pd.Series(x) if x is not None else x)
        if self.prob_categories.isnull().all():
            warnings.warn("Got all Nones in prob_categories. If simulation has Treatment variables in it, "
                          "this will throw an exception, as treatment variables must be categorical", UserWarning)

        # Check that all treatment variables are categorical:
        for i in self.treatment_indices:
            if self.prob_categories[i] is None:
                raise ValueError("Only categorical treatment is currently supported. However, treatment variable {t} "
                                 "is not categorical. Please specify corresponding category_probabilities".format(t=i))

        self.snr = self._convert_scalars_to_vectors(x=snr, default_value=1, x_type="snr")

        self.link_types = self._convert_scalars_to_vectors(x=link_types, default_value=DEFAULT_LINK_TYPE,
                                                           x_type="link_type")
        # if not all([x in self.VALID_LINK_TYPES for x in self.link_types]):
        all_linking_types = list(self.G_LINKING_METHODS.keys()) + list(self.O_LINKING_METHODS.keys())
        if not self.link_types.isin(all_linking_types).all():
            raise ValueError("link type must be one of {}, "
                             "got {} instead.".format(list(all_linking_types),
                                                      list(set(link_types) - set(all_linking_types))))

        self.treatment_methods = self._map_properties_to_variables(values=treatment_methods,
                                                                   keys=self.treatment_indices, var_type="treatment",
                                                                   value_type="methods")
        # if not all([x in TREATMENT_METHODS.keys() for x in self.treatment_methods.values()]):
        if not self.treatment_methods.isin(list(self.TREATMENT_METHODS.keys())).all():
            raise ValueError("link type must be one of {}, "
                             "got {} instead.".format(list(self.TREATMENT_METHODS.keys()),
                                                      list(
                                                          set(treatment_methods) - set(self.TREATMENT_METHODS.keys()))))

        self.treatment_importances = self._map_properties_to_variables(values=treatment_importances,
                                                                       keys=self.treatment_indices,
                                                                       var_type="treatment", value_type="importance")

        self.outcome_types = self._map_properties_to_variables(values=outcome_types, keys=self.outcome_indices,
                                                               var_type="outcome", value_type="type")
        for i in self.outcome_indices:
            if self.outcome_types[i] is CONTINUOUS and self.prob_categories[i] is not None:
                raise ValueError("Continuous outcome must be associated with None category probability. "
                                 "This was not the case in variable {outcome_var}. "
                                 "Might lead to undefined behaviour.".format(outcome_var=i))
            if self.outcome_types[i] is CATEGORICAL and self.prob_categories[i] is None:
                raise ValueError("Categorical outcome must be associated with category probability. However, None was"
                                 "associated with variable {outcome_var}".format(outcome_var=i))

        self.effect_sizes = self._map_properties_to_variables(values=effect_sizes, keys=self.outcome_indices,
                                                              var_type="outcome", value_type="effect size")

        # map survival_related properties to survival outcome and their corresponding censor variables.
        survival_outcome_variables = self.outcome_types[self.outcome_types.eq("survival")].index
        self.survival_distribution = self._map_properties_to_variables(values=survival_distribution,
                                                                       keys=survival_outcome_variables,
                                                                       var_type="outcome",
                                                                       value_type="survival_distribution")
        self.survival_distribution[self.survival_distribution.isnull()] = "expon"  # default is exponent distribution
        self.survival_baseline = self._map_properties_to_variables(values=survival_baseline,
                                                                   keys=survival_outcome_variables, var_type="outcome",
                                                                   value_type="survival_baseline")
        self.survival_baseline[self.survival_baseline.isnull()] = np.abs(np.random.normal(
            loc=0.0, scale=1.0, size=self.survival_baseline.isnull().sum()))
        for i in survival_outcome_variables:
            topology_predecessors = list(self.graph_topology.predecessors(i))
            censor_predecessors = self.censor_indices.intersection(topology_predecessors)
            if len(censor_predecessors) > 0:
                censor_predecessors = censor_predecessors[0]
                # match between the outcome value and it's matching censor variable:
                self.survival_distribution[censor_predecessors] = self.survival_distribution[i]
                self.survival_baseline[censor_predecessors] = self.survival_baseline[i]

        # self.params = params if params is not None else dict(zip(self.var_names, [None] * self.var_names.size))
        self.params = params if params is not None else {}

    # ### Initializing helper functions ### #
    def _convert_scalars_to_vectors(self, x, default_value, x_type):
        """
        Converts scalars (e.g. float, int, str, etc.) into vectors. Mapping between variable names to the desired value.
        In context: If arguments given to the class init are scalar (i.e. float, int, str, etc.), converts them into
                    vector shape - mapping every variable to the given value

        Args:
            x (Any): the value wished to map to the variables.
                     if supplied with some sequence (e.g. list, array, Series, etc.) it will map the sequence to
                     variable names. if supplied with a scalar - it will duplicate the single value to all vars.
            default_value (str|float|int|None): in case x=None (no value is supplied), map default_value to all vars
            x_type (str): The type of value that currently being processed (e.g. the variable name in the python code),
                          so in case there is an error, it can display the python-variable that caused the error.

        Returns:
            x (pd.Series): A Series mapping between variable name and a some wanted value.

        Raises:
            ValueError: If a sequence is given, but its length doesn't match the number of variables in topology.
        """
        if np.isscalar(x) or x is None:  # a scalar, not a sequence
            if x is None:  # put default value
                x = pd.Series(data=default_value, index=self.var_names)
            else:  # a scalar is given, map it to all variables
                x = pd.Series(data=x, index=self.var_names)
        else:
            # a sequence has been provided:
            if len(x) != self.m:
                raise ValueError("{x_type} should have same size as number of variables."
                                 "Got {emp} instead of {sup}".format(x_type=x_type, emp=len(x), sup=self.m))
            if isinstance(x, pd.Series) and x.index.difference(self.var_names).empty:
                # if supplied with a Series which has it own indexing, and it matches the the topology variables, then
                # keep it as is.
                x = x
            else:
                # either a simpler sequence or a Series with bad indexing, map to variable names.
                x = pd.Series(data=x, index=self.var_names)
        return x

    @staticmethod
    def _map_properties_to_variables(values, keys, var_type, value_type):
        """
        Maps between covariate variables properties to these properties.

        Args:
            values (Any): some property of some variable (e.g. 0.7 for treatment_importance or
                          "binary" for outcome_type)
            keys (Sequence[Any]): The names indices to map the given properties (values) (e.g. treatment_indices)
            var_type (str {"covariate", "hidden", "treatment", "outcome", "censor"}):  The type of variable the
                        properties being mapped to (e.g. "treatment", "outcome", "covariate")
            value_type (str): The name type that the property belongs to. (e.g. the variable name in the python code),
                              so in case there's an error, it can display the python-variable that caused the error.

        Returns:
            res (pd.Series): A map between the given keys (some covariate variable names indices) to the given values

        Raises:
            ValueError: When a Sequence is given as values (e.g. list of properties) but it does not match the length
                        of the keys.

        Warnings:
            UserWarning: If a values is a dict, it can may not be touched, unless its keys' do not match the variable
                         names. A warning is issued.

        Examples:
            Where effect_sizes is a Sequence or a float, outcome_indices are the indices names of the outcome variables
             in the graph. the variable type discussed is "outcome" (since it is effect-size). The python variable name
             is effect_size, thus the value_type is effect_size.
             map_properties_to_variables(values=effect_sizes, keys=self.outcome_indices, var_type="outcome",
                                         value_type="effect size")
        """
        if np.isscalar(values) or values is None:
            # values is a single value (i.e. int ot string), map its value to all given treatment variables:
            res = dict(list(zip(keys, [values] * len(keys))))
        else:
            # some sequence provided
            if len(keys) != len(values):
                raise ValueError("The number of {var_t} variables: {n_keys} does not match the size of the list "
                                 "depicting the {val_t} of creating each {var_t} variable: "
                                 "{n_vals}".format(var_t=var_type, n_keys=len(keys),
                                                   val_t=value_type, n_vals=len(values)))
            # values = values.values() if isinstance(values, dict) else values
            if isinstance(values, dict):
                # if given property is given by a dictionary, make sure this dict keys matches to the indices it
                # suppose to map to:
                res = values
                if list(values.keys()) != keys:
                    warnings.warn("{var_t} {val_t} was given as dictionary but its keys ({val}) does not match the "
                                  "{var_t} indices provided in topology ({keys}). You may expect "
                                  "undefined behaviour".format(var_t=var_type, val_t=value_type,
                                                               val=list(values.keys()), keys=keys), UserWarning)
            else:
                res = dict(list(zip(keys, values)))
        res = pd.Series(res, dtype=np.dtype(object))
        res = res.infer_objects()
        return res

    # ### Main functionality ### #
    def generate_data(self, X_given=None, num_samples=None, random_seed=None):
        """
        Generates tables of dataset given the object's initial parameters.

        Args:
            num_samples (int): Number of samples that will be in the dataset.
            X_given (pd.DataFrame): A baseline dataset to generate from. This dataset may contain only some of variables
                                    stated in the initialized topology. The rest of the dataset (variables which are
                                    stated in the topology and not in this dataset) will be generated.
                                    **Notes**: The data given will not be overwritten and will be taken as is. It is
                                     user responsibility to see that the given table has no dependant variables since
                                     they will not be re-generated according to the graph.
            random_seed (int): A seed for the pseudo-random-number-generator in order to reproduce results.

        Returns:
            (pd.DataFrame, pd.DataFrame, pd.DataFrame): 3-element tuple containing:

            - **X** (*pd.DataFrame*): A (num_samples x num_covariates) matrix of all covariates
                                      (including treatments and outcomes) over samples.
            - **propensities** (*pd.DataFrame*): A (num_samples x num_treatments) matrix (or vector) of propensity
                                                 values of every treatment.
            - **counterfactuals** (*pd.DataFrame*): A (num_samples x num_outcomes) matrix -
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if num_samples is None and X_given is None:
            raise ValueError("Must supply either a dataset (X) or number of samples to generate")
        if num_samples is not None and X_given is not None:
            warnings.warn("Got both number of samples (num_samples) and a baseline dataset (X_given). "
                          "Number of samples will be ignored and only X_given will be used.", UserWarning)

        if X_given is None:
            num_samples = num_samples
            patients_index = list(range(num_samples))
        else:
            num_samples = X_given.index.size
            patients_index = X_given.index

        # generate latent continuous covariates - every variable is guaranteed to have a population variance of 1.0
        # X_latent = pd.DataFrame(index=patients_index, columns=self.var_types.index)
        X = pd.DataFrame(index=patients_index, columns=self.var_types.index, dtype=float)
        if X_given is not None:  # if a dataset is given, integrate it to the current dataset being build.
            X.loc[:, X_given.columns] = X_given
            for col in X_given.columns:
                X.loc[:, col] = X[col].astype(X_given.dtypes[col])  # insist of keeping original types.
        propensities = pd.DataFrame(index=patients_index,
                                    columns=pd.MultiIndex.from_tuples([(i, j) for i in self.treatment_indices
                                                                       for j in self.prob_categories[i].index]))
        cf_columns = []
        for outcome in self.outcome_indices:
            predecessors = list(self.graph_topology.predecessors(outcome))
            treatment_predecessor = self.treatment_indices.intersection(predecessors)
            if not treatment_predecessor.empty:
                treatment_predecessor = treatment_predecessor[0]
                for j in self.prob_categories[treatment_predecessor].index:
                    cf_columns.append((outcome, j))
            else:
                cf_columns.append((outcome, "null"))
        counterfactuals = pd.DataFrame(index=patients_index, columns=pd.MultiIndex.from_tuples(cf_columns))

        # create the variables according to their topological order to avoid creating variables before their
        # dependencies are created:
        for i in nx.topological_sort(self.graph_topology):
            # i = self.var_names[i]     # get the name corresponding to the i'th location in topology
            if X.loc[:, i].notnull().any():
                # current column has non-NAN values meaning it has some data in it so it will not be overwritten
                continue
            var_type = self.var_types[i]
            X_parents = X.loc[:, self.topology[self.var_names[self.var_names == i].index[0], :]]
            if var_type == COVARIATE or var_type == HIDDEN or var_type == EFFECT_MODIFIER:

                X_signal, beta = self.generate_covariate_col(X_parents=X_parents, link_type=self.link_types[i],
                                                             snr=self.snr[i], prob_category=self.prob_categories[i],
                                                             num_samples=num_samples, var_name=i)
            elif var_type == TREATMENT:
                X_signal, propensity, beta = self.generate_treatment_col(X_parents=X_parents,
                                                                         link_type=self.link_types[i],
                                                                         snr=self.snr[i],
                                                                         method=self.treatment_methods[i],
                                                                         prob_category=self.prob_categories[i],
                                                                         var_name=i)
                propensities[i] = propensity
            elif var_type == OUTCOME:
                X_signal, cf, beta = self.generate_outcome_col(X_parents=X_parents, link_type=self.link_types[i],
                                                               snr=self.snr[i], prob_category=self.prob_categories[i],
                                                               effect_size=self.effect_sizes[i],
                                                               outcome_type=self.outcome_types[i],
                                                               survival_distribution=self.survival_distribution.get(i),
                                                               survival_baseline=self.survival_baseline.get(i),
                                                               var_name=i)
                counterfactuals[i] = cf
                # print 'mean treatment effect: %0.3f' % (np.mean(cf1 - cf0))

            elif var_type == CENSOR:
                outcome_successor = self.outcome_indices.intersection(self.graph_topology.successors(i))[0]
                treatment_predecessor = self.treatment_indices.intersection(self.graph_topology.predecessors(i))
                treatment_predecessor = treatment_predecessor[0] if len(treatment_predecessor) > 0 else None
                X_signal, beta = self.generate_censor_col(X_parents=X_parents, link_type=self.link_types[i],
                                                          snr=self.snr[i], prob_category=self.prob_categories[i],
                                                          outcome_type=self.outcome_types[outcome_successor],
                                                          treatment_importance=self.treatment_importances.
                                                          get(treatment_predecessor),
                                                          survival_distribution=self.survival_distribution.get(i),
                                                          survival_baseline=self.survival_baseline.get(i),
                                                          var_name=i)

            else:
                raise ValueError("{c_type} is not supported type of variable. "
                                 "Supported types are {s_types}".format(c_type=var_type, s_types=VALID_VAR_TYPES))
            X.loc[:, i] = X_signal
            self.linking_coefs[i] = beta

        # print X_latent.var(axis=0, ddof=1)
        # print X.var(axis=0, ddof=1)
        return X, propensities, counterfactuals

    def generate_covariate_col(self, X_parents, link_type, snr, prob_category, num_samples, var_name=None):
        """
        Generates a single signal (covariate) column

        Args:
            X_parents (pd.DataFrame): Sub-dataset containing only the relevant columns (features which are topological
                                      parents to the current covariate being created)
            link_type (str): How the parents variables (parents covariate columns) influence the current generated
                             column. What relation is there between them.
            snr (float): Signal to noise ratio that controls the amount of noise to add (value of 1.0 will not generate
                         noise)
            prob_category (pd.Series|None): A vector which length states the number of classes (number of discrete
                                            values) and every value is fractional - the probability of the corresponding
                                            class.
                                             **Notes**: vector must sum to 1 If None - the covariate column is left
                                             untouched (i.e. continuous)
            num_samples (int): number of samples to generate
            var_name (int|str): The name of the variable currently being generated. Optional.

        Returns:
            (pd.Series, pd.Series): 2-element tuple containing:

            - **X_final** (*pd.Series*): The final (i.e. noised and discretize [if needed]) covariate column.
            - **beta** (*pd.Series*): The coefficients used to generate current variable from it predecessors.

        Raises:
            ValueError: if the given link_type is not a valid link_type. (Supported link types are placed in
                        self.G_LINKING_METHODS)

        """
        # if variable has no parents - just sample from normal Gaussian distribution:
        if X_parents.empty:
            X_new = pd.Series(np.random.normal(loc=0.0, scale=1.0, size=num_samples), index=X_parents.index)
            beta = pd.Series(dtype=np.float64)
        else:
            # generate covariate column based on the parents' variables
            linking_method = self.G_LINKING_METHODS.get(link_type)
            if linking_method is None:
                raise KeyError("link type must be one of {},got {} instead.".format(list(self.G_LINKING_METHODS.keys()),
                                                                                    link_type))
            beta = self.linking_coefs.get(var_name)
            X_new, beta = linking_method(X_parents, beta=beta)

        # noise the sample
        X_noised_cont, _, _ = self._noise_col(X_new, snr=snr)

        # discretize variables if required:
        X_final = self._discretize_col(X_noised_cont, prob_category)

        return X_final, beta

    def generate_treatment_col(self, X_parents, link_type, snr, prob_category, method="logistic", var_name=None):
        """
        Generates a single treatment variable column.

        Args:
            X_parents (pd.DataFrame): Sub-dataset containing only the relevant columns (features which are topological
                                      parents to the current covariate being created)
            link_type (str): How the parents variables (parents covariate columns) influence the current generated
                             column. What relation is there between them.
            snr (float): Signal to noise ratio that controls the amount of noise to add (value of 1.0 will not generate
                         noise)
            prob_category (pd.Series|None): A k-length distribution vector over k-1 treatments with the probability
                                              of being untreated in prob_category[0] (prob_category.iloc[0]) and all
                                              other k-1 probabilities corresponds to k-1 treatments.
                                               **Notes**: vector must sum to 1. If None - the covariate column is left
                                                          untouched (i.e. continuous)
            method (str): A type of method to generate the treatment signal and the corresponding propensities.
            var_name (int|str): The name of the variable currently being generated. Optional.

        Returns:
            (pd.Series, pd.DataFrame, pd.Series): 3-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment to each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).
            - **beta** (*pd.Series*): The coefficients used to generate current variable from it predecessors.

        Raises:
            ValueError: if prob_category is None (treatment must be categorical)
            ValueError: If prob_category is not a legitimate probability vector (non negative, sums to 1)
        """
        # Check input validity:
        if prob_category is None:
            raise ValueError("Treatment variable must be categorical, therefore it must have a legitimate distribution "
                             "over its possible values. Got None instead.")
        CausalSimulator3._check_for_legitimate_probabilities(prob_category)

        # generate only the continuous signal since it is later processed (therefore prob_category = None)
        x_continuous, beta = self.generate_covariate_col(X_parents=X_parents, link_type=link_type, snr=snr,
                                                         prob_category=None, num_samples=X_parents.index.size,
                                                         var_name=var_name)

        generation_method = self.TREATMENT_METHODS.get(method)
        if generation_method is None:
            raise KeyError("The given method {method} is not supported, "
                           "only {valid_methods}.".format(valid_methods=list(self.TREATMENT_METHODS.keys()),
                                                          method=method))
        else:
            params = self.params.get(var_name, {})
            propensity, treatment = generation_method(x_continuous, prob_category, snr=snr, params=params)

        return treatment.astype(int), propensity.astype(float), beta

    def generate_outcome_col(self, X_parents, link_type, snr, prob_category, outcome_type, treatment_importance=None,
                             effect_size=None, survival_distribution=None, survival_baseline=None, var_name=None):
        """
        Generates a single outcome variable column.

        Args:
            X_parents (pd.DataFrame): Sub-dataset containing only the relevant columns (features which are topological
                                      parents to the current covariate being created)
            link_type (str): How the parents variables (parents covariate columns) influence the current generated
                             column. What relation is there between them.
            treatment_importance (float): The effect power of the treatment on the current generated outcome variable,
                                          as opposed to other variables that may influence on it.
            snr (float): Signal to noise ratio that controls the amount of noise to add (value of 1.0 will not generate
                         noise)
            prob_category (pd.Series|None): A k-length distribution vector over k-1 treatments with the probability
                                              of being untreated in prob_category[0] (prob_category.iloc[0]) and all
                                              other k-1 probabilities corresponds to k-1 treatments.
                                               **Notes**: vector must sum to 1. If None - the covariate column is left
                                                          untouched (i.e. continuous)
            effect_size (float): wanted mean effect size.
            outcome_type (str): Type of outcome variable. Either categorical (and continuous) or survival
            survival_distribution (str): The type of the distribution of which to sample the survival time from.
                                         relevant only if outcome_type is "survival"
            survival_baseline: The baseline value of the the cox ph model. relevant only if outcome_type is "survival"
            var_name (int|str): The name of the variable currently being generated. Optional.

        Returns:
            (pd.Series, pd.DataFrame, pd.DataFrame): 3-element tuple containing:

            - **x_outcome** (*pd.Series*): Outcome assignment for each sample.
            - **cf** (*pd.DataFrame*): Holding the counterfactuals for every possible treatment category of the
                                       outcome's treatment predecessor variable.
            - **beta** (*pd.DataFrame*): The coefficients used to generate current variable from it predecessors.

        Raises:
            ValueError: if the given link_type is not a valid link_type. (Supported link types are placed in
                        self.G_LINKING_METHODS)
            ValueError: if prob_category is neither None nor a legitimate distribution vector.
        """
        # drop censor indices as they do not affect the actual values of the outcome, only the masking later:
        X_parents = X_parents.drop(self.censor_indices, axis='columns')  # type: pd.DataFrame

        if X_parents.columns.size == 0:
            raise ValueError("Outcome variable cannot be independent variable (i.e. have no parent in graph topology)")

        # get effect modifiers:
        effect_modifier = self.effmod_indices.intersection(X_parents.columns)
        X_effmod = X_parents.loc[:, effect_modifier]  # type: pd.DataFrame
        X_covariates = X_parents.drop(effect_modifier, axis="columns")  # type: pd.DataFrame

        # get the treatment variable that affect current outcome.
        treatment_parent = self.treatment_indices.intersection(X_covariates.columns)
        if len(treatment_parent) > 1:  # outcome variable is dependent on more than one treatment
            raise ValueError(
                "Outcome should have only one treatment affecting it. The current topology has outcome"
                " variable dependant on {n_parent_treat} treatment parents which are: "
                "{treatment_parents}".format(n_parent_treat=len(treatment_parent),
                                             treatment_parents=treatment_parent))
        else:
            try:  # len(treatment_parents) == 0 outcome variable is dependent on exactly one treatment
                treatment_parent = treatment_parent[0]
                X_treatment = X_covariates.loc[:, treatment_parent]  # type: pd.Series
                X_covariates = X_covariates.drop(treatment_parent, axis="columns")  # type: pd.DataFrame
            except IndexError:  # len(treatment_parents) == 0 outcome variable is independent of treatment variables
                treatment_parent = None
                X_treatment = pd.Series(dtype=np.float64)
        has_treatment_parent = not X_treatment.empty
        treatment_importance = treatment_importance or self.treatment_importances.get(treatment_parent)

        original_treatment_categories = X_treatment.unique().astype(int)  # before being manipulated

        # convexly re-weight variables according if treatment has different importance than the covariates:
        if treatment_importance is not None:
            # !knowingly not weighting (especially weighting-down) effect modifiers! (so only re-weighting covariates)
            X_treatment *= treatment_importance  # how much the treatment affects the outcome
            if not X_covariates.columns.empty:  # how much non-treatments (regular covariates) affect outcome
                X_covariates *= float(float(1 - treatment_importance) / X_covariates.columns.size)
            X_parents = pd.concat([X_covariates, X_effmod, X_treatment], axis="columns", ignore_index=False)

        if link_type in list(self.G_LINKING_METHODS.keys()):
            # generate counterfactuals
            treatment_importance = 1 if treatment_importance is None else treatment_importance
            cf = {}
            for treatment_cat in original_treatment_categories:
                cf[treatment_cat] = X_parents.drop(treatment_parent, axis="columns")
                cf[treatment_cat].loc[:, treatment_parent] = treatment_cat * treatment_importance

            linking_method = self.G_LINKING_METHODS.get(link_type)
            beta = self.linking_coefs.get(var_name)
            x_outcome, beta = linking_method(X_parents, beta=beta)
            cf = {i: linking_method(cf[i], beta=beta)[0] for i in list(cf.keys())}

        elif link_type in self.O_LINKING_METHODS:
            linking_method = self.O_LINKING_METHODS.get(link_type)
            beta = self.linking_coefs.get(var_name)
            x_outcome, cf, beta = linking_method(X_covariates, X_effmod, X_treatment, beta=beta)
            cf = {col: cf[col] for col in cf.columns}

        else:
            raise KeyError("link type: {lt} is not a supported type of linking".format(lt=link_type))

        # noise the sample:
        x_outcome, cov_std, noise = self._noise_col(x_outcome, snr=snr)
        cf = {i: self._noise_col(cf[i], snr, cov_std, noise)[0] for i in list(cf.keys())}

        if effect_size is not None:
            warnings.warn("Stating effect size is not yet supported. Supplying it has no effect on results",
                          UserWarning)
            # TODO: support given effect size
            pass

        # aggregate according to type:
        if outcome_type == CATEGORICAL:
            x_outcome, bins = self._discretize_col(x_outcome, prob_category, retbins=True)
            # redefine bins edges so it could accommodate for values in the cfs that weren't present in the outcome:
            bins.iloc[0] = -np.inf
            bins.iloc[-1] = np.inf
            cf = {i: self._discretize_col(cf[i], prob_category, bins=bins) if has_treatment_parent else cf[i]
                  for i in list(cf.keys())}

        elif outcome_type == CONTINUOUS:
            pass

        elif outcome_type == PROBABILITY:
            x_outcome = self._sigmoid(x_outcome)
            cf = {i: self._sigmoid(cf[i]) for i in list(cf.keys())}

        elif outcome_type == SURVIVAL:
            if survival_distribution == "expon":
                rnd_state = np.random.randint(low=0, high=999999)
                param = survival_baseline * np.exp(x_outcome.astype(float))
                x_outcome = pd.Series(
                    stats.expon(loc=0.0, scale=(1.0 / param)).rvs(x_outcome.size, random_state=rnd_state),
                    index=x_outcome.index)
                cf = {i: pd.Series(
                    stats.expon(
                        loc=0.0,
                        scale=(1 / (survival_baseline * np.exp(cf[i].astype(float))))).rvs(
                            x_outcome.size,
                            random_state=rnd_state
                    ),
                    index=x_outcome.index)
                if has_treatment_parent else cf[i] for i in list(cf.keys())}
                # Supplying the random state assures that the resulting outcome and cfs is consistent while sampling rvs
            else:
                raise ValueError("survival distribution: {0}, is not supported".format(survival_distribution))
        else:
            raise ValueError("outcome type: {0}, is not supported outcome type".format(outcome_type))

        if not cf:  # dictionary is empty - outcome variable has no treatment parent
            cf = {"null": pd.DataFrame(data=None, index=X_parents.index, columns=["null"])}
        cf = pd.DataFrame(cf)
        return x_outcome, cf, beta

    def generate_censor_col(self, X_parents, link_type, snr, prob_category, outcome_type,
                            treatment_importance=None, survival_distribution=None, survival_baseline=None,
                            var_name=None):
        """
        Generates a single censor variable column.

        Args:
            X_parents (pd.DataFrame): Sub-dataset containing only the relevant columns (features which are topological
                                      parents to the current covariate being created)
            link_type (str): How the parents variables (parents covariate columns) influence the current generated
                             column. What relation is there between them.
            snr (float): Signal to noise ratio that controls the amount of noise to add (value of 1.0 will not generate
                         noise)
            prob_category (Sequence | None): A k-length distribution vector over k-1 treatments with the probability
                                              of being untreated in prob_category[0] (prob_category.iloc[0]) and all
                                              other k-1 probabilities corresponds to k-1 treatments.
                                               **Notes**: vector must sum to 1. If None - the covariate column is left
                                               untouched (i.e. continuous)
            outcome_type (str): The type of the outcome variable that is dependent on the current censor variable.
                                The censoring mechanism varies given different types of outcome variables.
            treatment_importance (float): The effect power of the treatment on the current generated outcome
                                                    variable, as opposed to other variables that may influence on it.
            survival_distribution (str): The type of the distribution of which to sample the survival time from.
                                         relevant only if outcome_type is "survival"
            survival_baseline: The baseline value of the the cox ph model. relevant only if outcome_type is "survival"
            var_name (int|str): The name of the variable currently being generated. Optional.

        Returns:
            (pd.Series, pd.Series): 2-element tuple containing:

            - **x_censor** (*pd.Series*): a column describing the censor variable
            - **beta** (*pd.Series*): The coefficients used to generate current variable from it predecessors.
        """
        if prob_category is None or len(prob_category) != 2:
            raise ValueError("Censor mechanism must be dichotomous (either censored or not-censored). However, Got the "
                             "following category probabilities instead: {0}".format(prob_category))
        if treatment_importance is not None:
            warnings.warn("treatment importance is not yet supported in generating censor variables", UserWarning)
            X_parents = X_parents.copy(deep=True)  # type: pd.DataFrame
            X_parents.loc[:, self.treatment_indices] *= treatment_importance
            non_treatment_parents = X_parents.columns.drop(self.treatment_indices)
            if not non_treatment_parents.empty:
                X_parents.loc[:, non_treatment_parents] *= float((float(1 - treatment_importance) /
                                                                  non_treatment_parents.size))

        if outcome_type in {CATEGORICAL, CONTINUOUS}:
            x_censor, beta = self.generate_covariate_col(X_parents=X_parents, link_type=link_type, snr=snr,
                                                         prob_category=prob_category, num_samples=X_parents.index.size,
                                                         var_name=var_name)
        elif outcome_type == SURVIVAL:
            x_signal, beta = self.generate_covariate_col(X_parents=X_parents, link_type=link_type, snr=snr,
                                                         prob_category=None, num_samples=X_parents.index.size,
                                                         var_name=var_name)
            if survival_distribution == "expon":
                # param = survival_baseline * (prob_category.iloc[0]/prob_category.loc[1]) * np.exp(x_signal)  # Cox ph
                param = survival_baseline * np.exp(x_signal.astype(float))  # Cox ph model
                survival_distribution = stats.expon(loc=0.0, scale=(1.0 / param))
                x_censor = pd.Series(survival_distribution.rvs(size=x_signal.size), index=x_signal.index)
                # scale values with censoring proportions - 0 is non censored, 1 is censored:
                x_censor *= (prob_category.iloc[0] / prob_category.loc[1])
            elif survival_distribution == "logistic":
                survival_distribution = stats.expon(loc=0.0, scale=(1.0 / survival_baseline))
                if X_parents.empty:  # censor variable is independent
                    probabilities = pd.Series(data=np.random.uniform(low=0, high=1, size=X_parents.index.size),
                                              index=X_parents.index)
                else:
                    x_signal, _ = self.generate_covariate_col(X_parents=X_parents, link_type=link_type, snr=snr,
                                                              prob_category=None, num_samples=X_parents.index.size)
                    t = x_signal.quantile(prob_category.iloc[1], interpolation="higher")
                    probabilities = 1.0 / (1 + np.exp(x_signal - np.repeat(t, x_signal.size)))
                x_censor = survival_distribution.ppf(probabilities)
            else:
                raise ValueError("survival distribution: {0}, is not supported".format(survival_distribution))
        else:
            raise ValueError("Unsupported censoring mechanism for type of outcome: {0}".format(outcome_type))

        return x_censor, beta

    # ### TREATMENT GENERATION METHODS ### #
    @staticmethod
    def _treatment_random(x_continuous, prob_category):
        """
        Assign treatment to samples completely at random.
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).
        """
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(data=np.tile(prob_category, (len(index_names), 1)),
                                  index=index_names, columns=columns_names)
        treatment = pd.Series(data=np.random.choice(a=prob_category.index, size=len(index_names), replace=True,
                                                    p=prob_category), index=index_names)
        return propensity, treatment

    @staticmethod
    def _treatment_gaussian_dichotomous(x_continuous, prob_category, snr):
        """
        Assign treatment to samples by sampling percentiles from a normal distribution
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.
            snr (float): signal to noise ratio.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).

        Raises:
            ValueError: If given more than to categories. This method supports dichotomous treatment only.
        """
        if prob_category.size != 2:  # this method suited for dichotomous outcome only
            raise ValueError("logistic method supports only binary treatment. Got the distribution vector "
                             "{p_vec} of length {n_cat}".format(n_cat=prob_category.size, p_vec=prob_category))
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(index=index_names, columns=columns_names)
        # compute propensities:
        t = stats.norm(loc=0, scale=1).ppf(prob_category.iloc[1])  # percentile given a distribution
        cur_propensity = stats.norm(loc=x_continuous, scale=(1 - snr)).sf(t)  # sf is 1 - CDF
        # discretize values:
        treatment = CausalSimulator3._discretize_col(x_continuous, prob_category)
        propensity.loc[:, columns_names[1]] = cur_propensity
        propensity.loc[:, columns_names[0]] = np.ones(cur_propensity.size) - cur_propensity
        return propensity, treatment

    @staticmethod
    def _treatment_logistic_dichotomous(x_continuous, prob_category, params=None):
        """
        Assign treatment to samples using a logistic model.
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.
            params (dict | None): Parameters that will be used in the generation function, e.g. sigmoid slope.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).

        Raises:
            ValueError: If given more than to categories. This method supports dichotomous treatment only.
        """
        if prob_category.size != 2:  # this method suited for dichotomous outcome only
            raise ValueError("logistic method supports only binary treatment. Got the distribution vector "
                             "{p_vec} of length {n_cat}".format(n_cat=prob_category.size, p_vec=prob_category))
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(index=index_names, columns=columns_names)
        # compute propensities:
        t = x_continuous.quantile(prob_category.iloc[1], interpolation="higher")
        slope = params.get("slope", 1.0) if params is not None else 1.0
        cur_propensity = 1.0 / (1 + np.exp(slope * (x_continuous - np.repeat(t, x_continuous.size)).astype(float)))
        # assign the propensity values:
        propensity.loc[:, columns_names[1]] = cur_propensity
        propensity.loc[:, columns_names[0]] = np.ones(cur_propensity.size) - cur_propensity
        treatment = CausalSimulator3._sample_from_row_stochastic_matrix(propensity)
        return propensity, treatment

    @staticmethod
    def _treatment_odds_ratio(x_continuous, prob_category, snr):
        """
        Assign treatment proportional to the odds ratio of the categories.
        Each category is assigned with it's odds ratio independently (based on logistic function) and are later sampled
        proportional to these odds ratio.
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.
            snr (float) - signal to noise ratio.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).
        """
        x_continuous = x_continuous.astype(float)
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(index=index_names, columns=columns_names)
        # start with filling up the odds ratio:
        for cur_category, p in prob_category.items():
            t = x_continuous.quantile(p, interpolation="higher")
            cur_propensity = (1.0 / (1 + np.exp((x_continuous - np.repeat(t, x_continuous.size)))))  # type: pd.Series
            cur_propensity = cur_propensity.div(np.ones_like(cur_propensity) - cur_propensity)
            cur_propensity += np.abs(np.random.normal(loc=0.0, scale=1 - snr, size=cur_propensity.size))
            # cur_propensity += np.random.exponential(scale=np.sqrt(snr), size=cur_propensity.size)
            propensity.loc[:, cur_category] = cur_propensity

        # normalize into probabilities:
        propensity = propensity.div(propensity.sum(axis="columns"), axis="rows")
        # treatment assignment is drawn according to marginal propensities:
        treatment = CausalSimulator3._sample_from_row_stochastic_matrix(propensity)
        return propensity, treatment

    @staticmethod
    def _treatment_quantile_gauss_fit(x_continuous, prob_category, snr):
        """
        Assign treatment by quantiling and shuffling.
        The signal is divided into quantiles according to the given probability (proportions). A gaussian distribution
        is fitted for each quantile. A score is calculated for each sample based on the pdf of the fitted gaussian.
        The scores are then rescaled to function as propensities to that category, while the complement (one minus the
        propensity) is distributed proportionally among the rest of the categories.
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.
            snr(float): signal to noise ratio.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).
        """
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(index=index_names, columns=columns_names)
        # section the signal into bins based on the probabilities (quantiles)
        x_continuous = x_continuous.astype(float)
        bins = pd.qcut(
            x=x_continuous,
            q=np.cumsum(pd.concat([pd.Series(0, index=["null"]), prob_category])),
            labels=columns_names
        )
        for cur_category in columns_names:
            cur_samples_mask = (bins == cur_category)
            cur_samples = x_continuous[cur_samples_mask]
            fit_mu, fit_sigma = stats.norm.fit(cur_samples)
            # fits.loc[cur_category, :] = {"mean": fit_mu, "var": fit_sigma}
            cur_pdfs = cur_samples.apply(stats.norm(loc=fit_mu, scale=fit_sigma).pdf)  # type:pd.Series
            # rescale:
            max_p = 1.0 - (1.0 - snr)
            min_p = cur_pdfs.div(cur_pdfs.sum()).min()
            cur_propensity = (max_p - min_p) * (cur_pdfs - cur_pdfs.min()) / \
                             (cur_pdfs.max() - cur_pdfs.min()) + min_p  # type: pd.Series
            # assign the propensity to the assigned category:
            propensity.loc[cur_samples_mask, cur_category] = cur_propensity
            # assign the propensity to the other, not assigned, categories:
            left_over_ps = prob_category.drop(cur_category)  # type: pd.Series
            left_over_ps = left_over_ps.div(left_over_ps.sum())
            not_propensity = pd.DataFrame(data=np.tile(np.ones_like(cur_propensity) - cur_propensity,
                                                       (left_over_ps.size, 1)).transpose(),
                                          index=cur_propensity.index, columns=left_over_ps.index)
            not_propensity = not_propensity.mul(left_over_ps)
            propensity.loc[cur_samples_mask, left_over_ps.index] = not_propensity
        # propensity = propensity.astype(np.float)
        # treatment assignment is drawn according to marginal propensities:
        treatment = CausalSimulator3._sample_from_row_stochastic_matrix(propensity)
        return propensity, treatment

    # ### HELPER FUNCTIONS ### #
    @staticmethod
    def _sample_from_row_stochastic_matrix(propensity):
        """
        Given a row-stochastic matrix (DataFrame) sample one support from each row.
        Args:
            propensity (pd.DataFrame): A row-stochastic DataFrame (i.e. all rows sums to one and non negative).

        Returns:
            treatment (pd.Series): A vector (length of propensity.index) of the resulted sampling.
        """
        categories_names = propensity.columns
        prop_cdf = propensity.cumsum(axis="columns")
        r = np.random.uniform(low=0, high=1, size=(propensity.index.size, 1))
        categories = prop_cdf.le(np.tile(r, (1, categories_names.size))).sum(axis="columns")
        treatment = pd.Series(categories_names[categories].values, index=propensity.index)
        # treatment = pd.Series(index=index_names)
        # for i in treatment.index:
        #     treatment[i] = np.random.choice(prob_category.index, [propensity.loc[i, :]])
        return treatment

    @staticmethod
    def _discretize_col(x_col, prob_category, method="empiric", retbins=False, bins=None):
        """
        If needed, turns the continuous covariate column into a discrete one (having discrete values as specified by the
        length of prob_category).

        Args:
            x_col (pd.Series): A covariate column (vector)
            prob_category (pd.Series|None): A vector which length states the number of classes (number of discrete
                                              values) and every value is fractional - the probability of the
                                              corresponding class.
                                               **Notes**: vector must sum to 1 If None - the covariate column is left
                                                untouched (i.e. continuous)
            method (str): Method by which discretization will be applied.
            retbins (bool): whether to return bins (if applicable)
            bins (pd.Series): discretize values according to these given bins.
                              for k categories, the bins is a k+1 Series if index being: ("null", 0, 1, ... k), i.e.
                              the first index of the Series is a "null" one which will be disregarded when labeling the
                              discretized values.

        Returns:
            res (pd.Series): A continuous covariate column if prob_category is None, else a discrete column according
                             to the given probabilities.
            bins (pd.Series): the bins

        Raises:
            ValueError: If prob_category is not a legitimate probability vector (non negative, sums to 1)
        """
        if prob_category is None or x_col.nunique() <= prob_category.size:
            res = x_col
            bins = None
        else:  # should perform discretization
            if bins is None:  # should create new bins
                CausalSimulator3._check_for_legitimate_probabilities(prob_category)
                # make k-1 thresholds (based on quantiles of the cdf, count how many thresholds each samples crosses:
                # see: https://en.wikipedia.org/wiki/Quantile_function#Definition
                if method == "gaussian":  # discretize according to percentiles drawn from normal distribution
                    bins = stats.norm(loc=0, scale=1).ppf(np.cumsum(prob_category)[:-1])
                    cutoffs = pd.DataFrame([x_col > thresh for thresh in bins]).T
                    res = cutoffs.sum(axis="columns")
                elif method == "empiric":  # discretize according to percentiles from the empirical data itself
                    try:
                        cumulative_ps = pd.concat(
                            [pd.Series(0, index=["null"]), prob_category]
                        ).cumsum()
                        res, bins = pd.qcut(x=x_col.astype(float), q=cumulative_ps,
                                            labels=prob_category.index, retbins=True)
                        bins = pd.Series(data=bins, index=cumulative_ps.index)
                        # TODO: maybe noise this a little?
                    except ValueError as _:
                        warnings.warn("Error occurred while discretizing column using pd.qcut. "
                                      "Probably the columns' values where already discrete (probably because it's "
                                      "cause variables were discrete too) and they current quantization has different "
                                      "proportions, causing the quantization process to have two identical bins and "
                                      "thus crash."
                                      "Using now a more robust (less accurate) method to discretize the column.")
                        bins = pd.Series(data=None, index=pd.Index(["null"]).append(prob_category.index))
                        bins["null"] = 0
                        res = pd.Series(data=None, index=x_col.index)
                        for category, cum_p in prob_category.cumsum().items():
                            # whomever is applicable because it's value is below upper quantile limit:
                            below_quantile_limit = x_col.sort_values().iloc[:int(cum_p * x_col.size)].index
                            # whomever is applicable because it hasn't been assigned yet:
                            not_yet_assigned = res[res.isnull()].index
                            # intersection of both conditions is the samples applicable to the current quantile:
                            current_quantile = below_quantile_limit.intersection(not_yet_assigned)
                            res[current_quantile] = category
                            bins[category] = x_col[below_quantile_limit].max()
                else:  # no legitimate discretization method was given.
                    raise ValueError("Method {m} for discretize the covariate is not supported".format(m=method))
            else:  # should use given bins
                res = pd.cut(x_col, bins, labels=bins.index.drop("null"))
            res = res.cat.codes if res.dtype.name == "categorical" else res.astype(np.int8)
        return (res, bins) if retbins else res

    @staticmethod
    def _check_for_legitimate_probabilities(prob_category):
        """
        Check if a given probability vector is a legitimate one (non negative and sums to 1.0)

        Args:
            prob_category (Sequence[float]): A vector describing a distribution over discrete categories.

        Returns: True if prob_category is legitimate probability vector. Otherwise raises relevant exception.

        Raises:
            ValueError: If prob_category does not sum to 1.0
            ValueError: If prob_category has negative entries.
        """
        p = np.array(prob_category)
        if p.sum() != 1.0:
            raise ValueError("probabilities for covariate categories do not sum to one. "
                             "{prob} sums to {sum} instead".format(prob=prob_category, sum=np.sum(prob_category)))
        if np.any(p < 0):
            raise ValueError("probabilities for covariate categories are negative.")
        return True

    @staticmethod
    def _noise_col(X_signal, snr, cov_std=None, noise=None):
        """
        Noising the given signal according to its size and the given snr and normalizing it to have standard-deviation
        of 1.
        Args:
            X_signal (pd.Series): Covariate column to noise.
            snr (float): Signal to noise ratio that controls the amount of noise to add (value of 1.0 will not generate
                         noise)
            cov_std (float): a given standard deviation
            noise (pd.Series): Gaussian white noise vector.
        Returns:
            (pd.Series, float, pd.Series): 3-element tuple containing:

            - **X_noised** (*pd.Series*): The signal after the noising process
            - **cov_std** (*float*): Standard deviation of the original un-noised signal
            - **noise** (*pd.Series*): The additive noise randomly generated.
        """
        n = X_signal.index.size
        cov_std = cov_std or X_signal.std(ddof=1)  # type: float
        X_signal *= np.sqrt(snr)
        # X_signal /= cov_std
        noise = np.random.normal(loc=0, scale=1, size=n) * np.sqrt(1 - snr) if noise is None else noise
        X_noised = X_signal + noise
        return pd.Series(X_noised, index=X_signal.index), cov_std, pd.Series(noise, index=X_signal.index)

    @staticmethod
    def _noise_col_shuffle(x_outcome, cf, snr):
        # randomly sample the same samples to be shuffle:
        rnd_state_sample = np.random.random_integers(low=0, high=999999)
        shuffled_samples = x_outcome.sample(frac=1 - snr, random_state=rnd_state_sample)
        sampled_cf = {i: cf[i].sample(frac=1 - snr, random_state=rnd_state_sample) for i in list(cf.keys())}
        # shuffle and keep the exact order among the counterfactuals:
        rnd_state_shuffle = np.random.random_integers(low=0, high=999999)
        shuffled_samples[:] = shuffled_samples.sample(frac=1, random_state=rnd_state_shuffle).values
        for i in list(sampled_cf.keys()):
            sampled_cf[i][:] = sampled_cf[i].sample(frac=1, random_state=rnd_state_shuffle).values
        # assign back to the signal:
        x_noised_continuous = x_outcome
        x_noised_continuous.loc[shuffled_samples.index, :] = shuffled_samples
        for i in list(cf.keys()):
            cf[i].loc[sampled_cf[i].index] = sampled_cf[i]

    @staticmethod
    def _sigmoid(x, slope=1):
        return 1.0 / (1.0 + np.exp(-slope * x))

    def reset_coefficients(self, variables=None):
        """
        Delete the linking coefficients that accumulated in the generating model so far.

        Args:
            variables (list|None): list of variables to reset the coefficients linking *into* them (Not from them).
                                   if None - all the available coefficients will be deleted.
        """
        if variables is None:
            variables = self.var_names  # self.linking_coefs.clear()
        variables = variables.intersection(list(self.linking_coefs.keys()))
        for variable in variables:
            del self.linking_coefs[variable]

    # ### LINKING PARENTS TO VARIABLE ### #
    @staticmethod
    def _linear_link(X_parents, beta=None):
        """
        creates a variable linearly dependant on its parents.
        Args:
            X_parents (pd.DataFrame): a (num_samples x num_parents) matrix containing the data (over all samples or
                                      samples or patients) of the variables which are topological parents of the current
                                      variable
            beta (pd.Series): Optional, a given Series which index corresponds to the parents variables
                              (X_parents.columns)

        Returns:
            (pd.Series, pd.Series): 2-element tuple containing:

            - **x_new** (*pd.Series*): Newly created signal.
            - **beta** (*pd.Series*): The coefficients used to create the linear link.
        """
        if beta is None:
            num_parents = X_parents.columns.size
            beta = pd.Series(data=np.random.normal(loc=0.0, scale=1.0, size=num_parents) / np.sqrt(num_parents),
                             index=X_parents.columns)
        x_new = X_parents.dot(beta)  # type: pd.Series
        return x_new, beta

    @staticmethod
    def _affine_link(X_parents, beta=None):
        """
        creates a variable of affine dependence on its parents (meaning linear + intercept)
        Args:
            X_parents (pd.DataFrame): a (num_samples x num_parents) matrix containing the data (over all samples or
                                      samples or patients) of the variables which are topological parents of the current
                                      variable
            beta (pd.Series): Optional, a given Series which index corresponds to the parents variables
                              (X_parents.columns)

        Returns:
            (pd.Series, pd.Series): 2-element tuple containing:

            - **X_new** (*pd.Series*): Newly created signal.
            - **beta** (*pd.Series*): The coefficients used to create the linear link.
        """
        X_parents = X_parents.copy()  # type: pd.DataFrame
        X_parents["intercept"] = 1
        return CausalSimulator3._linear_link(X_parents, beta=beta)

    @staticmethod
    def _exp_linking(X_parents, beta=None):
        """
        creates a variable linearly dependant on its parents and then exponent it: exp(beta*X)
        Args:
            X_parents (pd.DataFrame): a (num_samples x num_parents) matrix containing the data (over all samples or
                                      samples or patients) of the variables which are topological parents of the current
                                      variable
            beta (pd.Series): Optional, a given Series which index corresponds to the parents variables
                              (X_parents.columns)

        Returns:
            (pd.Series, pd.Series): 2-element tuple containing:

            - **x_new** (*pd.Series*): Newly created signal.
            - **beta** (*pd.Series*): The coefficients used to create the linear link.
        """
        x_new, beta = CausalSimulator3._affine_link(X_parents=X_parents, beta=beta)
        x_new = np.exp(x_new)  # type: pd.Series
        return x_new, beta

    @staticmethod
    def _log_linking(X_parents, beta=None):
        """
        creates a variable linearly dependant on its parents and then log it: log(beta*X)
        Args:
            X_parents (pd.DataFrame): a (num_samples x num_parents) matrix containing the data (over all samples or
                                      samples or patients) of the variables which are topological parents of the current
                                      variable
            beta (pd.Series): Optional, a given Series which index corresponds to the parents variables
                              (X_parents.columns)

        Returns:
            (pd.Series, pd.Series): 2-element tuple containing:

            - **x_new** (*pd.Series*): Newly created signal.
            - **beta** (*pd.Series*): The coefficients used to create the linear link.
        """
        x_new, beta = CausalSimulator3._affine_link(X_parents=X_parents, beta=beta)
        x_new = np.log(np.abs(x_new))  # type: pd.Series
        return x_new, beta

    @staticmethod
    def _poly_linking(X_parents, beta=None):
        """
        creates a variable polynomially dependant on its parents: x[i,0]**beta[0] + ... + x[i,d]**beta[d]
        Args:
            X_parents (pd.DataFrame): a (num_samples x num_parents) matrix containing the data (over all samples or
                                      samples or patients) of the variables which are topological parents of the current
                                      variable
            beta (pd.DataFrame): Optional, a given matrix of (degree x num_parents) specifying the coefficients where
                                 each row corresponds to a degree (e.g. the first (zeroth) row corresponds to the
                                 zeroth order of the polynomial and each column correspond to a parent variable.
                                 The overall calculation is: x_new = sum( beta[i,j] * x_j^i )  [x_j to the ith power]

        Returns:
            (pd.Series, pd.DataFrame): 2-element tuple containing:

            - **x_new** (*pd.Series*): Newly created signal.
            - **beta** (*pd.DataFrame*): The coefficients matrix used to create the linking: (degree x num_parents)
                                         matrix. beta[i,j] is the coefficient of parent variable x_j when raised to the
                                         ith power.
        """
        if beta is None:
            degree = np.random.randint(low=2, high=5)
            beta = pd.DataFrame(data=np.random.normal(loc=0.0, scale=4.0, size=(degree, X_parents.columns.size)),
                                columns=X_parents.columns, index=np.arange(degree))

        result_polynomial = pd.DataFrame(data=None, index=X_parents.index, columns=X_parents.columns, dtype=float)
        degrees = beta.index.to_series()
        # Apply a polynomial to every parent variable
        for var_name, col in X_parents.items():
            a = pd.concat([col] * len(degrees), axis="columns", ignore_index=True)  # type: pd.DataFrame
            a = a.pow(degrees, axis="columns")
            a = a.multiply(beta.loc[:, var_name], axis="columns")
            a = a.sum(axis="columns")
            result_polynomial.loc[:, var_name] = a
        x_new = result_polynomial.sum(axis="columns")

        return x_new, beta

    @staticmethod
    def _marginal_structural_model_link(X_covariates, X_effmod, X_treatment, beta=None):
        """
        Generate outcome variable based on marginal structural model (see Hernan and Robin sections 12.4 and 12.5)

        Args:
            X_covariates (pd.DataFrame): Causing covariates.
            X_effmod (pd.DataFrame): Causing effect modifiers
            X_treatment (pd.Series): Causing treatment variable.
            beta (pd.DataFrame): The coefficients used to generate current variable from it predecessors. Optional.
                                (num_parents_variables x num_treatment_categories) matrix.

        Returns:
            (pd.Series, pd.DataFrame, pd.DataFrame): 3-element tuple containing:

            - **x_outcome** (*pd.Series*): Newly created outcome signal.
            - **cf** (*pd.DataFrame*): Corresponding counterfactuals
            - **beta** (*pd.DataFrame*): The coefficients used to generate current variable from it predecessors.
                                         (num_parents_variables x num_treatment_categories) matrix.
        """
        # treatment_categories = np.concatenate([X_treatment[col].unique().astype(int) for col in X_treatment.columns])
        treatment_categories = X_treatment.unique().astype(int)
        # cf = pd.DataFrame(data=None, index=X_covariates.index, columns=treatment_categories)
        x_treatment_intercept = pd.Series(data=1, index=X_treatment.index, name=X_treatment.name)
        X_parents = pd.concat([X_covariates, X_effmod, x_treatment_intercept], axis="columns")  # type: pd.DataFrame
        if beta is None:  # no linking coefficients were supplied, create the coefficient matrix to be used
            # create a totally random matrix:
            beta = pd.DataFrame(data=np.random.normal(loc=0.0, scale=4.0,
                                                      size=(X_parents.columns.size, treatment_categories.size)),
                                index=X_parents.columns, columns=treatment_categories)
            # enforce the same coefficients for the regular covariates (neither treatment nor effect modifiers)
            covariates_coefs = np.tile(np.random.normal(loc=0.0, scale=4.0, size=(X_covariates.columns.size, 1)),
                                       (1, treatment_categories.size))
            beta.loc[X_covariates.columns, :] = covariates_coefs

        cf = X_parents.dot(beta)  # type: pd.DataFrame
        x_outcome = robust_lookup(cf, X_treatment)
        return x_outcome, cf, beta

    # ### SAVING DATASET ### #
    def format_for_training(self, X, propensities, cf, headers_chars=None, exclude_hidden_vars=True):
        """
        prepare to output. merge the data into two DataFrames - an observed one and one gathering the counterfactuals.

        Args:
            X (pd.DataFrame): Containing the data (covariates) , treatment and outcomes
            propensities (pd.DataFrame): Containing the propensity values for the treatmetns
            cf (pd.DataFrame): Containing the counterfactuals results for all possible treatments.
            headers_chars (dict): Optional. Containing the column header prefix for different types of variables.
                                  Examples: {"covariate": "x", "treatment": "t", "outcome": "y"}
            exclude_hidden_vars: If to exclude hidden variables from the resulting dataset.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **df_X** (*pd.DataFrame*): The observed dataset (if hidden variables are excluded).
            - **df_cf** (*pd.DataFrame*): Containing the two counterfactuals, treatments and propensities.

        """
        if headers_chars is None:
            headers_chars = {}

        covariate_char = headers_chars.get(COVARIATE) or "x"
        treatment_char = headers_chars.get(TREATMENT) or "t"
        outcome_char = headers_chars.get(OUTCOME) or "y"
        propensity_char = headers_chars.get("propensity") or "p"
        counterfact_char = headers_chars.get("counterfactual") or "cf"
        censor_char = headers_chars.get("censor") or "c"
        effmod_char = headers_chars.get("effect_modifier") or "m"

        # exclude hidden variables if needed:
        hidden_vars = self.hidden_indices if exclude_hidden_vars else pd.Index([])
        X = X.drop(hidden_vars, axis="columns")  # type: pd.DataFrame

        # partition the different elements (variables) of the dataset to different DataFrames:
        assert isinstance(propensities.columns, pd.MultiIndex)
        assert isinstance(cf.columns, pd.MultiIndex)
        propensities = propensities.copy()  # type: pd.DataFrame
        cf = cf.copy()  # type: pd.DataFrame
        X_covariates = X.loc[:, self.covariate_indices.difference(hidden_vars)]
        X_treatment = X.loc[:, self.treatment_indices.difference(hidden_vars)]
        X_outcome = X.loc[:, self.outcome_indices.difference(hidden_vars)]
        X_effmod = X.loc[:, self.effmod_indices.difference(hidden_vars)]
        X_censor = X.loc[:, self.censor_indices.difference(hidden_vars)]
        # censor outcome columns if needed
        for i in X_outcome.columns:
            censor_predecessor = self.censor_indices.intersection(self.graph_topology.predecessors(i))
            if censor_predecessor.size > 0:  # current outcome variable has a censoring variable.
                censor_predecessor = censor_predecessor[0]
                if self.outcome_types[i] in {CATEGORICAL, CONTINUOUS}:
                    censored_samples = X.loc[:, censor_predecessor].astype(bool)
                    X_outcome.loc[censored_samples, i] = np.nan
                elif self.outcome_types[i] == SURVIVAL:
                    last_event_last_observed = pd.concat([X.loc[:, i], X.loc[:, censor_predecessor]],
                                                         axis="columns")  # type: pd.DataFrame
                    X_outcome.loc[:, i] = last_event_last_observed.min(axis="columns")
                else:
                    warnings.warn("Was not able to censor outcome properly due to outcome type not being supported",
                                  UserWarning)

        # rename columns:
        # TODO: if x is already a string (i.e. given header from file) so to avoid col name: x_x_1 can do: (*below*) - however - might create two columns with same name (e.g. if int(1) is a column and x_1 is a given column it will create two x_1 columns. solution - different char in covariate_char dictionary.
        # [covariate_char + "_" + str(x) if type(x) != str else x for x in X_covariates.columns]
        # TODO: maybe do sequentially: in range(X_#####.columns) to avoid examples where the only treatment variable is t_1 rather than t_4 (if 4 is treatment variable index)
        # [covariate_char + "_" + str(x) if type(x) != str else x for x in range(X_covariates.columns)]
        X_covariates.columns = [covariate_char + "_" + str(x) for x in X_covariates.columns]
        X_treatment.columns = [treatment_char + "_" + str(x) for x in X_treatment.columns]
        X_outcome.columns = [outcome_char + "_" + str(x) for x in X_outcome.columns]
        X_censor.columns = [censor_char + "_" + str(x) for x in X_censor.columns]
        X_effmod.columns = [effmod_char + "_" + str(x) for x in X_effmod.columns]

        propensities.columns = ["_".join([propensity_char, str(treatment), str(category)])
                                for treatment, category in propensities.columns.values]
        cf.columns = ["_".join([counterfact_char, str(outcome), str(treatment_category)])
                      for outcome, treatment_category in cf.columns.values]

        df_X = pd.concat([X_covariates, X_effmod, X_treatment, X_outcome], axis="columns")
        df_cf = pd.concat([X_treatment, propensities, cf, X_censor], axis="columns")

        return df_X, df_cf

    @staticmethod
    def to_csv(data, out_file=None):
        if out_file is not None:
            data.to_csv(out_file, index=False)


def idx2var_vector(num_vars, args):
    res = pd.Series(index=list(range(num_vars)))
    for indices, name in args.items():
        res[indices] = name
    return res


def generate_random_topology(n_covariates, p, n_treatments=1, n_outcomes=1, n_censoring=0, given_vars=(),
                             p_hidden=0.0):
    """
    Creates a random graph topology, suitable for describing a causal graph model.
    Generation is based on a G(n,p) random graph model (each edge independently generated or not by a coin toss).

    Args:
        n_covariates (int): Number of simple covariates to generate
        p (float): Probability to generate an edge.
        n_treatments (int): Number of treatment variables.
        n_outcomes (int): Number of outcome variables.
        n_censoring (int): Number of censoring variables.
        given_vars (Sequence[Any]): Vector of names of given variables. These variables are considered independent.
                               These suppose to mimic a situation where a partial dataset can be supplied to the
                               generation process. Those names will correspond to the variable names in this existing
                               baseline dataset.
        p_hidden (float): The probability to convert a simple covariate variable into a latent (i.e. hidden)
                          variable.

    Returns:
        (pd.DataFrame, pd.Series): 2-element tuple containing:

            - **topology** (*pd.DataFrame*): A boolean matrix describing graph dependencies.
                                             Where T[i,j] = True iff j is a predecessor of i.
            - **var_types** (*pd.Series*): A Series which index holds variable names and values are variable types.
                                           (e.g. "treatment", "covariate", "hidden', "outcome"...)
                                           The given_vars will be the first variable, followed by the generated vars
                                           (covariates, then treatment, then outcome, then censors)
    """
    # Check input validity:
    if n_treatments != n_outcomes:
        raise ValueError("Number of treatment variables ({t}) must match "
                         "the number of outcome variables ({o})".format(t=n_treatments, o=n_outcomes))

    given_vars = pd.Series(given_vars, dtype=np.dtype(object))  # Series' index is range(n_given_vars)
    n_given_vars = len(given_vars)

    # the given_vars will be the first in the variable vector and then the generated variables will follow in the
    # following order: covariates, treatments, outcomes, censoring.
    covariates = list(range(n_given_vars, n_given_vars + n_covariates))
    treatments = list(range(n_given_vars + n_covariates, n_given_vars + n_covariates + n_treatments))
    outcomes = list(range(n_given_vars + n_covariates + n_treatments,
                          n_given_vars + n_covariates + n_treatments + n_outcomes))
    censoring = list(range(n_given_vars + n_covariates + n_treatments + n_outcomes,
                           n_given_vars + n_covariates + n_treatments + n_outcomes + n_censoring))
    # # Makethe order of variables shuffled
    # generated_vars = range(n_covariates + n_treatments + n_outcomes + n_censoring)
    # sample_from_gen_vars = np.array(generated_vars)
    # treatments = np.random.choice(generated_vars, size=n_treatments, replace=False).tolist()
    # for i in treatments:
    #     generated_vars.remove(i)
    # outcomes = np.random.choice(generated_vars, size=n_outcomes, replace=False).tolist()
    # for i in outcomes:
    #     generated_vars.remove(i)
    # censoring = np.random.choice(generated_vars, size=n_censoring, replace=False).tolist()
    # for i in censoring:
    #     generated_vars.remove(i)
    # covariates = generated_vars

    n_generated_vars = n_covariates + n_treatments + n_outcomes + n_censoring
    generated_vars = covariates + treatments + outcomes + censoring
    generated_vars = pd.Series(data=generated_vars, index=generated_vars)

    total_vars = pd.concat([
        given_vars if not given_vars.empty else None,
        generated_vars if not generated_vars.empty else None,
    ])
    topology = pd.DataFrame(data=0, index=total_vars, columns=total_vars, dtype=bool)

    # generate between the independent given set to generated set:
    topology.loc[generated_vars, given_vars] = np.random.binomial(n=1, p=p,
                                                                  size=(n_generated_vars, n_given_vars)).astype(bool)
    # generate between generated-covariate to all generated set:
    topology.loc[generated_vars, covariates] = np.random.binomial(n=1, p=p,
                                                                  size=(n_generated_vars, n_covariates)).astype(bool)

    # generate between treatment to censoring variables:
    topology.loc[censoring, treatments] = np.random.binomial(n=1, p=p, size=(n_censoring, n_treatments)).astype(bool)

    # verify dtypes where not changed during the generation (as it would if not all variable types are present)
    topology[topology.select_dtypes(exclude=[bool]).columns] = topology.select_dtypes(exclude=[bool]).astype(bool)

    # enforce DAG by enforcing lower triangular matrix. since every edge is iid - nullifying the upper triangular
    # introduces no bias. Maybe should shuffle columns to un-confound the bias from ordering the variables by type?
    # topology = topology.loc[np.random.permutation(topology.index), np.random.permutation(topology.columns)]
    topology.values[np.triu_indices(n=total_vars.size)] = False

    # match between treatments and outcomes:
    matches = list(zip(outcomes, np.random.permutation(treatments)))
    for outcome_idx, treatment_idx in matches:
        topology.loc[outcome_idx, treatment_idx] = True

    # match between censoring and outcomes:
    matches = list(zip(outcomes, np.random.permutation(censoring)))
    for outcome_idx, censoring_idx in matches:
        topology.loc[outcome_idx, censoring_idx] = True

    generated_types = pd.Series(index=generated_vars, dtype=object)
    generated_types[covariates] = COVARIATE
    generated_types[treatments] = TREATMENT
    generated_types[outcomes] = OUTCOME
    generated_types[censoring] = CENSOR
    given_types = pd.Series(data=COVARIATE, index=given_vars)
    # Nullify empty Frames/Series before concat to satisfy pandas FutureWarning:
    given_types = None if given_types.empty else given_types
    generated_types = None if generated_types.empty else generated_types
    var_types = pd.concat([given_types, generated_types])

    # create a hidden variables mask:
    covariates = pd.Series(covariates)
    # Nullify empty Frames/Series before concat to satisfy pandas FutureWarning:
    given_vars = None if given_vars.empty else given_vars
    covariates = None if covariates.empty else covariates
    hidden_vars = pd.concat([given_vars, covariates]).sample(frac=p_hidden)
    var_types[hidden_vars] = HIDDEN

    return topology, var_types
