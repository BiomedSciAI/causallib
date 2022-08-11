#!/usr/bin/env python3

import pandas as pd
from typing import Optional, Any
from causallib.time_varying.base import GMethodBase
from causallib.time_varying.treament_strategy import TreatmentStrategy
from causallib.utils import general_tools as g_tools
import numpy as np


class GFormula(GMethodBase):
    """
        GFormula class that is based on Monte Carlo Simulation.
    """
    def __init__(self, treatment_model, covariate_models, outcome_model, refit_models, random_state, n_obsv, n_sims, n_steps, mode, resid_val):
        super(GFormula, self).__init__(treatment_model, covariate_models, outcome_model, refit_models)
        self.random_state = random_state
        self.n_obsv = n_obsv
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.mode = mode
        self.resid_val = resid_val
        self.id_col = 'id'

        self.covariate_cols = None
        self.treatment_cols = None
        self.time_col = None
        self.y_col = None
        self.prev_covariate_cols = None
        self.prev_treatment_cols = None
        self.all_cols = None

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: Optional[pd.Series] = None,
            y: Optional[Any] = None,
            refit_models: bool = True,
            id_col: str = None,
            **kwargs
            ):

        # TODO calculate residual_values
        if kwargs is None:
            kwargs = {}

        self._set_cols(X, a, t, y)
        treatment_X, treatment_Y, covariate_X, covariate_Y, outcome_X, outcome_Y = self._prepare_data(X, a, t, y)

        treatment_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.treatment_model)
        if refit_models or treatment_model_is_not_fitted:
            self.treatment_model.fit(treatment_X, treatment_Y, **kwargs)

        for cov in self.covariate_models:
            cov_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.covariate_models[cov])
            if refit_models or cov_model_is_not_fitted:
                self.covariate_models[cov].fit(covariate_X[cov], covariate_Y[cov],  **kwargs)

        if y:
            outcome_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.outcome_model)
            if refit_models or outcome_model_is_not_fitted:
                self.outcome_model.fit(outcome_X, outcome_Y, **kwargs)
            self.id_col = id_col if id_col else self.id_col
        return self

    def estimate_individual_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: Optional[pd.Series] = None,
                                    y: Optional[Any] = None,
                                    treatment_strategy: TreatmentStrategy = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
            Returns dataframe (individual estimated curves) for each subject in (t * (X and a))

        Steps:
            1. For each sample,
                a. Get the simulation outputs ('covariates', 'actions' & 'time')
                   from _estimate_individual_outcome_single_sample
                b. For covariate (n_sim * n_steps * cov_cols) & action (n_sim * n_steps * act_cols),
                   take mean across 'n_sim', which will result (n_steps * cov_cols) & (n_steps * act_cols)
                c. Concatenate these two arrays across column
                d. Convert the result to df
                e. Add 'sample_id' and 'time' to df
                f. Append each sample to list
            2. Finally, concatenate list of dfs and return
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)
        unique_sample_ids = X[self.id_col].unique()
        all_sim_result = []
        for sample_id in unique_sample_ids:
            sample_X = X.loc[X[self.id_col] == sample_id]
            sample_a = a.loc[a[self.id_col] == sample_id]
            sample_y = y.loc[y[self.id_col] == sample_id] if y else None
            sample_sim = self._estimate_individual_outcome_single_sample(X=sample_X,
                                                                         a=sample_a,
                                                                         t=t,
                                                                         y=sample_y,
                                                                         treatment_strategy=treatment_strategy,
                                                                         timeline_start=timeline_start,
                                                                         timeline_end=timeline_end)

            sample_sim_cov = sample_sim['covariates'].mean(axis=0)  # n_steps * cov_cols
            sample_sim_act = sample_sim['actions'].mean(axis=0)  # n_steps * act_cols

            sample_sim_res = pd.DataFrame(np.concatenate([sample_sim_cov, sample_sim_act], axis=1),
                                          columns=list(self.covariate_cols) + list(self.treatment_cols))
            sample_sim_res[self.id_col] = sample_id
            sample_sim_res[self.time_col] = sample_sim['time']
            all_sim_result.append(sample_sim_res)
        return pd.concat(all_sim_result)

    def _estimate_individual_outcome_single_sample(self, X, a, t, y, treatment_strategy, timeline_start, timeline_end) -> dict:
        """
            Simulates the outcome for each sample across 't' steps.
            Returns:
                 sample_sim = [
                {
                    'actions' = n_sim * n_steps * cov_cols
                    'covariates' = n_sim * n_steps * act_cols
                    'time' = 1 * n_steps
                    'pat_id' = str
                }
            ]
        """
        # TODO remove redundancy : n_obsv and n_steps
        if t is None:
            t = X['time']
        min_time = timeline_start if timeline_start is not None else int(t.min())
        max_time = timeline_end if timeline_end is not None else int(t.max())
        contiguous_times = pd.Series(data=range(min_time, max_time + 1), name=t.name)
        n_steps = len(contiguous_times)

        simulation = dict(actions=list(),
                          covariates=list(),
                          time=list(),
                          pat_id=X.iloc[0][self.id_col],
                          )
        X = X.drop([self.id_col, t.name], axis=1)
        a = a.drop([self.id_col], axis=1)
        y = y.drop([self.id_col], axis=1) if y else None

        X = np.expand_dims(X.to_numpy(), axis=0)
        a = np.expand_dims(a.to_numpy(), axis=0)
        y = np.expand_dims(y.to_numpy(), axis=0) if y else None
        lengths = np.array([len(X), ])

        # assert y.shape[1] >= self.n_obsv, "n_obsv bigger than observed data"

        X = X.repeat(self.n_sims, 0)                 # N_sims * T * F
        a = a.repeat(self.n_sims, 0)                 # N_sims * T * F
        y = y.repeat(self.n_sims, 0) if y else None  # N_sims * T * F

        x_t = X[:, :self.n_obsv, :]
        a_t = a[:, :self.n_obsv, :]
        y_t = y[:, :self.n_obsv, :] if y else None

        act_t = treatment_strategy(prev_x=x_t[:, -1, :], all_x=x_t[:, :-1, :], prev_a=a_t[:, -1, :])
        a_t = np.concatenate((a_t[:, :-1, :], act_t), axis=1)

        # init all the models
        self._init_models()  # TODO

        # Simulate
        for _idx in range(self.n_steps):
            _idx = _idx + self.n_obsv
            sim_t = self._predict(x_t, a_t, y_t, _idx)  # TODO add support for RNN later

            # update x_t and a_t
            x_t = np.concatenate([x_t, sim_t], axis=1)

            # get a new treatment_action
            act_t = treatment_strategy(prev_x=x_t[:, -1, :], all_x=x_t[:, :-1, :], prev_a=a_t[:, -1, :])
            # a_t = np.concatenate((a_t[:, :-1, :], act_t), axis=1)
            a_t = np.concatenate([a_t, act_t], axis=1)
            
            simulation['actions'].append(act_t)
            simulation['covariates'].append(sim_t)
            simulation['time'].append(_idx)

        simulation['actions'] = np.concatenate(simulation['actions'], axis=1)  # N_sim * n_steps * F-act
        simulation['covariates'] = np.concatenate(simulation['covariates'], axis=1)  # N_sim * n_steps * F-act
        return simulation

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series = None,
                                    y: Optional[Any] = None,
                                    treatment_strategy: TreatmentStrategy = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:

        """
            Calculates population outcome for each covariate across 't' times.
            Backlog: Support for multiple treatment strategies. Like a list of  "always_treated"  and "never_treated".
        """
        individual_prediction_curves = self.estimate_individual_outcome(X=X,
                                                                        a=a,
                                                                        t=t,
                                                                        y=y,
                                                                        treatment_strategy=treatment_strategy,
                                                                        timeline_start=timeline_start,
                                                                        timeline_end=timeline_end)

        # Averaging across time
        res = individual_prediction_curves.groupby(individual_prediction_curves.time).mean()
        return res

    def _apply_noise(self, out, t, box_type='float'):

        # TODO: Convert Torch to np
        # TODO: infer box_type from data

        residuals = self.resid_val
        mode = self.mode if self.mode else "empirical"

        """adding noise to a box output
           out: bs * 1 * <F-box>
           t: time
           box: box id
           mode: mode of operation
           """
        #  _device = out.device
        # (first_box_var, last_box_var, _, _) = self.model.box_dim[box]
        #  box_type = self.model.box_type[box]

        if box_type == 'boolean':
            sim = (np.random.rand(*out.shape) < out).astype('int')
            #  sim = T.from_numpy(_sim).float().to(out.device)  # , requires_grad=False).to(device)
            #  sim.requires_grad_(False)

        elif box_type == 'float':
            if mode == 'empirical':
                _resid_dist = residuals[t, :, :]  # bs * F
                sim_noise = self._batch_choice(_resid_dist, num_samples=out.shape[0])
                #  _sim_noise_t = T.from_numpy(sim_noise).float()  # , requires_grad=False)  # bs * <F-box>
                #  _sim_noise_t.requires_grad_(False)

                #TODO: change the unsqueeze to numpy to
                _sim_noise_t = sim_noise.unsqueeze_(1)  # bs * 1 * <F-box>
                # clamping values between 0 and 1
                # sim.clamp_(0, 1)
            elif mode == 'normal':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                _sim_noise_t = 1.0 * np.random.randn(*out.shape)
            elif mode == 'tdist':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                #  _dist =  T.distributions.StudentT(df=residuals.shape[1] - 1)
                raise NotImplementedError()
                # TODO: chech the shape of the generated t-dist.
                # https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html
                _sim_noise_t = 1.0 * np.random.standard_t(df=residuals.shape[1] -1, size=out.shape)  # _dist.sample(out.shape)
            elif mode == 'emp_std':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                _std = residuals[t, :, :].std(axis=0)
                # TODO: chech the shape of the generated random variable
                _sim_noise_t = _std * np.random.randn(*out.shape)
            elif mode == 'emp_mean_std':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                _std = residuals[t, :, :].std(axis=0)
                _mean = residuals[t, :, :].mean(axis=0)
                # TODO: chech the shape of the generated random variable
                _sim_noise_t = _mean + _std * np.random.randn(*out.shape)
            #  _sim_noise_t = _sim_noise_t.to(_device)
            sim = out + _sim_noise_t
        else:
            raise AttributeError()
        return sim

    def _batch_choice(self, arr, num_samples):
        val_arr = arr[~np.isnan(arr).any(axis=1)]
        idx = val_arr.shape[0]
        choice = np.random.choice(idx, num_samples)
        sample = val_arr[choice, :]
        return sample

    def _prepare_data(self, X, a, t, y):

        data = X.join(a)
        for cov in self.covariate_cols:
            data['prev_' + cov] = data.groupby(self.id_col)[cov].shift(1)
        for a in self.treatment_cols:
            data['prev_' + a] = data.groupby(self.id_col)[a].shift(1)

        data.dropna(inplace=True)  # dropping first row
        data = data[self.all_cols]

        X_treatment, Y_treatment = self._extract_treatment_model_data(data,
                                                                      self.all_cols,
                                                                      self.treatment_cols
                                                                      )
        X_covariates, Y_covariates = self._extract_covariate_models_data(data,
                                                                         self.all_cols,
                                                                         self.prev_covariate_cols,
                                                                         self.prev_treatment_cols
                                                                         )
        X_outcome, Y_outcome = self._extract_outcome_model_data(data,
                                                                self.all_cols,
                                                                self.y_col
                                                                ) if y else None, None

        return X_treatment, Y_treatment, X_covariates, Y_covariates, X_outcome, Y_outcome

    def _init_models(self, *args, **kwargs):
        return

        # TODO Write actual code for model initialization
        # TODO Initially do it for only sklearn
        # for model in self.covariate_models:
        #     model.init()
        # self.treatment_model.init()
        # self.outcome_model.init()

        # self.treatment_model = sklearn.base.clone(self.treatment_model)
        # todo add args

    def _predict(self, X, a, y, step):

        _input = np.concatenate((X, a), axis=2)
        # roll/shift right to number of 'col_dim'
        _input = np.concatenate([_input, np.roll(_input, -_input.shape[2])], axis=2)
        _input = pd.DataFrame(_input[:, -1, :], columns=self.all_cols)

        X_sim = []
        d_type_dict = dict(_input.dtypes)
        default_cols = self.prev_covariate_cols + self.prev_treatment_cols
        _input = _input[default_cols]
        for i, cov in enumerate(self.covariate_models):
            if d_type_dict[cov] == 'float':
                _pred = self.covariate_models[cov].predict(_input)
            elif d_type_dict[cov] == 'bool':
                _pred = self.covariate_models[cov].predict_proba(_input)
            else:
                raise ValueError("Data type error. {0}, is not supported for {1}".format(d_type_dict[cov], cov))

            _pred = np.expand_dims(_pred, axis=1)
            _pred = self._apply_noise(_pred, step, d_type_dict[cov])  # bs * 1 * 1
            _input = pd.DataFrame(np.concatenate((_input, _pred), axis=1), columns=self.all_cols[:len(default_cols)+i+1])
            X_sim.append(_pred)

        X_sim = np.concatenate(X_sim, axis=1)
        X_sim = np.expand_dims(X_sim, axis=1)
        return X_sim

    def _extract_treatment_model_data(self, X, all_columns, treatment_cols):
        _columns = [col for col in all_columns if col not in treatment_cols]
        X_treatment = X[_columns]
        Y_treatment = X[treatment_cols]
        return X_treatment, Y_treatment

    def _extract_covariate_models_data(self, X, all_columns, prev_covariates, prev_treatments):
        X_covariates = {}
        Y_covariates = {}
        default_cols = prev_covariates + prev_treatments
        for i, cov in enumerate(self.covariate_models):
            _columns = all_columns[: len(default_cols) + i]
            X_covariates[cov] = X[_columns]
            Y_covariates[cov] = X[cov]
        return X_covariates, Y_covariates

    def _extract_outcome_model_data(self, X, all_columns, y_cols):
        _columns = [col for col in all_columns if col not in y_cols]
        X_outcome = X[_columns]
        Y_outcome = X[y_cols]
        return X_outcome, Y_outcome

    def _set_cols(self, X, a, t=None, y=None):
        self.covariate_cols = list(self.covariate_models.keys())
        self.treatment_cols = list('A')
        self.time_col = t.name if t else 'time'
        self.y_col = y.name if y else None

        self.prev_covariate_cols = ['prev_' + cov for cov in self.covariate_cols]
        self.prev_treatment_cols = ['prev_' + a for a in self.treatment_cols]
        # self.index_cols = [self.id_col, time_col]
        self.all_cols = self.prev_covariate_cols + self.prev_treatment_cols + self.covariate_cols + self.treatment_cols





