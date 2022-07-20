#!/usr/bin/env python3

import pandas as pd
from typing import Optional, Any
from causallib.time_varying.base import GMethodBase
from causallib.time_varying.treament_strategy import TreatmentStrategy
from causallib.utils import general_tools as g_tools
import numpy as np
import torch as T  # TODO remove torch dependency


class GFormula(GMethodBase):
    """
        GFormula class that is based on Monte Carlo Simulation for creating the noise.
    """
    def __init__(self, treatment_model, covariate_models, outcome_model, refit_models, seed, n_obsv, n_sims, n_steps, mode, resid_val, group_by):
        super(GFormula, self).__init__(treatment_model, covariate_models, outcome_model, refit_models)
        self.seed = seed
        self.n_obsv = n_obsv
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.mode = mode
        self.resid_val = resid_val
        self.group_by = group_by


    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: Optional[pd.Series] = None,
            y: Optional[Any] = None,
            refit_models: bool = True,
            **kwargs
            ):

        raise NotImplementedError

        if kwargs is None:
            kwargs = {}

        # TODO More to work on preparing data to be fed into the model
        treatment_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.treatment_model.learner)
        if refit_models or treatment_model_is_not_fitted:
            self.treatment_model.fit(X, a, y, **kwargs)

        for cov in self.covariate_models:
            cov_model = self.covariate_models[cov]
            cov_model_is_not_fitted = not g_tools.check_learner_is_fitted(cov_model.learner)

            if refit_models or cov_model_is_not_fitted:
                cov_model.fit(X, a, y, **kwargs)

        self.outcome_model.fit(X, a, y, **kwargs)
        return

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
            Returns individual estimated curves for each subject row in X/a/t

        Steps:
            1. For each sample,
                i. get the simulation outcome (n_sim * n_steps * X-dim) from _estimate_individual_outcome_single_sample
                ii. take mean across n_sim and then drop that axis, which will result (n_steps * X-dim)
            2. The result from #1 is appended to list for all the samples, i.e n_samples * n_steps * X-dim
            3. Repeat #1 and #2 for treatment (a) as well
            4. Finally, merge these two results from #2 and #3 across last dimension,
                which results (n_sub * n_steps * (X-dim + a-dim))
            5. Finally, return the result from #4
            #TODO may be add more details
        """

        unique_sample_ids = X[self.group_by].unique()
        all_global_sim = []
        all_global_actions = []
        for sample_id in unique_sample_ids:
            sample_data = X.loc[X[self.group_by] == sample_id]
            sample_a = a.loc[a[self.group_by] == sample_id]
            sample_y = y.loc[y[self.group_by] == sample_id]
            sample_sim = self._estimate_individual_outcome_single_sample(X=sample_data,
                                                                         a=sample_a,
                                                                         t=t,
                                                                         y=sample_y,
                                                                         treatment_strategy=treatment_strategy,
                                                                         timeline_start=timeline_start,
                                                                         timeline_end=timeline_end)

            all_global_sim.append(sample_sim['covariates'].mean(axis=0).squeeze(axis=0))
            all_global_actions.append(sample_sim['actions'].mean(axis=0).squeeze(axis=0))
        # TODO add column names
        # TODO add sample_id column
        res = pd.concat([pd.DataFrame(all_global_sim), pd.DataFrame(all_global_actions)], axis=2)
        return res

    def _estimate_individual_outcome_single_sample(self, X, a, t, y, treatment_strategy, timeline_start, timeline_end) -> dict:
        """
            Simulates the outcome for each sample across 't' steps.
            Returns:
                 sample_sim = [
                {
                    'actions' = N_sim * n_steps * dim(act)
                    'covariates' = N_sim * n_steps * dim(X-cov)
                    'time' = 1 * n_steps
                    'pat_id' = str
                }
            ]
        """
        min_time = timeline_start if timeline_start is not None else int(t.min())
        max_time = timeline_end if timeline_end is not None else int(t.max())
        contiguous_times = pd.Series(data=range(min_time, max_time + 1), name=t.name)
        n_steps = len(contiguous_times)

        X = X.unsqueeze(0)
        a = a.unsqueeze(0)
        lengths = np.array([len(X), ])

        assert y.shape[1] >= self.n_obsv, "n_obsv bigger than observed data"

        simulation = dict(actions=list(),
                          covariates=list(),
                          time=list(),
                          pat_id=X['id'][0],
                          )

        X = X.repeat(self.n_sims, 1, 1)  # N_sims * T * F
        a = a.repeat(self.n_sims, 1, 1)  # N_sims * T * F
        lengths = lengths.repeat(self.n_sims)

        t = self.n_obsv
        x_t = X[:, :t, :].clone()  # .unsqueeze(1)
        a_t = a[:, :t, :].clone()  # .unsqueeze(1)
        act_t = treatment_strategy(x_t[:, -1, :], x_t[:, :-1, :], a_t[:, -1, :])
        a_t = T.cat(a_t[:, -1, :], act_t)

        # init all the models
        self._init_models()  # TODO

        # Simulate
        with T.no_grad():
            for _idx in range(self.n_steps):
                # TODO add support for RNN later
                sim_t, act_t = self._predict(x_t[:, -1, :], a_t, n_steps, self.n_obsv, treatment_strategy)  # TODO
                simulation['actions'].append(act_t)
                simulation['covariates'].append(sim_t)
                simulation['time'].append(t)

                # update x_t and a_t
                x_t = np.concatenate([x_t, sim_t], axis=1)
                a_t = np.concatenate([a_t, act_t], axis=1)
                # if t <= self.n_obsv:
                #     print(T.cat(simulation['covariates'], dim=1).squeeze())

            simulation['actions'] = np.concatenate(simulation['actions'], axis=1)  # N_sim * n_steps * F-act
            simulation['covariates'] = np.concatenate(simulation['covariates'], axis=1)  # N_sim * n_steps * F-act
        return simulation

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    treatment_strategy: TreatmentStrategy = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:

        """
            Calculates population outcome for each subgroup stratified by treatment assignment.
            Backlog: Support for multiple treatment strategies. Like a list of  "always_treated"  and "never_treated".
        """
        individual_prediction_curves = self.estimate_individual_outcome(X=X,
                                                                        a=a,
                                                                        t=t,
                                                                        y=y,
                                                                        treatment_strategy=treatment_strategy,
                                                                        timeline_start=timeline_start,
                                                                        timeline_end=timeline_end)
        res = individual_prediction_curves.mean(axis=0).squeeze(axis=0)  # returns n_steps * (X-dim * a-dim)
        res = pd.DataFrame(res)
        return res

    def _apply_noise(self, out, t, box_type='float'):
        raise NotImplementedError()

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
        _device = out.device
        # (first_box_var, last_box_var, _, _) = self.model.box_dim[box]
        #  box_type = self.model.box_type[box]

        if box_type == 'boolean':
            _sim = (np.random.rand(*out.shape) < out.data.cpu().numpy()).astype('int')
            sim = T.from_numpy(_sim).float().to(out.device)  # , requires_grad=False).to(device)
            sim.requires_grad_(False)

        elif box_type == 'float':
            if mode == 'empirical':
                _resid_dist = residuals[t, :, :]  # bs * F
                sim_noise = self._batch_choice(_resid_dist, num_samples=out.shape[0])
                _sim_noise_t = T.from_numpy(sim_noise).float()  # , requires_grad=False)  # bs * <F-box>
                _sim_noise_t.requires_grad_(False)
                _sim_noise_t.unsqueeze_(1)  # bs * 1 * <F-box>
                # clamping values between 0 and 1
                # sim.clamp_(0, 1)
            elif mode == 'normal':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                _sim_noise_t = 1.0 * T.randn(*out.shape)
            elif mode == 'tdist':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                _dist = T.distributions.StudentT(df=residuals.shape[1] - 1)
                _sim_noise_t = 1.0 * _dist.sample(out.shape)
            elif mode == 'emp_std':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                _std = T.Tensor(residuals[t, :, :].std(axis=0))
                _sim_noise_t = _std * T.randn(*out.shape)
            elif mode == 'emp_mean_std':
                #  import ipdb; ipdb.set_trace()  # BREAKPOINT
                _std = T.Tensor(residuals[t, :, :].std(axis=0))
                _mean = T.Tensor(residuals[t, :, :].mean(axis=0))
                _sim_noise_t = _mean + _std * T.randn(*out.shape)
            _sim_noise_t = _sim_noise_t.to(_device)
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

    def _prepare_data(self, X, a):
        return NotImplementedError()
        # TODO Write actual code for data manipulation
        X = pd.concat([a, X], join="outer", axis="columns")
        return X

    def _init_models(self):
        raise NotImplementedError()

        # TODO Write actual code for model initialization
        # TODO Initially do it for only sklearn
        for model in self.covariate_models:
            model.init()
        self.treatment_model.init()
        self.outcome_model.init()

    def _predict(self, X, a, t, n_margin, treatment_strategy):
        raise NotImplementedError()

        # TODO Convert torch to np
        # TODO Debug with actual sklearn model and data

        all_cov = _prepare_data(X, a)
        d_type_dict = dict(all_cov.dtypes)

        for cov in self.covariate_models:
            _input = all_cov.drop(cov, axis=1)
            if d_type_dict[cov] == 'float':
                _pred = self.covariate_models[cov].predict(_input)
            elif d_type_dict[cov] == 'bool':
                _pred = self.covariate_models[cov].predict_proba(_input)
            else:
                raise ValueError("Data type error. {0}, is not supported".format(d_type_dict[cov]))

            if t < n_margin - 1:
                sim_t = _pred[:, -1, :].unsqueeze(1)
                # concatenate newly simulated value of each covariate in original input maintaining order
                # _input =
                # act_t = take the next action
            else:
                sim_t = self._apply_noise(_pred[:, -1, :].unsqueeze(1), t)  # bs * 1 * 1
                # concatenate newly simulated value of each covariate in original input maintaining order
                # _input =
                act_t = treatment_strategy(_input[:, -1, :], _input[:, :, :], _input[:, -1, 1])
            # concatenate act_t in original input
            # _input =

        a_pred = self.treatment_model.predict(_input)
        # drop treatment from _input
        sim_all_cov = _input.drop(a.name, axis=1)
        return sim_all_cov, a_pred
