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
    def __init__(self, treatment_model, covariate_models, outcome_model, refit_models, seed, n_obsv, n_sims, n_steps, mode, resid_val):
        super(GFormula, self).__init__(treatment_model, covariate_models, outcome_model, refit_models)
        self.seed = seed
        self.n_obsv = n_obsv
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.mode = mode
        self.resid_val = resid_val


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

        raise NotImplementedError
        group_by_field = 'pat_id' # TODO define somewhere or give user flexibility
        unique_sample_ids = X[group_by_field].unique()
        all_sim = []
        for sample_id in unique_sample_ids:
            sample_data = X.loc[X[group_by_field] == sample_id]
            sample_a = a.loc[a[group_by_field] == sample_id]
            sample_y = y.loc[y[group_by_field] == sample_id]
            sample_sim = self._estimate_individual_outcome_single_sample(X=sample_data,
                                                                                           a=sample_a,
                                                                                           t=t,
                                                                                           y=sample_y,
                                                                                           timeline_start=timeline_start,
                                                                                           timeline_end=timeline_end)
            """
            sample_sim = [
                {
                    'actions' = N_sim * n_steps * dim(act)
                    'covariates' = N_sim * n_steps * dim(X-cov)
                    'prediction' =  N_sim * n_steps * dim(X-cov)
                    'time' = 1 * n_steps
                    'pat_id' = str
                }            
            ]"""
            all_sim.append(sample_sim)

        res = {}
        """ 
        assume,
         dim(X-cov) = 3 with x1, x2, x3 covariates, we'll compute 
         all_global_sim[cov] = N_sim * n_steps * 1, for each cov
        """
        all_global = {}
        for i, cov in enumerate(self.covariate_models):
            cov_pos = i-len(self.covariate_models)  # position of covariate
            each_cov = [tmp['covariates'][:, :, cov_pos] for tmp in all_sim]  # for x1
            all_global[cov] = T.cat([tmp.squeeze(-1) for tmp in each_cov], axis=0).to('cpu')  # for x1
            res[cov+'_sim'] = all_global_sim[cov].mean(axis=0)

        all_global_pred = {}
        for i, cov in enumerate(self.covariate_models):
            cov_pos = i - len(self.covariate_models)
            each_cov = [tmp['prediction'][:, :, cov_pos] for tmp in all_sim]  # for x1
            all_global_pred[cov] = T.cat([tmp.squeeze(-1) for tmp in each_cov], axis=0).to('cpu')  # for x1
            res[cov+'_pred'] = all_global_pred[cov].mean(axis=0)

        all_global_ground = {}
        for i, cov in enumerate(self.covariate_models):
            cov_pos = i - len(self.covariate_models)
            each_cov = [tmp['prediction'][:, :, cov_pos] for tmp in all_sim]  # for x1
            all_global_ground[cov] = T.cat([tmp.squeeze(-1) for tmp in each_cov], axis=0).to('cpu')  # for x1
            res[cov+'_ground'] = all_global_ground[cov].mean(axis=0)

        #TODO do it for "a" as well

        return pd.DataFrame(res)

    def _estimate_individual_outcome_single_sample(self, X, a, t, y, treatment_strategy) -> pd.DataFrame:

        # min_time = timeline_start if timeline_start is not None else int(t.min())
        # max_time = timeline_end if timeline_end is not None else int(t.max())
        # contiguous_times = pd.Series(data=range(min_time, max_time + 1), name=t.name)  # contiguous time steps for inference
        # n_steps = len(contiguous_times)
        # unique_treatment_values = a.unique()
        # res = pd.DataFrame()
        # if self.seed is not None:
        #     pl.seed_everything(self.seed)
        simulation = dict(actions=list(),
                          # expectations=list(),
                          prediction=list(),
                          ground_truth=list(),
                          covariates=list(),
                          time=list(),
                          pat_id=str,
                          )

        n_margin = self.n_obsv
        n_obsv = 1
        X = X.unsqueeze(0)
        a = a.unsqueeze(0)
        lengths = np.array([len(X), ])
        pat_id = X['id'][0]

        assert y.shape[1] >= n_obsv, "n_obsv bigger than observed data"

        simulation["ground_truth"] = y.clone()[:, n_obsv - 1: n_obsv - 1 + self.n_steps, ]  # test dataset from n_obsv
        simulation["pat_id"] = pat_id

        X = X.repeat(self.n_sims, 1, 1)  # N_sims * T * F
        a = a.repeat(self.n_sims, 1, 1)  # N_sims * T * F
        lengths = lengths.repeat(self.n_sims)

        t = n_obsv
        x_t = X[:, :t, :].clone()  # .unsqueeze(1)
        a_t = a[:, :t, :].clone()  # .unsqueeze(1)

        act_t = treatment_strategy(x_t[:, -1, :], x_t[:, :-1, :], a_t[:, -1, :])

        x_t = T.cat([x_t, a_t], axis=1)
        last_t = T.cat([x_t[:, -1, :-1], act_t], axis=1)  # bs * F
        x_t = T.cat([x_t[:, :-1, :], last_t.unsqueeze(1)], axis=1)  # bs * n_obsv * F

        # init all the models
        self._init_models()  # TODO

        # Simulate
        with T.no_grad():
            for _idx in range(self.n_steps):
                t = _idx + n_obsv - 1  # + 1
                # out = self._predict(x_t, a)  # TODO
                # # this is X --> becomes prev_X for next round
                # if t < n_margin - 1:
                #     sim_t = out[:, -1, :].unsqueeze(1)
                #     new_t = X[:, (t + 1), :].unsqueeze(1)
                #     act_t = new_t[..., -1].view(-1, 1, 1)  # bs * 1 * 1
                # else:
                #     sim_t = self._apply_noise(out[:, -1, :].unsqueeze(1), t)  # bs * 1 * 1
                #
                #     # this is A becomes prev_A
                #     prev_act = x_t[:, -1, -1].unsqueeze(1).unsqueeze(1)  # bs * 1 * 1
                #     act_t = treatment_strategy(sim_t[:, -1, 0], sim_t[:, :, 0], a_t[:, -1, 1])
                #     new_t = T.cat([sim_t, prev_act, act_t], axis=-1)  # .unsqueeze(1) # bs * 1 * F
                # x_t = T.cat([x_t, new_t], axis=1)  # bs * (t + 1) * F
                sim_t, act_t = self._predict(x_t[:, -1, :], a, t, n_margin, treatment_strategy)  # TODO

                # prediction
                actual_t = X[:, :(t + 1), :]
                # out = self._predict(actual_t, a, t)
                out = self._predict(actual_t, a, t, n_margin, treatment_strategy)

                pred_t = out[:, -1, :].unsqueeze(1)

                simulation['actions'].append(act_t)
                simulation['covariates'].append(sim_t)
                simulation['prediction'].append(pred_t)
                simulation['time'].append(t)
                if t <= n_margin:  # and debug:  # == 11:
                    # print(T.allclose(hidden[0], hidden_p[0]))
                    # print(T.allclose(hidden[1], hidden_p[1]))
                    print(T.cat(simulation['prediction'], dim=1).squeeze())
                    print(T.cat(simulation['covariates'], dim=1).squeeze())

            simulation['actions'] = T.cat(simulation['actions'], dim=1)  # N_sim * n_steps * F-act
            simulation['prediction'] = T.cat(simulation['prediction'], dim=1)
            simulation['covariates'] = T.cat(simulation['covariates'], dim=1)  # N_sim * n_steps * F-act
        # return simulation

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    treatment_strategy: TreatmentStrategy = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:

        unique_treatment_values = a.unique()
        res = {}
        for treatment_value in unique_treatment_values:
            assignment = pd.Series(data=treatment_value, index=X.index, name=a.name)
            individual_prediction_curves = self.estimate_individual_outcome(X=X, a=assignment, t=t, y=y,
                                                                            treatment_strategy=treatment_strategy,
                                                                            timeline_start=timeline_start,
                                                                            timeline_end=timeline_end)
            # for key in individual_prediction_curves:
            res[treatment_value] = individual_prediction_curves
        res = pd.DataFrame(res)

        # Setting index/column names
        res.index.name = t.name
        res.columns.name = a.name
        return res

    def _apply_noise(self, out, t, box_type='float'):
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
        return NotImplementedError
        X = pd.concat([a, X], join="outer", axis="columns")
        return X

    def _init_models(self):
        raise NotImplementedError

        for model in self.covariate_models:
            model.init()
        self.treatment_model.init()
        self.outcome_model.init()

    def _predict(self, X, a, t, n_margin, treatment_strategy):
        raise NotImplementedError

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
