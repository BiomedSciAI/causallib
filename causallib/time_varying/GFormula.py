#!/usr/bin/env python3

import pandas as pd
from typing import Optional, Any, Callable
from causallib.time_varying.base import GMethodBase
from causallib.utils import general_tools as g_tools
from causallib.time_varying.treament_strategy import TreatmentStrategy
import numpy as np


class GFormula(GMethodBase):
    """
        GFormula class that is based on Monte Carlo Simulation for creating the noise.
    """
    def __init__(self, outcome_model, treatment_model, covariate_models, refit_models, seed, n_obsv, n_sims, n_steps, mode, resid_val):
        super(GFormula, self).__init__(outcome_model, treatment_model, covariate_models, refit_models)
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

        #TODO More to work on preparing data to be fed into the model
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

        # min_time = timeline_start if timeline_start is not None else int(t.min())
        # max_time = timeline_end if timeline_end is not None else int(t.max())
        # contiguous_times = pd.Series(data=range(min_time, max_time + 1), name=t.name)  # contiguous time steps for inference
        # n_steps = len(contiguous_times)
        # unique_treatment_values = a.unique()
        # res = pd.DataFrame()


        n_margin = self.n_obsv
        n_obsv = 1

        # if self.seed is not None:
        #     pl.seed_everything(self.seed)

        # if model is None:
        #     raise AttributeError("model needs to be passed")
            #  model = pl_model

        # source_data, target_data, lengths, pat_id = pat_data
        source_data = X['prev_A', 'prev_X', 'A']
        target_data = y
        lengths = np.array([len(source_data), ])

        source_data = X.unsqueeze(0)
        target_data = y.unsqueeze(0)
        pat_id = None

        assert target_data.shape[1] >= n_obsv, "n_obsv bigger than observed data"

        # Starting simulation
        simulation = dict(actions=list(),
                          # expectations=list(),
                          prediction=list(),
                          ground_truth=list(),
                          covariates=list(),
                          time=list(),
                          pat_id=pat_id,
                          )
        n_obsv = 1
        #  import ipdb; ipdb.set_trace()  # BREAKPOINT
        simulation["ground_truth"] = target_data.clone()[:, n_obsv - 1: n_obsv - 1 + self.n_steps, ]  # test dataset from n_obsv

        source_data = source_data.repeat(self.n_sims, 1, 1)  # N_sims * T * F
        lengths = lengths.repeat(self.n_sims)

        # till observation window: Forward in teacher-mode
        # required to get the hidden variables from previous steps
        # T: 0 1 2 3 4 5
        # n_obsv = 4
        # source data: 0 1 2 3
        # target data: 1 2 3 4

        # last source_data (t=3): cov_<obsv>, treatment_<obsv>
        # last target_data (t=4): cov_<obsv>, treatment_<counterfactual>
        # Updating target data to reflect countrefactual regime

        t = n_obsv

        # Update action at t
        # ['prev_X', 'prev_A', 'A']
        source_t = source_data[:, :t, :].clone()  # .unsqueeze(1)

        act_t = treatment_strategy.get_action(X[:, -1, :], source_t[:, :-1, :], a[:, -1, :])

        import torch as T
        last_t = T.cat([source_t[:, -1, :-1], act_t], axis=1)  # bs * F
        source_t = T.cat([source_t[:, :-1, :], last_t.unsqueeze(1)], axis=1)  # bs * n_obsv * F

        # act_t, base_t = self.get_act_base(target_data[:, :-1, :], sim_t, t)

        # Get Simulation at time t
        # sim_t = self._combine_cov_act_base(sim_t, act_t, base_t)

        # Update target data
        # target_data = T.cat([target_data[:, :-1, :], sim_t], dim=1).contiguous()

        # update hidden vector with teacher forcing till t = 3 (n_obsv - 1)
        #FIXME
        # if init_hidden is not None:
        #     if hasattr(model.model, 'init_hidden'):
        #         hidden = model.model.init_hidden(batch_size=n_sims, device=model.device)
        #         hidden_p = model.model.init_hidden(batch_size=n_sims, device=model.device)
        #     else:
        #         # assuming callable
        #         hidden = init_hidden(n_sims, model.model._hidden_size)  # None
        #         hidden_p = init_hidden(n_sims, model.model._hidden_size)  # None
        # else:
        #     hidden = None
        #     hidden_p = None

        # Simulate
        with T.no_grad():
            for _idx in range(self.n_steps):

                if hasattr(self.covariate_models[0].model, 'reset_mask'):
                    self.covariate_models[0].model.reset_mask()
                # import ipdb; ipdb.set_trace();
                if t == 0:
                    (out, hidden) = self.covariate_models[0].forward(source_t, hidden=hidden, lengths=None)  # all data is true till n_obsv
                else:
                    (out, hidden) = self.covariate_models[0].forward(source_t[:, -1, :].unsqueeze(1), hidden=hidden, lengths=None)  # all data is true till n_obsv
                # out.clamp_(min=-2., max=5.)
                # (out, hidden) = model.forward(source_t, hidden=hidden, lengths=None)  # all data is true till n_obsv

                # this is X --> becomes prev_X for next round
                # import ipdb; ipdb.set_trace();
                if t < n_margin - 1:
                    # sim_t = target_data[:, t, :].unsqueeze(1).repeat(n_sims, 1, 1)
                    sim_t = out[:, -1, :].unsqueeze(1)
                    new_t = source_data[:, (t + 1), :].unsqueeze(1)
                    act_t = new_t[..., -1].view(-1, 1, 1)  # bs * 1 * 1
                else:
                    # import ipdb; ipdb.set_trace();
                    sim_t = self._apply_noise(out[:, -1, :].unsqueeze(1), t) # bs * 1 * 1
                    # sim_t = out[:, -1, :].unsqueeze(1)
                    # import ipdb; ipdb.set_trace();

                    # this is A becomes prev_A
                    prev_act = source_t[:, -1, -1].unsqueeze(1).unsqueeze(1)  # bs * 1 * 1

                    act_t = treatment_strategy.get_action(source_t[:, -1, 0], source_t[:, :, 0], source_t[:, -1, 1])
                    new_t = T.cat([sim_t, prev_act, act_t], axis=-1)  # .unsqueeze(1) # bs * 1 * F
                source_t = T.cat([source_t, new_t], axis=1)  # bs * (t + 1) * F

                # hidden = None
                # if t >= 0:  import ipdb; ipdb.set_trace();
                # prediction
                actual_t = source_data[:, :(t + 1), :]
                if t == 0:
                    (out, hidden_p) = self.covariate_models[0].forward(actual_t, hidden=hidden_p, lengths=None)  # all data is true till n_obsv
                else:
                    (out, hidden_p) = self.covariate_models[0].forward(actual_t[:, -1, :].unsqueeze(1), hidden=hidden_p, lengths=None)  # all data is true till n_obsv

                pred_t = out[:, -1, :].unsqueeze(1)
                # hidden_p = None

                simulation['actions'].append(act_t)
                simulation['prediction'].append(pred_t)
                simulation['covariates'].append(sim_t)
                simulation['time'].append(t)
                debug = False
                if (t <= n_margin) and debug:  # == 11:
                    print(T.allclose(hidden[0], hidden_p[0]))
                    print(T.allclose(hidden[1], hidden_p[1]))
                    print(T.cat(simulation['prediction'], dim=1).squeeze())
                    print(T.cat(simulation['covariates'], dim=1).squeeze())

            simulation['actions'] = T.cat(simulation['actions'], dim=1)  # N_sim * n_steps * F-act
            # try:
            #     simulation['unmodeled'] = T.cat(simulation['unmodeled'], dim=1)      # N_sim * n_steps * F-base
            # except TypeError:
            #     pass
            simulation['prediction'] = T.cat(simulation['prediction'], dim=1)
            simulation['covariates'] = T.cat(simulation['covariates'], dim=1)  # N_sim * n_steps * F-act
        # return res

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    treatment_strategy: Callable = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:

        raise NotImplementedError

        unique_treatment_values = a.unique()
        res = {}
        for treatment_value in unique_treatment_values:
            assignment = pd.Series(data=treatment_value, index=X.index, name=a.name)
            individual_survival_curves = self.estimate_individual_outcome(X=X, a=assignment, t=t,
                                                                          timeline_start=timeline_start,
                                                                          timeline_end=timeline_end)
            res[treatment_value] = individual_survival_curves.mean(axis='columns')
        res = pd.DataFrame(res)

        # Setting index/column names
        res.index.name = t.name
        res.columns.name = a.name
        return res

    def _apply_noise(self, out, t, box_type='float'):
        residuals = self.resid_val
        mode = self.mode if self.mode else "empirical"

        import torch as T #TODO remove torch dependency
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

    def _prepare_data(self, X, a, t, y):
        pass

    @staticmethod
    def _predict_trajectory(self, X, a, t) -> pd.DataFrame:
        pass



