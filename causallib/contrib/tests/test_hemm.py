import numpy as np
import pandas as pd
import torch
import unittest

from causallib.contrib.hemm import HEMM
from causallib.contrib.hemm.hemm_utilities import genSplits, returnIndices, getMeanandStd
from causallib.contrib.hemm.hemm_outcome_models import genMLPModule, genLinearModule, BalancedNet
from causallib.contrib.hemm.gen_synthetic_data import gen_montecarlo


class TestHemmEstimator(unittest.TestCase):
    def experiment(self, data, i, comp, response, outcome_model, lr, batch_size):
        np.random.seed(0)

        Xtr = data['TRAIN']['x'][:, :, i]
        Ttr = data['TRAIN']['t'][:, i]
        Ytr = data['TRAIN']['yf'][:, i]

        Ytr_ = np.ones_like(Ytr)
        splits = genSplits(Ttr, Ytr_)
        train, dev = returnIndices(splits)
        n = Xtr.shape[0]

        Xte = data['TEST']['x'][:, :, i]
        Tte = data['TEST']['t'][:, i]
        Yte = data['TEST']['yf'][:, i]

        mu, std = getMeanandStd(Xtr)

        Xdev = Xtr[dev]  # Numpy array
        Ydev = torch.from_numpy(Ytr[dev].astype('float64'))
        Tdev = Ttr[dev]  # Numpy array

        Xtr = pd.DataFrame(Xtr[train])  # Train covariates as a data frame
        Ytr = torch.from_numpy(Ytr[train].astype('float64'))
        Ttr = pd.Series(Ttr[train])  # Train treatment assignments as a series

        Xte = torch.from_numpy(Xte.astype('float64'))
        Yte = torch.from_numpy(Yte.astype('float64'))
        Tte = torch.from_numpy(Tte.astype('float64'))

        if outcome_model == 'MLP':
            outcome_model = genMLPModule(Xte.shape[1], Xte.shape[1] / 5, 2)
        elif outcome_model == 'linear':
            outcome_model = genLinearModule(Xte.shape[1], 2)
        elif outcome_model == 'CF':
            outcome_model = BalancedNet(Xte.shape[1], Xte.shape[1], 1)

        estimator = HEMM(
            Xte.shape[1],
            comp,
            mu=mu,
            std=std,
            bc=6,
            lamb=0.,
            spread=0.,
            outcome_model=outcome_model,
            epochs=500,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=1e-4,
            metric='LL',
            response=response,
            imb_fun='wass'
        )
        estimator.fit(Xtr, Ttr, Ytr, validation_data=(Xdev, Tdev, Ydev))

        Xtr = data['TRAIN']['x'][:, :, i]
        Ttr = data['TRAIN']['t'][:, i]
        Ytr = data['TRAIN']['yf'][:, i]

        Xtr = torch.from_numpy(Xtr.astype('float64'))
        Ytr = torch.from_numpy(Ytr.astype('float64'))
        Ttr = torch.from_numpy(Ttr.astype('float64'))

        in_estimations = estimator.estimate_individual_outcome(Xtr, Ttr)
        out_estimations = estimator.estimate_individual_outcome(Xte, Tte)

        group_proba = estimator.get_groups_proba(Xte)
        self.assertEqual(group_proba.shape, (data['TEST']['x'][:, :, 1].shape[0], comp))

        group_assignment = estimator.get_groups(Xte)
        pd.testing.assert_series_equal(group_assignment, group_proba.idxmax(axis="columns"))

        group_effect = estimator.get_groups_effect(Xte, Tte)
        self.assertEqual(group_effect.shape, (comp,))

        group_sizes = estimator.group_sizes(Xte)
        self.assertEqual(len(group_sizes.keys()), comp)

        return in_estimations, out_estimations

    def test_hemm_estimator(self):
        data = gen_montecarlo(1000, 24, 2)
        in_estimations, out_estimations = self.experiment(data, 1, 3, 'cont', 'CF', 1e-3, 10)
        self.assertEqual(in_estimations.shape, (data['TRAIN']['x'][:, :, 1].shape[0], 2))
        self.assertEqual(out_estimations.shape, (data['TEST']['x'][:, :, 1].shape[0], 2))
