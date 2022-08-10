from abc import abstractmethod, ABC
from typing import Callable
import numpy as np


class TreatmentStrategy(ABC):

    def __call__(self, prev_x, all_x, prev_a):
        return self.get_action(prev_x, all_x, prev_a)

    @abstractmethod
    def get_action(self,  prev_x, all_x, prev_a):
        """
            Returns the Treatment action
        """
        raise NotImplementedError


class Observational(TreatmentStrategy):
    """
        Observational Treatment Strategy
    """
    def __init__(self,
                 inverse_transform: Callable,
                 **kwargs):
        """
            inverse_transform (Callable): Scaler class that compute the means and std to be used for later scaling.
                                        eg: sklearn.StandardScaler()
            kwargs (dict): Optional kwargs for init call in TreatmentStrategy
        """
        super(TreatmentStrategy, self).__init__(**kwargs)
        self.inverse_transform = inverse_transform

    def get_action(self,
                   prev_x,
                   all_x,
                   prev_a):
        if self.inverse_transform is not None:
            prev_x = self.inverse_transform(prev_x.data)
            all_x = self.inverse_transform(all_x.data)

        # rbinom(1,1,invlogit((X[i-1]-mean_x)/10.-A[i-1]))
        prev_x = prev_x[:, [0]]
        all_x = all_x[:, :, [0]]
        # _diff_x = np.atleast_2d(prev_x - all_x.mean(axis=1))
        x = (prev_x - all_x.mean(axis=1))/10 - prev_a
        p = np.exp(x) / (1 + np.exp(x))
        out = np.random.binomial(1, p)
        out = np.expand_dims(out, axis=2)
        return out


class CFBernoulli(TreatmentStrategy):
    """
           CounterFactual Bernoulli Treatment Strategy
    """
    def __init__(self,
                 p: float = 0.25,
                 **kwargs):
        """
                 p (float): Probability constant to be used in bernoulli distribution
                 kwargs (dict): Optional kwargs for init call in TreatmentStrategy
        """
        super(TreatmentStrategy, self).__init__(**kwargs)
        self.p = p

    def get_action(self, prev_x, all_x, prev_a):
        _prob_act_t = self.p * (np.ones_like(np.expand_dims(prev_a, axis=1)))
        act_t = np.random.binomial(1, _prob_act_t)
        return act_t




