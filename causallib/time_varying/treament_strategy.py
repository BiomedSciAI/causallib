from abc import abstractmethod, ABC
from typing import Callable
import numpy as np
import torch as T


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
        _device = prev_x.device
        raise NotImplementedError

        import torch as T  # TODO Need to remove pytorch dependency
        if self.inverse_transform is not None:
            prev_x = T.Tensor(self.inverse_transform(prev_x.data.cpu())).to(_device)
            all_x = T.Tensor(self.inverse_transform(all_x.data.cpu())).to(_device)
        # rbinom(1,1,invlogit((X[i-1]-mean_x)/10.-A[i-1]))
        x = (prev_x - all_x.mean(axis=1)) / 10. - prev_a
        p = T.exp(x) / (1 + T.exp(x))
        out = T.bernoulli(p)  # T.round(p) #
        return out.unsqueeze(1)


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




