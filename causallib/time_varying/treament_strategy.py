from abc import abstractmethod, ABC
from typing import Callable


class TreatmentStrategy(ABC):

    def __call__(self, source_t):
        self.get_action(source_t)

    @abstractmethod
    def get_action(self, source_t):
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
                   source_t):
        raise NotImplementedError

        prev_X = self.source_t[:, -1, 0]
        all_X = self.source_t[:, :, 0]
        prev_A = self.source_t[:, -1, 1]

        # if inverse_transform is not None:
            # prev_x = T.Tensor(inverse_transform(prev_x.data.cpu())).to(_device)
            # all_x = T.Tensor(inverse_transform(all_x.data.cpu())).to(_device)

        # rbinom(1,1,invlogit((X[i-1]-mean_x)/10.-A[i-1]))
        # x = (prev_x - all_x.mean(axis=1)) / 10. - A
        # p = T.exp(x) / (1 + T.exp(x))
        # out = T.bernoulli(p)  # T.round(p) #

        # return out.unsqueeze(1)


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

    def get_action(self,
                   source_t):
        raise NotImplementedError

        # _prob_act_t = self.p * (T.ones_like(source_t[:, -1, 2].unsqueeze(1)))
        # # cf_action(source_t[:, -1, :], n_obsv) # bs * 1
        # act_t = T.bernoulli(_prob_act_t)  # bs * 1




