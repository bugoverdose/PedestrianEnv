import numpy as np
import pandas as pd

from adopy.base import Engine as EngineADO
# from adopy.functions import get_random_design_index
from adopy.functions import get_nearest_grid_index
from scipy.special import logsumexp

class Engine(EngineADO):
    def __init__(self, task, model, grid_design, grid_param,grid_response, lambda_et=1, epsilon=0.001):
        super(Engine, self).__init__(
            task, model, grid_design, grid_param, grid_response)
        self.lambda_et = 1
        self.epsilon = 0.001

    def reset(self):
        """
        Reset the engine as in the initial state.
        """
        self._log_lik = None
        self._marg_log_lik = None
        self._ent = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self.log_post = np.copy(self.log_prior)
        self._update_mutual_info()

        self.eligibility_trace = np.zeros(self.grid_design.shape[0])

    def get_design(self, kind='optimal'):
        if kind == 'optimal':
            self._update_mutual_info()

            # Compute design weights
            w_design = np.exp(np.log(self.epsilon) * self.eligibility_trace)

            # Choose a (sub)optimal design maximizing weighted mutual info
            idx_design = np.argmax(self.mutual_info * w_design)

        elif kind == 'random':
            idx_design = np.random.randint(self.n_d)

        else:
            raise ValueError(
                'The argument kind should be "optimal" or "random".')

        return self.grid_design.iloc[idx_design].to_dict()

    def update(self, design, response):

        if isinstance(design, list) != isinstance(response, list):
            raise ValueError(
                'The number of observations (pairs of design and response) '
                'should be matched in the design and response arguments.')

        if isinstance(design, list) and isinstance(response, list):
            if len(design) != len(response):
                raise ValueError(
                    'The length of designs and responses should be the same.')

            _designs = [
                pd.Series(d, index=self.task.designs, dtype=self.dtype)
                for d in design
            ]

            _responses = [
                pd.Series(r, index=self.task.responses, dtype=self.dtype)
                for r in response
            ]

        else:
            _designs = [
                pd.Series(design, index=self.task.designs, dtype=self.dtype)
            ]

            _responses = [
                pd.Series(response, index=self.task.responses,
                          dtype=self.dtype)
            ]

        for d, r in zip(_designs, _responses):
            i_d = get_nearest_grid_index(d.values, self.grid_design.values)
            i_y = get_nearest_grid_index(r.values, self.grid_response.values)
            self.log_post += self.log_lik[i_d, :, i_y]


        if self.lambda_et:
            self.eligibility_trace *= self.lambda_et
            self.eligibility_trace[i_d] += 1

        self.log_post -= logsumexp(self.log_post)

        self._marg_log_lik = None
        self._ent_marg = None
        self._ent_cond = None
        self._mutual_info = None

        self._update_mutual_info()