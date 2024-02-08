# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Main MCMC class for mcmc setup."""

from copy import copy
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from tqdm import tqdm

from openmcmc.model import Model
from openmcmc.sampler.metropolis_hastings import MetropolisHastings
from openmcmc.sampler.sampler import MCMCSampler


@dataclass
class MCMC:
    """Class for running Markov Chain Monte Carlo on a Model object to do parameter inference.

    Args:
        state (dict): initial state of sampler any parameters not
                     specified will be sampler from prior distributions
        samplers (list): list of the samplers to be used for each parameter to be estimated
        n_burn (int, optional): number of initial burn in these iterations are not stored, default 5000
        n_iter (int, optional): number of iterations which are stored in store, default 5000

    Attributes:
        state (dict): initial state of sampler any parameters not
                     specified will be sampler from prior distributions
        samplers (list): list of the samplers to be used for each parameter to be estimated.
        n_burn (int): number of initial burn in these iterations are not stored.
        n_iter (int): number of iterations which are stored in store.
        store (dict): dictionary storing MCMC output as np.array for each inference parameter.

    """

    state: dict
    samplers: list[MCMCSampler]
    model: Model
    n_burn: int = 5000
    n_iter: int = 5000
    store: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Convert any state values to at least 2D np.arrays and sample any missing states from the prior distributions, and set up storage arrays for the sampled values.

        Ensures that all elements of the initial state are in an appropriate format for running
        the sampler:
            - sparse matrices are left unchanged.
            - all other data types are coerced (if possible) to np.ndarray.
            - any scalars or existing np.ndarray with only one dimension are forced to be at
                least 2D.

        Also initialises an item in the storage dictionary for each of the sampled values,
        for any data fitted values, and for the log-posterior value.

        """
        self.state = copy(self.state)

        for key, term in self.state.items():
            if sparse.issparse(term):
                continue

            if not isinstance(term, np.ndarray):
                term = np.array(term, ndmin=2, dtype=np.float64)
                if np.shape(term)[0] == 1:
                    term = term.T
            elif term.ndim < 2:
                term = np.atleast_2d(term).T

            self.state[key] = term

        for sampler in self.samplers:
            if sampler.param not in self.state:
                self.state[sampler.param] = sampler.model[sampler.param].rvs(self.state)
            self.store = sampler.init_store(current_state=self.state, store=self.store, n_iterations=self.n_iter)
        if self.model.response is not None:
            for response in self.model.response.keys():
                self.store[response] = np.full(shape=(self.state[response].size, self.n_iter), fill_value=np.nan)
        self.store["log_post"] = np.full(shape=(self.n_iter, 1), fill_value=np.nan)

    def run_mcmc(self):
        """Runs MCMC routine for model specification loops for n_iter+ n_burn iterations sampling the state for each parameter and updating the parameter state.

        Runs a first loop over samplers, and generates a sample for all corresponding variables in the state. Then
        stores the value of each of the sampled parameters in the self.store dictionary, as well as the data fitted
        values and the log-posterior value.

        """
        for i_it in tqdm(range(-self.n_burn, self.n_iter)):
            for sampler in self.samplers:
                self.state = sampler.sample(self.state)

            if i_it < 0:
                continue

            for sampler in self.samplers:
                self.store = sampler.store(current_state=self.state, store=self.store, iteration=i_it)

            self.store["log_post"][i_it] = self.model.log_p(self.state)
            if self.model.response is not None:
                for response, predictor in self.model.response.items():
                    self.store[response][:, [i_it]] = getattr(self.model[response], predictor).predictor(self.state)

        for sampler in self.samplers:
            if isinstance(sampler, MetropolisHastings):
                print(f"{sampler.param}: {sampler.accept_rate.get_acceptance_rate()}")
