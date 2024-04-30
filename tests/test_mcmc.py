# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Testing for the main MCMC class."""

import numpy as np
import pytest

from openmcmc.distribution.distribution import Gamma
from openmcmc.distribution.location_scale import Normal
from openmcmc.mcmc import MCMC
from openmcmc.model import Model
from openmcmc.parameter import LinearCombination, ScaledMatrix
from openmcmc.sampler.sampler import NormalGamma, NormalNormal


@pytest.fixture(name="model")
def fix_model():
    """Fix the model structure to be used in the tests."""
    model = Model(
        [
            Normal(
                "y", mean=LinearCombination(form={"beta": "X"}), precision=ScaledMatrix(matrix="P_tau", scalar="tau")
            ),
            Normal("beta", mean="mu", precision="sigma"),
            Gamma("tau", shape="a", rate="b"),
            Gamma("sigma", shape="c", rate="d"),
        ]
    )
    return model


@pytest.fixture(params=[1, 2, 3], ids=["n_smp=1", "n_smp=2", "n_smp=3"], name="sampler")
def fix_sampler(request, model):
    """Define the set of models to be used in MCMC class."""
    n_samplers = request.param
    sampler = [NormalNormal("beta", model)]

    if n_samplers >= 2:
        sampler.append(NormalGamma("tau", model))
    if n_samplers >= 3:
        sampler.append(NormalGamma("sigma", model))
    return sampler


@pytest.fixture(
    params=[
        (int(0), int(0)),
        (int(2.5), int(1.5)),
        (float(2.5), float(1.5)),
        (np.array([1.1, 3.2, 5.3]), np.array([2.4, 4.1, 6.2])),
        ([1.1, 3.2, 5.3], [2.4, 4.1, 6.2]),
    ],
    ids=["all_zero", "tau_beta_integer", "tau_beta_float", "tau_beta_np_array", "tau_beta_list"],
    name="state",
)
def fix_state(request):
    """Define the initial state for the MCMC."""
    [beta, tau] = request.param
    state = {"count": 0, "beta": beta, "tau": tau, "sigma": 10, "P_tau": np.eye(np.array(beta, ndmin=2).shape[0])}
    return state


@pytest.fixture(
    params=[(0, 4000, 1), (2000, 4000, 5), (0, 6000, 10), (2000, 6000, 1)],
    ids=[
        "n_burn=0,n_iter=4000, n_thin=1",
        "n_burn=non-zero,n_iter=4000,n_thin=5",
        "n_burn=0,n_iter=6000, n_thin=10",
        "n_burn=non-zero,n_iter=6000,n_thin=1",
    ],
    name="mcmc_settings",
)
def fix_mcmc_settings(request):
    """Define the initial state for the MCMC."""
    [n_burn, n_iter, n_thin] = request.param
    fix_mcmc_settings = {"nburn": n_burn, "niter": n_iter, "nthin": n_thin}

    return fix_mcmc_settings


def test_run_mcmc(state: dict, sampler: list, model: Model, mcmc_settings: dict, monkeypatch):
    """Test run_mcmc function Checks size is correct for the output parameters of the function (state and store) based
    on the number of iterations (n_iter) and number of burn (n_burn), i.e.,

    Args:
        state: dictionary
        model: Model input
        mcmc_settings: dictionary of mcmc settings
        monkeypatch object for avoiding computationally expensive mcmc sampler.

    """

    # set up samplers
    def mock_sample(self, state_in):
        state_in["count"] = state_in["count"] + 1
        return state_in

    def mock_store(self, current_state, store, iteration):
        store["count"] = store["count"] + 1
        return store

    def mock_log_p(self, current_state):
        return 0

    monkeypatch.setattr(NormalNormal, "sample", mock_sample)
    monkeypatch.setattr(NormalNormal, "store", mock_store)
    monkeypatch.setattr(NormalGamma, "sample", mock_sample)
    monkeypatch.setattr(NormalGamma, "store", mock_store)
    monkeypatch.setattr(Model, "log_p", mock_log_p)

    M = MCMC(
        state,
        sampler,
        model,
        n_burn=mcmc_settings["nburn"],
        n_iter=mcmc_settings["niter"],
        n_thin=mcmc_settings["nthin"],
    )
    M.store["count"] = 0
    M.run_mcmc()
    assert M.state["count"] == (M.n_iter + M.n_burn) * len(sampler) * M.n_thin
    assert M.store["count"] == M.n_iter * len(sampler)


def test_post_init(state: dict, sampler: list, model: Model, mcmc_settings: dict):
    """This function test __pos__init function to check returned store and state parameters are np.array of the
    dimension n * 1

    Args:
        state: dictionary
        mcmc_settings: integer
        model:

    """
    M = MCMC(state, sampler, model, n_iter=mcmc_settings["niter"])

    assert isinstance(M.state["count"], np.ndarray)
    assert M.state["count"].ndim == 2

    assert isinstance(M.state["beta"], np.ndarray)
    assert M.state["beta"].ndim == 2
    assert (len(M.store) - 1) * (M.store["beta"]).shape[1] == len(sampler) * mcmc_settings["niter"]
    assert M.store["log_post"].size == mcmc_settings["niter"]

    if len(sampler) > 1:
        assert isinstance(M.state["tau"], np.ndarray)
        assert isinstance(M.store["tau"], np.ndarray)
        assert M.state["tau"].ndim == 2
