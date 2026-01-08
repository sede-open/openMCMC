# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit testing for the sampler module.

For the main sampler tests, a standard model is created (form chosen so that we can test the majority of the conjugate
and non-conjugate sampler types). Then we perform both sampler-agnostic tests, and sampler-specific tests for each case.

"""

from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import gamma, norm

from openmcmc import parameter
from openmcmc.distribution.distribution import Categorical, Gamma
from openmcmc.distribution.location_scale import Normal
from openmcmc.model import Model
from openmcmc.sampler.metropolis_hastings import AcceptRate, ManifoldMALA, RandomWalk
from openmcmc.sampler.sampler import (
    MCMCSampler,
    MixtureAllocation,
    NormalGamma,
    NormalNormal,
)


@pytest.fixture(name="accept_rate")
def fix_accept_rate():
    """Fix the acceptance counter."""
    accept_rate = AcceptRate()
    return accept_rate


def test_increment_accept(accept_rate):
    """Test the increment_accept function in the AcceptRate class.

    Tests that if we initialise the acceptance count to 0 and then call increment_accept(), the resulting acceptance
    count is 1.

    """
    accept_rate.count["accept"] = 0
    accept_rate.increment_accept()
    assert accept_rate.count["accept"] == 1


def test_increment_proposal(accept_rate):
    """Test the increment_proposal function in the AcceptRate class.

    Tests that if we initialise the proposal count to 0 and then call increment_proposal(), the resulting proposal count
    is 1.

    """
    accept_rate.count["proposal"] = 0
    accept_rate.increment_proposal()
    assert accept_rate.count["proposal"] == 1


def test_acceptance_rate(accept_rate):
    """Test the acceptance_rate function of the AcceptRate class.

    Tests that we get an acceptance of 100% if both the proposal and acceptance counts are 1.

    """
    accept_rate.count["proposal"] = 1
    accept_rate.count["accept"] = 1
    assert np.isclose(accept_rate.acceptance_rate, 100.0)


def test_get_acceptance_rate(accept_rate):
    """Test get_acceptance_rate function of the AcceptRate class.

    Tests that get_acceptance_rate() returns 'Acceptance rate 100%' when the proposal and acceptance counts are both set
    to 1.

    """
    accept_rate.count["proposal"] = 1
    accept_rate.count["accept"] = 1
    assert accept_rate.get_acceptance_rate() == "Acceptance rate 100%"


@pytest.fixture(
    params=[(1, 1, 1), (1, 10, 1), (10, 1, 10), (10, 10, 1), (10, 10, 10)],
    ids=["n=1, p=1, c=1", "n=1, p=10, c=1", "n=10, p=1, c=10", "n=10, p=10, c=1", "n=10, p=10, c=10"],
    name="state",
)
def fix_state(request):
    """Fix the state for the MCMC sampler tests."""
    [n, p, n_cat] = request.param

    rng = np.random.default_rng(0)
    state = {}
    state["prefactor_matrix"] = rng.random((n, p))
    state["parameter"] = rng.random((p, 1))
    state["parameter_n"] = rng.random((n, 1))
    state["response"] = state["prefactor_matrix"] @ state["parameter"]
    state["prior_mean"] = rng.random((n_cat, 1))
    state["precision_matrix"] = np.diag(rng.random(n) + 0.1)
    state["prior_precision_vector"] = 0.1 + rng.random(n_cat)
    state["prior_precision_matrix"] = np.eye(p)
    state["prior_precision_scalar"] = 0.1 + rng.random((1, 1))
    state["gamma_shape"] = 1e-3 * np.ones(shape=(n_cat,))
    state["gamma_rate"] = 1e-3 * np.ones(shape=(n_cat,))
    state["allocation"] = rng.integers(low=0, high=n_cat, size=(p, 1))
    state["prior_allocation_prob"] = rng.random((1, n_cat))
    state["prior_allocation_prob"] = state["prior_allocation_prob"] / np.sum(state["prior_allocation_prob"])
    return state


@pytest.fixture(name="model")
def fix_model(state):
    """Create the model to be fed into the sampler object.

    The model contains the following:
        - A Normally-distributed response (with LinearCombination and ScaledMatrix parameters).
        - A Normally parameter prior (with MixtureAllocation parameters).
        - A gamma prior for the parameter prior precision.
        - A categorical distribution prior for the mixture allocation cases.

    """
    if state["prior_mean"].shape[0] == 1:
        state["prior_mean"] = state["prior_mean"] * np.ones(state["parameter"].shape)
        mean_parameter = parameter.Identity(form="prior_mean")
        precision_parameter = parameter.ScaledMatrix(matrix="prior_precision_matrix", scalar="prior_precision_vector")
    else:
        mean_parameter = parameter.MixtureParameterVector(param="prior_mean", allocation="allocation")
        precision_parameter = parameter.MixtureParameterMatrix(param="prior_precision_vector", allocation="allocation")
    model = Model(
        [
            Normal(
                response="response",
                mean=parameter.LinearCombination(form={"parameter": "prefactor_matrix"}),
                precision=parameter.Identity("precision_matrix"),
            ),
            Normal(response="parameter", mean=mean_parameter, precision=precision_parameter),
            Gamma(
                response="prior_precision_vector",
                shape=parameter.Identity(form="gamma_shape"),
                rate=parameter.Identity(form="gamma_rate"),
            ),
            Categorical(response="allocation", prob="prior_allocation_prob"),
        ]
    )
    return model


@pytest.fixture(
    params=[
        ("parameter", NormalNormal),
        ("parameter", ManifoldMALA),
        ("parameter", RandomWalk),
        ("prior_precision_vector", NormalGamma),
        ("allocation", MixtureAllocation),
    ],
    ids=["mean_NormalNormal", "mean_ManifoldMALA", "mean_RandomWalk", "precision_NormalGamma", "MixtureAllocation"],
    name="sampler_object",
)
def fix_sampler_object(request, model):
    """Create the sampler using the model specified in the model fixture.

    The fixture parameters specify the parameter to be sampled, and the sampler class to be used. In the
    MixtureAllocation sampler case, this can only be used when we have MixtureParameter classes for the parameter prior,
    so the sampler is set to None and the tests skipped when the parameters are incompatible.

    """
    [param, param_sampler] = request.param
    if issubclass(param_sampler, MixtureAllocation) and isinstance(model["parameter"].mean, parameter.Identity):
        sampler_object = None
    elif issubclass(param_sampler, MixtureAllocation) and isinstance(
        model["parameter"].mean, parameter.MixtureParameterVector
    ):
        sampler_object = param_sampler(param=param, model=model, response_param="parameter")
    else:
        sampler_object = param_sampler(param=param, model=model)
    return sampler_object


def test_sample(sampler_object: MCMCSampler, state: dict):
    """Test the sample function.

    Performs the following checks:
        1) Checks that the shape of the sampled parameter is the same before and after the sample generation.
        2) Checks that the other elements of the state (apart from self.param) have not been modified.

    """
    if sampler_object is None:
        return
    state_before = deepcopy(state)
    state = sampler_object.sample(state)
    assert state_before[sampler_object.param].shape == state[sampler_object.param].shape

    remaining_keys = list(state.keys())
    remaining_keys.remove(sampler_object.param)
    for key in remaining_keys:
        assert np.allclose(state_before[key], state[key])


def test_sampler_specific(sampler_object: MCMCSampler, state: dict, monkeypatch):
    """Specific tests for each of the samplers.

    For all tests, np.random.standard_normal is patched to always generate vectors of zeros, to enable standard results
    for testing.

    For details of the specific checking done in each of the sampler cases, see the relevant sub-functions.

    """

    def mock_standard_normal(size: tuple):
        """Replace numpy.random.standard_normal with a function that just generates a vector of zeros."""
        return np.zeros(shape=size)

    def mock_norm_rvs(size: tuple, scale: float):
        """Replace scipy.stats.norm.rvs with a function that just generates a vector of zeros."""
        return np.zeros(shape=size)

    monkeypatch.setattr(np.random, "standard_normal", mock_standard_normal)
    monkeypatch.setattr(norm, "rvs", mock_norm_rvs)

    if isinstance(sampler_object, RandomWalk):
        check_randomwalk(sampler_object, state)
    elif isinstance(sampler_object, ManifoldMALA):
        check_manifoldmala(sampler_object, state)
    elif isinstance(sampler_object, NormalNormal):
        check_normalnormal(sampler_object, state, monkeypatch)
    elif isinstance(sampler_object, NormalGamma):
        check_normalgamma(sampler_object, state, monkeypatch)
    elif isinstance(sampler_object, MixtureAllocation):
        check_mixtureallocation(sampler_object, state)


def check_randomwalk(sampler_object: RandomWalk, state: dict):
    """Bespoke checking for the RandomWalk case.

    Performs the following checks:
        1) that the shape of state[self.param] is the same before and after the proposal.
        2) that state[self.param] is the same before and after the proposal (given fixing or random variables to 0).
        3) that the log-proposal densities are both equal.

    """
    current_state = deepcopy(state)
    prop_state, logp_pr_g_cr, logp_cr_g_pr = sampler_object.proposal(current_state)
    assert current_state[sampler_object.param].shape == prop_state[sampler_object.param].shape
    assert np.allclose(current_state[sampler_object.param], prop_state[sampler_object.param], rtol=1e-5, atol=1e-8)
    assert logp_pr_g_cr == logp_cr_g_pr


def check_manifoldmala(sampler_object: ManifoldMALA, state: dict):
    """Bespoke checking for the ManifoldMALA case.

    Performs the following checks:
        1) that the shape of state[self.param] is the same before and after the proposal.
        2) that we can recover the correct gradient from the (non-random) proposal.

    """
    current_state = deepcopy(state)
    prop_state, _, _ = sampler_object.proposal(current_state)
    assert current_state[sampler_object.param].shape == prop_state[sampler_object.param].shape
    grad_cr, hessian_cr = sampler_object.model.grad_log_p(current_state, sampler_object.param)
    r = prop_state["parameter"] - current_state["parameter"]
    grad_recover = (hessian_cr @ r) * 2 / np.power(sampler_object.step, 2)
    assert np.allclose(grad_cr, grad_recover, rtol=1e-5, atol=1e-8)


def check_normalnormal(sampler_object: NormalNormal, state: dict, monkeypatch):
    """Bespoke checking for the NormalNormal case.

    Performs the following checks:
        1) that if state["prefactor_matrix"] is set to be all-zero, then the sample function (with randomness switched
            off) returns the parameter prior predictor.
        2) that if state["prior_precision_vector"] is set to be all-zero, then we recover the standard regression
            solution as the sample (with randomness switched off).
        3) that the expected result is returned when we set both contributions to the mean term to be zero, and the
            vector of random variables to be all ones.

    """
    test_state = deepcopy(state)
    test_state["prefactor_matrix"] = np.zeros(shape=test_state["prefactor_matrix"].shape)
    updated_state = sampler_object.sample(test_state)
    assert np.allclose(updated_state[sampler_object.param], sampler_object.model["parameter"].mean.predictor(state))

    if state["response"].shape[0] > 1:
        test_state = deepcopy(state)
        test_state["prior_precision_vector"] = np.zeros(shape=test_state["prior_precision_vector"].shape)
        updated_state = sampler_object.sample(test_state)
        response_precision = sampler_object.model["response"].precision.predictor(state)
        comparison = np.linalg.solve(
            state["prefactor_matrix"].T @ response_precision @ state["prefactor_matrix"],
            state["prefactor_matrix"].T @ response_precision @ state["response"],
        )
        assert np.allclose(updated_state[sampler_object.param], comparison)

    def mock_sample_ones(size: tuple):
        """Replace numpy.random.standard_normal with a function that just generates a vector of ones."""
        return np.ones(shape=size)

    monkeypatch.setattr(np.random, "standard_normal", mock_sample_ones)

    test_state = deepcopy(state)
    test_state["response"] = np.zeros(shape=test_state["response"].shape)
    test_state["prior_mean"] = np.zeros(shape=test_state["prior_mean"].shape)
    updated_state = sampler_object.sample(test_state)
    response_precision = sampler_object.model["response"].precision.predictor(state)
    comparison = np.linalg.solve(
        np.linalg.cholesky(
            state["prefactor_matrix"].T @ response_precision @ state["prefactor_matrix"]
            + sampler_object.model["parameter"].precision.predictor(state)
        ).T,
        np.ones(shape=state["parameter"].shape),
    )
    assert np.allclose(updated_state["parameter"], comparison)


def check_normalgamma(sampler_object: NormalGamma, state: dict, monkeypatch):
    """Bespoke checking for the NormalGamma case.

    Mocks scipy.stats.gamma.rvs to always return the expected value (a * scale = a / b), then checks that for each prior
    precision parameter (one for each category of the allocation), the reciprocal of the sampled value is equal to the
    mean of the squared residuals.

    """

    def mock_gamma_sample(a, scale):
        """Patch gamma sampler so that it always returns the expected value."""
        no_warning_scale = np.where(scale == np.inf, 1, scale)
        no_warning_sample = np.where(scale == np.inf, np.inf, a * no_warning_scale)
        return no_warning_sample

    monkeypatch.setattr(gamma, "rvs", mock_gamma_sample)

    test_state = deepcopy(state)
    test_state["gamma_shape"] = np.zeros(shape=test_state["gamma_shape"].shape)
    test_state["gamma_rate"] = np.zeros(shape=test_state["gamma_rate"].shape)
    updated_state = sampler_object.sample(test_state)

    residuals = test_state[sampler_object.model[sampler_object.normal_param].response] - sampler_object.model[
        sampler_object.normal_param
    ].mean.predictor(test_state)
    for k in range(test_state[sampler_object.param].shape[0]):
        component_index = test_state["allocation"] == k
        if np.sum(component_index) > 0:
            assert np.allclose(
                1 / updated_state[sampler_object.param][k], np.mean(np.power(residuals[component_index], 2))
            )


def check_mixtureallocation(sampler_object: MixtureAllocation, state: dict):
    """Bespoke checking for the MixtureAllocation case.

    Sets the prior Normal mean for each of the allocation categories to be {0, 1, 2,..., (n_cat - 1)}, and sets the
    corresponding prior precision to be large. Then ensures that each element of the parameter vector is randomly
    assigned one of these values.

    Under these circumstances, the conditional allocation sample should return the category corresponding to its value
    with probability 1: this behaviour is checked.

    """
    rng = np.random.default_rng(0)
    test_state = deepcopy(state)
    test_state["prior_mean"] = np.array(
        np.arange(start=0, stop=test_state["prior_mean"].shape[0]), ndmin=2, dtype=float
    ).T
    test_state["parameter"] = rng.choice(test_state["prior_mean"].flatten(), size=test_state["parameter"].shape)
    test_state["prior_precision_vector"] = 1e4 * np.ones(shape=test_state["prior_precision_vector"].shape)

    updated_state = sampler_object.sample(test_state)
    assert np.allclose(updated_state["allocation"], test_state["parameter"])
