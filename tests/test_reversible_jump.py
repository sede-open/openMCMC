# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bespoke tests for the reversible jump MCMC sampler."""

from typing import Tuple

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import chisquare, gamma, norm, poisson, randint, truncnorm, uniform

from openmcmc import parameter
from openmcmc.distribution.distribution import Gamma, Poisson, Uniform
from openmcmc.distribution.location_scale import Normal, NullDistribution
from openmcmc.mcmc import MCMC
from openmcmc.model import Model
from openmcmc.sampler.metropolis_hastings import ManifoldMALA, RandomWalkLoop
from openmcmc.sampler.reversible_jump import ReversibleJump


def make_basis(data_locations: np.ndarray, knots: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Create a Gaussian kernel basis from the data locations and knots supplied as inputs.

    Args:
        data_locations (np.ndarray): locations of observed data values.
        knots (np.ndarray): knot locations for basis formation.
        scales (np.ndarray): scales for each of the Gaussian basis functions.

    Returns:
        np.ndarray: [n_data x n_knot] basis matrix, with one column per Gaussian kernel.

    """
    basis_matrix = np.full(shape=(data_locations.shape[0], knots.shape[1]), fill_value=np.nan)
    for k in range(knots.shape[1]):
        basis_matrix[:, [k]] = norm.pdf(data_locations, loc=knots[:, k], scale=scales[:, k])
    return basis_matrix


def move_function(state: dict, update_column: int) -> Tuple[dict, int, int]:
    """Update the basis matrix in the state to take account of the relocation of a knot.

    Assumes that the supplied state has at least the following elements:
        "X": locations of the observed data points.
        "theta": locations of the basis knots.
        "omega": widths (standard deviations) of the Gaussian kernels.

    Args:
        state (dict): dictionary containing current state.
        update_column (int): defunct parameter.

    Returns:
        state (dict): state dictionary with updated basis matrix

    """
    state["B"] = make_basis(state["X"], knots=state["theta"], scales=state["omega"])
    return state, 0, 0


def birth_multiple_jump_function(current_state: dict, prop_state: dict) -> Tuple[dict, float, float]:
    """Augment the basis and update the allocation parameters in response to a birth move for the situation in which
    multiple jump parameters need to be updated.

    Assumes that the supplied state has at least the following elements:
        "theta": locations of the basis knots.
        "omega": widths (standard deviations) of the Gaussian kernels.
        "alloc_beta": null allocation vector.

    Args:
        current_state (dict): dictionary containing the current state information.
        prop_state (dict): dictionary containing the proposed state information.

    Returns:
        Tuple[dict, float, float]: tuple consisting of the following elements:
            prop_state (dict): proposed state with updated basis matrix and basis parameters.
            logp_pr_g_cr (float): transition probability for move from current state to proposed state.
            logp_cr_g_pr (float): transition probability for move from proposed state to current state.

    """
    prop_state["B"] = make_basis(prop_state["X"], prop_state["theta"], prop_state["omega"])
    prop_state["alloc_beta"] = np.concatenate((prop_state["alloc_beta"], np.array([0], ndmin=2)), axis=0)
    logp_pr_g_cr = 0.0
    logp_cr_g_pr = 0.0
    return prop_state, logp_pr_g_cr, logp_cr_g_pr


def death_multiple_jump_function(
    current_state: dict, prop_state: dict, deletion_index: int
) -> Tuple[dict, float, float]:
    """Update basis matrix and allocation parameter in reponse to a death move for the situation in which multiple jump
    parameters need to be updated.

    Assumes that the supplied state has at least the following elements:
        "theta": locations of the basis knots.
        "B": basis matrix.
        "alloc_beta": null allocation vector.

    Args:
        current_state (dict): dictionary containing the current state information.
        prop_state (dict): dictionary containing the proposed state information.
        deletion_index (int): index of the basis component to be deleted in the overall set of components.

    Returns:
        Tuple[dict, float, float]: tuple consisting of the following elements:
            prop_state (dict): proposed state with updated basis matrix and basis parameters.
            logp_pr_g_cr (float): transition probability for move from current state to proposed state.
            logp_cr_g_pr (float): transition probability for move from proposed state to current state.

    """
    prop_state["B"] = np.delete(prop_state["B"], obj=deletion_index, axis=1)
    prop_state["alloc_beta"] = np.delete(prop_state["alloc_beta"], obj=deletion_index, axis=0)
    logp_pr_g_cr = 0.0
    logp_cr_g_pr = 0.0
    return prop_state, logp_pr_g_cr, logp_cr_g_pr


@pytest.fixture(name="basis_limits")
def fix_basis_limits():
    """Fix the basis limits to be used for the tests."""
    return np.array([-10, 10])


@pytest.fixture(name="scale_limits")
def fix_scale_limits():
    """Fix the scale limits to be used for the tests."""
    return np.array([0.5, 2])


@pytest.fixture(name="state")
def fix_state(basis_limits):
    """Define the state for the tests."""
    n_basis = 4
    n_data = 50

    basis_knots = basis_limits[0] + (basis_limits[1] - basis_limits[0]) * uniform.rvs(size=n_basis).reshape(
        (1, n_basis)
    )
    data_locations = basis_limits[0] + (basis_limits[1] - basis_limits[0]) * np.sort(
        uniform.rvs(size=n_data).reshape((n_data, 1)), axis=0
    )
    basis_scales = 1.0 * np.ones(shape=(1, n_basis))
    B = make_basis(data_locations=data_locations, knots=basis_knots, scales=basis_scales)

    tau_beta = 1.0 / (2.0**2)
    tau_y = 1.0 / (0.1**2)
    beta_real = np.sqrt(1.0 / tau_beta) * norm.rvs(size=(n_basis, 1))
    y = B @ beta_real + np.sqrt(1.0 / tau_y) * norm.rvs(size=(n_data, 1))

    state = {
        "y": y,
        "beta": np.ones((n_basis, 1)),
        "tau_y": tau_y,
        "P": sparse.eye(n_data),
        "B": B,
        "n_basis": n_basis,
        "X": data_locations,
        "theta": basis_knots,
        "omega": basis_scales,
        "mu_beta": np.zeros((1, 1)),
        "tau_beta": tau_beta * np.ones((1, 1)),
        "rho": 8,
        "alloc_beta": np.zeros((n_basis, 1), dtype=int),
        "a_omega": 3.0 * np.ones((1, 1)),
        "b_omega": 2.0 * np.ones((1, 1)),
    }
    return state


@pytest.fixture(name="model")
def fix_model(basis_limits):
    """Set up the model for the reversible jump unit tests.

    Model specification has the following components:
        - response_distribution: a Null distribution for the response "y", which gives a 0 contribution to the
            log-posterior (and to the gradient etc.).
        - beta_prior: Normal prior for the basis parameters "beta", with mixture parameter priors to account for the
            changing shape of the basis parameter.
        - knot_num_prior: Poisson prior for the number "n_basis" of knots in the model.
        - knot_loc_prior: Uniform prior for the locations "theta" of the individual knots in the model.

    """
    response_mean = parameter.LinearCombination(form={"beta": "B"})
    response_precision = parameter.ScaledMatrix(matrix="P", scalar="tau_y")
    response_distribution = NullDistribution(response="y", mean=response_mean, precision=response_precision)

    beta_mean = parameter.MixtureParameterVector(param="mu_beta", allocation="alloc_beta")
    beta_precision = parameter.MixtureParameterMatrix(param="tau_beta", allocation="alloc_beta")
    beta_prior = Normal(response="beta", mean=beta_mean, precision=beta_precision)

    knot_num_prior = Poisson(response="n_basis", rate="rho")
    knot_loc_prior = Uniform(
        response="theta",
        domain_response_lower=np.array([basis_limits[0]], ndmin=2),
        domain_response_upper=np.array([basis_limits[1]], ndmin=2),
    )
    width_prior = Gamma("omega", shape="a_omega", rate="b_omega")

    model = Model([response_distribution, beta_prior, knot_num_prior, knot_loc_prior, width_prior])
    model.response = {"y": "mean"}
    return model


@pytest.fixture(name="samplers")
def fix_samplers(model, basis_limits, scale_limits):
    """Set up the samplers for the reversible jump unit tests.

    Sampler specification has the following components:
        - ManifoldMALA sampler for the basis coefficients. The Null likelihood distribution means that only the prior
            contributes to the gradient and Hessian.
        - RandomWalkLoop sampler for the locations of the basis knots.
        - ReversibleJump sampler for the number of knots in the basis.

    """
    n_basis_max = 20
    matching_params = {"variable": "beta", "matrix": "B", "scale": 1.0, "limits": [-10.0, 10.0]}
    samplers = [
        ManifoldMALA(param="beta", model=model, step=np.array(0.5), max_variable_size=n_basis_max),
        RandomWalkLoop(
            param="theta",
            model=model,
            step=np.array(0.1),
            max_variable_size=n_basis_max,
            domain_limits=np.array(basis_limits, ndmin=2),
            state_update_function=move_function,
        ),
        RandomWalkLoop(
            param="omega",
            model=model,
            step=np.array(0.1),
            max_variable_size=n_basis_max,
            domain_limits=np.array(scale_limits, ndmin=2),
            state_update_function=move_function,
        ),
        ReversibleJump(
            param="n_basis",
            model=model,
            associated_params=["theta", "omega"],
            n_max=n_basis_max,
            state_birth_function=birth_multiple_jump_function,
            state_death_function=death_multiple_jump_function,
            matching_params=matching_params,
        ),
    ]
    return samplers


def test_prior_recovery(state, model, samplers):
    """Run the sampler with the null likelihood (data-free).

    Checks that with the null likelihood, the sampler approximately recovers the prior distribution for the number of
    knots. This is checked by using a chi-squared goodness of fit test for the correspondence between the true Poisson
    prior and the MCMC samples, for bins where the expected count is at least 5.

    """
    solver = MCMC(state=state, samplers=samplers, model=model, n_burn=0, n_iter=5000)
    solver.run_mcmc()

    idx_thin = np.arange(start=0, stop=solver.n_iter, step=50)
    sample_n_knot = solver.store["n_basis"][:, idx_thin]

    num = np.arange(start=1, stop=21, step=1)
    bin_edges = np.linspace(start=0.5, stop=20.5, num=21)
    expected_count = sample_n_knot.size * poisson.pmf(num, state["rho"])
    observed_count, bin_edges = np.histogram(sample_n_knot.flatten(), bins=bin_edges)

    big_enough = expected_count >= 5
    observed_count_test = observed_count[big_enough]
    expected_count_test = expected_count[big_enough] * np.sum(observed_count_test) / np.sum(expected_count[big_enough])
    _, p_val = chisquare(observed_count_test, expected_count_test)
    assert p_val >= 0.001


@pytest.fixture
def mock_gmrf_normal_sampler(monkeypatch):
    """Replace np.random.normal with a function that just returns the mean, so that gmrf.sample_normal will also do the
    same."""

    def sample_zeros(size: tuple) -> np.ndarray:
        return np.zeros(shape=size)

    monkeypatch.setattr(np.random, "standard_normal", sample_zeros)


@pytest.fixture
def mock_gmrf_truncated_sampler(monkeypatch):
    """Replace truncnorm with a function that just returns the mean, so that gmrf.sample_normal will also do the
    same."""

    def sample_zeros(a, b, loc, scale, size):
        return loc * np.ones(shape=size)

    monkeypatch.setattr(truncnorm, "rvs", sample_zeros)


@pytest.fixture
def mock_gamma_sampler(monkeypatch):
    """Replace gamma with a function which always returns the 1, so the birth move always return an omega equalling
    one."""

    def sample_ones(shape, scale, size):
        return 1 * np.ones(shape=size)

    monkeypatch.setattr(gamma, "rvs", sample_ones)


@pytest.fixture
def mock_knot_midpoint(monkeypatch):
    """Replace the uniform random sampler with a function which always returns 0.5, so that the birth move always
    returns a knot in the centre of the domain."""

    def sample_midpoint(size: int, n=1):
        return 0.5 * np.ones((size, n))

    monkeypatch.setattr(np.random, "rand", sample_midpoint)


@pytest.fixture(name="mock_knot_endpoint")
def fix_mock_knot_endpoint(monkeypatch):
    """Replace the uniform random sampler with a function which always returns 0.5, so that the birth move always
    returns a knot at the upper end of the domain."""

    def sample_endpoint(size: int, n=1):
        return 1.0 * np.ones((size, n))

    monkeypatch.setattr(np.random, "rand", sample_endpoint)


@pytest.fixture(name="mock_knot_selection")
def fix_mock_knot_selection(monkeypatch):
    """Replace the numpy.random.randint with something that always selects the highest integer, to always select
    the final knot for deletion."""

    def select_final_knot(low: int, high: int, size=1):
        return high - 1

    monkeypatch.setattr(randint, "rvs", select_final_knot)


def test_birth_overlap(state, samplers, mock_knot_endpoint, mock_gmrf_truncated_sampler, mock_gamma_sampler):
    """Test the functionality which matches the predictions before and after the birth transition.

    Create a new knot in exactly the same location as one of the existing ones: the coefficient at the existing location
    should have 50% assigned to each of the concurrent locations in the new state.

    The initial state has knots at x=[-10, -5, 5, 10]. The mock_knot_endpoint patch forces np.random.rand to return 1.0
    in order that the proposed knot coincides the existing one at x=10. The mock_gmrf_truncated_sampler patch ensures
    that there is no randomness on the returned parameter.

    The parameters in the current state are all set to be 1, so that the in the proposed state, the two parameters
    associated with the knot at x=10 should both be 0.5.

    Also checks that the log-transition densities are returned as expected:
        - log(p(theta*|theta)) = logp_pr_g_cr = truncated Gaussian density evaluated at central point.
        - log(p(theta|theta*)) = logp_cr_g_pr = log(|F|) = log(0.5) in this situation

    """
    state["theta"] = np.array([-10, -5, 5, 10], ndmin=2)
    state["omega"] = np.array([1, 1, 1, 1], ndmin=2)
    state["B"] = make_basis(state["X"], state["theta"], state["omega"])
    prop_state, _, _ = samplers[3].birth_proposal(state)
    assert np.allclose(prop_state["beta"][-1], 0.5)
    assert np.allclose(prop_state["beta"][-2], 0.5)
    assert np.allclose(np.sum(prop_state["beta"]), state["theta"].size)

    prop_state, logp_pr_g_cr, logp_cr_g_pr = samplers[3].matched_birth_transition(state, prop_state, 0.0, 0.0)
    assert np.allclose(logp_pr_g_cr, -0.5 * np.log(2.0 * np.pi) * samplers[3].matching_params["scale"])
    assert np.allclose(logp_cr_g_pr, np.log(0.5))


def test_birth_no_overlap(state, samplers, mock_knot_midpoint, mock_gmrf_truncated_sampler, mock_gamma_sampler):
    """Test the functionality which matches the predictions before and after the birth transition.

    Create a new knot which doesn't overlap with any of the others, then we expect the existing knots to have no
    influence over the value of the new one.

    """
    state["theta"] = np.array([-10, -5, 5, 10], ndmin=2)
    state["omega"] = np.array([1, 1, 1, 1], ndmin=2)
    state["B"] = make_basis(state["X"], state["theta"], state["omega"])
    prop_state, logp_pr_g_cr, logp_cr_g_pr = samplers[3].birth_proposal(state)
    assert np.allclose(prop_state["beta"][-1], 0.0)
    assert np.allclose(np.sum(prop_state["beta"]), state["theta"].size)

    prop_state, logp_pr_g_cr, logp_cr_g_pr = samplers[3].matched_birth_transition(state, prop_state, 0.0, 0.0)
    assert np.allclose(logp_pr_g_cr, -0.5 * np.log(2.0 * np.pi) * samplers[3].matching_params["scale"])
    assert np.allclose(logp_cr_g_pr, 0.0)


def test_death_overlap(state, samplers, mock_knot_selection):
    """Test the functionality which matches the predictions before and after a death transition, in the edge case where
    there are overlapping basis knots.

    Test is effectively the opposite of the one run by test_death_overlap(). See that function for further information.

    """
    state["theta"] = np.array([-10, -5, 10, 10], ndmin=2)
    state["B"] = make_basis(state["X"], state["theta"], state["omega"])
    prop_state, _, _ = samplers[3].death_proposal(state)
    assert np.allclose(prop_state["beta"][-1], 2.0)
    assert np.allclose(np.sum(prop_state["beta"]), state["theta"].size)

    prop_state, logp_pr_g_cr, logp_cr_g_pr = samplers[3].matched_death_transition(
        state, prop_state, 0.0, 0.0, deletion_index=3
    )
    assert np.allclose(logp_pr_g_cr, np.log(0.5))
    assert np.allclose(logp_cr_g_pr, -0.5 * np.log(2.0 * np.pi) * samplers[3].matching_params["scale"])


def test_death_no_overlap(state, samplers, mock_knot_selection):
    """Test the functionality which matches the predictions before and after a death transition, in the edge case where
    the basis knots are fully spatially distinct.

    Test is effectively the opposite of the one run by test_birth_no_overlap().

    """
    state["theta"] = np.array([-10, -5, 5, 10], ndmin=2)
    state["beta"] = np.array([1, 1, 1, 0], ndmin=2).T
    state["B"] = make_basis(state["X"], state["theta"], state["omega"])
    prop_state, logp_pr_g_cr, logp_cr_g_pr = samplers[3].death_proposal(state)
    assert np.allclose(prop_state["beta"], state["beta"][:-1])

    prop_state, logp_pr_g_cr, logp_cr_g_pr = samplers[3].matched_death_transition(
        state, prop_state, 0.0, 0.0, deletion_index=3
    )
    assert np.allclose(logp_pr_g_cr, 0.0)
    assert np.allclose(logp_cr_g_pr, -0.5 * np.log(2.0 * np.pi) * samplers[3].matching_params["scale"])
