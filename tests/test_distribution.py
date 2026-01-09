# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for distributions."""

from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse, stats

from openmcmc.distribution.distribution import (
    Categorical,
    Distribution,
    Gamma,
    Poisson,
    Uniform,
)
from openmcmc.distribution.location_scale import LogNormal, Normal
from openmcmc.model import Model
from openmcmc.parameter import (
    Identity,
    LinearCombination,
    MixtureParameterMatrix,
    MixtureParameterVector,
    ScaledMatrix,
)


@pytest.fixture(
    params=[(1, 1, 1), (10, 5, 7), (10, 5, 1), (10, 1, 7), (1, 5, 7)],
    ids=["all_size_1", "all > 1", "p=1", "d=1", "n=1"],
    name="state",
)
def fix_state(request):
    """Fixture Defining a state vector which has all possible types of data for use with any parameter type.

    Args:
        request.param defines tuple (n, p, p2) for different size combinations of the inputs

    n is used in to represent a "number of observations" type parameter
    d is used in to represent a "number of dimensions" type parameter
    p is used in to represent a different "number of coefficients" type parameters in cases
    such as the linear combination where we might want two different regressor terms with different numbers of
    coefficients

    Returns
        state (dict) dictionary of parameter values

    """
    [n, d, p] = request.param
    rng = np.random.default_rng(0)
    state = {}
    state["scalar"] = rng.random((1, 1)) + 1
    state["scalar_2"] = rng.random((1, 1)) + 1
    state["observation_1_n"] = rng.random((1, n)) + 1
    state["observation_d_n"] = rng.random((d, n)) + 1
    state["vector_d"] = rng.random((d, 1)) + 1
    state["vector_p"] = rng.random((p, 1)) + 1
    state["count_1"] = rng.integers(10, size=(1, 1))
    state["count_d"] = rng.integers(10, size=(d, 1))
    state["sparse_identity"] = sparse.eye(d, format="csr")
    state["identity"] = np.eye(d)
    state["matrix"] = rng.random((d, p))
    state["allocation"] = np.mod(np.array(range(d), ndmin=2).T, p)
    state["probability_d"] = stats.dirichlet.rvs(np.ones(p), size=d)
    state["allocation_d_n"] = rng.integers(p, size=(d, n))
    return state


# Normal Parameters
@pytest.fixture(
    params=[
        Normal("observation_1_n", mean="scalar", precision="scalar_2"),
        Normal("observation_d_n", mean=Identity("vector_d"), precision=Identity("identity")),
        Normal("observation_d_n", mean=LinearCombination(form={"vector_p": "matrix"}), precision="sparse_identity"),
        Normal(
            "observation_d_n",
            mean=MixtureParameterVector("vector_p", "allocation"),
            precision=ScaledMatrix("sparse_identity", "scalar_2"),
        ),
        Normal("observation_d_n", mean="vector_d", precision=MixtureParameterMatrix("vector_p", "allocation")),
        Normal("observation_1_n", mean="scalar", precision="scalar_2", domain_response_lower=np.array(-2)),
        Normal("observation_d_n", mean="vector_d", precision="sparse_identity", domain_response_upper=np.array(10)),
        LogNormal("observation_1_n", mean="scalar", precision="scalar_2"),
        LogNormal("observation_d_n", mean="vector_d", precision="identity"),
        LogNormal("observation_d_n", mean=LinearCombination(form={"vector_p": "matrix"}), precision="sparse_identity"),
        LogNormal(
            "observation_d_n",
            mean=MixtureParameterVector("vector_p", "allocation"),
            precision=ScaledMatrix("sparse_identity", "scalar_2"),
        ),
        LogNormal("observation_d_n", mean="vector_d", precision=MixtureParameterMatrix("vector_p", "allocation")),
        Gamma("observation_1_n", shape="scalar", rate="scalar_2"),
        Gamma("observation_d_n", shape=LinearCombination(form={"vector_p": "matrix"}), rate="scalar_2"),
        Gamma("observation_d_n", shape="scalar", rate=MixtureParameterVector("vector_p", "allocation")),
        Poisson("count_1", rate="scalar"),
        Poisson("count_d", rate=LinearCombination(form={"vector_p": "matrix"})),
        Poisson("count_d", rate=MixtureParameterVector("vector_p", "allocation")),
        Uniform("observation_d_n", domain_response_lower=1, domain_response_upper=2),
        Categorical("allocation_d_n", "probability_d"),
    ],
    ids=[
        "UnivariateNormal",
        "MVNormal",
        "LinCombSparseMVN",
        "MixMeanScaledMatrixMVN",
        "MixtureMatrixMVN",
        "TruncateNormal",
        "TruncateMVN",
        "UnivariateLognormal",
        "MVLogNormal",
        "LinCombSparseLogNorm",
        "MixMeanScaledMatrixLogNorm",
        "MixtureMatrixLogNorm",
        "ScalarGamma",
        "LinCombGamma",
        "MixRateGamma",
        "ScalarPoisson",
        "LinCombPoisson",
        "MixRatePoisson",
        "Uniform",
        "Categorical",
    ],
    name="distribution",
)
def fix_distribution(request):
    """Define distribution to test.

    Returns     Distribution

    """
    return request.param


def test_log_p(distribution: Distribution, state: dict):
    """Log_p test.

    Test 1. Check the by observation log_p is correct shape
    Test 2. Check the summed log_p is size 1
    Test 3. Generate Random numbers from true distribution and check profile likelihood is
            at maximum around the true parameters
    Test 4. Generate Random numbers from true distribution and check gradient is +ve below true parameters and -ve above

    Generate random numbers from distribution and compute likelihood then go
    through parameters in inference_param and varies up and down to check likelihood gets worse.

    Args:
        distribution (Distribution): distribution object defined by fix_distribution
        state (dict): state object defined by fix_state

    """

    p, n = state[distribution.response].shape

    log_p_all = distribution.log_p(state, by_observation=True)
    assert log_p_all.size == n

    log_p_tru = distribution.log_p(state)
    assert log_p_tru.size == 1

    n = 300
    state_profile = deepcopy(state)
    state_profile[distribution.response] = distribution.rvs(state_profile, n=n)

    assert state_profile[distribution.response].shape == (p, n)
    log_p_tru = distribution.log_p(state_profile)

    if isinstance(distribution, Categorical):
        assert np.max(state_profile[distribution.response]) <= state_profile["probability_d"].shape[1] - 1

        # shift probability not response
        state_profile["probability_d"] = np.roll(state_profile["probability_d"], 1, axis=1)

        log_p_perm = distribution.log_p(state_profile)
        assert log_p_tru >= log_p_perm

    else:
        for param in distribution.param_list:
            if param in [distribution.response, "allocation"]:
                continue

            state_profile_high = deepcopy(state_profile)
            state_profile_high[param] = state_profile_high[param] * 10
            log_p_high = distribution.log_p(state_profile_high)
            assert log_p_tru > log_p_high

            state_profile_low = deepcopy(state_profile)
            state_profile_low[param] = state_profile_low[param] * 0.1
            log_p_low = distribution.log_p(state_profile_low)
            assert log_p_tru > log_p_low


def test_grad_log_p(distribution: Distribution, state: dict):
    """grad_log_p test: Test 1. Check grad_log_p is correct size Test 2. Check grad_log_p is matches finite difference
    (for cases where analytical gradients exist)

    Only perform test of calculating gradients for non-integer type parameters

    Generate random numbers from distribution and compute likelihood then go
    through parameters in inference_param and varies up and down to check likelihood gets worse.

    Args:
        distribution (Distribution): distribution object defined by fix_distribution
        state (dict): state object defined by fix_state

    """

    for param in distribution.param_list:
        if param in ["allocation", "allocation_d_n", "count_1", "count_d", "probability_d"]:
            continue

        grad_log_p = distribution.grad_log_p(state, param, hessian_required=False)
        assert grad_log_p.shape == state[param].shape

        if isinstance(distribution, (Normal, LogNormal)):
            grad_log_p_diff = distribution.grad_log_p_diff(state, param)
            assert np.allclose(grad_log_p, grad_log_p_diff, rtol=1e-3)


def test_hessian_log_p(distribution: Distribution, state: dict):
    """hessian_log_p test: Test 1. Check hessian_log_p is correct size Test 2. Check hessian_log_p is symmetric Test 3.
    Check hessian_log_p is matches finite difference (for cases where analytical gradients exist)

    Generate random numbers from distribution and compute likelihood then go
    through parameters in inference_param and varies up and down to check likelihood gets worse.

    Args:
        distribution (Distribution): distribution object defined by fix_distribution
        state (dict): state object defined by fix_state

    """

    for param in distribution.param_list:
        if param in ["allocation", "allocation_d_n", "count_1", "count_d", "probability_d"]:
            continue

        _, hessian_log_p = distribution.grad_log_p(state, param)
        p = np.prod(state[param].shape)
        assert hessian_log_p.shape == (p, p)

        if sparse.issparse(hessian_log_p):
            hessian_log_p = hessian_log_p.toarray()

        assert np.linalg.norm(hessian_log_p - hessian_log_p.T) < 1e-4

        if isinstance(distribution, (Normal, LogNormal)):
            hessian_log_p_diff = distribution.hessian_log_p_diff(state, param)
            assert np.linalg.norm(hessian_log_p - hessian_log_p_diff) <= 1e-3


def test_model_conditional():
    """Check that model conditional returns right number of elements."""
    model = Model(
        [
            Normal("A", mean="B", precision="C"),
            Normal("B", mean="B_mean", precision="B_precision"),
            Gamma("C", rate="C_rate", shape="C_shape"),
        ]
    )

    assert isinstance(model, dict)

    assert len(model.conditional("A")) == 1
    assert len(model.conditional("B")) == 2
    assert len(model.conditional("C")) == 2
    assert len(model.conditional("B_mean")) == 1
    assert len(model.conditional("B_precision")) == 1
    assert len(model.conditional("C_rate")) == 1
    assert len(model.conditional("C_shape")) == 1


def test_model_log_p(state: dict):
    """Test log likelihood and grad_log_p is computed correctly.

    state (dict): state object defined by fix_state

    """
    model = Model(
        [
            Normal("observation_d_n", mean=LinearCombination(form={"vector_p": "matrix"}), precision="sparse_identity"),
            Normal("observation_1_n", mean="scalar", precision="scalar_2"),
            Gamma("observation_1_n", shape="scalar", rate="scalar_2"),
            Poisson("count_1", rate="scalar"),
        ]
    )

    assert model.log_p(state).size == 1

    p = state["vector_p"].shape[0]

    grad_1, hessian = model.grad_log_p(state, "vector_p", hessian_required=True)
    grad_2 = model.grad_log_p(state, "vector_p", hessian_required=False)

    assert np.allclose(grad_1, grad_2, 1e-10)
    assert grad_1.shape == (p, 1)
    assert hessian.shape == (p, p)
