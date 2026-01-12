# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test for Model class, which combines multiple distributions."""

import numpy as np
import pytest

from openmcmc.distribution.location_scale import Normal
from openmcmc.model import Model
from openmcmc.parameter import LinearCombination, ScaledMatrix


@pytest.fixture(
    params=[(1, 1), (1, 7), (13, 1), (13, 7)], ids=["n=1 p=1", "n=1 p=7", "n=13 p=1", "n=13 p=7"], name="state"
)
def fix_state(request):
    """Fix state for use in the tests."""
    rng = np.random.default_rng(0)
    [n, p] = request.param
    state = {}
    state["theta"] = rng.random((p, 1))
    state["Q_response"] = (1 / 0.01**2) * np.eye(n)
    state["basis_matrix"] = rng.random((n, p))
    state["response"] = state["basis_matrix"] @ state["theta"] + np.linalg.solve(
        np.sqrt(state["Q_response"]), rng.normal(size=(n, 1))
    )
    state["prior_mean"] = np.zeros(shape=(p, 1))
    state["tau"] = np.array([1 / 10**2], ndmin=2)
    state["prior_matrix"] = np.eye(p)
    return state


@pytest.fixture(name="model")
def fix_model():
    """Fix the model for testing.

    Model consists of a normal distribution for response given parameter, and a normal prior distribution for the
    parameter. Measurement error precision and prior normal parameters are all fixed.

    """
    response_mean = LinearCombination(form={"theta": "basis_matrix"})
    prior_precision = ScaledMatrix(matrix="prior_matrix", scalar="tau")
    return Model(
        [
            Normal(response="response", mean=response_mean, precision="Q_response"),
            Normal(response="theta", mean="prior_mean", precision=prior_precision),
        ]
    )


def test_gradient(model, state):
    """Test the combined gradient function for the model.

    Checks that the gradient and Hessian returned by Model.grad_log_p() are indeed the sum of the gradients from the two
    components of the supplied model.

    """
    grad_from_model, hess_from_model = model.grad_log_p(state, param="theta", hessian_required=True)
    grad_resp, hess_resp = model["response"].grad_log_p(state, param="theta", hessian_required=True)
    grad_prior, hess_prior = model["theta"].grad_log_p(state, param="theta", hessian_required=True)
    assert np.allclose(grad_from_model, grad_resp + grad_prior)
    assert np.allclose(hess_from_model, hess_resp + hess_prior)


def test_log_p(model, state):
    """Test the combined log-density function for the model.

    Checks that Model.log_p() returns the same as the sum of the two components of the supplied model.

    """
    log_p_model = model.log_p(state)
    log_p_resp = model["response"].log_p(state)
    log_p_prior = model["theta"].log_p(state)
    assert np.allclose(log_p_model, log_p_resp + log_p_prior)
