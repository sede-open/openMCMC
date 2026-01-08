# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit testing for the parameter module.

There are two fixtures for settings up a parameter object and a state dictionary for setting up testing

Not yet tested is parameters to exclude

"""

from copy import deepcopy
from typing import Union

import numpy as np
import pytest
from scipy import sparse

from openmcmc.parameter import (
    Identity,
    LinearCombination,
    LinearCombinationWithTransform,
    MixtureParameter,
    MixtureParameterMatrix,
    MixtureParameterVector,
    Parameter,
    ScaledMatrix,
)


@pytest.fixture(
    params=[(1, 1, 1), (10, 9, 7), (10, 9, 1), (10, 1, 7), (1, 9, 7)],
    ids=["all_size_1", "all > 1", "p2=1", "p=1", "n=1"],
    name="state_tuple",
)
def fix_state(request):
    """Fixture Defining a state vector which has all possible types of data for use with any parameter type.

    Args:
        request.param defines tuple (n, p, p2) for different size combinations of the inputs

    n is used in to represent a "number of observations" type parameter
    p is used in to represent a "number of coefficients" type parameter
    p2 is used in to represent a different "number of coefficients" type parameters in cases
    such as the linear combination where we might want two different regressor terms with different numbers of
    coefficients

    Returns
        tuple of (state, n, p)

    """
    [n, p, p2] = request.param
    rng = np.random.default_rng(0)
    state = {}
    state["scalar"] = rng.random((1, 1))
    state["vector"] = rng.random((p, 1))
    state["matrix"] = rng.random((n, p))
    state["vector_2"] = rng.random((p2, 1))
    state["matrix_2"] = rng.random((n, p2))
    state["diagonal_matrix"] = np.diag(rng.random(p))
    state["square_matrix"] = rng.random((p, p))
    state["square_matrix_2"] = rng.random((p2, p2))
    # 0:p-1 repeated up to length n
    state["allocation"] = np.mod(np.array(range(n), ndmin=2).T, p)

    return state, n, p


@pytest.fixture(
    params=[
        Identity(form="scalar"),
        Identity(form="vector"),
        Identity(form="matrix"),
        LinearCombination(form={"vector": "matrix"}),
        LinearCombination(form={"vector": "matrix", "vector_2": "matrix_2"}),
        LinearCombinationWithTransform(form={"vector": "matrix"}, transform={"vector": True}),
        LinearCombinationWithTransform(form={"vector": "matrix"}, transform={"vector": False}),
        LinearCombinationWithTransform(
            form={"vector": "matrix", "vector_2": "matrix_2"}, transform={"vector": True, "vector_2": True}
        ),
        LinearCombinationWithTransform(
            form={"vector": "matrix", "vector_2": "matrix_2"}, transform={"vector": True, "vector_2": False}
        ),
        LinearCombinationWithTransform(
            form={"vector": "matrix", "vector_2": "matrix_2"}, transform={"vector": False, "vector_2": True}
        ),
        LinearCombinationWithTransform(
            form={"vector": "matrix", "vector_2": "matrix_2"}, transform={"vector": False, "vector_2": False}
        ),
        ScaledMatrix(scalar="scalar", matrix="matrix"),
        ScaledMatrix(scalar="scalar", matrix="diagonal_matrix"),
        ScaledMatrix(scalar="scalar", matrix="square_matrix"),
        MixtureParameterVector(param="vector", allocation="allocation"),
        MixtureParameterMatrix(param="vector", allocation="allocation"),
    ],
    ids=[
        "Identity_scalar",
        "Identity_vector",
        "Identity_matrix",
        "LinearCombination_1term",
        "LinearCombination_2terms",
        "LinearCombinationTransform_1term_T",
        "LinearCombinationTransform_1term_F",
        "LinearCombinationTransform_2terms_TT",
        "LinearCombinationTransform_2terms_TF",
        "LinearCombinationTransform_2terms_FT",
        "LinearCombinationTransform_2terms_FF",
        "ScaledMatrix_matrix",
        "ScaledMatrix_diagonal_matrix",
        "ScaledMatrix_square_matrix",
        "MixtureParameterVector",
        "MixtureParameterMatrix",
    ],
    name="parameter",
)
def fix_parameter(request):
    """Fixture for defining different parameter types.

    Returns:
        Parameter: particular parameter type

    """
    return request.param


def is_diag(A: Union[np.ndarray, sparse.csr_matrix]) -> bool:
    """Checks if a matrix is diagonal.

    Args:
        A (Union[np.ndarray, sparse.csr_matrix]): Matrix

    Returns:
        bool: True if matrix is diagonal

    """
    if A.size == 1:
        return True

    if sparse.issparse(A):
        A = A.toarray()

    return np.count_nonzero(A - np.diag(np.diagonal(A))) == 0


def test_predictor(parameter: Parameter, state_tuple: tuple):
    """Compute predictor given parameter and state object.

    Test size is as expected. This is different for each class, so is defined per case.

    Args:
        parameter : Parameter choice defined by fix_parameter
        state_tuple (tuple): a tuple (dict, n , p) where dict is a dictionary of state values,
                                    n and p are sizes. For more detail see fix_state

    """
    state, n, _ = state_tuple

    predictor = parameter.predictor(state)

    if isinstance(parameter, Identity):
        assert predictor.shape == state[parameter.form].shape

    elif isinstance(parameter, ScaledMatrix):
        assert predictor.shape == state[parameter.matrix].shape

    elif isinstance(parameter, (LinearCombination, LinearCombinationWithTransform, MixtureParameterVector)):
        assert predictor.shape == (n, 1)

    elif isinstance(parameter, MixtureParameterMatrix):
        assert predictor.shape == (n, n)
        assert is_diag(predictor)

    else:
        raise TypeError("parameter type not recognised")


def test_predictor_conditional(parameter: Parameter, state_tuple: tuple):
    """Test predictor condition in LinearCombination cases.

    Returns immediately if the parameter is not of LinearCombination type.

    Performs 2 tests:
        1. Excludes all terms in the linear combination and tests that the
            predict function returns a vector of zeros as expected.
        2. For a case where there are exactly more than one term in the linear combination:
            tests that when we exclude each parameter in turn and sum the with the predictor
            with all other parameter except the excluded one, we recover the full predictor.

    Args:
        parameter : Parameter choice defined by fix_parameter
        state_tuple (tuple): a tuple (dict, n , p) where dict is a dictionary of state values,
                                    n and p are sizes. For more detail see fix_state

    """

    if not isinstance(parameter, LinearCombination):
        return

    state, _, _ = state_tuple

    exclude_terms = list(parameter.form.keys())
    predictor = parameter.predictor_conditional(state, term_to_exclude=exclude_terms)
    assert predictor == 0

    if len(exclude_terms) > 1:
        for param in exclude_terms:
            predictor_exclude = 0.0
            predictor_exclude += parameter.predictor_conditional(state, term_to_exclude=param)

            full_keys = deepcopy(exclude_terms)
            full_keys.remove(param)
            predictor_exclude += parameter.predictor_conditional(state, term_to_exclude=full_keys)

            assert np.all(predictor_exclude == parameter.predictor(state))


def test_get_param_list(parameter):
    """Compute parameter list.

    Test size is as expected. This is different for each class, so is defined per case.

    Args:
        parameter (Parameter): Parameter choice defined by fix_parameter

    """

    param_list = parameter.get_param_list()

    assert isinstance(param_list, list)

    if isinstance(parameter, Identity):
        assert len(param_list) == 1

    elif isinstance(parameter, (LinearCombination, LinearCombinationWithTransform)):
        assert len(param_list) == len(parameter.form) * 2

    elif isinstance(parameter, (ScaledMatrix, MixtureParameterVector, MixtureParameterMatrix)):
        assert len(param_list) == 2
    else:
        raise TypeError("parameter type not recognised")


def test_grad(parameter: Parameter, state_tuple: tuple):
    """Compute predictor given parameter and state object.

    Test size is as expected. This is different for each class so the test is defined by case.

    Also checks the values of the gradient for some of the cases where this is simple to achieve.

    Args:
        parameter (Parameter): Parameter choice defined by fix_parameter
        state_tuple (tuple): a tuple (dict, n , p) where dict is a dictionary of state values,
                                    n and p are sizes. For more detail see fix_state

    """
    state, n, p = state_tuple

    if isinstance(parameter, Identity) and (state[parameter.form].shape[1] > 1):
        with pytest.raises(ValueError):
            parameter.grad(state, parameter.form)
    elif isinstance(parameter, Identity):
        gradient = parameter.grad(state, parameter.form)
        q = state[parameter.form].size
        assert gradient.shape == (q, q)
        assert np.all(gradient == np.eye(q))
    elif isinstance(parameter, (LinearCombination, LinearCombinationWithTransform)):
        for param, matrix in parameter.form.items():
            gradient = parameter.grad(state, param)
            assert gradient.shape == state[matrix].T.shape

            if isinstance(parameter, LinearCombinationWithTransform) and parameter.transform[param]:
                g = np.multiply(np.exp(state[param]), state[matrix].T)
                assert np.all(gradient == g)
            else:
                assert np.all(gradient == state[matrix].T)
    elif isinstance(parameter, ScaledMatrix):
        gradient = parameter.grad(state, parameter.scalar)
        assert gradient.shape == state[parameter.matrix].shape
        assert np.all(gradient == state[parameter.matrix])
    elif isinstance(parameter, MixtureParameterVector):
        gradient = parameter.grad(state, parameter.param)
        assert gradient.shape == (p, n)
    elif isinstance(parameter, MixtureParameterMatrix):
        with pytest.raises(TypeError):
            parameter.grad(state, parameter.param)
    else:
        raise TypeError("parameter type not recognised")


@pytest.mark.parametrize(
    "parameter",
    [
        MixtureParameterVector(param="vector", allocation="allocation"),
        MixtureParameterMatrix(param="vector", allocation="allocation"),
    ],
)
def test_get_element_match(parameter: MixtureParameter, state_tuple: tuple):
    """Test get element match function.

    Checks size is correct and that over all matches exactly n matches are found. i.e. every data point has an
    allocation.

    In the test, the allocation is defined as mod(0:n-1, p); there is then also a test to check that the matches are
    found in the right place.

    Args:
        parameter (MixtureParameter): parameter object of mixture type
        state_tuple (tuple): a tuple (dict, n , p) where dict is a dictionary of state values, n and p are sizes. For
            more detail see fix_state

    """
    state, n, p = state_tuple

    total = 0
    for i in range(p):
        match = parameter.get_element_match(state, i)
        assert match.shape == (n, 1)
        assert np.sum(match) >= np.floor(n / p)
        assert np.sum(match) <= np.ceil(n / p)
        if n >= p:
            assert match[i]

        total = total + sum(match)

    assert total == n
