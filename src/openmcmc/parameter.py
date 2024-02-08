# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collection of possible parameter specifications for the distribution objects.

Example choices defined:

Identity:  f = x
LinearCombination:  f = X @ beta + Y @ gamma
LinearCombinationWithTransform: f = X @ exp(beta) + Y @ gamma
ScaledMatrix f = lam * P
MixtureParameterVector  f= X[I]
MixtureParameterMatrix  f= np.diag(lam[I])

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import sparse


@dataclass
class Parameter(ABC):
    """Abstract base class for parameter."""

    @abstractmethod
    def predictor(self, state: dict) -> np.ndarray:
        """Create predictor from the state dictionary using the functional form defined in the specific subclass.

        Args:
            state (dict): dictionary object containing the current state information

        Returns:
            (np.ndarray): predictor vector

        """

    @abstractmethod
    def get_param_list(self) -> list:
        """Extract list of components from parameter specification.

        Returns:
            (list): parameter included as part of predictor

        """

    @abstractmethod
    def get_grad_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """

    @abstractmethod
    def grad(self, state: dict, param: str) -> np.ndarray:
        """Compute gradient of single parameter.

        Args:
            state (dict): Dictionary object containing the current state information
            param (str): Compute derivatives WRT this variable

        Returns:
            (np.ndarray): [n_param x n_data] array, gradient with respect to param

        """


@dataclass
class Identity(Parameter):
    """Class specifying a simple predictor in a single term.

    Predictor has the functional form:
        f = x

    The gradient should only be used for scalar and vector inputs

    Args:
        form (str): string specifying the element of state which determines the parameter

    Attributes:
        form (str): string specifying the element of state which determines the parameter.

    """

    form: str

    def predictor(self, state: dict) -> np.ndarray:
        """Create predictor from the state dictionary using the functional form defined in the specific subclass.

        Args:
            state (dict): dictionary object containing the current state information

        Returns:
            (np.ndarray): predictor vector

        """
        return state[self.form]

    def get_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return [self.form]

    def get_grad_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return [self.form]

    def grad(self, state: dict, param: str) -> np.ndarray:
        """Compute gradient of single parameter.

        Args:
            state (dict): Dictionary object containing the current state information
            param (str): Compute derivatives WRT this variable

        Returns:
            (np.ndarray): [n_param x n_data] array, gradient with respect to param

        """
        if state[self.form].shape[1] > 1:
            raise ValueError("Gradient in Identity should not be used for variables 2D and above.")
        p = state[self.form].size
        if param == self.form:
            grad = np.eye(p)
        else:
            grad = np.zeros(shape=(p, p))
        return grad


@dataclass
class LinearCombination(Parameter):
    """Class specifying linear combination form .

    This Parameter type is typically in the mean of a Normal distribution in a linear regression type case.

    Predictor has the form
        predictor  = sum_i (value[i] @ key[i])
    using the form dictionary input

    Attributes:
        form (dict): dict specifying the term and prefactor in the linear combination.
            example: {'beta': 'X', 'alpha': 'A'} produces linear combination X @ beta + A @ alpha.

    """

    form: dict

    def predictor(self, state: dict) -> np.ndarray:
        """Create predictor from the state dictionary using the functional form defined in the specific subclass.

        Args:
            state (dict): dictionary object containing the current state information

        Returns:
            (np.ndarray): predictor vector

        """
        return self.predictor_conditional(state)

    def predictor_conditional(self, state: dict, term_to_exclude: Union[str, list] = None) -> np.ndarray:
        """Extract predictor from the state dictionary using the functional form defined in the specific subclass excluding parameters.

        Used when estimating conditional distributions of those parameters.

        Args:
            state (dict): dictionary object containing the current state information
            term_to_exclude (Union[str, list]): terms to exclude from predictor

        Returns:
            (np.ndarray): predictor vector

        """
        if term_to_exclude is None:
            term_to_exclude = []

        if isinstance(term_to_exclude, str):
            term_to_exclude = [term_to_exclude]

        sum_terms = 0
        for prm, prefactor in self.form.items():
            if prm not in term_to_exclude:
                sum_terms += state[prefactor] @ state[prm]
        return sum_terms

    def get_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return list(self.form.keys()) + list(self.form.values())

    def get_grad_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return list(self.form.keys())

    def grad(self, state: dict, param: str) -> np.ndarray:
        """Compute gradient of single parameter.

        Args:
            state (dict): Dictionary object containing the current state information
            param (str): Compute derivatives WRT this variable

        Returns:
            (np.ndarray): [n_param x n_data] array, gradient with respect to param

        """
        return state[self.form[param]].T


@dataclass
class LinearCombinationWithTransform(LinearCombination):
    """Linear combination of parameters from the state, with optional exponential transformation for the parameter elements.

    Currently, the only allowed transformation is the exponential transform.

    This Parameter type is typically in the mean of a Normal distribution and could be
    used to impose positivity of the parameters

    Predictor has the form
        predictor  = sum_i (value[i] @ transform(key[i]))
    using the form dictionary input

    Attributes:
        transform (dict): dict with logicals specifying whether exp(.) transform should
            be applied to parameter
            example: form={'beta': X}, transform={'beta': True} will produce X @ np.exp(beta)

    """

    transform: dict

    def predictor_conditional(self, state: dict, term_to_exclude: Union[str, list] = None) -> np.ndarray:
        """Extract predictor from the state dictionary using the functional form defined in the specific subclass excluding parameters.

        Used when estimating conditional distributions of those parameters.

        Args:
            state (dict): dictionary object containing the current state information
            term_to_exclude (list): terms to exclude from predictor

        Returns:
            (np.ndarray): predictor vector

        """
        if term_to_exclude is None:
            term_to_exclude = []

        if isinstance(term_to_exclude, str):
            term_to_exclude = [term_to_exclude]

        sum_terms = 0
        for prm, prefactor in self.form.items():
            if prm not in term_to_exclude:
                param = state[prm]
                if self.transform[prm]:
                    param = np.exp(param)
                sum_terms += state[prefactor] @ param
        return sum_terms

    def grad(self, state: dict, param: str) -> np.ndarray:
        """Compute gradient of single parameter.

        Args:
            state (dict): Dictionary object containing the current state information
            param (str): Compute derivatives WRT this variable

        Returns:
            (np.ndarray): [n_param x n_data] array, gradient with respect to param

        """
        if self.transform[param]:
            if sparse.issparse(state[self.form[param]]):
                return state[self.form[param]].multiply(np.exp(state[param]).flatten()).T
            return np.exp(state[param]) * (state[self.form[param]].T)

        return state[self.form[param]].T


@dataclass
class ScaledMatrix(Parameter):
    """Defines parameter a scalar factor in front of a matrix.

    This is often used in case where we have a scalar variance in front of an unscaled precision matrix.
    Where we have a gamma distribution for the scalar parameter which wish to estimate

    Linear combinations have the form:
        predictor = scalar * matrix

    Attributes:
        matrix (str): variable name of the un-scaled matrix
        scalar (str): variable name of the scalar term

    """

    matrix: str
    scalar: str

    def predictor(self, state: dict) -> np.ndarray:
        """Create predictor from the state dictionary using the functional form defined in the specific subclass.

        Args:
            state (dict): dictionary object containing the current state information

        Returns:
            (np.ndarray): predictor vector

        """
        return float(state[self.scalar].item()) * state[self.matrix]

    def get_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return [self.scalar, self.matrix]

    def get_grad_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return [self.scalar]

    def grad(self, state: dict, param: str) -> np.ndarray:
        """Compute gradient of single parameter.

        Args:
            state (dict): Dictionary object containing the current state information
            param (str): Compute derivatives WRT this variable

        Returns:
            (np.ndarray): [n_param x n_data] array, gradient with respect to param

        """
        return state[self.matrix]

    def precision_unscaled(self, state: dict, _) -> np.ndarray:
        """Return the precision matrix un-scaled by the scalar precision parameter.

        Args:
            state (dict): state dictionary
            _ (int): argument unused but matches with version in MixtureParameterMatrix where element is needed

        Returns:
            (np.ndarray): unscaled precision matrix

        """
        return state[self.matrix]


@dataclass
class MixtureParameter(Parameter, ABC):
    """Abstract Parameter class for a mixture distribution.

    Subclasses implemented for both:

    - vector-valued parameter (MixtureParameterVector)
    - diagonal matrix-valued parameter (MixtureParameterMatrix)
    where the elements of the vector or matrix diagonal are allocated based
    on the allocation parameter.

    """

    param: str
    allocation: str

    def get_element_match(self, state: dict, element_index: Union[int, np.ndarray]) -> np.ndarray:
        """Extract the parts of self.allocation which have given element number.

        used in the gradient function to pull out gradient for given element.

        Args:
            state (dict): state vector
            element_index (int, np.array): element index or set of integers

        Returns:
            (np.array(dtype=int)): element matches with 1 where there is a match and 0 where there isn't

        """
        if isinstance(element_index, np.ndarray) and element_index.size > 1:
            element_index = element_index.reshape((1, -1))

        return np.array(state[self.allocation] == element_index, dtype=int, ndmin=2)

    def get_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return [self.param, self.allocation]


@dataclass
class MixtureParameterVector(MixtureParameter):
    """Vector parameter: elements of the vector are obtained from sub-parameter 'param' according to the allocation.

    The allocation parameter defines a mapping between a R^m and R^n where typically m<=n and m is the
    number true underlying number of parameters in the model but due to the representation/algebra in
    other parts of the model this is expanded out to an n parameter model where the values of m are copied
    according to the index vector

    predictor = param [allocation]

    Attributes:
        param (str): name of underlying state component used to generate parameter.
        allocation (np.ndarray): name of allocation parameter within state dict.

    """

    def predictor(self, state: dict) -> np.ndarray:
        """Create predictor from the state dictionary using the functional form defined in the specific subclass.

        Args:
            state (dict): dictionary object containing the current state information

        Returns:
            (np.ndarray): predictor vector

        """
        return state[self.param][state[self.allocation].flatten()]

    def grad(self, state: dict, param: str):
        """Compute gradient of single parameter.

        Args:
            state (dict): Dictionary object containing the current state information
            param (str): Compute derivatives WRT this variable

        Returns:
            (np.ndarray): [n_param x n_data] array, gradient with respect to param

        """
        element_index = np.arange(0, state[param].size)

        return self.get_element_match(state, element_index).astype(np.float64).T

    def get_grad_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return [self.param]


@dataclass
class MixtureParameterMatrix(MixtureParameter):
    """Diagonal matrix parameter: elements of the diagonal are obtained from sub-parameter 'param' according to the allocation index vector.

    The allocation parameter defines a mapping between a R^m and R^n where typically m<=n and m is the
    number true underlying number of parameters in the model but due to the representation/algebra in
    other parts of the model this is expanded out to an n parameter model where the values of m are copied
    according to the index vector

    predictor = np.diag( param [allocation] )

    Attributes:
        param (str): name of underlying state component used to generate parameter.
        allocation (np.ndarray): name of allocation parameter within state dict.

    """

    def predictor(self, state: dict) -> sparse.csc_matrix:
        """Create predictor from the state dictionary using the functional form defined in the specific subclass.

        Args:
            state (dict): dictionary object containing the current state information

        Returns:
            (sparse.csc_matrix): predictor vector

        """
        return sparse.diags(diagonals=state[self.param][state[self.allocation]].flatten(), offsets=0, format="csc")

    def grad(self, state: dict, param: str):
        """Compute gradient of single parameter.

        Args:
            state (dict): Dictionary object containing the current state information
            param (str): Compute derivatives WRT this variable

        Returns:
            (np.ndarray): [n_param x n_data] array, gradient with respect to param

        """
        raise TypeError("Not defined in this case")

    def get_grad_param_list(self) -> list:
        """Extract list of components from parameter specification that grad is defined for.

        Returns:
            (list): parameter that grad is defined for.

        """
        return []

    def precision_unscaled(self, state: dict, element_index: int) -> np.ndarray:
        """Return the precision matrix un-scaled by the scalar precision parameter.

        Args:
            state (dict): state dictionary
            element_index (int): index of element to subset

        Returns:
            (np.ndarray): unscaled precision matrix

        """
        return sparse.diags(diagonals=self.get_element_match(state, element_index).flatten(), offsets=0, format="csc")
