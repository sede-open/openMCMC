# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""LocationScale module.

This module provides a class definition of the LocationScale class an abstract base class for distributions defined by a
mean and a precision such as the Normal and Lognormal.

"""

from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from scipy import sparse

from openmcmc import gmrf
from openmcmc.distribution.distribution import Distribution
from openmcmc.parameter import (
    Identity,
    LinearCombination,
    MixtureParameterMatrix,
    MixtureParameterVector,
    ScaledMatrix,
)


@dataclass
class LocationScale(Distribution, ABC):
    """Abstract base class for distributions defined by a mean and a precision such as the Normal and Lognormal.

    Attributes:
       mean (Union[str, Identity, LinearCombination, MixtureParameterVector]): mean parameter (of class Parameter).
       precision (Union[str, Identity, ScaledMatrix, MixtureParameterMatrix]): precision parameter (of class Parameter).

    """

    mean: Union[str, Identity, LinearCombination, MixtureParameterVector]
    precision: Union[str, Identity, ScaledMatrix, MixtureParameterMatrix]

    @property
    def _dist_params(self) -> list:
        """Return the full list of state elements used in the mean and precision parameters."""
        lst = self.mean.get_param_list() + self.precision.get_param_list()
        return lst

    def __post_init__(self):
        """Parse any str parameter inputs as Parameter classes."""
        if isinstance(self.mean, str):
            self.mean = Identity(self.mean)

        if not isinstance(self.mean, (Identity, LinearCombination, MixtureParameterVector)):
            raise TypeError("mean expected to be one of [Identity, LinearCombination, MixtureParameterVector]")

        if isinstance(self.precision, str):
            self.precision = Identity(self.precision)

        if not isinstance(self.precision, (Identity, ScaledMatrix, MixtureParameterMatrix)):
            raise TypeError("precision expected to be one of [Identity, ScaledMatrix, MixtureParameterMatrix]")


class NullDistribution(LocationScale):
    """Null distribution, which returns 0 for the log-likelihood, a zero vector for the gradient and a zero matrix for the Hessian.

    Used in prior recovery testing for reversible jump sampler.

    """

    def log_p(self, state: dict, by_observation: bool = False) -> float:
        """Null log-density function: returns 0.

        Args:
            state (dict): dictionary object containing the current state information. state[distribution.response]
                is expected to be p x n where: p is the number of responses; n is the number of independent
                replicates/observations.
            by_observation (bool, optional): If True, the log-likelihood is returned for each of the p responses of
                the distribution separately. Defaults to False.

        Returns:
            (float): 0.0.

        """
        return 0.0

    def grad_log_p(
        self, state: dict, param: str, hessian_required: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Null gradient function returning an all-zero vector for the gradient, and an all-zero matrix for the Hessian.

        Args:
            state (dict): current state information.
            param (str): name of the parameter for which we compute derivatives.
            hessian_required (bool): flag for whether the Hessian should be calculated and supplied as an output.

        Returns:
            (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): if hessian_required=True, then a tuple of (gradient,
                hessian) is returned. If hessian_required=False, then just a gradient vector is returned. The returned
                values are as follows:
                grad (np.ndarray): all-zero vector. shape=(d, 1), where d is the dimensionality of param.
                hessian (np.ndarray): all-zero matrix. shape=(d, d), where d is the dimensionality of param.

        """
        if hessian_required:
            return np.zeros(state[param].shape), np.zeros((state[param].shape[0], state[param].shape[0]))

        return np.zeros(state[param].shape)

    def rvs(self, state: dict, n: int = 1) -> None:
        """Null random sampling function.

        Args:
            state (dict): dictionary object containing the current state information.
            n (int, optional): specifies the number of replicate samples required. Defaults to 1.

        Returns:
            (None): simply returns None value.

        """
        return None


@dataclass
class Normal(LocationScale):
    """Multivariate normal distribution class.

    Supports both standard multivariate normal and truncated normal distribution cases. By default, no truncation is
    assumed. To truncate the distribution, one or both of self.domain_response_lower or self.domain_response_upper must
    be specified.

    Attributes:
        domain_response_lower (np.array, optional): check lower bound domain to implement truncated sampling. Defaults
            to None.
        domain_response_upper (np.array, optional): check upper bound domain to implement truncated sampling. Defaults
            to None.

    """

    domain_response_lower: np.ndarray = None
    domain_response_upper: np.ndarray = None

    def log_p(self, state: dict, by_observation: bool = False) -> Union[np.ndarray, float]:
        """Compute the log of the probability density for a given state.

        NOTE: This function simply computes the non-truncated Gaussian density: i.e. the extra normalization for the
        truncation is NOT accounted for. Relative densities (differences of log-probabilities) are still valid when
        comparing different response parameter values (with fixed mean and precision parameter values). Comparisons
        for different mean or precision parameters are not valid, since such changes would affect the normalization.

        Args:
            state (dict): dictionary object containing the current parameter information.
            by_observation (bool, optional): indicates whether log-density should be computed for each individual
                response in the distribution. Defaults to False (i.e. the overall log-density is computed).

        Returns:
            (Union[np.ndarray, float]): log-density computed using the values in state.

        """
        Q = self.precision.predictor(state)
        mu = self.mean.predictor(state)
        if self.check_domain_response(state):
            return -np.inf
        log_p = gmrf.multivariate_normal_pdf(x=state[self.response], mu=mu, Q=Q, by_observation=by_observation)
        return log_p

    def check_domain_response(self, state: dict) -> bool:
        """Checks whether the distributional response lies OUTSIDE the defined limits.

        Returns True if the current value of self.response in the supplied state lies OUTSIDE the stated domain;
        returns False otherwise.

        Args:
            state (dict): dictionary object containing the current parameter information.

        Returns:
            (bool): True when the response lies OUTSIDE the valid response domain; False when it lies INSIDE.

        """
        if self.domain_response_lower is not None:
            if np.any(state[self.response] < self.domain_response_lower):
                return True
        if self.domain_response_upper is not None:
            if np.any(state[self.response] > self.domain_response_upper):
                return True
        return False

    def grad_log_p(
        self, state: dict, param: str, hessian_required: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Gradient and Hessian of the log-Gaussian density, with respect to a given parameter.

        See also distribution.grad_log_p() for more information.

        Handles three possibilities:
            1) param is the response of the distribution, in which case the standard gradient of the log-density is
                returned.
            2) param is a parameter used in the computation of the mean (through a parameter object) and not in the
                computation of the precision, in which case the gradient is computed through application of the chain
                rule. Note that the Hessian calculated in this case is only valid if the dependence of self.mean on
                param is linear.
            3) neither of the above conditions is True, in which case the default finite-difference gradient is
                calculated (using self.grad_log_p_diff() and self.hessian_log_p_diff()). Note that as per those
                docstrings, it is only possible to compute gradients with respect to scalar or vector parameters.

        Args:
            state (dict): current state information.
            param (str): name of the parameter for which we compute derivatives.
            hessian_required (bool): flag for whether the Hessian should be calculated and supplied as an output.

        Returns:
            (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): if hessian_required=True, then a tuple of (gradient,
                hessian) is returned. If hessian_required=False, then just a gradient vector is returned. The returned
                values are as follows:
                grad (np.ndarray): vector gradients of the POSITIVE log-pdf with respect to param. shape=(n_param, 1)
                hessian (np.ndarray): array of NEGATIVE second derivatives of the log-pdf with respect to param.
                shape=(n_param, n_param)

        """
        if param in self.response:
            Q = self.precision.predictor(state)
            r = state[self.response] - self.mean.predictor(state)
            grad = -Q @ r
            if hessian_required:
                hessian = Q
                if state[param].shape[1] > 1 and sparse.issparse(Q):
                    hessian = sparse.kron(Q, sparse.eye(state[param].shape[1]))
                elif state[param].shape[1] > 1:
                    hessian = np.kron(Q, np.eye(state[param].shape[1]))
                return grad, hessian

        elif param in self.mean.get_grad_param_list() and param not in self.precision.get_grad_param_list():
            Q = self.precision.predictor(state)
            r = np.sum(state[self.response] - self.mean.predictor(state), axis=1, keepdims=True)
            grad_param = self.mean.grad(state, param)
            grad_times_prec = grad_param @ Q
            grad = grad_times_prec @ r
            if hessian_required:
                hessian = state[self.response].shape[1] * grad_times_prec @ grad_param.T
                return grad, hessian

        else:
            grad = self.grad_log_p_diff(state, param)
            if hessian_required:
                hessian = self.hessian_log_p_diff(state, param)
                return grad, hessian

        return grad

    def rvs(self, state: dict, n: int = 1) -> np.ndarray:
        """Generate random samples from the multivariate Gaussian distribution.

        Args:
            state (dict): dictionary object containing the current state information.
            n (int, optional): specifies the number of replicate samples required. Defaults to 1.

        Returns:
            (np.ndarray): random variables generated from distribution returned as p x n where p is the
                dimensionality of the response.

        """
        mean = self.mean.predictor(state)
        precision = self.precision.predictor(state)

        if self.domain_response_lower is None and self.domain_response_upper is None:
            return gmrf.sample_normal(mu=mean, Q=precision, n=n)

        return gmrf.sample_truncated_normal(
            mu=mean, Q=precision, lower=self.domain_response_lower, upper=self.domain_response_upper, n=n
        )


@dataclass
class LogNormal(LocationScale):
    """Multivariate log-normal distribution class."""

    def log_p(self, state: dict, by_observation: bool = False) -> np.ndarray:
        """Compute the log of the probability density (for current parameter settings).

        Args:
            state (dict): dictionary object containing the current state information. state[distribution.response]
                is expected to be p x n where: p is the number of responses; n is the number of independent
                replicates/observations.
            by_observation (bool, optional): If True, the log-likelihood is returned for each of the p responses of
                the distribution separately. Defaults to False.

        Returns:
            (Union[np.ndarray, float]): POSITIVE log-density evaluated using the supplied state dictionary.

        """
        Q = self.precision.predictor(state)
        mu = self.mean.predictor(state)
        log_p = gmrf.multivariate_normal_pdf(x=np.log(state[self.response]), mu=mu, Q=Q, by_observation=True) - np.sum(
            np.log(state[self.response]), axis=0
        )
        if not by_observation:
            log_p = np.sum(log_p)
        return log_p

    def grad_log_p(
        self, state: dict, param: str, hessian_required: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate vector of derivatives of the log-pdf with respect to a given parameter, and if required, also generate the Hessian.

        See also distribution.grad_log_p() for more information.

        Handles 3 possibilities:
            1) param is the response of the distribution, in which case the standard gradient of the log-density is
                returned.
            2) param is a parameter used in the computation of the mean (through a parameter object) and not in the
                computation of the precision, in which case the gradient is computed through application of the chain
                rule. Note that the Hessian calculated in this case is only valid if the dependence of self.mean on
                param is linear.
            3) neither of the above conditions is True, in which case the default finite-difference gradient is
                calculated (using self.grad_log_p_diff() and self.hessian_log_p_diff()). Note that as per those
                docstrings, it is only possible to compute gradients with respect to scalar or vector parameters.

        Args:
            state (dict): current state information.
            param (str): name of the parameter for which we compute derivatives.
            hessian_required (bool): flag for whether the Hessian should be calculated and supplied as an output.

        Returns:
            (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): if hessian_required=True, then a tuple of (gradient,
                hessian) is returned. If hessian_required=False, then just a gradient vector is returned. The returned
                values are as follows:
                grad (np.ndarray): vector gradients of the POSITIVE log-pdf with respect to param. shape=(d, 1), where
                d is the dimensionality of param.
                hessian (np.ndarray): array of NEGATIVE second derivatives of the log-pdf with respect to param.
                shape=(d, d), where d is the dimensionality of param.

        """
        Q = self.precision.predictor(state)
        if param in self.response:
            r = np.log(state[self.response]) - self.mean.predictor(state)
            grad = -(1 / state[self.response]) * (1 + Q @ r)
        elif param in self.mean.get_grad_param_list() and param not in self.precision.get_grad_param_list():
            r = np.sum(np.log(state[self.response]) - self.mean.predictor(state), axis=1, keepdims=True)
            grad_param = self.mean.grad(state, param)
            grad = grad_param @ Q @ r
        else:
            grad = self.grad_log_p_diff(state, param)

        if hessian_required:
            hessian = self.hessian_log_p(state, param)
            return grad, hessian

        return grad

    def hessian_log_p(self, state: dict, param: str) -> np.ndarray:
        """Compute Hessian of the log-density with respect to a given parameter.

        Handles 3 possibilities:
            1) param is the response of the distribution, in which case the Hessian of the log-density is computed
                directly.
            2) param is a parameter used in the computation of the mean (through a parameter object) and not in the
                computation of the precision, and the dependence of the mean parameter on param is linear. The chain
                rule is used to determine the Hessian.
            3) neither of the above conditions is True, in which case the default finite-difference gradient is
                calculated (using self.hessian_log_p_diff()). Note that as per the docstring of
                self.hessian_log_p_diff(), it is only possible to compute gradients with respect to scalar or vector
                parameters.

        NOTE: sparse implementation of response hessian currently converts Q from sparse.

        Args:
            state (dict): contains current state information.
            param (str): name of the parameter for which we compute derivatives.

        Returns:
            (np.ndarray): Hessian of log-density wrt the specified param.

        """
        if param in self.response:
            Q = self.precision.predictor(state)
            r = np.log(state[self.response]) - self.mean.predictor(state)
            reciprocal = 1 / state[self.response]

            if sparse.issparse(Q):
                hess_p = -sparse.diags((np.power(reciprocal, 2) * (1 + Q @ r)).flatten(), offsets=0)
                Q = Q.toarray()
            else:
                hess_p = -np.diagflat(np.power(reciprocal, 2) * (1 + Q @ (r)))

            dim, n = state[self.response].shape
            out = np.zeros((n, dim, n, dim))
            diag = np.einsum("ijik->ijk", out)
            np.einsum("ik, ij, jk -> kij", reciprocal, Q, reciprocal, out=diag)
            out = out.transpose((1, 0, 3, 2))
            out = out.reshape((n * dim, n * dim))
            hess_p = out + hess_p

        elif param in self.mean.get_grad_param_list() and param not in self.precision.get_grad_param_list():
            Q = self.precision.predictor(state)
            grad_param = self.mean.grad(state, param)
            hess_p = state[self.response].shape[1] * grad_param @ Q @ grad_param.T
        else:
            hess_p = self.hessian_log_p_diff(state, param)

        return hess_p

    def rvs(self, state: dict, n: int = 1) -> np.ndarray:
        """Generate random samples from the multivariate log-Gaussian distribution.

        Args:
            state (dict): dictionary object containing the current state information.
            n (int, optional): specifies the number of replicate samples required. Defaults to 1.

        Returns:
            (np.ndarray): random variables generated from distribution returned as p x n where p is the
                dimensionality of the response.

        """
        mean = self.mean.predictor(state)
        precision = self.precision.predictor(state)
        return np.exp(gmrf.sample_normal(mu=mean, Q=precision, n=n))
