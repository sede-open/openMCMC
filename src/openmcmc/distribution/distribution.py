# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collection of distributions for use with openMCMC code.

General assumptions about code functionality:
    - The first dimension of a parameter array is assumed to represent the dimensionality of the parameter vector; the
        second dimension is assumed to represent independent realizations of the parameter set. For example: an array
        with shape=(d, n) would be assumed to hold n replicates of a d-dimensional parameter vector.
    - self.response is a string containing the name of the response parameter for the distribution. For example, when
        self.response="y", all functions within the class will perform calculations using the value stored in
        state["y"].

"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from scipy import sparse, stats

from openmcmc.parameter import Identity, LinearCombination, MixtureParameterVector


@dataclass
class Distribution(ABC):
    """Abstract superclass for handling distribution objects.

    Attributes:
        response (str): specifies the name of the response variable of the distribution.

    """

    response: str

    @abstractmethod
    def log_p(self, state: dict, by_observation: bool = False) -> Union[np.ndarray, float]:
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

    @abstractmethod
    def rvs(self, state: dict, n: int = 1) -> np.ndarray:
        """Generate random samples from the distribution.

        Args:
            state (dict): dictionary object containing the current state information.
            n (int, optional): specifies the number of replicate samples required. Defaults to 1.

        Returns:
            (np.ndarray): random variables generated from distribution returned as p x n where p is the
                dimensionality of the response.

        """

    @property
    @abstractmethod
    def _dist_params(self) -> list:
        """Get list of parameter labels across all Parameter objects in distribution (EXCLUDING the response).

        Returns:
            (list): list of parameter labels.

        """

    @property
    def param_list(self) -> list:
        """Get list of all parameter labels in model (INCLUDING the response).

        Returns:
            (list): list of parameter labels

        """
        lst = [self.response] + self._dist_params
        return lst

    def grad_log_p(
        self, state: dict, param: str, hessian_required: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate vector of derivatives of the log-pdf with respect to a given parameter, and if required, also generate the Hessian.

        Function only defined for scalar- and vector-valued parameters param. If hessian_required=True, this function
        returns a tuple of (gradient, Hessian). If hessian_required=False, this function returns a np.ndarray (just
        the gradient of the log-density).

        As a default, the individual gradients are computed by finite-differencing the log_p function defined for the
        distribution. Where analytical forms for the gradient exist, these will be defined in distribution-specific
        subclasses.

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
        grad = self.grad_log_p_diff(state=state, param=param)
        if hessian_required:
            hessian = self.hessian_log_p_diff(state=state, param=param)
            return grad, hessian
        return grad

    def grad_log_p_diff(self, state: dict, param: str, step_size: float = 1e-4) -> np.ndarray:
        """Compute vector of derivatives of the POSITIVE log-pdf (with respect to param) using central differences.

        Args:
            state (dict): current state information.
            param (str): name of the parameter for which we compute derivatives.
            step_size (float, optional): step size to use for the finite difference derivatives. Defaults to 1e-4.

        Returns:
            (np.ndarray): vector of log-pdf gradients with respect to param. shape=(d, 1), where d is the dimensionality
                of param.

        """
        n_param = np.prod(state[param].shape)
        grad_param = np.full(shape=n_param, fill_value=np.nan)
        for k in range(n_param):
            state_plus = deepcopy(state)
            state_minus = deepcopy(state)

            if sparse.issparse(state[param]):
                m, n = state[param].shape
                step_temp = sparse.csr_array(
                    (np.array([step_size / 2]), np.unravel_index(np.array([k]), (m, n))), shape=(m, n)
                )
                state_plus[param] = state_plus[param] + step_temp
                state_minus[param] = state_minus[param] - step_temp
            else:
                state_plus[param][np.unravel_index(k, state[param].shape)] += step_size / 2
                state_minus[param][np.unravel_index(k, state[param].shape)] += -step_size / 2

            log_p_plus = self.log_p(state=state_plus)
            log_p_minus = self.log_p(state=state_minus)

            grad_param[k] = (log_p_plus - log_p_minus) / step_size
        return grad_param.reshape(state[param].shape)

    def hessian_log_p_diff(self, state: dict, param: str, step_size: float = 1e-4) -> np.ndarray:
        """Compute Hessian matrix of second derivatives of the NEGATIVE log-pdf (with respect to param) using finite differences.

        Args:
            state (dict): current state information.
            param (str): name of the parameter for which we compute derivatives.
            step_size (float, optional): step size to use for the finite difference derivatives. Defaults to 1e-4

        Returns:
            (np.ndarray): matrix of log-pdf second derivatives with respect to param. shape=(d, d), where d is the
                dimensionality of param.

        """
        n_param = np.prod(state[param].shape)
        hess_param = np.full(shape=(n_param, n_param), fill_value=np.nan)
        for k in range(n_param):
            state_plus = deepcopy(state)
            state_minus = deepcopy(state)

            if sparse.issparse(state[param]):
                m, n = state[param].shape
                step_temp = sparse.csr_array(
                    (np.array([step_size / 2]), np.unravel_index(np.array([k]), (m, n))), shape=(m, n)
                )
                state_plus[param] = state_plus[param] + step_temp
                state_minus[param] = state_minus[param] - step_temp
            else:
                state_plus[param][np.unravel_index(k, state[param].shape)] += step_size / 2
                state_minus[param][np.unravel_index(k, state[param].shape)] += -step_size / 2

            grad_p_plus = self.grad_log_p(state_plus, param, hessian_required=False)
            grad_p_minus = self.grad_log_p(state_minus, param, hessian_required=False)

            hess_param[:, k] = (grad_p_minus - grad_p_plus).flatten() / step_size

        return hess_param


@dataclass
class Gamma(Distribution):
    """Gamma distribution class defined using shape and rate convention.

    f(x) = x^(shape-1) * exp(-rate*x) * rate^shape / Gamma(shape)

    Attributes:
        shape (Union[str, Identity, LinearCombination, MixtureParameterVector]): Gamma shape parameter.
        rate (Union[str, Identity, LinearCombination, MixtureParameterVector]): Gamma rate parameter.

    """

    shape: Union[str, Identity, LinearCombination, MixtureParameterVector]
    rate: Union[str, Identity, LinearCombination, MixtureParameterVector]

    def __post_init__(self):
        """Parse any str parameter inputs as Parameter.Identity, and check the parameter types."""
        if isinstance(self.shape, str):
            self.shape = Identity(self.shape)

        if not isinstance(self.shape, (Identity, LinearCombination, MixtureParameterVector)):
            raise TypeError("shape expected to be one of [Identity, LinearCombination, MixtureParameterVector]")

        if isinstance(self.rate, str):
            self.rate = Identity(self.rate)

        if not isinstance(self.rate, (Identity, LinearCombination, MixtureParameterVector)):
            raise TypeError("rate expected to be one of [Identity, LinearCombination, MixtureParameterVector]")

    @property
    def _dist_params(self) -> list:
        """Get list of parameter labels across all Parameter objects in distribution (EXCLUDING the response).

        Returns:
            (list): list of parameter labels.

        """
        lst = self.shape.get_param_list() + self.rate.get_param_list()
        return lst

    def log_p(self, state: dict, by_observation: bool = False) -> Union[np.ndarray, float]:
        """Compute the log of the probability density (for current parameter settings).

        Args:
            state (dict): dictionary object containing the current state information. state[distribution.response]
                is expected to be p x n where: p is the number of parameters; n is the number of independent
                replicates/observations.
            by_observation (bool, optional): If True, the log-likelihood is returned for each of the p parameters of
                the distribution separately. Defaults to False.

        Returns:
            (Union[np.ndarray, float]): POSITIVE log-density evaluated using the supplied state dictionary.

        """
        log_p = np.sum(
            stats.gamma.logpdf(state[self.response], self.shape.predictor(state), scale=1 / self.rate.predictor(state)),
            axis=0,
        )
        if not by_observation:
            log_p = np.sum(log_p)
        return log_p

    def rvs(self, state, n: int = 1) -> np.ndarray:
        """Generate random samples from the Gamma distribution.

        Args:
            state (dict): dictionary object containing the current state information.
            n (int, optional): specifies the number of replicate samples required. Defaults to 1.

        Returns:
            (np.ndarray): random variables generated from distribution returned as p x n where p is the
                dimensionality of the response.

        """
        shape = self.shape.predictor(state)
        rate = self.rate.predictor(state)
        p = max(shape.shape[0], rate.shape[0])
        return stats.gamma.rvs(shape, scale=1 / rate, size=(p, n))


@dataclass
class Categorical(Distribution):
    """Categorical distribution: equivalent to a single trial of a multinomial distribution.

    A 2-category categorical distribution is equivalent to a Bernoulli distribution.

    The response of this distribution is a category index in {0, 1, 2,..., n_cat}: thus, state[self.response] is
    expected to be a np.array with dtype=int. As per other distributions, the expected shape of state[self.response]
    is (p, n), where p=dimensionality of response, and n=number of replicates.

    The prior probability parameter is expected to be a np.ndarray with shape=(p, n_cat).

    Attributes:
        prob (Identity, str): allocation probability parameter.

    """

    prob: Union[str, Identity]

    def __post_init__(self):
        """Parse any str parameter inputs as Parameter.Identity(), and check the parameter types."""
        if isinstance(self.prob, str):
            self.prob = Identity(self.prob)

        if not isinstance(self.prob, Identity):
            raise TypeError("prob expected to be Identity")

    @property
    def _dist_params(self) -> list:
        """Get list of parameter labels across all Parameter objects in distribution (EXCLUDING the response).

        Returns:
            (list): list of parameter labels.

        """
        return self.prob.get_param_list()

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
        n_categories = self.prob.predictor(state).shape[1]
        n = state[self.response].shape[1]

        if n > 1:
            x = np.atleast_3d(state[self.response])
            x = np.equal(np.transpose(x, (0, 2, 1)), np.atleast_3d(range(n_categories)))
        else:
            x = state[self.response] == range(n_categories)

        if by_observation:
            if n > 1:
                prob = np.transpose(np.atleast_3d(self.prob.predictor(state)), (0, 2, 1))
                log_p = stats.multinomial.logpmf(np.transpose(x, (0, 2, 1)), n=1, p=prob)
            else:
                log_p = stats.multinomial.logpmf(x, n=1, p=self.prob.predictor(state))
        else:
            if n > 1:
                x = np.sum(x, axis=2)
            log_p = stats.multinomial.logpmf(x, n=n, p=self.prob.predictor(state))

        return np.sum(log_p, axis=0)

    def rvs(self, state, n: int = 1) -> np.ndarray:
        """Generate a random sample from the distribution.

        Args:
            state (dict): dictionary object containing the current state information
            n (int, optional): specifies the number of random variables required. Defaults to 1

        Returns:
            (np.ndarray): random sample from the categorical distribution. shape=(p, n)

        """
        prob = self.prob.predictor(state)

        d, _ = prob.shape

        cat = np.empty((d, n))
        for i in range(d):
            Z = stats.multinomial.rvs(n=1, p=prob[i, :], size=n)
            _, cat[i, :] = np.nonzero(Z)

        return cat


@dataclass
class Uniform(Distribution):
    """Uniform distribution class for a p-dimensional hyper-rectangle.

    Attributes:
        domain_response_lower (Union[float, np.ndarray]): shape=(p, 1): lower limits for uniform distribution in each
            dimension. Defaults to 0.0.
        domain_response_upper (Union[float, np.ndarray]): shape=(p, 1) upper limits for uniform distribution in each
            dimension. Defaults to 1.0.

    """

    domain_response_lower: Union[float, np.ndarray] = 0.0
    domain_response_upper: Union[float, np.ndarray] = 1.0

    def __post_init__(self):
        """Convert any domain limits supplied as floats to np.ndarray."""
        self.domain_response_lower = np.array(self.domain_response_lower, ndmin=2)
        if self.domain_response_lower.shape[0] == 1:
            self.domain_response_lower = self.domain_response_lower.T
        self.domain_response_upper = np.array(self.domain_response_upper, ndmin=2)
        if self.domain_response_upper.shape[0] == 1:
            self.domain_response_upper = self.domain_response_upper.T

    @property
    def _dist_params(self) -> list:
        """Uniform distribution doesn't have parameters, so return an empty list."""
        return []

    def domain_range(self, state) -> np.ndarray:
        """Get the domain range (upper-lower) from domain_limits.

        Args:
            state (dict): dictionary with current state information.

        Returns:
            (np.ndarray): domain range. shape=(p, 1).

        """
        d = state[self.response].shape[0]
        domain_range = self.domain_response_upper - self.domain_response_lower
        if domain_range.size == 1:
            domain_range = np.ones((d, 1)) * domain_range
        return domain_range

    def log_p(self, state: dict, by_observation: bool = False) -> Union[np.ndarray, float]:
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
        n = state[self.response].shape[1]
        log_p = -np.sum(np.log(self.domain_range(state)))
        if by_observation:
            log_p = np.ones(n) * log_p
        else:
            log_p = n * log_p
        return log_p

    def rvs(self, state, n: int = 1) -> np.ndarray:
        """Generate random samples from the distribution.

        Args:
            state (dict): dictionary object containing the current state information.
            n (int, optional): specifies the number of replicate samples required. Defaults to 1.

        Returns:
            (np.ndarray): random variables generated from distribution returned as p x n where p is the
                dimensionality of the response.

        """
        standard_unif = np.random.rand(state[self.response].shape[0], n)
        return self.domain_response_lower + self.domain_range(state) * standard_unif


@dataclass
class Poisson(Distribution):
    """Poisson distribution for count data.

    Attributes:
        rate (Union[str, Identity, LinearCombination, MixtureParameterVector]): Poisson rate parameter.

    """

    rate: Union[str, Identity, LinearCombination, MixtureParameterVector]

    def __post_init__(self):
        """Parse any str parameter inputs as Parameter.Identity, and check the parameter types."""
        if isinstance(self.rate, str):
            self.rate = Identity(self.rate)

        if not isinstance(self.rate, (Identity, LinearCombination, MixtureParameterVector)):
            raise TypeError("rate expected to be one of [Identity, LinearCombination, MixtureParameterVector]")

    @property
    def _dist_params(self) -> list:
        """Get list of parameter labels across all Parameter objects in distribution (EXCLUDING the response).

        Returns:
            (list): list of parameter labels.

        """
        return self.rate.get_param_list()

    def log_p(self, state: dict, by_observation: bool = False) -> np.ndarray:
        """Compute the log of the probability density (for current parameter settings).

        Args:
            state (dict): dictionary object containing the current state information. state[distribution.response]
                is expected to be p x n where: p is the number of parameters; n is the number of independent
                replicates/observations.
            by_observation (bool, optional): If True, the log-likelihood is returned for each of the p parameters of
                the distribution separately. Defaults to False.

        Returns:
            (Union[np.ndarray, float]): POSITIVE log-density evaluated using the supplied state dictionary.

        """
        rate = self.rate.predictor(state)
        logpmf = np.sum(stats.poisson.logpmf(state[self.response], rate), axis=0)
        if not by_observation:
            logpmf = np.sum(logpmf)
        return logpmf

    def rvs(self, state: dict, n: int = 1) -> np.ndarray:
        """Generate random samples from the Poisson distribution.

        Args:
            state (dict): dictionary object containing the current state information.
            n (int, optional): specifies the number of replicate samples required. Defaults to 1.

        Returns:
            (np.ndarray): random variables generated from distribution returned as p x n where p is the
                dimensionality of the response.

        """
        rate = self.rate.predictor(state)
        return stats.poisson.rvs(mu=rate, size=(rate.shape[0], n))
