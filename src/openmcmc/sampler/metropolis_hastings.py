# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""MetropolisHastings module.

This module provides a class definition of the MetropolisHastings class an abstract base class for implementation of
Metropolis-Hastings-type sampling algorithms for a model.

"""

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
from scipy.stats import norm

from openmcmc import gmrf
from openmcmc.sampler.sampler import MCMCSampler


@dataclass
class AcceptRate:
    """Class for dealing with calculation of acceptance rates.

    Called from MetropolisHastings-type samplers.

    Attributes:
        count: counters of current number of proposals and accepted proposals from a MH chain

    """

    def __init__(self):
        self.count = {"accept": 0, "proposal": 0}

    @property
    def acceptance_rate(self) -> float:
        """Acceptance rate property, as calculated from counters.

        Returns:
            (float): percentage proposals accepted in chain

        """
        return self.count["accept"] / self.count["proposal"] * 100

    def get_acceptance_rate(self) -> str:
        """Return acceptance rate formatted as string.

        Returns:
            (str): acceptance rate string print out

        """
        if self.count["proposal"] == 0:
            return "No proposals"
        return f"Acceptance rate {self.acceptance_rate:.0f}%"

    def increment_accept(self):
        """Increment acceptance count."""
        self.count["accept"] += 1

    def increment_proposal(self):
        """Increment proposal count."""
        self.count["proposal"] += 1


@dataclass
class MetropolisHastings(MCMCSampler):
    """Abstract base class for implementation of Metropolis-Hastings-type sampling algorithms for a model.

    Subclasses include RandomWalk and ManifoldMALA.

    https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

    Attributes:
        step (np.ndarray): step size for Metropolis-Hastings proposals. Should either have shape=(p, 1) or shape=(p, n),
            where p is the dimension of the parameter, and n is the number of replicates.
        accept_rate (AcceptRate): Acceptance Rate counter to keep track of proposals.

    """

    step: np.ndarray = field(default_factory=lambda: np.array([0.2], ndmin=2), init=True)
    accept_rate: AcceptRate = field(default_factory=lambda: AcceptRate(), init=False)

    @abstractmethod
    def proposal(self, current_state: dict, param_index: int = None) -> Tuple[dict, float, float]:
        """Method which generates proposed state from current state, and computes corresponding transition probabilities.

        Args:
            current_state (dict): current state
            param_index (int): subset of parameter used in proposal, If none all parameters are used

        Returns:
            prop_state (dict): updated proposal_state dictionary.
            logp_pr_g_cr (float): log-density of proposed state given current state.
            logp_cr_g_pr (float): log-density of current state given proposed state.

        """

    def sample(self, current_state: dict) -> dict:
        """Generate a sample from the specified Metropolis-Hastings-type method.

        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

        generate proposal state x' from  current_state x and accept or reject proposal according to the probability:
        A(x',x) = min(1, (P(x')g(x|x'))/(P(x)g(x'|x)))
        where:
            - P(x) is the probability of the state x
            - g(x|x') is the probability of moving from state x to x'

        The exact method for the proposal (and therefore the form of the proposal distribution) is determined by the
        specific type of MetropolisHastings Sampler used.

        Args:
            current_state (dict): dictionary containing the current sampler state.

        Returns:
            current_state (dict): with updated sample for self.param.

        """
        prop_state, logp_pr_g_cr, logp_cr_g_pr = self.proposal(current_state)
        current_state = self._accept_reject_proposal(current_state, prop_state, logp_pr_g_cr, logp_cr_g_pr)
        return current_state

    def _accept_reject_proposal(
        self, current_state: dict, prop_state: dict, logp_pr_g_cr: float, logp_cr_g_pr: float
    ) -> dict:
        """Accept or Reject Metropolis-Hastings-type proposal.

        Computes the log posterior for the current and proposed states, and evaluates the log acceptance probability.
        Accepts the proposal with probability A(x, x'), and returns either the proposed or the current state
        accordingly.

        Increments self.acceptance_rate() to indicate that a proposal has been made, and also increments the acceptance
        counter if the proposal is subsequently accepted.

        Args:
            current_state (dict): current state dictionary
            prop_state (dict): proposal_state dictionary
            logp_pr_g_cr (float): log posterior of proposal given current state
            logp_cr_g_pr (float): log posterior of current state given proposals

        Returns:
            (dict): updated current state dictionary, after the proposal has either been accepted or rejected.

        """
        self.accept_rate.increment_proposal()
        logp_cs = 0
        logp_pr = 0
        for model in self.model.values():
            logp_cs += model.log_p(current_state)
            logp_pr += model.log_p(prop_state)
        log_accept = logp_pr + logp_cr_g_pr - (logp_cs + logp_pr_g_cr)

        if self.accept_proposal(log_accept):
            current_state = prop_state
            self.accept_rate.increment_accept()
        return current_state

    @staticmethod
    def accept_proposal(log_accept: float) -> bool:
        """Decide to accept or reject proposal based on log acceptance probability.

        Args:
            log_accept (np.float64): log acceptance probability.

        Returns:
            (bool): True for accept, False for Reject.

        """
        return np.log(np.random.rand()) < log_accept


@dataclass
class RandomWalk(MetropolisHastings):
    """Subtype of MetropolisHastings sampler that uses Gaussian random Walk proposals.

    Supports both non-truncated and truncated Gaussian proposals: specifying self.domain limits leads to a truncated
    proposal mechanism.

    Allows for the possibility that other elements of the model state have a dependence on the value of self.param, and
    if so should change when this value changes. If supplied, the self.state_update_function() property is called by the
    proposal function to update any other elements of the state as required.

    Attributes:
        domain_limits (np.ndarray): array with shape=(p, 2), where p is the dimensionality of the parameter being
            sampled. The first column gives the lower limits for the proposal, the second column gives the upper limits.
        state_update_function (Callable): function which updates other elements of proposed state based on the proposed
            value for param.

    """

    domain_limits: np.ndarray = None
    state_update_function: Callable = None

    def __post_init__(self):
        """Derive conditional model instead of storing all distributions where things are simple.

        However, this should not be done in the case where a state_update_function is provided as we don't know  in
        general what/how parameters might change so need to keep full model to avoid incorrect conditioning.

        """
        if self.state_update_function is None:
            self.model = self.model.conditional(self.param)
        self.step = np.array(self.step, ndmin=2)

    def proposal(self, current_state: dict, param_index: int = None) -> Tuple[dict, float, float]:
        """Updates the current value of self.param using a (truncated) Gaussian random walk proposal.

        In the non-truncated case, the proposal mechanism is symmetric, i.e. logp_pr_g_cr = logp_cr_g_pr. In this
        instance, the function simply returns logp_pr_g_cr = logp_cr_g_pr = 0, since these terms would anyway cancel
        in the calculation of the acceptance ratio.

        Introducing a truncation into the proposal distribution means that the proposal is no longer symmetric, and so
        the log-proposal densities are computed in these cases.

        Enables 3 different possibilities for the step size:
            1) shape=(1, 1): scalar step size, identical for every element of the parameter.
            2) shape=(p, 1): step size with the same shape as the parameter being sampled (for one or many replicates).
            3) shape=(p, n): a p-dimensional step size for each of n-replicates.

        Args:
            current_state (dict): dictionary containing current parameter values.
            param_index (int): subset of parameter used in proposal, If none all parameters are used

        Returns:
            prop_state (dict): updated proposal_state dictionary.
            logp_pr_g_cr (float): log-density of proposed state given current state.
            logp_cr_g_pr (float): log-density of current state given proposed state.

        """
        prop_state = deepcopy(current_state)

        if param_index is None:
            mu = prop_state[self.param]
            step = self.step
        else:
            mu = prop_state[self.param][:, param_index]
            if self.step.shape[1] == 1:
                step = self.step.flatten()
            else:
                step = self.step[:, param_index].flatten()

        if self.domain_limits is None:
            z = mu + norm.rvs(size=prop_state[self.param].shape, scale=step)
            logp_pr_g_cr = logp_cr_g_pr = 0.0
        else:
            lb = self.domain_limits[:, 0]
            ub = self.domain_limits[:, 1]
            z = gmrf.truncated_normal_rv(mean=mu, scale=step, lower=lb, upper=ub, size=len(lb))
            logp_pr_g_cr = np.sum(gmrf.truncated_normal_log_pdf(z, mu, step, lower=lb, upper=ub))
            logp_cr_g_pr = np.sum(gmrf.truncated_normal_log_pdf(mu, z, step, lower=lb, upper=ub))

        if param_index is None:
            prop_state[self.param] = z
        else:
            prop_state[self.param][:, param_index] = z

        if callable(self.state_update_function):
            prop_state = self.state_update_function(prop_state, param_index)

        return prop_state, logp_pr_g_cr, logp_cr_g_pr


@dataclass
class RandomWalkLoop(RandomWalk):
    """Subtype of MetropolisHastings sampler which updates each of n replicates of a parameter one-at-a-time, rather than all simultaneously."""

    def sample(self, current_state: dict) -> dict:
        """Update each of n replicates of a given parameter in a loop, rather than simultaneously.

        Args:
            current_state (dict): dictionary containing the current sampler state.

        Returns:
            current_state (dict): with updated sample for self.param.

        """
        for param_index in range(current_state[self.param].shape[1]):
            prop_state, logp_pr_g_cr, logp_cr_g_pr = self.proposal(current_state, param_index)
            current_state = self._accept_reject_proposal(current_state, prop_state, logp_pr_g_cr, logp_cr_g_pr)
        return current_state


@dataclass
class ManifoldMALA(MetropolisHastings):
    """Class implementing manifold Metropolis-adjusted Langevin algorithm (mMALA) proposal mechanism.

    Reference: Riemann manifold Langevin and Hamiltonian Monte Carlo methods, Mark Girolami, Ben Calderhead,
    03 March 2011 https://doi.org/10.1111/j.1467-9868.2010.00765.x

    """

    def proposal(self, current_state: dict, param_index: int = None) -> Tuple[dict, np.ndarray, np.ndarray]:
        """Generate mMALA proposed state from current state using gradient and hessian, and compute corresponding log-transition probabilities.

        Args:
            current_state (dict): dictionary containing current parameter values.
            param_index (int): required input from superclass. Not used; defaults to None.

        Returns:
            prop_state (dict): updated proposal_state dictionary.
            logp_pr_g_cr (np.ndarray): log-density of proposed state given current state.
            logp_cr_g_pr (np.ndarray): log-density of current state given proposed state.

        """
        prop_state = deepcopy(current_state)

        mu_cr, chol_cr = self._proposal_params(current_state)
        prop_state[self.param] = gmrf.sample_normal(mu_cr, L=chol_cr)
        logp_pr_g_cr = self._log_proposal_density(prop_state, mu_cr, chol_cr)

        mu_pr, chol_pr = self._proposal_params(prop_state)
        logp_cr_g_pr = self._log_proposal_density(current_state, mu_pr, chol_pr)

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def _proposal_params(self, current_state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the mean vector and the Cholesky factorization of the precision matrix for the mMALA proposal.

        The density for either the forward or return proposal in an mMALA scheme is a Gaussian. In the case of the
        forward proposal, the density is as follows:
            q(prop | theta_0) ~ N(mu*, stp^2 * H ^-1 )
        where:
            mu* = theta_0 + 1/2 * stp^2 * H ^-1 @ G
            H = hessian(theta_0)
            G = gradient(theta_0)

        Args:
            current_state (dict): dictionary containing current parameter values.

        Returns:
            mu_cr (np.ndarray): mean for proposal distribution, shape=(p, 1).
            chol_cr (np.ndarray): lower triangular Cholesky factorization of precision matrix, shape=(p, p).

        """
        grad_cr, hessian_cr = self.model.grad_log_p(current_state, param=self.param, hessian_required=True)
        precision_cr = hessian_cr / (self.step**2)
        chol_cr = gmrf.cholesky(precision_cr)
        mu_cr = current_state[self.param] + (1 / 2) * gmrf.cho_solve((chol_cr, True), grad_cr).reshape(grad_cr.shape)
        return mu_cr, chol_cr

    def _log_proposal_density(self, state: dict, mu: np.ndarray, chol: np.ndarray) -> np.ndarray:
        """Evaluate the log-proposal density for the mMALA transition.

        log determinant calculated using:
        https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html


        A quadratic form can be expressed in terms of the Cholesky factorization of the matrix as:
            r' Q r = r' L L' r = w' w =sum(w^2)
        where:
            w = L' r
            r = prm - mu

        Args:
            state (dict): dictionary containing parameter values.
            mu (np.ndarray): mean vector, shape=(p, 1).
            chol (np.ndarray): LOWER triangular cholesky factorization of the precision matrix, shape=(p, p)

        Returns:
            (np.ndarray): log-transition probability.

        """
        w = chol.transpose() @ (state[self.param] - mu)
        return np.sum(np.log(chol.diagonal())) - 0.5 * w.T.dot(w)
