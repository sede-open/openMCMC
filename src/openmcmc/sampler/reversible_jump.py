# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""ReversibleJump module.

This module provides a class definition of the ReversibleJump class a class for reversible jump sampling for given
parameter and associated parameters.

"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
from scipy.stats import randint, uniform

from openmcmc import gmrf
from openmcmc.sampler.metropolis_hastings import MetropolisHastings


@dataclass
class ReversibleJump(MetropolisHastings):
    """Reversible jump sampling for given parameter and associated parameter.

    self.param corresponds to a number of elements, which will either increase of decrease by 1. self.associated_params
    corresponds to an associated set of self.param parameters, to which we either add or remove an element for a birth
    or death move.

    The attributes self.state_birth_function and self.state_death_function can be used to supply functions which
    implement problem-specific alterations to elements of the state on the occurrence of a birth or death move
    respectively. For example, it may be required to update a basis matrix in the state after a change in the number
    of knots/locations associated with the basis definition.

    The functions self.matched_birth_transition and self.matched_death_transition implement optional functionality which
    can be used to ensure consistency between sets of basis parameters before and after a transition. These work by
    ensuring that the basis predictions before and after the transition match, then applies Gaussian random noise (with
    a given standard deviation) to the coefficient of the new element.

    Attributes:
        associated_params (list or string): a list or a string associated with the dimension jump. List of additional
            parameters that need to be created/removed as part of the dimension change. The default behaviour is to
            sample the necessary additional values from the associated parameter prior distribution. Defaults to None.
        n_max (int): upper limit on self.param (lower limit is assumed to be 1).
        birth_probability (float): probability that a birth move is chosen on any given iteration of the algorithm
            (death_probability = 1 - birth_probability). Defaults to 0.5.
        state_birth_function (Callable): function which implements problem-specific requirements for updates to elements
            of the state as part of a birth function (e.g. updates to a problem-specific basis matrix based given
            additional location parameters). Defaults to None.
        state_death_function (Callable): function which implements problem-specific requirements for updates to elements
            of state as part of a death function. Should mirror the supplied state_birth_function. Defaults to None.
        matching_params (dict): dictionary of parameters required for the matched coefficient transitions- for details
            of what it should contain, see self.matched_birth_transition.

    """

    associated_params: Union[list, str, None] = None
    n_max: Union[int, None] = None
    birth_probability: float = 0.5
    state_birth_function: Union[Callable, None] = None
    state_death_function: Union[Callable, None] = None
    matching_params: Union[dict, None] = None

    def __post_init__(self):
        """Empty function to prevent super.__post_init__ from being run.

        The whole model should be attached in this instance, rather than simply those elements with a dependence on
        self.param.

        """
        if isinstance(self.associated_params, str):
            self.associated_params = [self.associated_params]

    def proposal(self, current_state: dict, param_index: int = None) -> Tuple[dict, float, float]:
        """Make a proposal, and compute related transition probabilities for the move.

        Args:
            current_state (dict): dictionary with current parameter values.
            param_index (int): not used, included for compatibility with superclass.

        Returns:
            prop_state (dict): dictionary updated with proposed value for self.param.
            logp_pr_g_cr (float): transition probability for proposed state given current state.
            logp_cr_g_pr (float): transition probability for current state given proposed state.

        """
        birth = self.get_move_type(current_state)
        if birth:
            prop_state, logp_pr_g_cr, logp_cr_g_pr = self.birth_proposal(current_state=current_state)
        else:
            prop_state, logp_pr_g_cr, logp_cr_g_pr = self.death_proposal(current_state=current_state)
        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def birth_proposal(self, current_state: dict) -> Tuple[dict, float, float]:
        """Make a birth proposal move: INCREASES state[self.param] by 1.

        Also makes a proposal for a new element of an associated parameter, state[self.associated_params], by generating a draw
        from the prior distribution for self.associated_params.

        self.state_birth_function() is a function which can be optionally specified for altering the dimensionality of
        any other parameters associated with the dimension change (e.g. a basis matrix, or an allocation parameter).

        If the self.matching_params dictionary is specified, self.matched_birth_transition() is used to generate a
        proposal for a set of basis parameters such that the predicted values match before and after the transition.

        NOTE: log-probability for deletion of a particular knot (-log(n + 1)) is cancelled by the contribution from
        the order statistics densities, log((n + 1)! / n!) = log(n + 1). Therefore, both contributions are omitted from
        the calculation. For further information, see Richardson & Green 1997, Section 3.2:
        https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf

        NOTE: log-probability density for the full model is obtained from summing the contribution of the log-density
        for the individual distributions corresponding to each jump parameter.

        Args:
            current_state (dict): dictionary with current parameter values.

        Returns:
            prop_state (dict): dictionary updated with proposed state.
            logp_pr_g_cr (float): transition probability for proposed state given current state.
            logp_cr_g_pr (float): transition probability for current state given proposed state.

        """
        prop_state = deepcopy(current_state)
        prop_state[self.param] = prop_state[self.param] + 1
        log_prop_density = 0

        for associated_key in self.associated_params:
            new_element = self.model[associated_key].rvs(state=current_state, n=1)
            prop_state[associated_key] = np.concatenate((prop_state[associated_key], new_element), axis=1)
            log_prop_density += self.model[associated_key].log_p(current_state, by_observation=True)
        if callable(self.state_birth_function):
            prop_state, logp_pr_g_cr, logp_cr_g_pr = self.state_birth_function(current_state, prop_state)
        else:
            logp_pr_g_cr, logp_cr_g_pr = 0.0, 0.0
        if self.matching_params is not None:
            prop_state, logp_pr_g_cr, logp_cr_g_pr = self.matched_birth_transition(
                current_state, prop_state, logp_pr_g_cr, logp_cr_g_pr
            )

        p_birth, p_death = self.get_move_probabilities(current_state, True)
        logp_pr_g_cr += np.log(p_birth) + log_prop_density[-1]
        logp_cr_g_pr += np.log(p_death)

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def death_proposal(self, current_state: dict) -> Tuple[dict, float, float]:
        """Make a death proposal move: DECREASES state[self.param] by 1.

        Also adjusts the associated parameter state[self.associated_params] by deleting a randomly-selected element.

        self.state_death_function() and self.matched_death_transition() can be used (optional) to specify transitions
        opposite to those used in the birth move.

        NOTE: log-probability density for the full model is obtained from summing the contribution of the log-density
        for the individual distributions corresponding to each jump parameter.

        For further information about the transition, see also self.birth_proposal().

        Args:
            current_state (dict): dictionary with current parameter values.

        Returns:
            prop_state (dict): dictionary updated with proposed state.
            logp_pr_g_cr (float): transition probability for proposed state given current state.
            logp_cr_g_pr (float): transition probability for current state given proposed state.

        """
        prop_state = deepcopy(current_state)
        prop_state[self.param] = prop_state[self.param] - 1
        log_prop_density = 0
        deletion_index = randint.rvs(low=0, high=current_state[self.param])
        for associated_key in self.associated_params:
            prop_state[associated_key] = np.delete(prop_state[associated_key], obj=deletion_index, axis=1)
            log_prop_density += self.model[associated_key].log_p(current_state, by_observation=True)

        if callable(self.state_death_function):
            prop_state, logp_pr_g_cr, logp_cr_g_pr = self.state_death_function(
                current_state, prop_state, deletion_index
            )
        else:
            logp_pr_g_cr, logp_cr_g_pr = 0.0, 0.0
        if self.matching_params is not None:
            prop_state, logp_pr_g_cr, logp_cr_g_pr = self.matched_death_transition(
                current_state, prop_state, logp_pr_g_cr, logp_cr_g_pr, deletion_index
            )

        p_birth, p_death = self.get_move_probabilities(current_state, False)
        logp_pr_g_cr += np.log(p_death)
        logp_cr_g_pr += np.log(p_birth) + log_prop_density[-1]

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def matched_birth_transition(
        self, current_state: dict, prop_state: dict, logp_pr_g_cr: float, logp_cr_g_pr: float
    ) -> Tuple[dict, float, float]:
        """Generate a proposal for coefficients associated with a birth move, using the principle of matching the predictions before and after the move.

        The parameter vector in the proposed state is computed as: beta* = F @ beta_aug, where:
        F = [G, 0
             0', 1]
        G = (X*' @ X*)^{-1} @ (X*' @ X)
        where X is the original basis matrix, and X* is the augmented basis matrix. For a detailed explanation of the
        approach, see: https://ygraigarw.github.io/ZnnEA1D19.pdf

        The basis matrix in the proposed state should already have been updated in self.state_birth_function(), before
        the call to this function (along with any other associated parameters that need to change shape).

        The following fields should be supplied as part of the self.matching_params dictionary:
            - "variable" (str): reference to the coefficient parameter vector in the state.
            - "matrix" (str): reference to the associated basis matrix in state.
            - "scale" (float): scale of Gaussian noise added to proposal.
            - "limits" (list): [lower, upper] limit for truncated Normal proposals.

        The proposal for the additional basis parameter can be either from:
            - a standard normal distribution (when self.matching_params["limits"] is passed as None).
            - a truncated normal distribution (when self.matching_params["limits"] is a two-element list of the lower
                and upper limits).

        Args:
            current_state (dict): current parameter state as dictionary.
            prop_state (dict): proposed state dictionary, with updated basis matrix.
            logp_pr_g_cr (float): transition probability for proposed state given current state.
            logp_cr_g_pr (float): transition probability for current state given proposed state.

        Returns:
            prop_state (dict): proposed state with updated parameter vector.
            logp_pr_g_cr (float): updated transition probability.
            logp_cr_g_pr (float): updated transition probability.

        """
        vector = self.matching_params["variable"]
        matrix = self.matching_params["matrix"]
        proposal_scale = self.matching_params["scale"]
        proposal_limits = self.matching_params["limits"]

        current_basis = current_state[matrix]
        prop_basis = prop_state[matrix]
        G = np.linalg.solve(
            prop_basis.T @ prop_basis + 1e-10 * np.eye(prop_basis.shape[1]), prop_basis.T @ current_basis
        )
        F = np.concatenate((G, np.eye(N=G.shape[0], M=1, k=-G.shape[0] + 1)), axis=1)
        mu_star = G @ current_state[vector]
        prop_state[vector] = deepcopy(mu_star)

        if proposal_limits is not None:
            prop_state[vector][-1] = gmrf.truncated_normal_rv(
                mean=mu_star[-1], scale=proposal_scale, lower=proposal_limits[0], upper=proposal_limits[1], size=1
            )
            logp_pr_g_cr += gmrf.truncated_normal_log_pdf(
                prop_state[vector][-1], mu_star[-1], proposal_scale, lower=proposal_limits[0], upper=proposal_limits[1]
            )
        else:
            Q = np.array(1 / (proposal_scale**2), ndmin=2)
            prop_state[vector][-1] = gmrf.sample_normal(mu=mu_star[-1], Q=Q, n=1)
            logp_pr_g_cr += gmrf.multivariate_normal_pdf(x=prop_state[vector][-1], mu=mu_star[-1], Q=Q)

        logp_cr_g_pr += np.log(np.linalg.det(F))

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def matched_death_transition(
        self, current_state: dict, prop_state: dict, logp_pr_g_cr: float, logp_cr_g_pr: float, deletion_index: int
    ) -> Tuple[dict, float, float]:
        """Generate a proposal for coefficients associated with a death move, as the reverse of the birth proposal in self.matched_birth_transition().

        See self.matched_birth_transition() for further details.

        Args:
            current_state (dict): current parameter state as dictionary.
            prop_state (dict): proposed state dictionary, with updated basis matrix.
            logp_pr_g_cr (float): transition probability for proposed state given current state.
            logp_cr_g_pr (float): transition probability for current state given proposed state.
            deletion_index (int): index of the basis element to be deleted

        Returns:
            prop_state (dict): proposed state with updated parameter vector.
            logp_pr_g_cr (float): updated transition probability.
            logp_cr_g_pr (float): updated transition probability.

        """
        vector = self.matching_params["variable"]
        matrix = self.matching_params["matrix"]
        proposal_scale = self.matching_params["scale"]
        proposal_limits = self.matching_params["limits"]

        current_basis = current_state[matrix]
        prop_basis = prop_state[matrix]
        G = np.linalg.solve(
            current_basis.T @ current_basis + 1e-10 * np.eye(current_basis.shape[1]), current_basis.T @ prop_basis
        )
        F = np.insert(G, obj=deletion_index, values=np.eye(N=G.shape[0], M=1, k=-deletion_index).flatten(), axis=1)
        mu_aug = np.linalg.solve(F, current_state[vector])
        param_del = mu_aug[deletion_index]
        prop_state[vector] = np.delete(mu_aug, obj=deletion_index, axis=0)

        logp_pr_g_cr += np.log(np.linalg.det(F))
        if proposal_limits is not None:
            logp_cr_g_pr += gmrf.truncated_normal_log_pdf(
                param_del, np.array(0), proposal_scale, lower=proposal_limits[0], upper=proposal_limits[1]
            )
        else:
            logp_cr_g_pr += gmrf.multivariate_normal_pdf(
                x=param_del, mu=np.array(0.0, ndmin=2), Q=np.array(1 / (proposal_scale**2), ndmin=2)
            )

        return prop_state, logp_pr_g_cr, logp_cr_g_pr

    def get_move_type(self, current_state: dict) -> bool:
        """Select the type of move (birth or death) to be made at the current iteration.

        Logic for the choice of move is as follows:
            - if state[self.param]=self.n_max, it is not possible to increase self.param, so a death move is chosen.
            - if state[self.param]=1, it is not possible to decrease self.param, so a birth move is chosen.
            - in any other state, a birth move is chosen with probability self.birth_probability, or a death move is
                chosen with probability (1 - self.birth_probability).

        Args:
            current_state (dict): dictionary with current parameter values.

        Returns:
            (bool): if True, make a birth proposal; if False, make a death proposal.

        """
        if current_state[self.param] == self.n_max:
            return False
        if current_state[self.param] == 1:
            return True

        return uniform.rvs() <= self.birth_probability

    def get_move_probabilities(self, current_state: dict, birth: bool) -> Tuple[float, float]:
        """Get the state-dependent probabilities of the forward and reverse moves, accounting for edge cases.

        Returns a tuple of (p_birth, p_death), where these should be interpreted as follows:
            Birth move: p_birth = probability of birth from CURRENT state.
                        p_death = probability of death from PROPOSED state.
            Death move: p_death = probability of death in CURRENT state.
                        p_birth = probability of birth in PROPOSED state.

        In standard cases (away from the limits, assumed to be at [1, n_max]):
            p_birth = q; p_death = 1 - q

        In edge cases (either where we are at one of the limits, or where our chosen move takes us into a limiting
        case), we adjust the probability of either the forward or the reverse move to account for this. E.g.: if n=2,
        q=0.5 and a death is proposed (i.e. proposed value n*=1), then p_death=0.5 (equal probabilities of birth/death
        in CURRENT state), and p_birth=1 (because death is not possible in PROPOSED state).

        Args:
            current_state (dict): dictionary with current parameter values.
            birth (bool): indicator for birth or death move.

        Returns:
            p_birth (float): state-dependent probability of birth move.
            p_death (float): state-dependent probability of death move.

        """
        p_birth = self.birth_probability
        p_death = 1.0 - self.birth_probability

        if current_state[self.param] == self.n_max:
            p_death = 1.0
        if current_state[self.param] == (self.n_max - 1) and birth:
            p_death = 1.0

        if current_state[self.param] == 1:
            p_birth = 1.0
        if current_state[self.param] == 2 and not birth:
            p_birth = 1.0
        return p_birth, p_death
