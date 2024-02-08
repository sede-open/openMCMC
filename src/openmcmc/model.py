# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Model module.

This module provides a class definition of the Model class, a dictionary-like collection of distributions to form a
model.

"""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from openmcmc.distribution.distribution import Distribution


@dataclass
class Model(dict):
    """Dictionary-like collection of distributions to form a model.

    self.keys() indexes the responses of the distributions in the collection; self.values() contain the individual
    distribution objects in the model, of type Distribution.

    Attributes:
        response (dict): dictionary with keys corresponding to the data values within state, and values corresponding
            to the desired predictor values within the data distributions (for storing fitted values).

    """

    def __init__(self, distributions: list[Distribution], response: dict = None):
        dist_dict = {}
        for dist in distributions:
            dist_dict[dist.response] = dist
        super().__init__(dist_dict)
        self.response = response

    def conditional(self, param: str):
        """Return sub-model which consists of the subset of distributions dependent on the supplied parameter.

        Args:
            param (str): parameter to find within the model distributions.

        Returns:
            (Model): model object containing only distributions which have a dependence on param.

        """
        conditional_dist = []
        for dst in self.values():
            if param in dst.param_list:
                conditional_dist.append(dst)
        return Model(conditional_dist)

    def log_p(self, state: dict) -> Union[float, np.ndarray]:
        """Compute the log-probability density for the full model.

        Args:
            state (dict): dictionary with current state information.

        Returns:
            (Union[float, np.ndarray]): POSITIVE log-probability density evaluated using the information in state.

        """
        log_prob = 0
        for dst in self.values():
            log_prob += dst.log_p(state)
        return log_prob

    def grad_log_p(
        self, state: dict, param: str, hessian_required: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate vector of derivatives of the log-pdf with respect to a given parameter, as the sum of the derivatives of all the individual components of the model. If required, also generate the Hessian.

        Function only defined for scalar- and vector-valued parameters param. If hessian_required=True, this function
        returns a tuple of (gradient, Hessian). If hessian_required=False, this function returns a np.ndarray (just
        the gradient of the log-density).

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
        grad_sum = np.zeros(shape=state[param].shape)
        if hessian_required:
            hessian_sum = np.zeros(shape=(state[param].shape[0], state[param].shape[0]))

        for dist in self.values():
            grad_out = dist.grad_log_p(state, param, hessian_required=hessian_required)
            if hessian_required:
                grad_sum += grad_out[0]
                hessian_sum += grad_out[1]
            else:
                grad_sum += grad_out

        if hessian_required:
            return grad_sum, hessian_sum

        return grad_sum
