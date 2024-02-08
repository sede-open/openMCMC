# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gmrf: gaussian Markov Random Field.

Reference: Rue, Held 2005 Gaussian Markov Random Fields

Helper functions for sampling and dealing with Multivariate normal distributions
defined by precision matrices which avoid the need for direct inversion and efficiently
reuse cholesky factorisations with sparse implementations

Notation:
b: conditional mean
Q: precision matrix
L: lower triangle cholesky factorisation of a precision matrix Q

"""

from typing import Union

import numpy as np
from pandas.arrays import DatetimeArray
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.stats import truncnorm


def sample_normal(
    mu: np.ndarray, Q: Union[np.ndarray, sparse.csc_matrix] = None, L: np.ndarray = None, n: int = 1
) -> np.ndarray:
    """Generate multivariate random variables from a precision matrix Q using lower cholesky factorisation to get L.

    Note: sparse_linalg.spsolve_triangular compared to sparse_linalg.spsolve, and
    it appears to be much slower.
    Algorithm 2.4 from Rue, Held 2005 Gaussian Markov Random Fields
     Sampling x ~ N(mu , Q^-1)
    1: Compute the lower Cholesky factorisation, Q = L @ L'
    2: Sample z ~ N(0,  I)
    3: Solve L' v = z
    4: Compute x = z + v
    5: Return x
    Args:
        mu (np.array): p x 1 mean
        Q (np.array, optional): p x p for precision matrix. Defaults to None.
        L (np.array, optional): p x p for lower triangular cholesky factorisation of
                                     precision matrix. Defaults to None.
        n (int, optional): number of samples. Defaults to 1.

    Returns:
        (np.array): p x n random normal values

    """
    size = [np.size(mu), n]

    z = np.random.standard_normal(size=size)

    if L is None:
        L = cholesky(Q)

    return solve(L.T, z).reshape(z.shape) + mu


def sample_truncated_normal(
    mu: np.ndarray,
    Q: Union[np.ndarray, sparse.csc_matrix] = None,
    L: np.ndarray = None,
    lower: np.ndarray = None,
    upper: np.ndarray = None,
    n: np.array = 1,
    method="Gibbs",
) -> np.ndarray:
    """Sample from multivariate truncated normal using either rejection sampling or Gibbs sampling.

    Gibbs sampling should be faster but is generated through a markov chain so samples may not be completely independent
    The Markov chain is set up for sampling from gibbs_canonical_truncated_normal which is thinned by every 10
    observations to get more i.i.d. samples

    Rejection sampling will work well for low dimensions and low amounts of truncated but will scale very poorly.

    Args:
        mu (np.array): p x 1 mean
        Q (np.array, optional): p x p for precision matrix. Defaults to None.
        L (np.array, optional): p x p for lower triangular cholesky factorisation of
                                     precision matrix. Defaults to None.
        lower (np.array, optional): lower bound
        upper (np.array, optional): upper bound
        n (int, optional): number of samples. Defaults to 1.
        method (str, optional): defines method to use for TN sampling Either 'Gibbs' or 'Rejection' Defaults to 'Gibbs'.

    Returns:
        (np.array): p x n random truncated normal values

    """
    if method == "Gibbs":
        d = mu.shape[0]
        b = Q @ mu
        Z = np.empty(shape=(d, n))
        Z[:, 0] = sample_truncated_normal_rejection(mu=mu, Q=Q, L=L, lower=lower, upper=upper, n=1).flatten()
        thin = 10
        for i in range(n - 1):
            x = Z[:, i].reshape(d, 1)
            for _ in range(thin):
                x = gibbs_canonical_truncated_normal(b=b, Q=Q, x=x, lower=lower, upper=upper)
                Z[:, i + 1] = x.flatten()
        return Z
    if method == "Rejection":
        return sample_truncated_normal_rejection(mu=mu, Q=Q, L=L, lower=lower, upper=upper, n=n)

    raise TypeError("method should be either Gibbs or Rejection")


def sample_truncated_normal_rejection(
    mu: np.ndarray,
    Q: Union[np.ndarray, sparse.csc_matrix] = None,
    L: np.array = None,
    lower: np.ndarray = None,
    upper: np.ndarray = None,
    n: np.array = 1,
) -> np.array:
    """Sample from multivariate truncated normal using rejection sampling.

    Rejection sampling will work well for low dimensions and low amounts of truncated but will scale very poorly.

    Args:
        mu (np.array): p x 1 mean
        Q (np.array, optional): p x p for precision matrix. Defaults to None.
        L (np.array, optional): p x p for lower triangular cholesky factorisation of
                                     precision matrix. Defaults to None.
        lower (np.array, optional): lower bound
        upper (np.array, optional): upper bound
        n (int, optional): number of samples. Defaults to 1.

    Returns:
        (np.array): p x n random truncated normal values

    """
    if L is None:
        L = cholesky(Q)

    n_bad = n

    if lower is None:
        lower = -np.inf

    if upper is None:
        upper = np.inf

    if np.any(lower >= upper):
        raise ValueError("Error lower bound must be strictly less than upper bound")

    samples = sample_normal(mu, L=L, n=n_bad)
    ind_bad = np.any(np.bitwise_or(samples < lower, samples > upper), axis=0)
    n_bad = np.sum(ind_bad)

    while n_bad > 0:
        sample_temp = sample_normal(mu, L=L, n=n_bad)

        samples[:, ind_bad] = sample_temp
        ind_bad = np.any(np.bitwise_or(samples < lower, samples > upper), axis=0)

        n_bad = np.sum(ind_bad)

    return samples


def sample_normal_canonical(b: np.ndarray, Q: np.ndarray = None, L: np.ndarray = None) -> np.ndarray:
    """Generate multivariate random variables canonical representation precision matrix using cholesky factorisation.

    Algorithm 2.5 from Rue, Held 2005 Gaussian Markov Random Fields:
    Sampling x ~ N( Q^-1 b, Q^-1)
        1: Compute the Cholesky factorisation, Q = L @ L'
        2: Solve L w = b
        3: Solve L' mu = w
        4: Sample z ~ N(0; I)
        5: Solve L' v = z
        6: Compute x = mu + v
        7: Return x

    Steps 2 and 3 are done in the function cho_solve and the output is thus mu.
    Steps 4, 5 and 6 are the algorithm 2.5 implemented in the function sample_normal

    Args:
        b (np.ndarray): p x 1 conditional mean
        Q (np.ndarray, optional): p x p for precision matrix. Defaults to None.
        L (np.ndarray, optional): p x p for lower triangular cholesky factorisation
                                    of precision matrix. Defaults to None.

    Returns:
        (np.ndarray): p x 1 random normal values

    """
    if L is None:
        L = sparse_cholesky(Q)

    mu = cho_solve((L, True), b).reshape(b.shape)

    return sample_normal(mu, L=L)


def gibbs_canonical_truncated_normal(
    b: np.ndarray,
    Q: Union[np.ndarray, sparse.csc_matrix],
    x: np.ndarray,
    lower: np.ndarray = -np.inf,
    upper: np.ndarray = np.inf,
) -> np.ndarray:
    """Generate truncated multivariate random variables from a precision matrix Q using lower cholesky factorisation to get L based on current state x using Gibbs sampling.

    subject to linear inequality constraints
    lower < X < upper

    Lemma 2.1 from Rue, Held 2005 Gaussian Markov Random Fields
     Sampling x ~ N_c( Q^-1 b , Q^-1)
     x_a | x_b ~ N_c( b_a - Q_ab x_b, Q_aa)

    Args:
        b (np.array): p x 1 mean
        Q (np.array): p x p for precision matrix. Defaults to None.
        x (np.array): p x 1 current state.
        lower (np.array, optional): p x 1 lower bound for each dimension
        upper (np.array, optional): p x 1 upper bound for each dimension

    Returns:
        (np.array): p x 1 random normal values

    """
    if (lower == -np.inf or lower is None) and (upper == np.inf or upper is None):
        return sample_normal_canonical(b, Q)

    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf

    p = np.size(x)
    temp_limit = np.full(shape=(p, 1), fill_value=np.inf)
    lower = np.maximum(lower, -temp_limit)
    upper = np.minimum(upper, temp_limit)

    if p == 1:
        if sparse.issparse(Q):
            Q = Q.toarray()
        return np.array(truncated_normal_rv(mean=b / Q, scale=1 / np.sqrt(Q), lower=lower, upper=upper), ndmin=2)

    if sparse.issparse(Q):
        Q_diag = Q.diagonal()
    else:
        Q_diag = np.diag(Q)

    for i in range(p):
        Q_ii = Q_diag[i]
        v_i = 1 / Q_ii
        scale_i = np.sqrt(v_i)

        if sparse.issparse(Q):
            cond_mean_i = v_i * (b[i] - Q.getrow(i) @ x + Q_ii * x[i])
        else:
            cond_mean_i = v_i * (b[i] - Q[i, :] @ x + Q_ii * x[i])

        x[i] = truncated_normal_rv(mean=cond_mean_i, scale=scale_i, lower=lower[i], upper=upper[i])

    return x


def truncated_normal_rv(
    mean: np.ndarray, scale: np.ndarray, lower: np.ndarray, upper: np.ndarray, size=1
) -> np.ndarray:
    """Wrapper for scipy.stats.truncnorm.rvs handles cases a, b not standard form.

    Args:
        mean (np.array): p x 1 mean for each dimension
        scale (np.array): p x 1 standard deviation for each dimension
        lower (np.array): p x 1 lower bound for each dimension
        upper (np.array): p x 1 upper bound for each dimension
        size (int): size of output array default = 1

    Returns:
        (np.ndarray): size x 1  truncated normal samples

    """
    if lower is None:
        lower = -np.inf

    if upper is None:
        upper = np.inf

    a, b = (lower - mean) / scale, (upper - mean) / scale
    return truncnorm.rvs(a, b, loc=mean, scale=scale, size=size)


def truncated_normal_log_pdf(
    x: np.ndarray, mean: np.ndarray, scale: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Wrapper for scipy.stats.truncnorm.logpdf handles cases a, b not standard form.

    Args:
        x (np.ndarray): values
        mean (np.ndarray): mean
        scale (np.ndarray): standard deviation
        lower (np.ndarray): lower bound
        upper (np.ndarray): upper bound

    Returns:
        (np.ndarray): truncated normal sample

    """
    if lower is None:
        lower = -np.inf

    if upper is None:
        upper = np.inf

    a, b = (lower - mean) / scale, (upper - mean) / scale
    return truncnorm.logpdf(x, a, b, loc=mean, scale=scale)


def multivariate_normal_pdf(
    x: np.ndarray, mu: np.ndarray, Q: Union[np.ndarray, sparse.csc_matrix], by_observation: bool = False
) -> Union[np.ndarray, float]:
    """Compute diagonalized log-pdf of a multivariate Gaussian distribution in terms of the precision matrix, can take sparse precision matrix inputs.

    Args:
        x (np.ndarray): dim  x n value for the distribution response. where dim is the number of dimensions and n
                        is the number of observations
        mu (np.ndarray): dim x 1 distribution mean vector.
        Q (np.ndarray, sparse.csc_matrix): dim x dim distribution precision matrix can be sparse or np.array
        by_observation (bool, optional): indicates whether we should sum over observations default= False

    Returns:
        (np.ndarray): log-pdf of the Gaussian distribution either:
                                    (1,) if by_observation = False or
                                    (n,) if by_observation = True

    """
    L = cholesky(Q)
    dim = L.shape[0]

    log_det_precision = 2 * np.sum(np.log(L.diagonal()))
    Q_residual = L.T @ (x - mu)
    log_p = (1 / 2) * (log_det_precision - dim * np.log(2 * np.pi) - np.sum(np.power(Q_residual, 2), axis=0))

    if not by_observation:
        log_p = np.sum(log_p)
    return log_p


def precision_temporal(
    time: DatetimeArray, unit_length: float = 1.0, is_sparse: bool = True
) -> Union[np.ndarray, sparse.csc_matrix]:
    """Generate temporal difference penalty matrix.

    Details can be found on pages 97-99 of 'Gaussian Markov Random Fields'
    [Rue, Held 2005], 'The first-order random walk for irregular locations'.

    Converts time to number of seconds then call precision_irregular

    Args:
        time (DatetimeArray): vector of times
        unit_length (float, optional): numbers seconds to define unit difference Defaults to 1 second
        is_sparse (bool, optional): Flag if generated as sparse. Defaults to True.

    Returns:
        P (Union[np.ndarray, sparse.csc_matrix]): un-scaled precision matrix

    """
    s = (time - time.min()).total_seconds() / unit_length

    return precision_irregular(s, is_sparse=is_sparse)


def precision_irregular(s: np.ndarray, is_sparse: bool = True) -> Union[np.ndarray, sparse.csc_matrix]:
    """Generate penalty matrix from irregular observations using first order random walk.

    Details can be found on pages 97-99 of 'Gaussian Markov Random Fields'
    [Rue, Held 2005], 'The first-order random walk for irregular locations'.

    Diagonal and off-diagonal elements of the precision found as follows:
                1/del_{i-1} + 1/del_{i},    j = i,
    Q_{ij} =    -1/del_{i},                 j = i+1,
                0,                          else.
    where del = [t_{i+1} - t_{i}]

    Args:
        s (np.ndarray): vector of locations.
        is_sparse (bool, optional): Flag if generated as sparse. Defaults to True.

    Returns:
        P ( Union[np.ndarray, sparse.csc_matrix]): un-scaled precision matrix

    """
    if s.ndim > 1:
        s = np.squeeze(s)

    if s.size > 1:
        delta_reciprocal = 1.0 / np.diff(s)

        d_0 = np.append(
            np.append(delta_reciprocal[0], delta_reciprocal[:-1] + delta_reciprocal[1:]), delta_reciprocal[-1]
        )
        if is_sparse:
            P = sparse.diags(diagonals=(-delta_reciprocal, d_0, -delta_reciprocal), offsets=[-1, 0, 1], format="csc")
        else:
            P = np.diag(d_0, k=0) - np.diag(delta_reciprocal, k=-1) - np.diag(delta_reciprocal, k=1)
    else:
        P = np.array(1, ndmin=2)

    return P


def solve(
    a: Union[np.ndarray, sparse.csc_matrix], b: Union[np.ndarray, sparse.csc_matrix]
) -> Union[np.ndarray, sparse.csc_matrix]:
    """Solve a linear matrix equation, or system of linear scalar equations.

    Computes the “exact” solution, x, of the well-determined, i.e., full rank, linear matrix equation ax = b.

    If inputs are sparse calls scipy.linalg.spsolve else calls np.linalg.solve

    Args:
        a (Union[np.ndarray, sparse.csc_matrix]): _description_
        b (Union[np.ndarray, sparse.csc_matrix]): _description_

    Returns
        Union(np.ndarray, sparse.csc_matrix) solution to the system in same format as input

    """
    if sparse.issparse(a) or sparse.issparse(b):
        return sparse_linalg.spsolve(a, b)

    return np.linalg.solve(a, b)


def cho_solve(c_and_lower: tuple, b: Union[np.ndarray, sparse.csc_matrix]) -> Union[np.ndarray, sparse.csc_matrix]:
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    If inputs are sparse calls sparse solvers otherwise uses scipy.linalg.cho_solve

    Args:
        c_and_lower ( tuple(Union(np.ndarray, sparse.csc_matrix), bool)): Cholesky factorization of A
                    and flag for if it is a lower Cholesky
        b (Union(np.ndarray, sparse.csc_matrix)): Right-hand side

    Returns
        (Union(np.ndarray, sparse.csc_matrix)) The solution to the system A x = b

    """
    if sparse.issparse(c_and_lower[0]) or sparse.issparse(b):
        if c_and_lower[1]:
            L = c_and_lower[0]
            U = c_and_lower[0].T
        else:
            L = c_and_lower[0].T
            U = c_and_lower[0]

        w = sparse_linalg.spsolve(L, b)
        return sparse_linalg.spsolve(U, w)

    return linalg.cho_solve(c_and_lower, b)


def cholesky(Q: Union[np.ndarray, sparse.csc_matrix], lower: bool = True) -> Union[np.ndarray, sparse.csc_matrix]:
    """Compute Cholesky factorization of input matrix.

    If it is sparse will use gmf.sparse_cholesky otherwise will use linalg.cholesky

    Args:
        Q (Union[np.ndarray, sparse.csc_matrix]):  precision matrix, for factorization
        lower (bool, optional): flag for lower triangular matrix, default is true

    Returns
       (Union[np.ndarray, sparse.csc_matrix]: Cholesky factorization of the input in the same format as the input

    """
    if sparse.issparse(Q):
        L = sparse_cholesky(Q)
    else:
        L = np.linalg.cholesky(Q)

    if lower:
        return L

    return L.T


def sparse_cholesky(Q: sparse.csc_matrix) -> sparse.csc_matrix:
    """Compute sparse Cholesky factorization of input matrix.

    Uses the scipy.sparse functionality for LU decomposition, and converts
    to Cholesky factorization. Approach taken from:
    https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d

    If the sparse matrix is identified as unsuitable for Cholesky factorization,
    the function attempts to compute the Chol of the dense matrix instead.

    Args:
        Q (sparse.csc_matrix): sparse precision matrix, for factorization

    Returns:
        (sparse.csc_matrix): Cholesky factorization of the input

    """
    m = Q.shape[0]
    n = Q.shape[1]
    if m != n:
        raise ValueError("Matrix is not square")

    if sparse.issparse(Q):
        if not isinstance(Q, sparse.csc_matrix):
            Q = Q.tocsc()
        fact_lu = sparse_linalg.splu(Q, diag_pivot_thresh=0, options={"RowPerm": False, "ColPerm": False})
        if (fact_lu.U.diagonal() > 0).all():
            return fact_lu.L.dot(sparse.diags(fact_lu.U.diagonal() ** 0.5))

        return np.linalg.cholesky(Q.toarray())

    return np.linalg.cholesky(Q)
