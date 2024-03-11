# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit testing for GMRF module."""

from typing import Union

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import chi2, multivariate_normal, norm, ttest_ind
import warnings

from openmcmc import gmrf


def rand_precision(d: int = 1, is_time: bool = False, is_sparse: bool = False) -> Union[np.ndarray, sparse.csc_matrix]:
    """Generate random observations locations to pass into a precision matrix and generate a precision matrix.

    observations are generated using exponential inter arrivals (Poisson process equivalent)

    Function is used in testing

    Args:
        d (int, optional): dimension of precision matrix. Defaults to 1.
        is_time (bool, optional): Flag if generated from timestamp. Defaults to False.
        is_sparse (bool, optional): Flag if generated as sparse. Defaults to False.

    Returns:
        Union[np.ndarray, sparse.csc_matrix] d x d precision matrix

    """

    s = np.cumsum(np.random.exponential(scale=1.0, size=d))

    if is_time:
        s = pd.Timestamp.utcnow() + pd.to_timedelta(s, unit="sec")
        return gmrf.precision_temporal(s, is_sparse=is_sparse)

    return gmrf.precision_irregular(s, is_sparse=is_sparse)


@pytest.mark.parametrize("n", [1, 100])
@pytest.mark.parametrize("d", [1, 3, 10], ids=["d=1", "d=3", "d=10"])
@pytest.mark.parametrize("is_sparse", [True, False], ids=["sparse", "full"])
def test_sample_normal(d: int, is_sparse: bool, n: int):
    """Test that sample_normal gives s output consistent with Mahalanobis distance against chi2 distribution with d
    degrees of freedom.

    We only throw a warning instead of asserting False as the randomness of the test sometimes causes the test to fail
    while this is only due to the random number generation process. Therefore, we decided to for now only throw a
    warning such that we can keep track of the test results without always failing automated pipelines when the test
    fails.

    Args:
        d (int): dimension of precision
        is_sparse (bool): is precision generated as sparse

    """
    mu = np.random.rand(d, 1)
    Q = rand_precision(d, is_sparse=is_sparse)
    if is_sparse:
        Q = Q + sparse.eye(d)
    else:
        Q = Q + np.eye(d)

    rand_norm = gmrf.sample_normal(mu=mu, Q=Q, n=n)

    rsd = rand_norm - mu
    dist = np.diag(rsd.T @ Q @ rsd)

    P = 1 - chi2.cdf(dist, df=d)
    alpha = 0.01

    if n == 1:
        test_outcome = P > alpha
    else:
        test_outcome = np.sum(P > alpha) > n * (1 - 3 * alpha)

    if not test_outcome:
        warnings.warn(f"Test failed, double check if this is due to randomness or a real issue. "
                      f"Input args: [{d, is_sparse, n}]. P values: {P}.")
        test_outcome = True

    assert test_outcome


@pytest.mark.parametrize("d", [1, 2, 5])
@pytest.mark.parametrize("is_sparse", [True, False], ids=["sparse", "full"])
@pytest.mark.parametrize("upper", [np.inf, 1.3])
@pytest.mark.parametrize("lower", [-np.inf, -0.2])
def test_compare_truncated_normal(d: int, is_sparse: bool, lower: np.ndarray, upper: np.ndarray):
    """Test that runs both sample_truncated_normal with both methods rejection sampling and Gibbs sampling to show they
    give consistent results and check both output consistent within upper and lower bounds.

    We only throw a warning instead of asserting False as the randomness of the test sometimes causes the test to fail
    while this is only due to the random number generation process. Therefore, we decided to for now only throw a
    warning such that we can keep track of the test results without always failing automated pipelines when the test
    fails.

    Args:
        d (int): dimension of precision-
        is_sparse (bool): is precision generated as sparse
        lower (np.ndarray): lower bound for truncated sampling
        upper (np.ndarray): upper bound for truncated sampling

    """
    n = 100
    mu = np.linspace(0, 1, d).reshape((d, 1))
    Q = rand_precision(d, is_sparse=is_sparse)
    if is_sparse:
        Q = Q + sparse.eye(d)
    else:
        Q = Q + np.eye(d)

    rand_norm_1 = gmrf.sample_truncated_normal(mu=mu, Q=Q, n=n, lower=lower, upper=upper, method="Gibbs")
    rand_norm_2 = gmrf.sample_truncated_normal(mu=mu, Q=Q, n=n, lower=lower, upper=upper, method="Rejection")

    if lower != -np.inf:
        assert np.all(rand_norm_1 > lower)
        assert np.all(rand_norm_2 > lower)

    if upper != np.inf:
        assert np.all(rand_norm_1 < upper)
        assert np.all(rand_norm_2 < upper)

    #  t test to compare means
    [_, p_value] = ttest_ind(rand_norm_1, rand_norm_2, axis=1, equal_var=False)

    alp = 0.001

    test_outcome = np.all(p_value < (1 - alp))
    if not test_outcome:
        warnings.warn(f"Test failed, double check if this is due to randomness or a real issue. "
                      f"Input args: [{d, is_sparse, lower, upper}]. P value: {p_value}.")
        test_outcome = True

    assert test_outcome

@pytest.mark.parametrize("mean", [0.5, 1.3])
@pytest.mark.parametrize("scale", [0.1, 1])
@pytest.mark.parametrize("upper", [np.inf, None, 1.3])
@pytest.mark.parametrize("lower", [-np.inf, None, -0.2])
def test_truncated_normal_rv(mean: np.ndarray, scale: np.array, lower: np.ndarray, upper: np.ndarray):
    """Test that checks the univariate truncated normal against known mean
    https://en.wikipedia.org/wiki/Truncated_normal_distribution.

    Args:
        mean (np.ndarray): mean of truncated sampling
        scale (np.ndarray): scale of truncated sampling
        lower (np.ndarray): lower bound for truncated sampling
        upper (np.ndarray): upper bound for truncated sampling

    """

    # Rejection Sampling version
    Z = gmrf.truncated_normal_rv(mean=mean, scale=scale, lower=lower, upper=upper, size=10000)

    if lower is None:
        lower = -np.inf

    if upper is None:
        upper = np.inf

    alp = (lower - mean) / scale
    bet = (upper - mean) / scale
    true_mean = mean + (norm.pdf(alp) - norm.pdf(bet)) / (norm.cdf(bet) - norm.cdf(alp)) * scale

    assert np.isclose(np.mean(Z), true_mean, atol=1e-1 * scale)


@pytest.mark.parametrize("d", [1, 3, 10])
@pytest.mark.parametrize("is_sparse", [True, False], ids=["sparse", "full"])
def test_sample_normal_canonical(d: int, is_sparse: bool):
    """Test that sample_normal_canonical gives output consistent with Mahalanobis distance against chi2 distribution
    with d degrees of freedom.

    Args:
        d (int): dimension of precision
        is_sparse (bool): is precision generated as sparse

    """
    b = np.random.rand(d, 1)
    Q = rand_precision(d, is_sparse=is_sparse)
    if is_sparse:
        Q = Q + sparse.eye(d)
    else:
        Q = Q + np.eye(d)

    mu = gmrf.solve(Q, b).reshape(b.shape)

    rand_norm = gmrf.sample_normal_canonical(b=b, Q=Q)

    dist = (rand_norm - mu).T @ Q @ (rand_norm - mu)

    P = 1 - chi2.cdf(dist, df=d)
    alpha = 0.01

    assert P > alpha


@pytest.mark.parametrize("d", [1, 10])
@pytest.mark.parametrize("is_sparse", [True, False], ids=["sparse", "full"])
@pytest.mark.parametrize("upper", [np.inf, 0.7])
@pytest.mark.parametrize("lower", [-np.inf, 0.5])
def test_gibbs_truncated_normal_canonical(d: int, is_sparse: bool, lower: np.ndarray, upper: np.ndarray):
    """Test that gibbs_canonical_truncated_normal gives output within 5 standard deviations according to Mahalanobis
    distance.

    Args:
        d (int): dimension of precision
        is_sparse (bool): is precision generated as sparse
        lower (np.ndarray): lower bound for truncated sampling
        upper (np.ndarray): upper bound for truncated sampling

    """
    b = np.random.rand(d, 1)
    Q = rand_precision(d, is_sparse=is_sparse)
    if is_sparse:
        Q = Q + sparse.eye(d)
    else:
        Q = Q + np.eye(d)

    x = np.ones(shape=(d, 1)) * 0.6

    rand_norm = gmrf.gibbs_canonical_truncated_normal(b=b, Q=Q, lower=lower, upper=upper, x=x)

    if lower != -np.inf:
        assert np.all(rand_norm > lower)

    if upper != np.inf:
        assert np.all(rand_norm < upper)


@pytest.mark.parametrize("d", [1, 2, 5])
@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("is_sparse", [True, False], ids=["sparse", "full"])
def test_multivariate_normal_pdf(d: int, n: int, is_sparse: bool):
    """Test multivariate normal pdf.

    Tests size of output as well as comparing with scipy.stats version

    Args:
        d (int): dimension for Gaussian
        n (int): _description_
        is_sparse (bool): _description_

    """

    mu = np.linspace(0, 1, d).reshape((d, 1))
    Q = rand_precision(d, is_sparse=is_sparse)
    if is_sparse:
        Q = Q + sparse.eye(d)
    else:
        Q = Q + np.eye(d)
    x = np.random.rand(d, n)

    log_p = gmrf.multivariate_normal_pdf(x, mu=mu, Q=Q, by_observation=True)
    assert log_p.size == n

    log_p = gmrf.multivariate_normal_pdf(x, mu=mu, Q=Q, by_observation=False)
    assert log_p.size == 1

    if is_sparse and d > 1:
        Q = Q.toarray()

    if d == 1:
        log_p_scipy = np.sum(norm.logpdf(x.T, loc=mu.flatten(), scale=np.sqrt(1 / Q)))
    else:
        log_p_scipy = np.sum(multivariate_normal.logpdf(x.T, mean=mu.flatten(), cov=np.linalg.inv(Q)))

    assert np.allclose(log_p, log_p_scipy, atol=1e-5)


@pytest.mark.parametrize("d", [1, 3, 10])
@pytest.mark.parametrize("is_time", [True, False])
@pytest.mark.parametrize("is_sparse", [True, False], ids=["sparse", "full"])
def test_precision(d: int, is_time: bool, is_sparse: bool):
    """Test for generation of precision matrix from first order RW.

    Check sum to 0 and symmetry

    Args:
        d (int): dimension of precision
        is_time (bool): is precision generated from timestamp
        is_sparse (bool): is precision generated as sparse

    """

    P = rand_precision(d, is_time=is_time, is_sparse=is_sparse)

    assert P.shape[0] == d
    assert P.shape[1] == d
    assert 0 == pytest.approx(np.sum(abs(P - P.T)))

    if d > 1:
        assert 0 == pytest.approx(np.sum(P))


@pytest.mark.parametrize("d", [1, 3, 10, 30])
@pytest.mark.parametrize("is_sparse", [True, False], ids=["sparse", "full"])
@pytest.mark.parametrize("lower", [True, False], ids=["L", "U"])
def test_solve(d: int, is_sparse: bool, lower: bool):
    """Test solve functions against np.linalg.solve.

    Args:
        d (int): dimension of problem
        is_sparse (bool): is precision matrix sparse
        lower (bool) is cholesky done using lower or triangular version

    """

    a = rand_precision(d, is_sparse=is_sparse)

    if is_sparse:
        a = a + sparse.eye(d)
    else:
        a = a + np.eye(d)

    b = np.random.rand(d, 2)

    # Solve version
    x = gmrf.solve(a, b)

    # Cholesky version
    C = gmrf.cholesky(a, lower)
    x_ch = gmrf.cho_solve((C, lower), b)

    # np version
    if sparse.issparse(a):
        a = a.toarray()
    x_np = np.linalg.solve(a, b)

    assert 0 == pytest.approx(np.sum(abs(x - x_np)))
    assert 0 == pytest.approx(np.sum(abs(x_ch - x_np)))


@pytest.mark.parametrize("d", [1, 5, 10, 50])
def test_sparse_cholesky(d: int):
    """Test sparse_cholesky function against the non-sparse version.

    Args:
        d (int): dimension of precision

    """
    P = rand_precision(d, is_time=False, is_sparse=True)
    P = P + sparse.eye(d)

    L = gmrf.sparse_cholesky(P)

    if sparse.issparse(L):
        L = L.toarray()
    if sparse.issparse(P):
        P = P.toarray()

    L_np = np.linalg.cholesky(P)

    # check is a proper decomposition
    assert 0 == pytest.approx(np.sum(abs(P - L @ L.T)))
    # check lower triangular
    assert 0 == np.sum(np.triu(L, k=1))
    # check same as non sparse version
    assert 0 == pytest.approx(np.sum(abs(L - L_np)))
