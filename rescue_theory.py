from math import *
import numpy as np
from scipy import optimize


#########################
# Probability of rescue #
#########################


def pois(mu, x):
    '''
    Poisson probability mass function with parameter mu and number of occurrences x.

    Parameters
    ----------
    mu : float
        Parameter of the Poisson distribution.
    x : int
        Number of occurrences.

    Returns
    -------
    float
        Probability.
    '''
    assert type(x) is int
    return exp(-mu) * mu ** x / factorial(x)


def phi(t, mu, nmax):
    '''
    Probability generating function of the number of offspring of a wildtype individual.
    Assumes Poisson offspring distribution.

    Parameters
    ----------
    t : float
        Parameter of pgf.
    mu : float
        Mean number of offspring.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    f = 0
    for j in range(nmax):
        P = pois(mu, j)
        f += (t ** j) * P
    return f


def G(t, r, s, nmax):
    '''
    Probability generating function of the number of offspring of a beneficial individual
    (Equation 4).

    Parameters
    ----------
    t : float
        Parameter of pgf.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    mu = (1 - r) * (1 + s)
    assert mu > 1
    return phi(t, mu, nmax)


def qb_fun(qb, r, s, nmax):
    '''
    Function used to calculate the probability of extinction of a population starting from one beneficial individual, qb.
    qb is estimated by minimizing the value of this function.

    Parameters
    ----------
    qb : float
        Probability of extinction.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Difference between G(qb) and qb.
    '''
    return G(qb, r, s, nmax) - qb


def get_qb(r, s, nmax):
    '''
    Estimate the probability of extinction of a population starting from one beneficial individual, qb.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Estimate of qb.
    '''
    sol = optimize.root_scalar(qb_fun, args=(r, s, nmax), bracket=(0, 1-1e-10), x0=.5, xtol=1e-12)
    return sol.root


def qw_fun(qw, r, s, u, nmax):
    '''
    Function used to calculate the probability of extinction of a population starting from one wildtype individual, qw.
    qw is estimated by minimizing the value of this function.

    Parameters
    ----------
    qw : float
        Probability of extinction.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Difference between pgf and qw.
    '''
    qb = get_qb(r, s, nmax)
    mu = 1 - r
    return phi(qw * (1 - u) + u * qb, mu, nmax) - qw


def qwu0_fun(qw, r, nmax):
    '''
    Function used to calculate the probability of extinction of a population starting from one wildtype individual, qw, assuming no mutation.

    qwu0 is estimated by minimizing the value of this function.

    Parameters
    ----------
    qw : float
        Probability of extinction.
    r : float
        Degree of maladaptation of wildtype individuals.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Difference between pgf and qw.
    '''
    mu = 1 - r
    return phi(qw, mu, nmax) - qw


def get_qw(r, s, u, nmax):
    '''
    Estimate the probability of extinction of a population starting from one wildtype individual, qw.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Estimate of qw.
    '''
    if u == 0:
        if qwu0_fun(1.0, r, nmax) < 1e-12:
            return 1.0
        else:
            return 'error'
    else:
        sol = optimize.root_scalar(qw_fun, args=(r, s, u, nmax), bracket=(0, 1-1e-10), x0=.5, xtol=1e-12)
        return sol.root


def prob_rescue(W0, B0, r, s, u, nmax, verbose):
    '''
    Calculate the probability of rescue of a population.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    B0 : int
        Initial number of mutant individuals
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    nmax : int
        Maximum number of individuals to consider in Poisson distribution.
    verbose : bool
        Whether to show the component probabilities and the weak mutation-weak selection approximations.

    Returns
    -------
    tuple of floats
        Probability of rescue (exact, approximate)
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    approx_pnew = 1 - exp(-2 * W0 * u * (s - r) / r)
    approx_stand = 1 - exp(-2 * B0 * (s - r))
    approx_ptotal = approx_stand + (1 - approx_stand) * approx_pnew
    if verbose:
        print('  prob: exact, approx')
        print('pstand:', 1 - qb ** B0, approx_stand)
        print('  pnew:', 1 - qw ** W0, approx_pnew)
        print('ptotal:', 1 - (qw ** W0) * (qb ** B0), approx_ptotal)
    return 1 - (qw ** W0) * (qb ** B0), approx_ptotal


###################
# Population size #
###################


def F(v, t, mu, u, nmax):
    '''
    Joint probability generating function of the number of wildtype and beneficial offspring of a wildtype individual.

    Parameters
    ----------
    v, t : float
        Parameters of pgf.
    mu : float
        Mean number of offspring.
    u : float
        Mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    return phi((1 - u) * v + u * t, mu, nmax)


def M(r, s, u):
    '''
    Reproduction matrix.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    '''
    return (1 - r) * np.array([[1 - u, u], [0, 1 + s]])


def Mn(r, s, u, n):
    '''
    nth power of reproduction matrix.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    n : int
        Power.
    '''
    return (1 - r) ** n * np.array([[(1 - u) ** n, u * ((1 + s) ** n - (1 - u) ** n) / (s + u)], [0, (1 + s) ** n]])


def WBn(W0, B0, r, s, u, n):
    '''
    Number of wildtype and mutant individuals in the population after n generations.
    Takes into account both rescued and extinct populations.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    B0 : int
        Initial number of mutant individuals
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    n : int
        Number of generations

    Returns
    -------
    array([Wn, Bn])
        Number of wildtype and mutant individuals after n generations.
    '''
    v = np.array([W0, B0])
    m = np.transpose(Mn(r, s, u, n))
    return np.dot(m, v)


def dphi(t, mu, nmax):
    '''
    Derivative of phi.

    Parameters
    ----------
    t : float
        Parameter of pgf.
    mu : float
        Mean number of offspring.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    f = 0
    for j in range(1, nmax):
        P = pois(mu, j)
        f += j * (t ** (j - 1)) * P
    return f


def dFv(v, t, mu, u, nmax):
    '''
    Derivative of F with respect to v.

    Parameters
    ----------
    v, t : float
        Parameters of pgf.
    mu : float
        Mean number of offspring.
    u : float
        Mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    return (1 - u) * dphi((1 - u) * v + u * t, mu, nmax)


def dFt(v, t, mu, u, nmax):
    '''
    Derivative of F with respect to t.

    Parameters
    ----------
    v, t : float
        Parameters of pgf.
    mu : float
        Mean number of offspring.
    u : float
        Mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    return u * dphi((1 - u) * v + u * t, mu, nmax)


def dG(t, r, s, nmax):
    '''
    Derivative of G.

    Parameters
    ----------
    t : float
        Parameter of pgf.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    mu = (1 - r) * (1 + s)
    assert mu > 1
    return dphi(t, mu, nmax)


def Mhat(r, s, u, nmax):
    '''
    Reproduction matrix.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    nmax : int
        Maximum number of offspring considered.
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    mu = 1 - r
    return np.array([[dFv(qw, qb, mu, u, nmax), (qb / qw) * dFt(qw, qb, mu, u, nmax)], [0, dG(qb, r, s, nmax)]])


def Mhatn(r, s, u, n, nmax):
    '''
    nth power of reproduction matrix.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    n : int
        Power.
    nmax : int
        Maximum number of offspring considered.
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    mu = 1 - r
    return np.array([[dFv(qw, qb, mu, u, nmax) ** n, (qb / qw) * dFt(qw, qb, mu, u, nmax) * (dG(qb, r, s, nmax) ** n - dFv(qw, qb, mu, u, nmax) ** n) / (dG(qb, r, s, nmax) - dFv(qw, qb, mu, u, nmax))], [0, dG(qb, r, s, nmax) ** n]])


def rescuedWBn(W0, B0, r, s, u, n, nmax, verbose):
    '''
    Number of wildtype and mutant individuals in the population after n generations.  Takes into account both rescued and extinct populations.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    B0 : int
        Initial number of mutant individuals
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
    n : int
        Number of generations

    Returns
    -------
    array([Wn, Bn])
        Number of wildtype and mutant individuals after n generations.
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    qq = (qw ** W0) * (qb ** B0)
    m = Mn(r, s, u, n)
    mhat = Mhatn(r, s, u, n, nmax)
    mm = (m - qq * mhat) / (1 - qq)
    v = np.array([W0, B0])
    if verbose:
        print(qb)
        print(qw)
        print(qq)
        print(m)
        print(mhat)
        print(mm)
    return np.dot(v, mm)
