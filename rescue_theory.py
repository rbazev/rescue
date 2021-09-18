###############################################################################
# Code for performing numerical calculations based on the theory presented    #
# in the paper:                                                               #
# "A Branching Process Model of Evolutionary Rescue".                         #
# Authors: Ricardo B. R. Azevedo & Peter Olofsson.                            #
# Last updated: September 17, 2021.                                           #
###############################################################################


from math import exp
from math import factorial
import numpy as np
from scipy.optimize import root_scalar


#########################
# 2.1 Branching process #
#########################


def pois(mu, x):
    '''
    Poisson probability mass function with parameter mu and number of
    occurrences x.

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
    Probability generating function (pgf) of the number of offspring (X) of a
    wildtype individual.

    Assumes Poisson offspring distribution.

    See Equation 2.

    Parameters
    ----------
    t : float
        Variable of pgf.
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
    for k in range(nmax):
        P = pois(mu, k)
        f += (t ** k) * P
    return f


def F(v, t, mu, u, nmax):
    '''
    Joint pgf of the number of wildtype (W) and beneficial (B) offspring of a
    wildtype individual. Note that X = W + B.

    Assumes Poisson offspring distribution.

    See Equation 3 and Lemma 2.1.

    Parameters
    ----------
    v, t : float
        Variables of pgf.
    mu : float
        Mean number of offspring.
    u : float
        Beneficial mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    return phi((1 - u) * v + u * t, mu, nmax)


def G(t, r, s, nmax):
    '''
    Pgf of the number of offspring of a beneficial individual.

    Assumes Poisson offspring distribution.

    See Equation 4.

    Parameters
    ----------
    t : float
        Variable of pgf.
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


##################
# 2.2 Extinction #
##################


def qb_fun(qb, r, s, nmax):
    '''
    Function used to calculate the probability of extinction of a population
    starting from one beneficial individual, qb.

    qb is estimated by finding the root of this function.

    Assumes Poisson offspring distribution.

    See text below Equation 5.

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
    Estimate the probability of extinction of a population starting from one
    beneficial individual, qb.

    Assumes Poisson offspring distribution.

    The values of r and s must be > tol, where tol = 1e-8.
    Solution must be in the interval [0, 1-tol].

    See text below Equation 5.

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
    tol = 1e-8
    assert r > tol
    assert s > tol
    sol = root_scalar(qb_fun, args=(r, s, nmax),
                      bracket=(0, 1-tol), x0=.5, xtol=tol*1e-4)
    return sol.root


def qw_fun(qw, r, s, u, nmax):
    '''
    Function used to calculate the probability of extinction of a population
    starting from one wildtype individual, qw.

    qw is estimated by finding the root of this function.

    Assumes Poisson offspring distribution.

    See Proposition 2.2.

    Parameters
    ----------
    qw : float
        Probability of extinction.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
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


def get_qw(r, s, u, nmax):
    '''
    Estimate the probability of extinction of a population starting from one
    wildtype individual, qw.

    Assumes Poisson offspring distribution.

    The values of r and s must be > tol, where tol = 1e-8.
    Solution must be in the interval [0, 1-tol].

    See Proposition 2.2.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Estimate of qw.
    '''
    tol = 1e-8
    assert r > tol
    assert s > tol
    if u == 0:
        return 1.0
    else:
        sol = root_scalar(qw_fun, args=(r, s, u, nmax),
                          bracket=(0, 1-tol), x0=.5, xtol=tol*1e-4)
        return sol.root


##############
# 2.3 Rescue #
##############


def prob_rescue(W0, B0, r, s, u, nmax):
    '''
    Calculate the probability of rescue of a population.

    Assumes Poisson offspring distribution.

    See Equation 6.

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
        Beneficial mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability of rescue.
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    return 1 - (qw ** W0) * (qb ** B0)


##################################################
# 2.4 Weak selection/weak mutation approximation #
##################################################


def approx_prob_rescue_new(W0, r, s, u):
    '''
    Calculate the probability of rescue from new mutations assuming weak
    selection / weak mutation.

    See Equation 14.

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
        Beneficial mutation rate.

    Returns
    -------
    floats
        Probability of rescue.
    '''
    return 1 - exp(-2 * W0 * u * (s - r) / r)


def approx_prob_rescue_sv(B0, r, s):
    '''
    Calculate the probability of rescue from standing variation assuming weak
    selection / weak mutation.

    See Section 2.4.2.

    Parameters
    ----------
    B0 : int
        Initial number of mutant individuals
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.

    Returns
    -------
    floats
        Probability of rescue.
    '''
    return 1 - exp(-2 * B0 * (s - r))


######################
# 3. Population size #
######################


def M(r, s, u):
    '''
    Reproduction matrix.

    See Equation 15.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    '''
    return (1 - r) * np.array([[1 - u, u], [0, 1 + s]])


def Mn(r, s, u, n):
    '''
    nth power of reproduction matrix.

    See Equation 16.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    n : int
        Power.
    '''
    return (1 - r) ** n * np.array(
        [[(1 - u) ** n,
          u * ((1 + s) ** n - (1 - u) ** n) / (s + u)],
         [0, (1 + s) ** n]])


def WBn(W0, B0, r, s, u, n):
    '''
    Expected number of wildtype and mutant individuals in the population after
    n generations.

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
        Beneficial mutation rate.
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


def Zn(W0, B0, r, s, u, n):
    '''
    Expected total number of individuals in the population after n generations.

    Takes into account both rescued and extinct populations.

    See Equation 17.

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
        Beneficial mutation rate.
    n : int
        Number of generations.

    Returns
    -------
    float
        Expected total number of individuals.
    '''
    return WBn(W0, B0, r, s, u, n).sum()


###########################
# 4. Beneficial mutations #
###########################


def K(W0, r, u):
    '''
    Expected total number of beneficial mutations arising in a population.

    Takes into account both rescued and extinct populations, both successful
    and unsuccessful mutations.

    See Proposition 4.2.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    r : float
        Degree of maladaptation of wildtype individuals.
    u : float
        Beneficial mutation rate.

    Returns
    -------
    float
        Number of mutations.

    '''
    mu = 1 - r
    return W0 * u * mu / (1 - (1 - u) * mu)


##########################
# 5. Rescued populations #
##########################


#######################
# 5.1 Population size #
#######################

def dphi(t, mu, nmax):
    '''
    Derivative of phi.

    Assumes Poisson offspring distribution.

    Parameters
    ----------
    t : float
        Variable of pgf.
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
    for k in range(1, nmax):
        P = pois(mu, k)
        f += k * (t ** (k - 1)) * P
    return f


def dFv(v, t, mu, u, nmax):
    '''
    Derivative of F with respect to v.

    Assumes Poisson offspring distribution.

    Parameters
    ----------
    v, t : float
        Variables of pgf.
    mu : float
        Mean number of offspring.
    u : float
        Beneficial mutation rate.
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

    Assumes Poisson offspring distribution.

    Parameters
    ----------
    v, t : float
        Variables of pgf.
    mu : float
        Mean number of offspring.
    u : float
        Beneficial mutation rate.
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

    Assumes Poisson offspring distribution.

    Parameters
    ----------
    t : float
        Variable of pgf.
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

    Assumes Poisson offspring distribution.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    nmax : int
        Maximum number of offspring considered.
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    mu = 1 - r
    return np.array([[dFv(qw, qb, mu, u, nmax),
                      (qb / qw) * dFt(qw, qb, mu, u, nmax)],
                     [0, dG(qb, r, s, nmax)]])


def Mhatn(r, s, u, n, nmax):
    '''
    nth power of reproduction matrix.

    Assumes Poisson offspring distribution.

    See Equation 23.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    n : int
        Power.
    nmax : int
        Maximum number of offspring considered.
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    mu = 1 - r
    A = (qb / qw) * dFt(qw, qb, mu, u, nmax)
    B1 = dG(qb, r, s, nmax) ** n
    B2 = dFv(qw, qb, mu, u, nmax) ** n
    C = dG(qb, r, s, nmax) - dFv(qw, qb, mu, u, nmax)
    return np.array([[dFv(qw, qb, mu, u, nmax) ** n, A * (B1 - B2) / C],
                     [0, dG(qb, r, s, nmax) ** n]])


def rescued_WBn(W0, B0, r, s, u, n, nmax):
    '''
    Expected number of wildtype and mutant individuals in a rescued population
    after n generations.

    Assumes Poisson offspring distribution.

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
        Beneficial mutation rate.
    n : int
        Number of generations.
    nmax : int
        Maximum number of offspring considered.

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
    return np.dot(v, mm)


def rescued_Zn(W0, B0, r, s, u, n, nmax):
    '''
    Expected total number of individuals in a rescued population after n
    generations.

    Assumes Poisson offspring distribution.

    See Equation 24.

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
        Beneficial mutation rate.
    n : int
        Number of generations.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Expected total number of individuals.
    '''
    return rescued_WBn(W0, B0, r, s, u, n, nmax).sum()


####################################################
# 5.2 Weak selection / weak mutation approximation #
####################################################


def approx_rescued_Zn_new(W0, r, s, u, n):
    '''
    Expected total number of individuals in a rescued population after n
    generations assuming weak selection / weak mutation and rescue caused by
    new mutations.

    See Section 5.2.1.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    n : int
        Number of generations.

    Returns
    -------
    float
        Expected total number of individuals.
    '''
    A = (1 - r) ** n
    pb = 2 * (s - r)
    return W0 * A * (1 - u) ** n + (r / s) * A * (1 + s) ** n / pb


####################################
# 5.3 Number of rescuing mutations #
####################################


def KS(W0, r, s, u, nmax):
    '''
    Expected number of rescuing mutations arising in a rescued population.

    See Equation 27.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    r : float
        Degree of maladaptation of wildtype individuals.
    u : float
        Beneficial mutation rate.

    Returns
    -------
    float
        Number of mutations.

    '''
    pb = 1 - get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    mu = 1 - r
    return W0 * pb * u * mu / ((1 - (1 - u) * mu) * (1 - qw ** W0))


##############################################
# 4.2 Waiting time for a beneficial mutation #
##############################################


def phitilde(t, r, s, u, nmax):
    '''
    Pgf of W + BS offspring of a wildtype individual, where W is the number of
    wildtype offspring and BS is the number of mutant offspring that are
    successful in rescuing the population.

    Assumes Poisson offspring distribution.

    See Equation after proof of Proposition 4.3.

    Parameters
    ----------
    t : float
        Variable of pgf.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    qb = get_qb(r, s, nmax)
    pb = 1 - qb
    mu = 1 - r
    return phi((1 - u + u * pb) * t + u * qb, mu, nmax)


def utilde(r, s, u, nmax):
    '''
    The conditional distribution of BS given W + BS is binomial with success
    probability utilde.

    Assumes Poisson offspring distribution.

    Parameters
    ----------
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    qb = get_qb(r, s, nmax)
    pb = 1 - qb
    return u * pb / (1 - u + u * pb)


def Htilde(t, r, s, u, n, nmax):
    '''
    Recursive function used in calculating the distribution of TS, the waiting
    time for a rescuing mutation.

    Assumes Poisson offspring distribution.

    See Proposition 4.3 and Corolary 4.3.1.

    Parameters
    ----------
    t : float
        Variable of pgf.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    n : int
        Number of iterations.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    h = 1
    for i in range(n):
        h = phitilde(t * h, r, s, u, nmax)
    return h


############################################
# 5.4 Waiting time for a rescuing mutation #
############################################


def prob_time(W0, r, s, u, n, nmax):
    '''
    Probability that TS > n generations conditional on rescue.

    Assumes Poisson offspring distribution.

    See Corolary 5.0.1.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    n : int
        Number of generations.
    nmax : int
        Maximum number of offspring considered.

    Returns
    -------
    float
        Probability.
    '''
    qw = get_qw(r, s, u, nmax)
    return ((Htilde(1 - utilde(r, s, u, nmax), r, s, u, n, nmax)) ** W0 - qw ** W0) / (1 - qw ** W0)


def get_time(W0, r, s, u, nmax, tol):
    '''
    Calculate expected waiting time for a rescuing muation in a rescued
    population.

    Assumes Poisson offspring distribution.

    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    nmax : int
        Maximum number of offspring considered.
    tol : float
        Tolerance.

    Returns
    -------
    float
        Expected waiting time.
    '''
    t = 0
    n = 0
    stalled = False
    while not stalled:
        newt = t + prob_time(W0, r, s, u, n, nmax)
        deltat = newt - t
        t = newt
        if deltat < tol:
            stalled = True
        n += 1
    return t
