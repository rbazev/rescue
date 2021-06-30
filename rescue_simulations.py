from math import *
import numpy as np
import numpy.random as rnd
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Simulations

def get_next_gen(W, B, r, s, u):
    '''
    Calculate the number of wildtype (W) and beneficial (B) individuals in the next generation.
    
    Parameters
    ----------
    W : int
        Number of wildtype individuals.
    B : int
        Number of beneficial individuals
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate.
        
    Returns
    -------
    tuple of ints
        Values of W and B in the next generation
    '''
    fitW = 1 - r
    assert fitW < 1
    fitB = (1 - r) * (1 + s)
    assert fitB > 1
    if B == 0:
        nextB = 0
    else:
        nextB = rnd.poisson(lam=fitB, size=B).sum()
    if W == 0:
        nextW = 0
    else:
        nextW = rnd.poisson(lam=fitW, size=W).sum()
        if nextW > 0:
            nextW_mutants = rnd.binomial(1, u, size=nextW).sum()
            nextW -= nextW_mutants
            nextB += nextW_mutants
    return nextW, nextB

def evolve(W, B, r, s, u, rescue_threshold):
    '''
    Simulate evolution of a population until it either undergoes extinction or rescue.
    
    Z = W + B is the total population size.
    
    Parameters
    ----------
    W : int
        Number of wildtype individuals.
    B : int
        Number of beneficial individuals
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Mutation rate. 
    rescue_threshold : int
        Population size above which a population is considered rescued.

    Returns
    -------
    tuple
        outcome, generation, arrays of W, B, and Z values
    '''
    Z = W + B
    assert Z < rescue_threshold
    extinct = False
    rescued = False
    t = 0
    WW = [W]
    BB = [B]
    ZZ = [Z]
    while (not extinct) and (not rescued):
        W, B = get_next_gen(W, B, r, s, u)
        Z = W + B
        WW.append(W)
        BB.append(B)
        ZZ.append(Z)
        t += 1
        if Z == 0:
            extinct = True
        elif Z > rescue_threshold:
            rescued = True
    WW = np.array(WW)
    BB = np.array(BB)
    ZZ = np.array(ZZ)
    if extinct:
        return 'extinct', t, WW, BB, ZZ
    elif rescued:
        return 'rescued', t, WW, BB, ZZ
    
def evolven(W, B, r, s, u, n):
    '''
    Simulate evolution of a population for a certain number of generations.
    
    Z = W + B is the total population size.
    
    Parameters
    ----------
    W : int
        Number of wildtype individuals.
    B : int
        Number of beneficial individuals
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
    tuple
        arrays of W, B, and Z values
    '''
    Z = W + B
    WW = [W]
    BB = [B]
    ZZ = [Z]
    for i in range(n):
        W, B = get_next_gen(W, B, r, s, u)
        Z = W + B
        WW.append(W)
        BB.append(B)
        ZZ.append(Z)
    WW = np.array(WW)
    BB = np.array(BB)
    ZZ = np.array(ZZ)
    return WW, BB, ZZ

# Theory

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

def phiB(t, r, s, nmax):
    '''
    Probability generating function of the number of offspring of a beneficial individual.
    
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
        Difference between pgf and qb.  qb is estimated by minimizing this value.
    '''
    return phiB(qb, r, s, nmax) - qb

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
        Difference between pgf and qw.  qw is estimated by minimizing this value.
    '''
    qb = get_qb(r, s, nmax)
    mu = 1 - r
    return phi(qw * (1 - u) + u * qb, mu, nmax) - qw

def qwu0_fun(qw, r, nmax):
    '''
    Function used to calculate the probability of extinction of a population starting from one wildtype individual, qw, assuming no mutation.
    
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
        Difference between pgf and qw.  qw is estimated by minimizing this value.
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
        Initial number of beneficial individuals
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
    float
        Probability of rescue.
    '''
    qb = get_qb(r, s, nmax)
    qw = get_qw(r, s, u, nmax)
    pnew = 1 - exp(-2 * W0 * u * (s - r) / r)
    pstand = 1 - exp(-2 * B0 * (s - r))
    orr = pstand + (1 - pstand) * pnew
    if verbose:
        print('pstand:', 1 - qb ** B0, pstand)
        print('  pnew:', 1 - qw ** W0, pnew)
        print('ptotal:', 1 - (qw ** W0) * (qb ** B0), orr)
    return 1 - (qw ** W0) * (qb ** B0), orr
    
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
    Number of wildtype and beneficial indiviuals in the population after n generations.  Takes into account both rescued and extinct populations.
    
    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    B0 : int
        Initial number of beneficial individuals
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
        Number of wildtype and beneficial indiviuals after n generations.
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
    Number of wildtype and beneficial indiviuals in the population after n generations.  Takes into account both rescued and extinct populations.
    
    Parameters
    ----------
    W0 : int
        Initial number of wildtype individuals.
    B0 : int
        Initial number of beneficial individuals
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
        Number of wildtype and beneficial indiviuals after n generations.
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