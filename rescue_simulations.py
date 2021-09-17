############################################################################
# Code for the simulation of population dynamics under evolutionary rescue #
# using the model described in the paper (not tracking mutations):         #
# "A Branching Process Model of Evolutionary Rescue"                       #
# Ricardo B. R. Azevedo & Peter Olofsson                                   #
# Last updated: September 17, 2021                                         #
############################################################################


import numpy as np
import numpy.random as rnd


def get_next_gen(W, B, r, s, u):
    '''
    Calculate the number of wildtype (W) and beneficial (B) individuals in the
    next generation.

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
        Beneficial mutation rate.

    Returns
    -------
    tuple of ints
        Values of W and B in the next generation.
    '''
    fitW = 1 - r
    assert fitW < 1
    fitB = fitW * (1 + s)
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
    Simulate evolution of a population until it either undergoes extinction or
    rescue.  Rescue is defined as the population size rising above a threshold.

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
        Beneficial mutation rate.
    rescue_threshold : int
        Population size above which a population is considered rescued.

    Returns
    -------
    tuple
        outcome, extinction time, arrays of W, B, and Z time series.
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
        Beneficial mutation rate.
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
