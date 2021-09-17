###############################################################################
# Code for the simulation of population dynamics under evolutionary rescue    #
# using the model described in the paper:                                     #
# "A Branching Process Model of Evolutionary Rescue".                         #
# Authors: Ricardo B. R. Azevedo & Peter Olofsson.                            #
# Note: tracks the appearance and spread of individual beneficial mutations.  #
# Last updated: September 17, 2021.                                           #
###############################################################################


import numpy as np
import numpy.random as rnd


def get_next_gen(population, appearance_times, disappearance_times, r, s, u, t):
    '''
    Calculate the number of individuals in each genotype class in the next
    generation.

    Parameters
    ----------
    population : list
        Number of individuals of each genotype: [W, B1, B2, ..., Bk], where W
        is the number of wildtype individuals, k is the number of independent
        beneficial mutations that have ever occurred in the population, and
        Bi is the number of individuals carrying the i-th beneficial mutation.
    appearance_times : list
        Time of appearance of genotypes W, B1, B2, ..., Bk.  Value is 0 if the
        genotype was present at the start of the simulation.
    disappearance_times : type
        Time of extinction of genotypes W, B1, B2, ..., Bk.  Value is -1 if the
        genotype has not gone extinct yet.
    r : float
        Degree of maladaptation of wildtype individuals.
    s : float
        Effect of a beneficial mutation.
    u : float
        Beneficial mutation rate.
    t : int
        Generation number.

    Returns
    -------
    tuple
        population, appearance_times, disappearance_times.
    '''
    fitW = 1 - r
    assert fitW < 1
    fitB = (1 - r) * (1 + s)
    assert fitB > 1
    ngenotypes = len(population)
    assert ngenotypes == len(appearance_times)
    assert ngenotypes == len(disappearance_times)
    if ngenotypes > 1:
        for i in range(1, ngenotypes):
            B = population[i]
            if B == 0:
                nextB = 0
            else:
                nextB = rnd.poisson(lam=fitB, size=B).sum()
                population[i] = nextB
                if nextB == 0:
                    disappearance_times[i] = t
    W = population[0]
    if W == 0:
        nextW = 0
    else:
        nextW = rnd.poisson(lam=fitW, size=W).sum()
        if nextW > 0:
            nextW_mutants = rnd.binomial(1, u, size=nextW).sum()
            if nextW_mutants > 0:
                nextW -= nextW_mutants
                for i in range(nextW_mutants):
                    population.append(1)
                    appearance_times.append(t)
                    disappearance_times.append(-1)
        else:
            disappearance_times[0] = t
        population[0] = nextW
    return population, appearance_times, disappearance_times


def evolve(W, B, r, s, u, rescue_threshold):
    '''
    Simulate evolution of a population until it either undergoes extinction or
    rescue.  Rescue is defined as the population size rising above a threshold.

    Z = W + B1 + B2 + ... + Bk is the total population size.

    Warning:

    Parameters
    ----------
    W : int
        Number of wildtype individuals.
    B : int
        Number of mutant individuals.
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
    extinct = False
    rescued = False
    t = 0
    if B == 0:
        population = [W]
        appearance_times = [0]
        disappearance_times = [-1]
    else:
        population = [W, B]
        appearance_times = [0, 0]
        disappearance_times = [-1, -1]
    Z = sum(population)
    assert 0 < Z < rescue_threshold
    WW = [W]
    ZZ = [Z]
    while (not extinct) and (not rescued):
        t += 1
        population, appearance_times, disappearance_times = \
        get_next_gen(population, appearance_times, disappearance_times, r, s,
            u, t)
        W = population[0]
        Z = sum(population)
        WW.append(population[0])
        ZZ.append(Z)
        if Z == 0:
            extinct = True
        elif (Z > rescue_threshold):
            rescued = True
    WW = np.array(WW)
    ZZ = np.array(ZZ)
    Ks = 0
    Ts = []
    for i in range(1, len(disappearance_times)):
        if disappearance_times[i] == -1:
            Ks += 1
            Ts.append(appearance_times[i])
    if extinct:
        return 'extinct', t, WW, ZZ - WW, ZZ, appearance_times, disappearance_times, Ks, Ts, population
    elif rescued:
        return 'rescued', t, WW, ZZ - WW, ZZ, appearance_times, disappearance_times, Ks, Ts, population
