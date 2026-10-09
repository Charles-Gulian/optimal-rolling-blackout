"""
Forced outage sampling for generators.

Each unit is a two-state Markov chain in hourly steps: available or on forced
outage. The chain is calibrated so its stationary down-probability equals the
unit's FOR from gen.csv and its mean down-time equals MTTR.

Reproducibility
---------------
Draws are generated on demand from a seed rather than stored. The seed for a
unit-year is derived by Cantor pairing the unit's index with the year, then
offset by a scenario seed, so that:

  * the same scenario seed always reproduces the same outages,
  * units and years are independent of one another,
  * changing the scenario seed gives an independent Monte Carlo replication,
  * adding or removing units does not reshuffle the others, provided their
    indices are stable (we use the unit's position in a sorted list of names).

Scope
-----
Only units with FOR > 0 in gen.csv are sampled: thermal (CT, STEAM, CC,
NUCLEAR) and hydro (HYDRO, ROR). PV, RTPV, WIND and STORAGE carry FOR = 0
upstream, since their availability is already represented by their profiles.
"""

import numpy as np

DEFAULT_SCENARIO_SEED = 0


def cantor_pair(k1, k2):
    """Encode two non-negative integers as one."""
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2


def outage_seed(unit_index, year, scenario_seed=DEFAULT_SCENARIO_SEED):
    """Deterministic, collision-free seed for one unit-year."""
    return cantor_pair(int(unit_index), int(year)) + int(scenario_seed)


def forced_outage_profile(FOR, MTTR, T, rng=None):
    """Binary hourly availability of length T. 1 = available, 0 = forced outage.

    Parameters
    ----------
    FOR : float
        Stationary forced outage rate, the long-run fraction of time down.
    MTTR : float
        Mean time to repair, in timesteps (hours here).
    T : int
        Number of timesteps.
    rng : numpy.random.Generator, optional
    """
    if not (0.0 < FOR < 1.0):
        raise ValueError("FOR must be in (0,1).")
    if MTTR <= 0.0:
        raise ValueError("MTTR must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    # Calibrate transition probabilities so the chain's stationary down
    # probability is FOR and its mean down-time is MTTR.
    p10 = 1.0 / MTTR                    # down -> up
    p01 = (FOR / (1.0 - FOR)) * p10     # up -> down
    if p01 >= 1.0:
        raise ValueError("Implied failure probability >= 1. Check FOR/MTTR consistency.")

    profile = np.empty(T, dtype=np.uint8)
    profile[0] = 1 if rng.random() < (1.0 - FOR) else 0   # start from stationary
    u = rng.random(T - 1)

    for t in range(1, T):
        if profile[t - 1] == 1:
            profile[t] = 0 if u[t - 1] < p01 else 1
        else:
            profile[t] = 1 if u[t - 1] < p10 else 0

    return profile
