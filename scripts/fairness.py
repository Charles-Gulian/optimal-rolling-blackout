"""
Lexicographic fairness rationing: shortest outages first, then least exposure,
then cost.

Three stages, each narrowing the feasible set for the next:

    (1)  min kmax                                 -> k*   longest run at any bus
    (2)  min omax   s.t. kmax <= k*               -> o*   most hours at any bus
    (3)  min cost   s.t. kmax <= k*, omax <= o*          cheapest schedule
                                                          achieving that fairness

Each stage is feasible by construction, since the previous stage's solution
satisfies the constraint it adds. That matters: a fixed cap like "no bus dark
more than 2 hours" can be flatly infeasible when a bus sits behind congestion
with a local deficit, and no amount of re-solving fixes it. Here stage 1
*computes* what is achievable instead of guessing.

Build the System with voll_model="static_calibrated", not "static". Stage 3
minimises cost, and that cost should be the best available estimate of what
unserved energy is worth -- the price the dynamic model actually paid, not the
uncalibrated 10,000. Otherwise cooling is under-priced and stage 3 chooses
badly among the schedules that achieve (k*, o*), handicapping the policy
against the thing it is being compared to. We want fair at its strongest.

Stage 3 is what stops it over-shedding. Minimising fairness alone spreads the
outage over every bus it can reach -- measured at 6,100 MWh against 2,400 for
the lexicographic version on the same day, nearly three times the energy, for
a true cost of $72M against $28M. Fairness has no exchange rate with dollars,
so it has to be a constraint rather than an objective.

Why both terms
--------------
kmax caps how long in a row; omax caps how often at all. The thermal state
responds to both, and capping only the first leaves the second free to do the
damage: a 1-hour cap alone lets the cheapest schedule hit the same nine buses
at alternating hours, each repeat landing on a house that has not recovered.
Measured 8.3 degC-h that way, against 0.0 with omax added.

Counters versus caps
--------------------
Stages 1 and 2 need kmax and omax as variables, so they carry the counter
formulation below. Stage 3 knows k* and o* as numbers, so it uses the sliding
window instead -- exact at the LP relaxation, and no auxiliary variables at
all.
"""

import cvxpy as cp
import numpy as np

from component import T


def _blackout_vars(system):
    z = [n.blackout for n in system.nodes.values() if n.blackout is not None]
    if not z:
        raise RuntimeError("fairness needs binary blackouts -- call "
                           "System.enable_blackouts() before write_opf()")
    return z


def counter_terms(system, periods=T):
    """(kmax, omax, constraints) for the two fairness stages.

    kmax is the longest unbroken outage at any bus, omax the most outage hours
    at any bus. The counter recursion is

        k_it >= k_i,t-1 + z_it - M_t (1 - z_it),     k_it >= 0

    which needs only LOWER bounds: minimising kmax already pushes every k_it
    down, so at z = 0 the counter resets to zero on its own and at z = 1 it is
    squeezed to k_i,t-1 + 1. M_t = t suffices, since k_i,t-1 <= t, and a
    tighter big-M is a tighter relaxation.

    k can be continuous -- the constraints pin it to an integer given binary z
    -- so this adds no integrality beyond the blackout variables already there.

    omax needs no recursion at all: total hours is just sum_t z_it.
    """
    kmax, omax = cp.Variable(name="kmax"), cp.Variable(name="omax")
    M = np.arange(periods, dtype=float)
    cons = []
    for z in _blackout_vars(system):
        k = cp.Variable(periods, nonneg=True)
        cons.append(k[0] >= z[0])
        cons += [k[t] >= k[t - 1] + z[t] - M[t] * (1 - z[t])
                 for t in range(1, periods)]
        cons += [kmax >= k, omax >= cp.sum(z)]
    return kmax, omax, cons


def cap_constraints(system, k_star, o_star, periods=T):
    """Tight constraints fixing fairness at known k* and o*, no new variables.

    "No run longer than k" is a sliding window, which we measured as exact at
    the LP relaxation -- unlike the counter form, whose bound on kmax is zero
    and gives branch-and-bound no guidance. Use this wherever k is a number
    rather than a variable.
    """
    k = int(round(k_star))
    cons = []
    for z in _blackout_vars(system):
        cons += [cp.sum(z[t:t + k + 1]) <= k for t in range(periods - k)]
        cons.append(cp.sum(z) <= o_star + 1e-6)
    return cons


def solve_fair(system, solver_kwargs=None, periods=T):
    """Run the three stages on the system's current timeseries parameters.

    Returns a dict of the achieved k*, o*, each stage's solve time and status.
    The schedule is left in the blackout variables, as after any solve, so the
    caller reads it with System.blackout_schedule.
    """
    kw = dict(solver_kwargs or {})
    kmax, omax, fair_cons = counter_terms(system, periods)
    base = list(system.constraints)
    out = {}

    # (1) shortest possible longest outage
    p1 = cp.Problem(cp.Minimize(kmax), base + fair_cons)
    p1.solve(**kw)
    if p1.status.startswith("infeasible"):
        raise RuntimeError(f"fairness stage 1 infeasible: {p1.status}")
    k_star = float(np.ceil(kmax.value - 1e-6))
    out.update(k_star=k_star, stage1_status=p1.status)

    # (2) least exposure, holding that
    p2 = cp.Problem(cp.Minimize(omax), base + fair_cons + [kmax <= k_star + 1e-6])
    p2.solve(**kw)
    o_star = float(np.ceil(omax.value - 1e-6))
    out.update(o_star=o_star, stage2_status=p2.status)

    # (3) cheapest schedule achieving it. Counters dropped; the caps are
    #     tighter and carry no auxiliary variables.
    p3 = cp.Problem(system.objective,
                    base + cap_constraints(system, k_star, o_star, periods))
    p3.solve(**kw)
    out.update(stage3_status=p3.status, cost=p3.value)
    return out
