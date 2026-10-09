"""
Find the coal retirement depth that brings RTS-ORB-HR's reliability back in
line with RTS-ORB.

RTS-ORB-HR adds 3 GW of wind, solar and storage, which makes it markedly more
reliable than RTS-ORB. Retiring coal STEAM walks that back. We want the rung of
RETIREMENT_LADDER whose 20-year LOLE is closest to RTS-ORB's.

Method
------
LOLE is the number of days with unserved energy, and a day has unserved energy
in the LP relaxation if and only if it does in the MILP. So the LP screen alone
gives LOLE exactly, and we never need the MILP pass while searching.

The system is built and compiled ONCE. Between rungs we only zero
`nameplate_capacity` on the retired units: ThermalResource.pmax is a
cp.Parameter refreshed from it each day, so the constraint structure never
changes and nothing is recompiled. Outage draws are untouched, since the unit
stays in the fleet and keeps its index -- every rung sees identical random
numbers, so a LOLE difference is purely the retirement.

The search is a bisection over rung index, seeded at the caller's first guess.

The committed config retires by deleting the row outright, which is equally
safe now that seeds key off Resource ID; the sweep zeroes instead only because
a compiled CVXPY problem has a fixed variable set. The two agree exactly -- a
retired unit contributes nothing either way, and a zeroed unit's own draw is
irrelevant because it cannot generate.

The sweep therefore expects the on-disk config at rung 0. Once a rung is
chosen, regenerate the committed config with `make_system_config.py <rung>` so
the CSV is the source of truth.
"""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data-scripts"))

from make_system_config import RETIREMENT_LADDER  # noqa: E402
from system import UNSERVED_TOL, System  # noqa: E402

YEARS = range(2001, 2021)


def _ladder_units(system):
    """Map rung index -> the resource objects it retires, in ladder order."""
    by_rung = []
    for bus, fuel in RETIREMENT_LADDER:
        units = [r for r in system.resources.values()
                 if r.node_ID == bus and r.unit_type == "STEAM"
                 and r.data["Fuel"] == fuel]
        if not units:
            raise KeyError(
                f"no {fuel} STEAM at bus {bus}. The sweep switches rungs in "
                f"memory on top of an unretired fleet, so the on-disk config "
                f"must be at rung 0: run `python data-scripts/make_system_config.py 0`.")
        by_rung.append(units)
    return by_rung


def set_rung(system, by_rung, rung):
    """Retire the first `rung` ladder entries; restore everything above."""
    retired = 0.0
    for i, units in enumerate(by_rung):
        for u in units:
            if i < rung:
                u.nameplate_capacity = 0.0
            else:
                u.nameplate_capacity = u.data["PMax MW"]
        if i < rung:
            retired += sum(u.data["PMax MW"] for u in units)
    return retired


def lole(system, years=YEARS, verbose=True):
    """20-year LOLE in days/yr, via the LP screen."""
    total_days = 0
    for year in years:
        system.read_timeseries(year)
        for day in system.available_days(year):
            system.solve_opf(day)
            if system.solve_summary["unserved_MWh"] > UNSERVED_TOL:
                total_days += 1
    n = len(list(years))
    if verbose:
        print(f"      {total_days} UE days over {n} years", flush=True)
    return total_days / n


def bisect_rung(system, target, first_guess=7, lo=0, hi=len(RETIREMENT_LADDER)):
    """Bisect on rung index for the LOLE closest to `target`.

    LOLE increases with rung (more coal retired), so this is a standard
    bisection on a monotone step function. We evaluate every rung we visit and
    return the closest, rather than the bracket endpoint, because the ladder is
    coarse -- neighbouring rungs can straddle the target by a wide margin.
    """
    by_rung = _ladder_units(system)
    seen = {}
    guess = first_guess

    while lo < hi:
        mw = set_rung(system, by_rung, guess)
        t0 = time.time()
        print(f"  rung {guess:2d} (-{mw:4.0f} MW): ", end="", flush=True)
        seen[guess] = lole(system)
        print(f"      -> LOLE {seen[guess]:.2f} days/yr  [{time.time() - t0:.0f}s]", flush=True)

        if seen[guess] < target:
            lo = guess + 1          # too reliable, retire more
        else:
            hi = guess              # overshot, retire less
        guess = (lo + hi) // 2
        if guess in seen:
            break

    best = min(seen, key=lambda r: abs(seen[r] - target))
    return best, seen


def build(system_config="RTS-ORB-HR", scenario_seed=0):
    """Build and compile once; rungs are applied in memory afterwards."""
    system = System(ROOT / "data", ROOT / "system-config", system_config,
                    scenario_seed=scenario_seed)
    system.disable_blackouts()
    system.write_opf()
    return system


def sweep(rungs, system_config="RTS-ORB-HR", years=YEARS, scenario_seed=0):
    """Evaluate LOLE at each rung in `rungs`, printing as it goes."""
    system = build(system_config, scenario_seed)
    by_rung = _ladder_units(system)
    print(f"{system_config}: {len(system.resources)} resources, "
          f"{len(list(years))} years, seed {scenario_seed}\n", flush=True)

    results = {}
    for rung in rungs:
        mw = set_rung(system, by_rung, rung)
        t0 = time.time()
        print(f"  rung {rung:2d} (-{mw:4.0f} MW coal):", flush=True)
        results[rung] = lole(system, years)
        print(f"      LOLE {results[rung]:.2f} days/yr  "
              f"[{time.time() - t0:.0f}s]\n", flush=True)

    print("SUMMARY")
    for rung, v in results.items():
        mw = sum(sum(u.data["PMax MW"] for u in by_rung[i]) for i in range(rung))
        print(f"  rung {rung:2d}  -{mw:5.0f} MW  LOLE {v:5.2f} days/yr")
    return results


if __name__ == "__main__":
    rungs = [int(a) for a in sys.argv[1:]] or [7]
    sweep(rungs)
