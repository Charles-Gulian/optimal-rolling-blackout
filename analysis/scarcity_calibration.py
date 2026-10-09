"""
Calibrate a load scaling factor that induces a target number of blackout days.

The RTS-ORB system as built is adequate: ~9.3 GW dispatchable against an ~8.7 GW
peak, so nothing sheds and the rolling-blackout model has nothing to schedule.
This sweeps a uniform load multiplier until scarcity appears at roughly the rate
we want to study.

Method
------
For each scaling factor, scale every load's peak by it, then solve a DC-OPF for
every day of July and August across ten years and count the days on which any
bus sheds. A "blackout day" is any day with unserved energy above a numerical
tolerance at any bus.

Shedding is left CONTINUOUS here -- no binary blackout decisions. We only need
to know *whether* a day is short of capacity, not how the outage would be
rotated, and an LP solves far faster than the MILP. Enabling binaries would not
change which days are short.

Timezone: profiles are on the UTC clock, and solve_opf slices 24 hours forward
from midnight UTC. "Days" here are therefore UTC days, which straddle two local
days (17:00 local to 16:00 local). That is fine for counting scarcity events but
would need care if we reported outage timing.
"""

import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from system import System  # noqa: E402

SCALING_FACTORS = (1.1, 1.2, 1.3, 1.4)
YEARS = range(2011, 2021)          # most recent ten years of the 2001-2020 window
MONTHS = (7, 8)                    # July and August
TARGET_BLACKOUT_DAYS = 3.0         # per year; stop once reached
UNSERVED_TOL_MWH = 1e-3            # ignore LP numerical dust


def summer_days(year, months=MONTHS):
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return [d for d in days if d.month in months]


def count_blackout_days(system, scaling_factor, years=YEARS, verbose=True):
    """Days with any unserved energy, per year, at the given load scaling."""
    base_peaks = {name: load.peak_load for name, load in system.loads.items()}
    for name, load in system.loads.items():
        load.peak_load = base_peaks[name] * scaling_factor

    system.write_opf()

    per_year, unserved_by_year = {}, {}
    try:
        for year in years:
            system.read_timeseries(year)
            days, unserved = 0, 0.0
            for day in summer_days(year):
                system.solve_opf(day)
                shed = sum(float(load.unserved_energy.value.sum())
                           for load in system.loads.values())
                if shed > UNSERVED_TOL_MWH:
                    days += 1
                    unserved += shed
            per_year[year] = days
            unserved_by_year[year] = unserved
            if verbose:
                print(f"      {year}: {days:2d} blackout days, {unserved:9.1f} MWh unserved",
                      flush=True)
    finally:
        for name, load in system.loads.items():   # always restore
            load.peak_load = base_peaks[name]

    return per_year, unserved_by_year


def main():
    system = System(ROOT / "data", ROOT / "system-config", "RTS-ORB")
    print(f"RTS-ORB: {len(system.loads)} loads, {len(system.resources)} resources")
    system.read_timeseries(YEARS[0] if isinstance(YEARS, list) else list(YEARS)[0])
    print(f"unscaled peak {system.peak_load:.0f} MW "
          f"({list(YEARS)[0]}); {len(list(summer_days(2015)))} days per summer, "
          f"{len(list(YEARS))} years\n")

    results = []
    for factor in SCALING_FACTORS:
        print(f"  scaling factor {factor:.1f}")
        t0 = time.time()
        per_year, unserved = count_blackout_days(system, factor)
        mean_days = sum(per_year.values()) / len(per_year)
        total_unserved = sum(unserved.values())
        results.append({"factor": factor, "mean_blackout_days": mean_days,
                        "total_unserved_MWh": total_unserved,
                        "years_with_none": sum(1 for v in per_year.values() if v == 0)})
        print(f"    -> {mean_days:.1f} blackout days/year, "
              f"{total_unserved:,.0f} MWh total unserved  [{time.time()-t0:.0f}s]\n",
              flush=True)
        if mean_days >= TARGET_BLACKOUT_DAYS:
            print(f"  reached the target of {TARGET_BLACKOUT_DAYS} days/year at "
                  f"scaling factor {factor:.1f}")
            break

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
