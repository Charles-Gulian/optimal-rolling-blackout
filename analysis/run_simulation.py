"""
Twenty-year reliability simulation for a system config, with results reporting.

Two passes, exploiting the fact that a day has unserved energy in the LP
relaxation if and only if it does in the MILP:

  1. LP screen -- solve every day with continuous shedding. Fast (~0.2 s/day).
     On days with no unserved energy the LP and MILP solutions are identical
     (both have u = 0, z = 0 and the same dispatch), so LP costs are exact there.

  2. MILP -- re-solve only the days the screen flagged, with binary per-bus
     blackout decisions. This gives the real unserved energy, which is much
     larger than the LP's: the LP can trim a fraction of a MW at one bus, while
     the MILP must de-energise a whole bus for a whole hour.

Why the screen is valid: "a zero-unserved-energy dispatch exists" is a property
of the network and fleet, not of how shedding is parameterised -- the two
formulations share an identical constraint set when u = 0. It holds as long as
VOLL exceeds the largest generation-cost saving shedding could buy (10,000 vs
179.5 $/MWh here) and nothing makes z = 0 infeasible.

Outputs, under results/<config>/:

    reliability_by_year.csv   EUE (MWh), NEUE (ppm), LOLE (days), LOLH (hours)
    costs_by_year.csv         generation and unserved-energy cost per year
    unserved_energy_days.csv  one row per UE day
    daily_summary.csv         one row per simulated day (LP screen)
    outages/<date>.csv        per UE day: timestamps x bus, 0/1 blackout

Timezone: profiles are on the UTC clock; a simulated "day" is one whole LOCAL
day, 00:00-23:00 local = 07:00 UTC to 06:00 UTC. That keeps the 17:00-21:00
local shedding block inside a single problem instead of splitting it across
two. `date` in the outputs is the local calendar date; the per-day outage
files are indexed in UTC, so subtract 7 h to read local timing.
"""

import pathlib
import sys
import time

import cvxpy as cp
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from system import UNSERVED_TOL, System  # noqa: E402

YEARS = range(2001, 2021)
SYSTEM_CONFIG = "RTS-ORB"
MIP_GAP = 1e-3
MIP_TIME_LIMIT = 120.0


def lp_screen(system, years=YEARS, verbose=True):
    """Pass 1: solve every day with continuous shedding."""
    system.disable_blackouts()
    system.write_opf()
    rows = []
    for year in years:
        t0 = time.time()
        system.read_timeseries(year)
        days = system.available_days(year)
        for day in days:
            system.solve_opf(day)
            rows.append(system.solve_summary)
        if verbose:
            flagged = sum(1 for r in rows[-len(days):] if r["unserved_MWh"] > UNSERVED_TOL)
            print(f"    {year}: {len(days)} days, {flagged} flagged  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return _indexed(rows)


def _indexed(rows):
    """Frame of solve summaries indexed by local date, empty-safe.

    A year with no blackout days at all leaves `rows` empty, and
    pd.DataFrame([]).set_index("date") raises KeyError because there are no
    columns to index on. Return a frame carrying an empty DatetimeIndex named
    "date" instead, so callers can use .index.year and len() uniformly whether
    or not anything was shed.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({"date": pd.to_datetime([])})
    return df.set_index("date")


def _longest_run(flags):
    """Longest unbroken run of 1s in a 0/1 sequence."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def milp_days(system, window_starts, out_dir, verbose=True):
    """Pass 2: re-solve the flagged days with binary blackout decisions.

    Takes UTC window starts, not local dates -- these get passed straight to
    update_timeseries_parameters(), which slices the profiles forward from
    them, so they must be the real start of the 24 h window.
    """
    system.enable_blackouts()
    system.write_opf()
    outage_dir = out_dir / "outages"
    outage_dir.mkdir(parents=True, exist_ok=True)

    rows, current_year = [], None
    for day in window_starts:
        if day.year != current_year:
            current_year = day.year
            system.read_timeseries(current_year)
        t0 = time.time()
        for obj in system.components:
            obj.update_timeseries_parameters(day)
        system.prob.solve(solver=cp.GUROBI, MIPGap=MIP_GAP,
                          TimeLimit=MIP_TIME_LIMIT, **system.solver_options)
        system.opf_date = day

        summary = system.solve_summary
        schedule = system.blackout_schedule
        summary["bus_hours_shed"] = int(schedule.values.sum())
        summary["buses_shed"] = int((schedule.sum(axis=0) > 0).sum())

        # Worst-bus outage duration, the two senses that matter for thermal
        # state. A bus out 17:00 and again 19:00 scores 2 and 1; one out
        # 17:00-19:00 scores 3 and 3. Only the consecutive figure can drive
        # indoor temperature past the discomfort threshold, so a day with a
        # high total but low consecutive count cannot exercise dynamic VOLL.
        # The bus attaining each maximum need not be the same bus.
        per_bus = schedule.sum(axis=0)
        summary["max_bus_unserved_hours"] = int(per_bus.max())
        summary["max_bus_consecutive_hours"] = int(
            max(_longest_run(schedule[bus].to_numpy()) for bus in schedule.columns))
        summary["solve_s"] = time.time() - t0
        rows.append(summary)

        # Name the file for the LOCAL day, not the UTC window start.
        schedule.to_csv(outage_dir / f"{summary['date'].date()}.csv")
        if verbose:
            print(f"    {summary['date'].date()}: {summary['unserved_MWh']:8.1f} MWh, "
                  f"{summary['bus_hours_shed']:3d} bus-hours, "
                  f"{summary['unserved_hours']} h  [{summary['solve_s']:.1f}s]", flush=True)
    return _indexed(rows)


def stamp(frames, config, seed):
    """Put config/seed on every table.

    Needed to concatenate across the runs of a multi-seed study, and to
    reproduce any single row: (config, seed) plus the date fully determines
    the outage draw, since outages are sampled from Resource ID x year x
    scenario_seed rather than stored.
    """
    for df in frames:
        if len(df) and "config" not in df.columns:
            df.insert(0, "config", config)
            df.insert(1, "seed", seed)


def aggregate(daily_lp, daily_milp, years=YEARS):
    """Annual reliability and cost tables.

    Costs combine the two passes: LP on clean days (where it is exact) and MILP
    on unserved-energy days. Reliability metrics come from the MILP alone, since
    the LP understates how much load binary shedding actually drops.
    """
    clean = daily_lp[daily_lp["unserved_MWh"] <= UNSERVED_TOL]

    reliability, costs = [], []
    for year in years:
        c = clean[clean.index.year == year]
        m = daily_milp[daily_milp.index.year == year] if len(daily_milp) else daily_milp
        reliability.append({
            "year": year,
            "EUE_MWh": float(m["unserved_MWh"].sum()) if len(m) else 0.0,
            "LOLE_days": int(len(m)),
            "LOLH_hours": int(m["unserved_hours"].sum()) if len(m) else 0,
            "bus_hours_shed": int(m["bus_hours_shed"].sum()) if len(m) else 0,
            "load_MWh": float(c["load_MWh"].sum() + (m["load_MWh"].sum() if len(m) else 0.0)),
        })
        gen = float(c["gen_cost"].sum() + (m["gen_cost"].sum() if len(m) else 0.0))
        ue = float(m["unserved_cost"].sum()) if len(m) else 0.0
        costs.append({"year": year, "gen_cost": gen, "unserved_cost": ue,
                      "total_cost": gen + ue})

    rel = pd.DataFrame(reliability)
    # Normalised EUE in parts per million of annual energy -- NERC's NEUE. Its
    # published risk bands: 0 is low, <=20 ppm medium, >20 ppm high.
    rel["NEUE_ppm"] = 1e6 * rel["EUE_MWh"] / rel["load_MWh"]
    return rel, pd.DataFrame(costs)


def main(system_config=SYSTEM_CONFIG, years=YEARS, tag=None, scenario_seed=0):
    # `tag` separates results for runs that share a config but differ in a knob
    # we are sweeping (load scaling, retirement rung, MC draw), so a tuning
    # sweep does not overwrite itself.
    out_dir = ROOT / "results" / (system_config if tag is None else f"{system_config}-{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    system = System(ROOT / "data", ROOT / "system-config", system_config,
                    scenario_seed=scenario_seed)
    print(f"{system_config}{'' if tag is None else f' [{tag}]'}: {len(system.loads)} loads, "
          f"{len(system.resources)} resources, {len(list(years))} years, "
          f"seed {scenario_seed}\n")

    print("PASS 1 -- LP screen (continuous shedding, every day)")
    t0 = time.time()
    daily_lp = lp_screen(system, years)
    # Ask for window_start explicitly. The table is indexed by local date,
    # which is the right label but the wrong thing to re-solve from.
    flagged = daily_lp.loc[daily_lp["unserved_MWh"] > UNSERVED_TOL, "window_start"]
    print(f"  {len(daily_lp)} days solved, {len(flagged)} flagged  "
          f"[{time.time() - t0:.0f}s]\n")

    print(f"PASS 2 -- MILP on {len(flagged)} flagged days (binary blackouts)")
    t0 = time.time()
    daily_milp = milp_days(system, flagged, out_dir)
    print(f"  [{time.time() - t0:.0f}s]\n")

    rel, costs = aggregate(daily_lp, daily_milp, years)

    stamp((daily_lp, daily_milp, rel, costs), system_config, scenario_seed)

    daily_lp.to_csv(out_dir / "daily_summary.csv")
    daily_milp.to_csv(out_dir / "unserved_energy_days.csv")
    rel.to_csv(out_dir / "reliability_by_year.csv", index=False)
    costs.to_csv(out_dir / "costs_by_year.csv", index=False)

    print("RELIABILITY BY YEAR")
    print(rel.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("\nCOSTS BY YEAR ($)")
    print(costs.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(f"\nwrote {out_dir.relative_to(ROOT)}/")
    return rel, costs, daily_milp


if __name__ == "__main__":
    main()
