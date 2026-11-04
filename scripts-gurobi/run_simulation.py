"""
Multi-year simulation of the system under random generator forced outages.

    system = System(data_dir, config_dir, "RTS-ORB")
    df = simulate_years(system, n_years=10)
    print(reliability_summary(df))
"""

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
import pathlib
import pandas as pd
from tqdm import tqdm

from system import System, UNSERVED_TOL

def simulate_years(system, n_years, year=2020, first_seed=0, days=None, verbose=True):
    """Solve every day of `year` under `n_years` independent outage draws.

    Returns one row per (replicate, day): generation cost, unserved energy cost, total unserved energy, loss-of-load hours
    """
    if not system.forced_outages:
        raise ValueError("system was built with forced_outages=False; there is nothing random to replicate")

    system.read_timeseries(year)
    system.write_opf()
    if days is None:
        days = system.available_days(year)

    rows = []
    bar = tqdm(total=n_years * len(days), unit="solve", disable=None if verbose else True,
               desc="simulating")
    for replicate in range(n_years):
        scenario_seed = first_seed + replicate
        system.scenario_seed = scenario_seed
        for resource in system.resources.values():
            resource.get_outage_profile(year, scenario_seed)

        lol_days, replicate_ue = 0, 0.0
        for date in days:
            system.solve_opf(date)
            if system.model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"replicate {replicate}, {date.date()}: "f"solver status {system.model.Status}")

            # (T, n_loads) MW; one load per bus, so columns are buses.
            ue_profile = system.unserved_energy_profile
            lmp_profile = system.lmp_profile
            ue_by_hour = ue_profile.sum(axis=1)
            lmp_by_hour = lmp_profile.max(axis=1)
            total_ue = float(ue_by_hour.sum())
            rows.append({
                "replicate": replicate,
                "scenario_seed": scenario_seed,
                "date": date,
                "ue_costs": float(system.total_unserved_energy_costs.getValue()),
                "total_ue": total_ue,
                "ue_hours": int((ue_by_hour > UNSERVED_TOL).sum()),
                "crit_hours": int((lmp_by_hour > 1e-2).sum()),
                "event_type": system.ue_event_type,
            })

            lol_days += total_ue > UNSERVED_TOL
            replicate_ue += total_ue
            bar.update(1)
            bar.set_postfix(rep=replicate, lol_days=lol_days,
                            EUE=f"{replicate_ue:,.0f} MWh", refresh=False)

        if verbose:
            bar.write(f"  replicate {replicate:3} (seed {scenario_seed:6}): "
                      f"{lol_days:3} loss-of-load days, "
                      f"{replicate_ue:12,.1f} MWh unserved")
    bar.close()

    return pd.DataFrame(rows)

def reliability_summary(df):
    """Standard reliability metrics per replicate, plus the mean over them.
    """
    per = df.groupby("replicate").agg(
        LOLE_days=("total_ue", lambda s: int((s > UNSERVED_TOL).sum())),
        LOLH_hours=("ue_hours", "sum"),
        crit_hours=("crit_hours", "sum"),
        EUE_MWh=("total_ue", "sum"),
        total_ue_costs=("ue_costs", "sum"),
        max_ue_hours=("ue_hours", "max"),
        max_crit_hours=("crit_hours", "max"),
    )
    per.loc["mean"] = per.mean()
    return per

if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    system = System(root / "data", root / "system-config", "RTS-ORB-HR")
    df_results = simulate_years(system, 2)
    df_summary = reliability_summary(df_results)
    print(df_summary)
    print(df_results["event_type"].value_counts())