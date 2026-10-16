"""
Score every saved blackout schedule under the TRUE dynamic VOLL cost.

run_experiments.py solves each catalogued day under several VOLL models and
saves the schedules. Those solves are not comparable to each other -- each
reports its own objective, and the exogenous objective charges less for the
same schedule by construction. This pass puts them on one footing: take each
schedule as given and ask what it costs under the dynamic model, which is the
one we believe.

    python analysis/evaluate_schedules.py --experiments results/experiments/voll-ladder

Writes <experiments>/evaluated.csv, one row per (config, voll_model, seed,
date), and prints the headline comparison.

Scoring is not an optimisation -- z is fixed, so x_t follows from the physics
and v(x_t) is just a parameter. It needs no solver and takes seconds.

Note the evaluation depends on the year (through demand and temperature) but
NOT on scenario_seed: generator outages change which schedule the optimiser
chose, not what a given schedule costs. So one System per config suffices,
stepping through years.
"""

import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

from evaluate import evaluate, load_schedule  # noqa: E402
from system import System  # noqa: E402

CONFIGS = ("RTS-ORB", "RTS-ORB-HR")
MODELS = ("dynamic", "exogenous")


def schedule_paths(experiments, config, model):
    """(seed, year, path) for every saved schedule of one config and model."""
    base = pathlib.Path(experiments) / config / model
    for p in sorted(base.glob("seed*/year*/outages/*.csv")):
        seed = int(p.parents[2].name.removeprefix("seed"))
        year = int(p.parents[1].name.removeprefix("year"))
        yield seed, year, p


def evaluate_config(experiments, config, models, verbose=True):
    system = System(ROOT / "data", ROOT / "system-config", config,
                    voll_model="dynamic")
    system.enable_blackouts()
    system.write_opf()

    work = []
    for model in models:
        work += [(model, s, y, p) for s, y, p in
                 schedule_paths(experiments, config, model)]
    rows, loaded = [], None
    # Sort by year so each year's profiles are read once.
    for model, seed, year, path in sorted(work, key=lambda t: (t[2], t[0], t[1])):
        if year != loaded:
            system.read_timeseries(year)
            loaded = year
        sched = load_schedule(path)
        r = evaluate(system, sched.index[0], sched)
        r.pop("per_load"), r.pop("trajectories")
        rows.append({"config": config, "voll_model": model, "seed": seed, **r})
    if verbose:
        print(f"  {config}: scored {len(rows)} schedules")
    return pd.DataFrame(rows)


def compare(df):
    """Dynamic-cost comparison between models, paired by (config, seed, date)."""
    key = ["config", "seed", "date"]
    wide = df.pivot_table(index=key, columns="voll_model",
                          values=["cost", "unserved_MWh", "degC_hours",
                                  "bus_hours_shed", "buses_affected",
                                  "max_bus_consecutive_hours"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()


def main(experiments, configs=CONFIGS, models=MODELS):
    experiments = pathlib.Path(experiments)
    df = pd.concat([evaluate_config(experiments, c, models) for c in configs],
                   ignore_index=True)
    out = experiments / "evaluated.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out}\n")

    print("TRUE DYNAMIC-VOLL COST OF EACH POLICY'S SCHEDULE")
    print(f"{'config':>12} {'model':>10} {'days':>6} {'cost $M':>10} "
          f"{'MWh':>10} {'degC-h':>9} {'bus-hrs':>8}")
    for (c, m), g in df.groupby(["config", "voll_model"]):
        print(f"{c:>12} {m:>10} {len(g):6d} {g['cost'].sum() / 1e6:10,.1f} "
              f"{g['unserved_MWh'].sum():10,.0f} {g['degC_hours'].sum():9,.0f} "
              f"{g['bus_hours_shed'].sum():8,.0f}")

    if set(models) >= {"dynamic", "exogenous"}:
        print("\nSAVING FROM DYNAMIC vs EXOGENOUS (same days, same weather)")
        w = compare(df)
        for c, g in w.groupby("config"):
            d, e = g["cost_dynamic"].sum(), g["cost_exogenous"].sum()
            dh_d, dh_e = g["degC_hours_dynamic"].sum(), g["degC_hours_exogenous"].sum()
            mwh_d, mwh_e = g["unserved_MWh_dynamic"].sum(), g["unserved_MWh_exogenous"].sum()
            print(f"  {c}: {len(g)} days")
            print(f"    cost        ${e / 1e6:,.1f}M -> ${d / 1e6:,.1f}M   "
                  f"{100 * (d / e - 1):+.2f}%  (saving ${(e - d) / 1e6:,.1f}M)")
            print(f"    discomfort  {dh_e:,.0f} -> {dh_d:,.0f} degC-h   "
                  f"{100 * (dh_d / dh_e - 1):+.1f}%")
            print(f"    unserved    {mwh_e:,.0f} -> {mwh_d:,.0f} MWh      "
                  f"{100 * (mwh_d / mwh_e - 1):+.1f}%")
            better = int((g["cost_dynamic"] < g["cost_exogenous"] - 1e-6).sum())
            worse = int((g["cost_dynamic"] > g["cost_exogenous"] + 1e-6).sum())
            print(f"    dynamic cheaper on {better} days, dearer on {worse}, "
                  f"tied on {len(g) - better - worse}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--experiments",
                   default=str(ROOT / "results" / "experiments" / "voll-ladder"))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    a = p.parse_args()
    main(a.experiments, a.configs, a.models)
