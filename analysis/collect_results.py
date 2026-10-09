"""
Concatenate the per-task results written by run_parallel.py.

Each task writes its own small tables under
results/<config>/seed<NN>/year<YYYY>/. This stacks them per config:

    results/<config>/unserved_energy_days.csv   the blackout-day catalog
    results/<config>/reliability_by_year.csv    one row per seed-year
    results/<config>/costs_by_year.csv          one row per seed-year

    python analysis/collect_results.py

daily_summary.csv is deliberately NOT concatenated: it is ~364 rows per task,
so 73k rows across a 100-year study, and nothing downstream reads it. It stays
in the per-task directories for diagnostics.

Reliability and cost rows are stacked rather than recomputed. aggregate() in
run_simulation.py loops year by year with no cross-year state, so a task's
single-row output is identical to what a central pass over everything would
produce -- there is no re-derivation to get wrong.

Also reports which (config, seed, year) tasks are missing, so an incomplete
run is obvious rather than silently producing short tables.
"""

import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from run_parallel import CONFIGS, SEEDS, YEARS, task_dir  # noqa: E402

STACKED = ("unserved_energy_days.csv", "reliability_by_year.csv",
           "costs_by_year.csv")


def collect_config(out_root, config, seeds, years, verbose=True):
    """Stack one config's per-task tables. Returns (tables, missing)."""
    missing, found = [], []
    for seed in seeds:
        for year in years:
            d = task_dir(out_root, config, seed, year)
            if (d / "reliability_by_year.csv").exists():
                found.append((seed, year, d))
            else:
                missing.append((seed, year))

    tables = {}
    for name in STACKED:
        frames = [pd.read_csv(d / name) for _, _, d in found
                  if (d / name).exists() and (d / name).stat().st_size > 0]
        frames = [f for f in frames if len(f)]
        if not frames:
            tables[name] = pd.DataFrame()
            continue
        df = pd.concat(frames, ignore_index=True)
        sort_cols = [c for c in ("config", "seed", "date", "year") if c in df.columns]
        tables[name] = df.sort_values(sort_cols).reset_index(drop=True)

    if verbose:
        print(f"{config}: {len(found)}/{len(found) + len(missing)} tasks")
        if missing:
            print(f"  MISSING {len(missing)}: " +
                  ", ".join(f"seed{s:02d}/{y}" for s, y in missing[:10]) +
                  (" ..." if len(missing) > 10 else ""))
    return tables, missing


def summarise(config, tables):
    rel = tables["reliability_by_year.csv"]
    ue = tables["unserved_energy_days.csv"]
    if not len(rel):
        print(f"  {config}: nothing to summarise")
        return
    n_years = len(rel)
    print(f"  blackout days      {len(ue):6d}   ({len(ue) / n_years:.2f}/yr over "
          f"{n_years} simulated years)")
    print(f"  EUE                {rel.EUE_MWh.sum() / n_years:>9,.0f} MWh/yr")
    print(f"  NEUE               {1e6 * rel.EUE_MWh.sum() / rel.load_MWh.sum():>9.1f} ppm")
    if "max_bus_consecutive_hours" in ue.columns and len(ue):
        h = ue.max_bus_consecutive_hours
        print(f"  longest outage at one bus: max {int(h.max())} h; "
              f"days with >=3 h: {int((h >= 3).sum())} ({100 * (h >= 3).mean():.0f}%)")
    # Divide each seed by ITS OWN year count -- seeds can have different
    # numbers of completed tasks in a partial run.
    per_seed = rel.groupby("seed").agg(days=("LOLE_days", "sum"),
                                       n=("year", "count"))
    print("  LOLE by seed (days/yr): " +
          ", ".join(f"{i}={r.days / r.n:.2f}" for i, r in per_seed.iterrows()))


def main(out_root=None, configs=CONFIGS, seeds=SEEDS, years=YEARS):
    out_root = pathlib.Path(out_root or ROOT / "results")
    any_missing = []
    for config in configs:
        tables, missing = collect_config(out_root, config, seeds, years)
        any_missing += [(config, s, y) for s, y in missing]
        dest = out_root / config
        dest.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            if len(df):
                df.to_csv(dest / name, index=False)
        summarise(config, tables)
        print(f"  wrote {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest}/\n")

    if any_missing:
        print(f"{len(any_missing)} tasks missing -- tables above are incomplete. "
              f"Rerun run_parallel.py to fill them in.")
    return any_missing


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out-root", default=None)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    a = p.parse_args()
    main(a.out_root, a.configs, a.seeds, a.years)
