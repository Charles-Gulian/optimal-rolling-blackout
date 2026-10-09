"""
Run the reliability simulation over many (config, seed, year) tasks in parallel.

Each task is one config, one outage scenario seed, one weather year -- fully
independent of every other, so there is no coordination beyond handing out
work. Tasks write their own results; analysis/collect_results.py concatenates
afterwards. That makes the whole run resumable: if a task dies, rerun it and
nothing else is affected.

    python analysis/run_parallel.py --seeds 0 1 2 3 4 --n-workers 6

Why processes and not threads: CVXPY holds the GIL around every solver call,
so threads serialise regardless of what Gurobi does in C.

Why Threads=1 per worker: Gurobi defaults to one thread per core, so N workers
each grabbing N threads fight each other. Measured 0.159 s/day solo against
0.222 s/day with only 3 workers on 8 cores.

Timezone: a task's "year" is the set of local days whose window starts in that
calendar year. The last local day of each year needs hours from the next
year's profiles, so it drops out -- 364 or 365 days per task, not 365/366.
"""

import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))

from run_simulation import aggregate, lp_screen, milp_days, stamp  # noqa: E402
from system import UNSERVED_TOL, System  # noqa: E402

CONFIGS = ("RTS-ORB", "RTS-ORB-HR")
SEEDS = (0, 1, 2, 3, 4)
YEARS = tuple(range(2001, 2021))

# One System per config, built on first use and reused for every later task in
# this worker. Under spawn each worker re-imports this module, so the cache
# starts empty per process and is never shared.
_SYSTEMS = {}


def _get_system(config):
    if config not in _SYSTEMS:
        system = System(ROOT / "data", ROOT / "system-config", config)
        system.solver_options = {"Threads": 1}
        _SYSTEMS[config] = system
    return _SYSTEMS[config]


def task_dir(out_root, config, seed, year):
    return pathlib.Path(out_root) / config / f"seed{seed:02d}" / f"year{year}"


def run_task(config, seed, year, out_root, force=False):
    """One (config, seed, year). Returns a short status tuple for the parent."""
    out_dir = task_dir(out_root, config, seed, year)
    marker = out_dir / "reliability_by_year.csv"
    if marker.exists() and not force:
        return (config, seed, year, "skip", 0.0, -1)

    t0 = time.time()
    system = _get_system(config)
    # read_timeseries reads self.scenario_seed at call time, so flipping it
    # here is enough -- the compiled problem is untouched and reused.
    system.scenario_seed = seed

    daily_lp = lp_screen(system, years=[year], verbose=False)
    flagged = daily_lp.loc[daily_lp["unserved_MWh"] > UNSERVED_TOL, "window_start"]
    daily_milp = milp_days(system, flagged, out_dir, verbose=False)
    rel, costs = aggregate(daily_lp, daily_milp, [year])
    stamp((daily_lp, daily_milp, rel, costs), config, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    daily_lp.to_csv(out_dir / "daily_summary.csv")
    daily_milp.to_csv(out_dir / "unserved_energy_days.csv")
    costs.to_csv(out_dir / "costs_by_year.csv", index=False)
    # Written last: it is the marker that says this task finished.
    rel.to_csv(marker, index=False)

    return (config, seed, year, "ok", time.time() - t0, len(flagged))


def main(configs=CONFIGS, seeds=SEEDS, years=YEARS, n_workers=4,
         out_root=None, force=False):
    out_root = pathlib.Path(out_root or ROOT / "results")
    tasks = [(c, s, y) for c in configs for s in seeds for y in years]
    print(f"{len(tasks)} tasks ({len(configs)} configs x {len(seeds)} seeds "
          f"x {len(years)} years) on {n_workers} workers -> {out_root}\n", flush=True)

    t0, done, failed, ue_days = time.time(), 0, [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_task, c, s, y, str(out_root), force): (c, s, y)
                   for c, s, y in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            done += 1
            try:
                config, seed, year, status, secs, flagged = fut.result()
            except Exception as exc:                       # noqa: BLE001
                failed.append((key, repr(exc)))
                print(f"  [{done}/{len(tasks)}] FAILED {key}: {exc!r}", flush=True)
                continue
            if status == "ok":
                ue_days += flagged
                print(f"  [{done}/{len(tasks)}] {config} seed{seed:02d} {year}: "
                      f"{flagged:3d} blackout days  [{secs:.0f}s]", flush=True)
            else:
                print(f"  [{done}/{len(tasks)}] {config} seed{seed:02d} {year}: "
                      f"already done, skipped", flush=True)

    mins = (time.time() - t0) / 60
    print(f"\n{done - len(failed)}/{len(tasks)} tasks ok, {ue_days} blackout days "
          f"total, {mins:.1f} min wall clock")
    if failed:
        print(f"{len(failed)} FAILED -- rerun to pick them up:")
        for key, exc in failed:
            print(f"  {key}: {exc}")
    return failed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--out-root", default=None)
    p.add_argument("--force", action="store_true",
                   help="redo tasks that already have results")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    failed = main(a.configs, a.seeds, a.years, a.n_workers, a.out_root, a.force)
    sys.exit(1 if failed else 0)
