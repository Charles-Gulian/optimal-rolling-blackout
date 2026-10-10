"""
Re-solve the catalogued blackout days under a given VOLL model.

The 100-year screen produced a catalog of days that have unserved energy
(results/<run>/<config>/unserved_energy_days.csv). Which days those are does
not depend on the VOLL model -- shedding always costs more than it saves -- so
the experiments need no screen. This just re-solves the known days.

    python analysis/run_experiments.py --voll-models dynamic exogenous \\
        --catalog results/orb-100yr --n-workers 20

One task is (config, voll_model, seed, year): every catalogued day in that
group, solved with one System built once. Tasks are independent and resumable,
so a failed or timed-out task is simply rerun.

This script SOLVES and saves schedules. It does not score them -- evaluation
under the true dynamic cost is a separate pass (analysis/evaluate.py), because
scoring a non-dynamic schedule needs a dynamic-mode System and carrying two
Systems per worker would double the memory for work that is pure arithmetic on
saved output.

Per day it records the achieved MIP gap and Gurobi status alongside the
result. That matters: at a 1% target some days do not converge, and a policy
comparison on a day whose own solve error exceeds the effect being measured is
not evidence. Those days have to be identifiable afterwards, not averaged in.
"""

import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cvxpy as cp
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from system import System  # noqa: E402

CONFIGS = ("RTS-ORB", "RTS-ORB-HR")
VOLL_MODELS = ("dynamic", "exogenous")
MIP_GAP = 1e-2
MIP_TIME_LIMIT = 1800.0

# One System per (config, voll_model), built on first use and reused for every
# later task in this worker. Under spawn each worker re-imports this module, so
# the cache starts empty per process and is never shared.
_SYSTEMS = {}


def _get_system(config, voll_model):
    key = (config, voll_model)
    if key not in _SYSTEMS:
        system = System(ROOT / "data", ROOT / "system-config", config,
                        voll_model=voll_model)
        system.solver_options = {"Threads": 1}
        system.enable_blackouts()      # needed by every model: z prices the outage
        system.write_opf()
        _SYSTEMS[key] = system
    return _SYSTEMS[key]


def task_dir(out_root, config, voll_model, seed, year):
    return (pathlib.Path(out_root) / config / voll_model
            / f"seed{seed:02d}" / f"year{year}")


def run_task(config, voll_model, seed, year, window_starts, out_root, force=False):
    """Solve every catalogued day in one (config, model, seed, year) group."""
    out_dir = task_dir(out_root, config, voll_model, seed, year)
    marker = out_dir / "days.csv"
    if marker.exists() and not force:
        return (config, voll_model, seed, year, "skip", 0.0, 0, 0)

    t_task = time.time()
    system = _get_system(config, voll_model)
    system.scenario_seed = seed
    system.read_timeseries(year)

    outage_dir = out_dir / "outages"
    outage_dir.mkdir(parents=True, exist_ok=True)

    rows, n_bad = [], 0
    for ws in window_starts:
        for obj in system.components:
            obj.update_timeseries_parameters(ws)
        t0 = time.time()
        try:
            system.prob.solve(solver=cp.GUROBI, MIPGap=MIP_GAP,
                              TimeLimit=MIP_TIME_LIMIT, **system.solver_options)
            m = system.prob.solver_stats.extra_stats
            gap, status, nodes = float(m.MIPGap), int(m.Status), float(m.NodeCount)
        except Exception as exc:                            # noqa: BLE001
            rows.append({"window_start": ws, "error": repr(exc)[:200],
                         "solve_s": time.time() - t0})
            n_bad += 1
            continue

        system.opf_date = ws
        summary = system.solve_summary
        schedule = system.blackout_schedule
        summary.update(config=config, seed=seed, voll_model=voll_model,
                       mip_gap=gap, gurobi_status=status, nodes=nodes,
                       solve_s=time.time() - t0,
                       bus_hours_shed=int(schedule.values.sum()))
        rows.append(summary)
        # Gurobi status 2 is OPTIMAL (tolerance met); anything else did not.
        if status != 2:
            n_bad += 1
        schedule.to_csv(outage_dir / f"{summary['date'].date()}.csv")

    df = pd.DataFrame(rows)
    df.to_csv(marker, index=False)   # written last: the resume marker
    return (config, voll_model, seed, year, "ok", time.time() - t_task,
            len(window_starts), n_bad)


def build_tasks(catalog, configs, voll_models, seeds=None, years=None):
    """Group catalogued days into (config, model, seed, year) tasks.

    `seeds` and `years` subset the catalog, for smoke tests and for splitting
    a large run across several jobs. None means take everything in it.
    """
    tasks = []
    for config in configs:
        path = pathlib.Path(catalog) / config / "unserved_energy_days.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} -- run the screen and collect first")
        ue = pd.read_csv(path, parse_dates=["window_start"])
        if seeds is not None:
            ue = ue[ue.seed.isin(seeds)]
        if years is not None:
            ue = ue[ue.window_start.dt.year.isin(years)]
        for model in voll_models:
            for (seed, year), grp in ue.groupby([ue.seed,
                                                 ue.window_start.dt.year]):
                tasks.append((config, model, int(seed), int(year),
                              sorted(grp.window_start)))
    return tasks


def main(catalog, configs=CONFIGS, voll_models=VOLL_MODELS, n_workers=4,
         out_root=None, force=False, seeds=None, years=None):
    out_root = pathlib.Path(out_root or ROOT / "results" / "experiments")
    tasks = build_tasks(catalog, configs, voll_models, seeds, years)
    n_days = sum(len(t[4]) for t in tasks)
    print(f"{len(tasks)} tasks, {n_days} day-solves "
          f"({len(configs)} configs x {len(voll_models)} models) "
          f"on {n_workers} workers\n  catalog {catalog}\n  -> {out_root}\n", flush=True)

    t0, done, failed, bad = time.time(), 0, [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_task, c, m, s, y, w, str(out_root), force):
                   (c, m, s, y) for c, m, s, y, w in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            done += 1
            try:
                c, m, s, y, status, secs, n, nb = fut.result()
            except Exception as exc:                        # noqa: BLE001
                failed.append((key, repr(exc)))
                print(f"  [{done}/{len(tasks)}] FAILED {key}: {exc!r}", flush=True)
                continue
            bad += nb
            if status == "ok":
                flag = f"  ({nb} did not reach {MIP_GAP:g})" if nb else ""
                print(f"  [{done}/{len(tasks)}] {c} {m} seed{s:02d} {y}: "
                      f"{n} days [{secs:.0f}s]{flag}", flush=True)
            else:
                print(f"  [{done}/{len(tasks)}] {c} {m} seed{s:02d} {y}: skipped",
                      flush=True)

    print(f"\n{done - len(failed)}/{len(tasks)} tasks ok, {n_days} day-solves, "
          f"{bad} did not reach the {MIP_GAP:g} tolerance, "
          f"{(time.time() - t0) / 60:.1f} min wall clock")
    if failed:
        print(f"{len(failed)} FAILED -- rerun to pick them up:")
        for key, exc in failed:
            print(f"  {key}: {exc}")
    return failed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--catalog", default=str(ROOT / "results" / "orb-100yr"),
                   help="directory holding <config>/unserved_energy_days.csv")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--voll-models", nargs="+", default=list(VOLL_MODELS))
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--out-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="subset the catalog; default is every seed in it")
    p.add_argument("--years", nargs="+", type=int, default=None,
                   help="subset the catalog; default is every year in it")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    failed = main(a.catalog, a.configs, a.voll_models, a.n_workers,
                  a.out_root, a.force, a.seeds, a.years)
    sys.exit(1 if failed else 0)
