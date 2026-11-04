"""
Parallel re-solve of the loss-of-load days found by run_simulation.py.

    python scripts-gurobi/run_ols_parallel.py \
        --results results/RTS-ORB-HR/simulation_results_100yrs.csv \
        --configs RTS-ORB RTS-ORB-HR --voll static dynamic \
        --beta 12000 -12000 --fair none 1 \
        --n-workers 20 --out-root results/ols/run1

One task is (config, voll, beta, fair, lex, scenario_seed): every loss-of-load
day for that seed, solved on one System built once per worker. Tasks are
independent and resumable -- rerunning skips any task whose results.csv
already exists.

Each day's total unserved energy is capped at the static LP's value from the
screen, which makes the dynamic solve a strict refinement of the static one and
removes the incentive to shed early to discount later shedding. --no-lex drops
the cap. With --fair the cap can make a day infeasible: that is reported as
gurobi_status 3, not as zero unserved energy.

Results go to <out-root>/<config>/<variant>/seed<NNN>/, where <variant> tags
the VoLL and fairness settings, and every row records beta, gamma and fair.

Saves load_shed and unserved_energy per day, so the dynamic-VoLL cost of any
schedule can be scored afterwards without re-solving.
"""

import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import gurobipy as gp
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from load import VoLL  # noqa: E402
from system import System, UNSERVED_TOL  # noqa: E402

YEAR = 2020

# One System per (config, voll, fairness on/off, lex on/off), built on first use
# and reused by every later task in this worker. beta and the fairness tolerance
# are set per task on the cached System, so they do not key the cache: beta is an
# objective coefficient and the tolerance is one coefficient per cap row. What
# does key the cache is whether the fairness and unserved energy cap rows exist.
# Each process re-imports the module, so nothing is shared.
_SYSTEMS = {}


def _get_system(config, voll, fair, lex, threads, mip_gap, time_limit):
    key = (config, voll, fair is not None, lex)
    if key not in _SYSTEMS:
        s = System(ROOT / "data", ROOT / "system-config", config,
                   dynamic_voll=(voll == "dynamic"), fair=fair, ue_cap=lex)
        s.solver_options = {"Threads": threads, "MIPGap": mip_gap,
                            "TimeLimit": time_limit}
        s.read_timeseries(YEAR)
        s.write_opf()
        _SYSTEMS[key] = s
    return _SYSTEMS[key]


def variant_label(voll, beta=None, fair=None, lex=False):
    """Directory name for one variant, e.g. "beta=12000_fair=1_lex=True"."""
    tag = f"{beta:g}" if voll == "dynamic" else "static"
    fair = f"{fair:g}" if fair is not None else None
    return f"beta={tag}_fair={fair}_lex={lex}"


def task_dir(out_root, config, variant, seed):
    return pathlib.Path(out_root) / config / variant / f"seed{seed:03d}"


def run_task(config, voll, beta, fair, lex, seed, days, out_root, force, threads,
             mip_gap, time_limit, save_dispatch):
    """Solve every given day for one variant and seed. Returns a status tuple."""
    variant = variant_label(voll, beta, fair, lex)
    out_dir = task_dir(out_root, config, variant, seed)
    marker = out_dir / "results.csv"
    if marker.exists() and not force:
        return (config, variant, seed, "skip", 0.0, 0, 0)

    t_task = time.time()
    s = _get_system(config, voll, fair, lex, threads, mip_gap, time_limit)
    if beta is not None:
        s.set_beta(beta)
    if fair is not None:
        s.set_fair(fair)
    s.scenario_seed = seed
    for r in s.resources.values():
        r.get_outage_profile(YEAR, seed)

    sched_dir = out_dir / "schedules"
    sched_dir.mkdir(parents=True, exist_ok=True)

    keys = {"config": config, "voll": voll, "variant": variant, "beta": beta,
            "gamma": VoLL.GAMMA, "fair": fair, "lex": lex, "scenario_seed": seed}
    rows, n_bad = [], 0
    for date, ue_static in days:
        t0 = time.time()
        if lex:
            s.set_ue_cap(ue_static)
        try:
            s.solve_opf(date)
        except Exception as exc:                                  # noqa: BLE001
            # An infeasible or unsolved day raises on ObjVal. Record the solver
            # status so it cannot be read later as zero unserved energy.
            rows.append({**keys, "date": date, "ue_static": ue_static,
                         "gurobi_status": int(s.model.Status),
                         "error": repr(exc)[:200],
                         "solve_s": time.time() - t0})
            n_bad += 1
            continue

        m = s.model
        status = int(m.Status)
        # MIPGap only exists once the model is a MIP; the static LP has none.
        try:
            gap = float(m.MIPGap)
        except (AttributeError, gp.GurobiError):
            gap = 0.0

        ue, shed = s.unserved_energy_profile, s.load_shed_profile
        row = {**keys, "date": date, "ue_static": ue_static,
               "cost": float(s.total_unserved_energy_costs.getValue()),
               "unserved_energy": float(ue.values.sum()),
               "ue_hours": int((ue.sum(axis=1) > UNSERVED_TOL).sum()),
               "gurobi_status": status, "mip_gap": gap,
               "nodes": float(m.NodeCount), "solve_s": time.time() - t0}
        # Duals exist only while the model stays an LP; a dynamic solve is a MIP.
        if voll != "dynamic":
            row["crit_hours"] = int((s.lmp_profile.max(axis=1) > 1e-2).sum())
            row["event_type"] = s.ue_event_type
        rows.append(row)

        stamp = pd.Timestamp(date).strftime("%Y-%m-%d")
        shed.to_csv(sched_dir / f"shed_{stamp}.csv")
        ue.to_csv(sched_dir / f"ue_{stamp}.csv")
        if save_dispatch:
            s.dispatch_results.to_csv(sched_dir / f"dispatch_{stamp}.csv")

        if status != gp.GRB.OPTIMAL:
            n_bad += 1

    pd.DataFrame(rows).to_csv(marker, index=False)   # written last: resume marker
    return (config, variant, seed, "ok", time.time() - t_task, len(days), n_bad)


def build_tasks(results_csv, configs, volls, betas, fairs, lex, seeds=None):
    """Group the loss-of-load days in `results_csv` into per-seed tasks."""
    df = pd.read_csv(results_csv, parse_dates=["date"])
    df = df[df.total_ue > UNSERVED_TOL]
    if seeds is not None:
        df = df[df.scenario_seed.isin(seeds)]
    tasks = []
    for config in configs:
        for voll in volls:
            # The static objective has no beta, so it solves once per fair.
            for beta in (betas if voll == "dynamic" else [None]):
                for fair in fairs:
                    for seed, grp in df.groupby("scenario_seed"):
                        # Each day carries the static LP's unserved energy,
                        # which is the lexicographic cap for that day.
                        days = sorted(zip(grp.date, grp.total_ue))
                        tasks.append((config, voll, beta, fair, lex,
                                      int(seed), days))
    return tasks


def collect(out_root):
    """Concatenate every task's results.csv into one frame."""
    parts = sorted(pathlib.Path(out_root).glob("*/*/seed*/results.csv"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in parts],
                   ignore_index=True)
    return df.sort_values(["config", "variant", "scenario_seed", "date"])


def main(results_csv, configs, volls, betas, fairs, lex, n_workers, out_root,
         force=False, seeds=None, threads=1, mip_gap=1e-4, time_limit=1800.0,
         save_dispatch=False):
    out_root = pathlib.Path(out_root)
    tasks = build_tasks(results_csv, configs, volls, betas, fairs, lex, seeds)
    n_days = sum(len(t[6]) for t in tasks)
    variants = sorted({variant_label(v, b, f, x) for _, v, b, f, x, _, _ in tasks})
    print(f"{len(tasks)} tasks, {n_days} day-solves "
          f"({len(configs)} configs x {len(variants)} variants) on {n_workers} "
          f"workers, {threads} Gurobi thread(s) each"
          f"\n  variants: {' '.join(variants)}\n  from {results_csv}"
          f"\n  -> {out_root}\n", flush=True)

    t0, done, failed, bad = time.time(), 0, [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_task, c, v, b, f, x, s, d, str(out_root), force,
                             threads, mip_gap, time_limit, save_dispatch):
                   (c, variant_label(v, b, f, x), s)
                   for c, v, b, f, x, s, d in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            done += 1
            try:
                c, v, s, status, secs, n, nb = fut.result()
            except Exception as exc:                              # noqa: BLE001
                failed.append((key, repr(exc)))
                print(f"  [{done}/{len(tasks)}] FAILED {key}: {exc!r}", flush=True)
                continue
            bad += nb
            if status == "ok":
                flag = f"  ({nb} not optimal)" if nb else ""
                print(f"  [{done}/{len(tasks)}] {c} {v} seed{s:03d}: "
                      f"{n} days [{secs:.0f}s]{flag}", flush=True)
            else:
                print(f"  [{done}/{len(tasks)}] {c} {v} seed{s:03d}: skipped",
                      flush=True)

    df = collect(out_root)
    if len(df):
        df.to_csv(out_root / "ols_results.csv", index=False)
    print(f"\n{done - len(failed)}/{len(tasks)} tasks ok, {n_days} day-solves, "
          f"{bad} not optimal, {(time.time() - t0) / 60:.1f} min wall clock")
    print(f"collected {len(df)} rows -> {out_root / 'ols_results.csv'}")
    if failed:
        print(f"{len(failed)} FAILED -- rerun to pick them up:")
        for key, exc in failed:
            print(f"  {key}: {exc}")
    return failed


def _fair(value):
    """--fair argument: a tolerance, or "none" for no fairness constraint."""
    return None if value.lower() == "none" else float(value)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--results", required=True,
                   help="simulation_results_*.csv from run_simulation.py")
    p.add_argument("--configs", nargs="+", default=["RTS-ORB", "RTS-ORB-HR"])
    p.add_argument("--voll", nargs="+", default=["static", "dynamic"],
                   choices=["static", "dynamic"])
    p.add_argument("--beta", nargs="+", type=float, default=[VoLL.BETA],
                   help="VoLL sensitivities to solve; dynamic VoLL only")
    p.add_argument("--fair", nargs="+", type=_fair, default=[None],
                   help='fairness tolerances; "none" for no fairness constraint')
    p.add_argument("--no-lex", dest="lex", action="store_false", default=True,
                   help="drop the cap on each day's total unserved energy")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=1,
                   help="Gurobi threads per worker; workers x threads <= cores")
    p.add_argument("--out-root", default=str(ROOT / "results" / "ols"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--time-limit", type=float, default=1800.0)
    p.add_argument("--save-dispatch", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    failed = main(a.results, a.configs, a.voll, a.beta, a.fair,
                  a.lex, a.n_workers, a.out_root, a.force, a.seeds, a.threads,
                  a.mip_gap, a.time_limit, a.save_dispatch)
    sys.exit(1 if failed else 0)
