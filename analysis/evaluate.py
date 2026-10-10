"""
Evaluate an arbitrary blackout schedule under the true dynamic VOLL cost.

The experiments compare rationing policies: solve a day under one VOLL model,
then ask what that schedule actually costs under the dynamic model, which is
the one we believe. That second step is not an optimisation -- z is fixed, so
x_t is determined and v(x_t) is just a parameter. This module does it.

    system = System(..., voll_model="dynamic")
    system.enable_blackouts(); system.write_opf(); system.read_timeseries(year)
    result = evaluate(system, window_start, schedule)

`schedule` is (T, n_buses) of 0/1 indexed by bus ID -- exactly what
System.blackout_schedule returns and what results/.../outages/<date>.csv
holds. It can come from a solve under any VOLL model, or off disk.

Why a System rather than loading files itself: a dynamic-mode System already
holds demand, phi, gbar, kappa, c, rho and VOLL for every load. Reading
load.csv and the profiles again here would duplicate Load's file handling to
no purpose. The evaluator still never consults what the optimiser decided --
it takes z as given and recomputes everything from the physics.

The dynamics come from scripts/thermal.py, the same implementation the
optimiser's bound uses, so an evaluation can never disagree with the model
about what a schedule does.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import thermal  # noqa: E402
from system import LOCAL_UTC_OFFSET, UNSERVED_TOL  # noqa: E402


def _runs(flags):
    """Lengths of each unbroken run of 1s in a 0/1 sequence."""
    out, run = [], 0
    for f in flags:
        if f:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def evaluate(system, window_start, schedule, update=True):
    """True dynamic-VOLL cost and thermal outcome of a fixed schedule.

    Parameters
    ----------
    system : System in voll_model="dynamic" with timeseries read for the year.
        Only its Parameters are read; nothing is solved and no Variable is
        touched, so a System that has never been solved works fine.
    window_start : UTC timestamp of local midnight -- the solve window start.
    schedule : (T, n_buses) DataFrame of 0/1, columns are bus IDs.
    update : set the day's Parameters first. Pass False if the caller has
        already done so for this window_start.

    Returns
    -------
    dict with per-load detail and system totals. `cost` is the dynamic-VOLL
    cost of this schedule, directly comparable across policies.
    """
    if update:
        for obj in system.components:
            obj.update_timeseries_parameters(window_start)

    cols = {int(c): c for c in schedule.columns}
    per_load, traj = [], {}
    for name, load in system.loads.items():
        col = cols.get(int(load.node_ID))
        z = (np.round(schedule[col].to_numpy()).astype(float)
             if col is not None else np.zeros(len(schedule)))
        d = load.demand.value
        u = d * z                      # Node enforces u = d * z in the model

        if load.needs_thermal:
            x = thermal.simulate(load.phi.value, z, load.a, load.gbar)
            v = load.VOLL + load.rho * (load.kappa * x + load.c * load.phi.value)
            traj[name] = x
        else:
            x = np.zeros_like(d)
            v = np.full_like(d, load.VOLL)

        per_load.append({
            "load": name,
            "bus": int(load.node_ID),
            "load_type": load.load_type,
            "unserved_MWh": float(u.sum()),
            "cost": float(v @ u),
            # Discomfort only accrues while actually unserved.
            "degC_hours": float((x * z).sum()),
            "peak_x": float(x.max()),
            "shed_hours": int(z.sum()),
        })

    df = pd.DataFrame(per_load)
    shed_runs = [n for name in traj
                 for n in _runs(np.round(
                     schedule[cols[int(system.loads[name].node_ID)]].to_numpy()) > 0.5)]
    buses_shed = int((schedule.sum(axis=0) > 0).sum())

    return {
        "window_start": pd.Timestamp(window_start),
        "date": pd.Timestamp(window_start + LOCAL_UTC_OFFSET).normalize(),
        "cost": float(df["cost"].sum()),
        "unserved_MWh": float(df["unserved_MWh"].sum()),
        "degC_hours": float(df["degC_hours"].sum()),
        "peak_x": float(df["peak_x"].max()),
        "bus_hours_shed": int(schedule.values.sum()),
        "buses_affected": buses_shed,
        "max_bus_consecutive_hours": max(shed_runs) if shed_runs else 0,
        **{f"unserved_{t}_MWh": float(g["unserved_MWh"].sum())
           for t, g in df.groupby("load_type")},
        **{f"cost_{t}": float(g["cost"].sum()) for t, g in df.groupby("load_type")},
        "per_load": df,
        "trajectories": traj,
    }


def average_voll(result, load_type="cooling"):
    """Realised VOLL for one load type, $/MWh: cost over unserved energy.

    This is the number a uniform static VOLL should be calibrated to for an
    apples-to-apples comparison -- the price the dynamic model actually paid
    per MWh, not an average of v over all hours (which would sit at VOLL,
    since x is zero whenever the AC is keeping up).
    """
    d = result["per_load"]
    d = d[d["load_type"] == load_type]
    mwh = d["unserved_MWh"].sum()
    return float(d["cost"].sum() / mwh) if mwh > UNSERVED_TOL else float("nan")


def load_schedule(path):
    """Read an outages/<date>.csv written by run_simulation.milp_days."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [int(c) for c in df.columns]
    return df
