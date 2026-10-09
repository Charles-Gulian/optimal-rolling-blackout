# Plan: 100-year simulation of ORB and ORB-HR

Goal: a catalog of test days (days with unserved energy) for the dynamic VOLL
experiments, over 100 system-years per config with forced outages.

---

## 0. Two decisions needed before any code

### 0.1 Window-start fix — BLOCKING

`analysis/run_simulation.py` is currently broken and will produce invalid
results. Carried over unanswered from 2026-09-30; see
`docs/session-review-2026-09-30.md` for the full write-up.

Short version: `solve_summary["date"]` is both the human-readable label and the
key pass 2 uses to re-enter the solve. Re-windowing to local midnight changed
it to a local calendar date, so the MILP pass re-solves `00:00 UTC` (17:00
local, the old window) on days the LP screen flagged using local-day windows.
Symptom: years with `LOLE_days > 0` and `EUE = 0`.

- **Option A** — add `window_start` (UTC) beside `date` (local); pass 2 keys off
  `window_start`. ~3 lines. Keeps outputs as they are.
- **Option B** — `date` reverts to the UTC window start; convert to local once
  in `aggregate()` before writing. Removes the failure mode rather than
  documenting it.

Recommended: **B**. Nothing else in this plan can be validated until this is
settled.

### 0.2 What "100 years" means

We hold 20 weather years (2001-2020) and no more. So 100 system-years must be

    5 scenario_seeds x 20 weather years = 100 config-seed-years

**This samples outage variance 5x but weather variance only 1x.** Each of the
5 replications sees the same 20 heat waves. Given that LOLE is driven by
coincidences of high load and unit outages, and that we measured per-year LOLE
correlation of only +0.22 between two outage draws, 5 seeds will tighten the
outage component materially. It will do nothing for weather uncertainty, and
the 20 observed heat waves remain the binding limit on what we can say about
tail risk.

Confirm this is the intent before proceeding. The alternative — bootstrapping
or resampling weather years — is a different and much more debatable design.

---

## 1. To-do triage

Split by whether the 100-year run is wrong without it.

### Must fix before the run

| # | Item | Why it blocks |
|---|------|---------------|
| 1 | Window-start fix (0.1) | Results are invalid |
| 2 | Re-validate the LP screen on a handful of days | The screen is the whole basis for cost; it has not been re-checked since re-windowing |
| 3 | Settle the results directory / naming schema | 200 runs land somewhere; `tag` is ad hoc and was invented mid-sweep |
| 4 | Delete the two invalid `-localday` result dirs | They will otherwise be mistaken for good data |

### Should fix before the run (cheap, and touches the run path)

| # | Item | Note |
|---|------|------|
| 5 | Document `StorageResource(duration=4.0)` | Load-bearing, overrides RTS-GMLC's 3 h, still undocumented |
| 6 | Delete `bisect_rung()` in `tune_orb_hr.py` | Written, never called |
| 7 | Delete or wire `CoolingLoad.THERMAL_COLUMNS` | Declared, never used, duplicates `has_thermal_model` |
| 8 | Check whether `System.simulate_year()` is dead | Superseded by `run_simulation.py`; if live, it also needs the window fix |
| 9 | Update `outages.py` docstring | Still describes seeding by sorted position; seeding moved to `Resource ID` |

### Parked for the dynamic VOLL experiments (not this run)

- **MIP gap trace on the McCormick formulation.** The dynamic-VOLL MILP is
  1-2 orders of magnitude slower than static: 2003-07-11 solved in 18 s static
  and took minutes with the discomfort terms, with RSS climbing 233 -> 532 MB
  as the tree grew. Hypothesis to test: the solver reaches a <1% gap quickly
  and then spends the bulk of the time proving optimality on the remaining
  tree. If so, raising MIPGap makes the experimental runs cheap. Get the trace
  off the Gurobi log in a single solve. Irrelevant to the 100-year screen,
  which is static VOLL only.
- **Cold load pickup.** Restoring a feeder draws well above its pre-outage
  load, because every AC compressor and motor restarts at once and thermostats
  have drifted off setpoint. It limits how fast blocks can be cycled, and it
  is the same physics the thermal state already tracks -- the hotter a bus got
  while dark, the bigger its restoration surge. The model currently treats
  restoration as free. A real omission worth considering once the basic
  experiments run.

### Can wait until after the run

- `scripts/thermal.py` is orphaned while `CoolingLoad` reimplements
  `simulate()` and `state_bound()`. Real duplication, but not on the run path.
- Comment-volume pass (`outages.py` 53%, `make_load_csv.py` 49%, `load.py` 44%,
  `make_system_config.py` 44%).
- `Resource.pmax` type switch (float on base, `cp.Parameter` after
  `create_parameters()`), with `nameplate_capacity` as the workaround.
- Extracting the results-reporting block out of `system.py` (454 lines).
- `design_temperature()` doing 1020 redundant CSV reads.
- Tests for the properties verified ad hoc: LP screen equivalence, McCormick
  exactness, CRN across configs.

### Deliberately NOT in scope for this run

**Dynamic VOLL stays off.** The set of days with unserved energy is identical
under static and dynamic VOLL: if a zero-UE dispatch exists it costs only
generation, and any shedding costs at least VOLL per MWh, which exceeds any
generation saving. Dynamic VOLL with `rho >= 0` only raises the cost of
shedding, so it cannot create UE where there was none. Generating the test-day
catalog is therefore a static-VOLL job, and the dynamic VOLL experiments run
afterwards on just those days — which is the whole point of building a catalog.

---

## 2. Run design

### Work unit

One `(config, scenario_seed, weather_year)` triple, fully independent of every
other. 2 configs x 5 seeds x 20 years = **200 work units**.

Each unit: LP screen over all ~364 local days, then MILP on the flagged days.

### Why this parallelises cleanly

`read_timeseries()` reads `self.scenario_seed` at call time
(`scripts/system.py:223`), so a worker can hold one compiled `System` and
switch seeds between tasks by assigning the attribute. No code change needed,
and no recompilation per task.

The one wrinkle: LP and MILP modes need separately compiled problems
(`disable_blackouts()` / `enable_blackouts()` then `write_opf()`). So a worker
should hold both for its config. Partitioning by config — one SLURM job per
config — keeps that to two problems per worker instead of four.

### Parallel structure

Follow the pattern in `multi-zonal-elcc/scripts/compute_shortfall_parallel.py`:

- `ProcessPoolExecutor`, not threads — CVXPY holds the GIL around solver calls,
  so threading serialises regardless of what Gurobi does in C.
- `initializer=_worker_init` builds the `System` and both compiled problems
  once per process, stored in module globals.
- Tasks carry only `(seed, year)`; workers return small summary frames plus
  write their own per-day outage CSVs. Keeps IPC negligible.

New file: `analysis/run_simulation_parallel.py`, wrapping the existing
`lp_screen` / `milp_days` / `aggregate` rather than reimplementing them.

### Output schema

Proposed, to be reviewed:

```
results/<config>/seed<NN>/
    reliability_by_year.csv
    costs_by_year.csv
    unserved_energy_days.csv
    daily_summary.csv
    outages/<local-date>.csv
results/<config>/
    test_days.csv          <- the deliverable
    reliability_by_seed.csv
```

`test_days.csv` is what the dynamic VOLL work consumes. One row per test day:

```
config, scenario_seed, local_date, window_start_utc, event_id, day_in_event,
event_length_days, unserved_MWh, unserved_hours, buses_affected, bus_hours_shed
```

Two things worth noting about this schema:

- **Outage realisations need not be stored.** They are a deterministic function
  of `Resource ID`, year and `scenario_seed`, so `(config, seed, date)` is
  sufficient to reproduce any test day exactly. The catalog stays small.
- **`event_id` matters more than it looks.** 88% of unserved energy sat in
  multi-day events, and thermal state carrying across days within an event is
  exactly the mechanism the dynamic VOLL model is about. Isolated days would
  throw that structure away, so events must be first-class in the catalog.

### Expected yield

At the last valid measurement (5.70 LOLE days/yr for ORB, 5.75 for ORB-HR),
100 years gives roughly **550-600 test days per config**. That number will move
once the window fix lands — re-windowing repartitions hours into days, so day
counts change even though the underlying hours do not. Treat 550-600 as an
order-of-magnitude expectation, not a target.

---

## 3. Cost and where to run it

### Measured timings (this Mac, single process)

| | LP screen | MILP | per config-seed-year |
|---|---|---|---|
| ORB | ~60 s/yr | ~1.1 s/day x ~6 days | ~66 s |
| ORB-HR | ~100 s/yr | ~1.1 s/day x ~6 days | ~107 s |

MILP is negligible: 114 days totalled 2.0 minutes.

### Serial totals

- ORB: 100 x 66 s = **1.8 h**
- ORB-HR: 100 x 107 s = **3.0 h**
- Both: **~4.8 h serial on this Mac**

### An honest note on SAVIO

At 4.8 h serial this job does not need a cluster. On 6 local workers it is
roughly 1 hour of wall clock. SAVIO's savio2 nodes are Xeon E5-2670 v3 (2014),
likely 1.5-2.5x slower per core than this Mac, so ~10 h serial there, or ~30
min on 20 cores.

The case for setting up SAVIO now is **not** this run — it is what comes next:
the dynamic VOLL sweeps re-solve every test day across multiple `rho` values
and both configs, with the McCormick terms making each MILP more expensive, and
the 500-year runs discussed earlier. Those genuinely need it.

Recommended sequencing: build the driver portable (a `--n-workers` flag, no
SLURM assumptions), validate it locally where the feedback loop is seconds, and
submit the same script to SAVIO. That way SLURM debugging is not entangled with
driver debugging.

### Local memory constraint

This Mac has 8 GB. Each worker holds its own `System` plus two compiled
problems plus one year of profiles. The ELCC code put a `System` at ~40 MB, but
ours has 178 resources and binary blackout variables, so the compiled MILP is
the unknown. **Measure one worker's RSS before choosing `--n-workers`** — 8 GB
may cap us at 3-4 locally, which would make local validation a partial run
rather than the full thing.

---

## 4. SAVIO plan

Mirror the two-script pattern from `multi-zonal-elcc`:

- `submit_simulation.sh` — thin wrapper: makes scratch dirs, `sbatch` with
  `--job-name` / `--output` / `--export=ALL,...`, forwards extra args.
- `run_simulation.sh` — the SLURM job script with the `#SBATCH` directives.

Starting point from the ELCC job script, with the deltas I would change:

```
#SBATCH --partition=savio2
#SBATCH --account=fc_power
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G          # up from 8G: 20 workers x compiled MILP, unmeasured
#SBATCH --time=4:00:00     # down from 24h: est. 30-60 min, 4h is ample headroom
```

One job per config, so two submissions. Scratch at
`/global/scratch/users/charlesgulian/optimal-rolling-blackout`.

### Things I cannot verify from here and that you should check first

1. **Gurobi licensing on compute nodes.** We use a WLS academic licence, which
   checks out over the internet. SAVIO compute nodes are typically walled off
   from outbound internet. The ELCC project apparently ran Gurobi there with 20
   workers, so there is presumably a working arrangement — but it needs
   confirming, and so does whether the licence permits 20 concurrent sessions.
   Fallback is HiGHS through CVXPY: fine for the LP screen, slower on the MILP,
   and it would require re-verifying the screen property under a different
   solver.
2. **Conda environment on SAVIO.** The ELCC script does
   `conda activate multi-zonal-elcc`. We need an equivalent carrying cvxpy,
   gurobipy, pandas, numpy. `environment.yml` exists in the repo root.
3. **Data staging.** `data/` holds 20 years of per-bus profiles plus the NSRDB
   weather inputs for 73 buses x 26 years. Needs measuring and copying to
   scratch; this may be the slowest part of the setup.
4. **Node core count** on the chosen partition, so `--cpus-per-task` and
   `--n-workers` agree.

---

## 5. Proposed order of work

1. Decide 0.1 (window fix A or B) and 0.2 (what 100 years means).
2. Apply the window fix; delete the two invalid `-localday` result dirs.
3. Re-validate the LP screen on ~8 days (4 with UE, 4 clean), as was done
   before — this is the one correctness property everything else rests on.
4. Short single-config, single-seed, 20-year run locally to confirm the fix
   produces internally consistent output (no `LOLE > 0, EUE = 0` rows).
5. Settle the output schema and `test_days.csv` columns.
6. Write `analysis/run_simulation_parallel.py`; measure worker RSS.
7. Local validation run: 1 config x 2 seeds x 20 years.
8. Items 5-9 from the "should fix" list while that runs.
9. SAVIO: environment, data staging, licence check, then the two full jobs.
10. Build `test_days.csv`; sanity-check the event structure against the
    per-seed reliability tables.

Steps 1-5 are a short session. Step 6 is the only substantial new code. Steps
9-10 depend entirely on how the licence question resolves.
