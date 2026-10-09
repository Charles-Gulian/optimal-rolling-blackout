# Code review to-do — session of 2026-09-30

Everything touched today, ordered so that each item is reviewable without
depending on a decision from a later one. Data-layer first (defines the
columns), then the model, then the drivers.

Comment density is measured as comment + docstring lines over total lines.
The base repo runs leaner than most of these; anything at 40%+ is a candidate
for cutting.

---

## 0. Blocking decision — resolve before reviewing the drivers

**`window_start` vs `date`** (see "The window-start issue" below). Item 7
(`run_simulation.py`) cannot be finalised until this is chosen, and the choice
also determines whether `solve_summary` in item 5 gains a field. Everything
else can be reviewed independently.

---

## 1. `data-scripts/make_system_config.py` — 218 lines, ~44% comment
**New file today.** Derives RTS-ORB and RTS-ORB-HR from the upstream RTS-GMLC.

Review for:
- The three long block comments (nuclear FOR, Resource ID, ORB-HR rationale)
  carry the *why* for decisions not recoverable from the code. Probably worth
  keeping in substance, but they are verbose.
- `RETIREMENT_LADDER` — the inline cumulative-MW comments will go stale if the
  fleet changes. Consider computing and printing them instead.
- `_add_units` — the `source_types` / `template_uid` fallback chain has three
  branches for what is really two cases (clone from this bus, or clone from a
  named template). May collapse.
- `make_rts_orb_hr(rungs=...)` signature: rung is a config-generation argument,
  which conflates "what the system is" with "what we were sweeping".

## 2. `data-scripts/make_load_csv.py` — 184 lines, ~49% comment
Modified: added the Wang thermal parameter block and per-bus `gbar`.

Review for:
- **Highest comment density of the data scripts.** The Wang parameter block is
  ~25 lines of comment for 5 constants.
- `design_temperature()` re-reads 20 CSVs per bus per call, and is called once
  per bus — 51 buses x 20 years = 1020 file reads. Works, but slow and easily
  cached.
- `thermal_params()` returns a dict that must stay in sync with
  `CoolingLoad.THERMAL_COLUMNS` in `scripts/load.py`. Two sources of truth.
- The `__main__` loop over `PEAK_SCALING` keys — I changed this today so one
  invocation regenerates every config. Confirm that is what you want.

## 3. `scripts/outages.py` — 83 lines, ~53% comment
**New file today**, built from the code you supplied.

Review for:
- Highest comment ratio in the repo. The module docstring explains the
  Cantor-pairing reproducibility argument at length; some of that is now
  obsolete because seeding moved to the explicit `Resource ID` column.
- **The docstring's reproducibility claims need re-checking against the
  Resource ID change** — it still describes seeding by sorted position.
- `forced_outage_profile` has a pure-Python loop over 8760 steps, called
  ~93 x 20 times per run. Vectorisable if sampling ever becomes a bottleneck.

## 4. `scripts/resource.py` — 218 lines, ~20% comment
Modified: forced outages, `nameplate_capacity`, time-varying thermal `pmax`.

Review for:
- Leanest of the modified files; probably least work.
- `Resource.pmax` is a float on the base class but a `cp.Parameter` on
  `ThermalResource` and `VariableResource` after `create_parameters()`. That
  type switch is a trap — `nameplate_capacity` exists only to work around it.
- `StorageResource(duration=4.0)` default is load-bearing (overrides the 3 h in
  RTS-GMLC's storage.csv) but undocumented. **Known gap, worth a comment.**
- `HydroResource.SOURCE_YEAR = 2020` calendar re-mapping.

## 5. `scripts/system.py` — 454 lines, ~30% comment
Modified heavily: forced outages, Resource ID seeding, results reporting,
dynamic VOLL toggle, local-day re-windowing.

Review for:
- **Largest file and the most churn. Review this one when fresh.**
- `LOCAL_UTC_OFFSET` block comment (~10 lines) explaining the re-windowing.
- `available_days()` — now returns UTC timestamps at local midnight. The
  docstring explains two separate coverage cases; check it is not overlong.
- `enable_dynamic_voll()` / `disable_dynamic_voll()` mirror the blackout
  enablers. Both carry the "call before `write_opf()`" constraint implicitly;
  neither enforces it.
- `solve_summary` — the field whose meaning the window-start issue turns on.
- The results-reporting block (`opf_index`, `unserved_energy_profile`,
  `blackout_schedule`, `solve_summary`, `dispatch_results`) has grown
  organically. Candidate for extraction into its own module.
- `simulate_year()` may now be dead code — superseded by
  `analysis/run_simulation.py`. Check and delete if so.

## 6. `scripts/load.py` — 287 lines, ~44% comment
Modified: split into `Load` + `CoolingLoad`, all Wang/dynamic-VOLL machinery.

Review for:
- **Newest design, least settled.** Written today and restructured once.
- `Load.dynamic_voll = False` as a class attribute on the base so `system.py`
  can avoid `isinstance` checks. Deliberate, but it is a subclass concern
  leaking upward — you may prefer the isinstance checks.
- `CoolingLoad.THERMAL_COLUMNS` is declared but never used. Dead; either wire
  it into `_opt`/`has_thermal_model` or delete it.
- `has_thermal_model` hardcodes the six attribute names, duplicating
  `THERMAL_COLUMNS`.
- The constraint comments in `write_constraints` are long (three numbered
  blocks, ~30 lines of comment). The maths is worth recording somewhere; maybe
  in `thermal.py` or a docs note rather than inline.
- `thermal_trajectory()` duplicates the recursion in `thermal.simulate()`.
  **`scripts/thermal.py` is currently unused by anything** — decide whether
  `CoolingLoad` should call into it or whether `thermal.py` should be archived.
- `_state_bound()` likewise duplicates `thermal.state_bound()`.

## 7. `analysis/run_simulation.py` — 181 lines, ~27% comment
Modified: added `tag` and `scenario_seed` to `main()`; timezone docstring.

**Blocked on the window-start decision.**

Review for:
- The two-pass LP-screen/MILP design and its validity argument in the
  docstring.
- The `tag` parameter I added for sweep runs — ad hoc, may not be how you want
  result directories named.
- `aggregate()` mixes LP costs on clean days with MILP costs on UE days.

## 8. `analysis/tune_orb_hr.py` — 160 lines, ~31% comment
**New file today.** Coal-retirement ladder sweep.

Review for:
- Lowest priority — a one-off tuning script, not library code.
- Switches rungs by zeroing `nameplate_capacity` in memory while the committed
  config deletes rows. Documented as equivalent, but it is two mechanisms for
  one concept.
- Requires the on-disk config at rung 0 or it raises. Fragile coupling.
- `bisect_rung()` was written but never used — the parallel sweep was used
  instead. Dead code; delete or keep deliberately.

---

## Cross-cutting

- **Comment volume.** `outages.py` 53%, `make_load_csv.py` 49%, `load.py` 44%,
  `make_system_config.py` 44%. Worth one pass with a consistent rule for what
  earns a comment.
- **`system-config/` and `results/` are untracked.** Both are regenerable, but
  nothing records that. Decide: `.gitignore` entries, or commit the configs.
- **`scripts/thermal.py` is orphaned.** Nothing imports it, yet `CoolingLoad`
  reimplements two of its functions.
- **No tests.** The properties verified ad hoc this session — LP screen
  equivalence, McCormick exactness, CRN across configs — are exactly the kind
  that should be pinned down.

---

# The window-start issue

## What happened

`run_simulation` is two passes. Pass 1 (LP screen) solves every day and indexes
its results by `solve_summary["date"]`. Pass 2 (MILP) takes the flagged index
values and feeds them back in as window starts:

```python
flagged = daily_lp.index[daily_lp["unserved_MWh"] > UNSERVED_TOL]
...
for obj in system.components:
    obj.update_timeseries_parameters(day)     # `day` came from that index
```

This worked while `date` *was* the window start. Re-windowing to local midnight,
I changed `date` to the local calendar date (`2003-07-12 00:00`) for readable
reporting — so pass 2 now re-solves a window starting `2003-07-12 00:00 UTC`,
which is 17:00 local: **the old window**. Pass 1 screens local days; pass 2
re-solves UTC days.

The root cause is that one field is doing two jobs: human-readable label and
machine re-entry key.

## How it shows up

- Years with `LOLE_days > 0` but `EUE = 0` (2004: 3 days, 0 MWh; also 2006,
  2008, 2013, 2014, 2018) — the MILP looked at hours the LP never flagged.
- Outage CSVs stamped `00:00-23:00 UTC` instead of `07:00-06:00`.
- The apparent 44% drop in shed bus-hours after re-windowing is an artifact of
  the two passes disagreeing, **not** a real effect.

Both re-runs from today (`RTS-ORB-scale1.10-localday`,
`RTS-ORB-HR-scale1.10-rung6-localday`) are invalid and should be deleted.
The pre-re-windowing results are internally consistent but carry date labels
shifted by one relative to the local evening they describe.

## Option A — add `window_start`, keep `date` local

`solve_summary` returns both:

```python
"window_start": pd.Timestamp(self.opf_date),                       # UTC, re-entry key
"date": pd.Timestamp(self.opf_date + LOCAL_UTC_OFFSET).normalize() # local, label
```

and `run_simulation` keys pass 2 off `window_start`:

```python
flagged = daily_lp.loc[daily_lp["unserved_MWh"] > UNSERVED_TOL, "window_start"]
```

- **For:** output CSVs carry the local date a reader expects; no mental
  conversion when reading results; `date` stays the natural index.
- **Against:** two time columns in every row, and a future caller can still
  pick the wrong one — the trap is documented rather than removed.
- Size: one line in `solve_summary`, two in `run_simulation`.

## Option B — `date` stays the UTC window start; convert only at the edge

`solve_summary["date"]` reverts to `self.opf_date` (07:00 UTC). Conversion to
local happens once, in whatever writes the final CSVs.

- **For:** one canonical key, so the bug class cannot recur; timestamps stay on
  one clock everywhere inside the model, matching the profiles; presentation
  lives at the boundary.
- **Against:** the intermediate CSVs carry `2003-07-12 07:00`, which reads as
  the wrong day at a glance; every consumer must remember to convert, and the
  daily files are exactly what gets eyeballed during analysis.
- Size: revert one line, add conversion in `run_simulation.main()` before
  writing.

## Recommendation

**Option B**, on the grounds that it removes the failure mode rather than
documenting it, and this failure mode already cost a session. The readability
objection is real but confined to intermediate files, and can be handled by
converting in `aggregate()` so every *written* CSV carries local dates while
everything in memory stays UTC.

That said, Option A is the smaller change and keeps the outputs as they are
today — reasonable if you would rather not touch the reporting layer again.
