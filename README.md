# Optimal Rolling Blackout — RTS-GMLC

A DC-OPF over the RTS-GMLC 73-bus test system in which **load shedding is a binary
per-bus decision** and the **value of lost load rises with accumulated indoor
heat**, so that scheduling a rolling blackout becomes an optimisation over
consumer thermal discomfort rather than over unserved MWh alone.

---

## 1. Quick start

```bash
# Use weather-env. The base anaconda python is broken (pyarrow/bottleneck were
# built against numpy 1.x, so `import pandas` raises _ARRAY_API not found).
PY=/opt/anaconda3/envs/weather-env/bin/python

$PY -m pytest tests/ -q          # 34 tests, ~60 s (3 MILP solves dominate)
```

```python
import sys; sys.path.insert(0, "scripts")
import rts_data, weather, rolling_blackout as rb
from thermal import ThermalParams

system = rts_data.load_rts(include_hydro=True)
wx     = weather.synthetic_weather(system)
tp     = ThermalParams.build(system.n_bus)

model  = rb.RollingBlackout(system, wx, tp, rb.RBConfig(scarcity_mw=100.0, rho=250.0))
result = model.solve("2020-08-26")
print(result.summary())
```

Requires Gurobi (WLS academic licence is configured and working).

---

## 2. Repository map — what is new, what predates this work

### New in this project

| file | role |
|---|---|
| `scripts/rts_data.py` | RTS-GMLC CSVs → DC-OPF matrices, nodal load, availability |
| `scripts/dcopf.py` | plain DC-OPF, continuous shedding — the pre-MILP checkpoint |
| `scripts/weather.py` | `Weather` container + synthetic temperature source |
| `scripts/thermal.py` | Wang 1R-1C building model: `a`, `φ`, `ḡ`, thermostat, state bound |
| `scripts/rolling_blackout.py` | the MILP: binary shedding + dynamic VoLL |
| `tests/` | 34 correctness tests |
| `figures/m1_m3_check.png` | 4-panel sanity figure for the peak day |

### Predates this project (untouched)

| path | what it is |
|---|---|
| `RTS-GMLC-master/` | upstream NREL test system (unmodified) |
| `data-scripts/*.py` | NSRDB / WTK-LED / ERA5 download + profile builders |
| `notebooks/*.ipynb` | earlier reliability simulation and data exploration |
| `scripts/power_system_simulation.py` | object-oriented CVXPY reliability simulator |
| `archive/` | IEEE 9-bus prototype of the dynamic-VoLL idea |

### Why the new code did not extend `power_system_simulation.py`

That module is a working LP reliability simulator and is left intact. Three
reasons the rolling-blackout model is separate:

1. **Binary shedding changes the problem class.** Gurobi returns no duals for a
   MIP, so `simulate_year`'s LMP collection breaks outright.
2. **Speed.** It loops over component objects to build constraints. For 73 buses
   × 120 lines × 24 h plus per-bus thermal state, a vectorised `(n, T)` matrix
   formulation builds far faster and makes the McCormick and thermal blocks
   natural array operations.
3. **It reads profiles from `load-data/profiles/…`, which does not exist in this
   repo.** Those live in `e2e-downscale` (§6).

`archive/dc_opf_cvxpy.py` is the direct ancestor of `rolling_blackout.py` — same
McCormick pattern — but it builds its matrices from a PyPSA network, and **PyPSA
is not installed in `weather-env`**. `rts_data.py` reads the CSVs directly and
drops the dependency.

---

## 3. The model

### State: indoor temperature above setpoint

Derived from Wang et al. (2021) *ERL* **16** 074003, eq. (1)
(PDF lives in the sibling repo: `../optimal-rolling-blackout-archive/papers/`):

```
C dT_in/dt = (T_out − T_in)/R + T_eq/R + Q_HVAC        Wang eq (1)

  ÷C,  τ := RC,  T_eff := T_out + T_eq
  solve exactly over one hour,  a := exp(−1/τ)
  shift  x := T_in − T_set,  φ_t := (1−a)(T_eff,t − T_set)
  thermostat: g = 0 if shed, else min(ḡ, a·x + φ)

x_{t+1} = max(0,  a·x_t + φ_t − ḡ·(1 − z_t))
```

Read it as: the house **keeps** fraction `a` of its excess, **gains** `φ_t`, and
the AC **removes** `ḡ` when powered.

**`z` enters additively, not multiplicatively.** That is the whole reason this is
tractable. Assuming the AC restores setpoint *instantaneously* would give
`x_{t+1} = z_t(a·x_t + φ_t)` — bilinear in `(x, z)`. Rate-limiting the AC is both
more realistic and what keeps the dynamics linear.

### Objective

```
min  c'p
   + Σ_it [ v̄ · d^HVAC_it · z_it  +  ρ · d^HVAC_it · w_it ]
   + Σ_it   v_other · d^other_it · z_it
```

VoLL is `v_t = v̄ + ρ·x_t`, so the cost term is `(v̄ + ρx)(d·z)` — **bilinear in
`(x, z)`**. Note the bilinearity lives in the *objective*, not the dynamics.

### The two relaxations, and what makes each exact

**1. `x_{t+1} ≥ …` instead of `= max(0, …)`.** Exact iff the objective is
nondecreasing in every `x`, which holds iff **`ρ ≥ 0`**. Since `a ≥ 0`, lowering
`x_t` only relaxes the bound on `x_{t+1}`, so the exact recursion is the
componentwise minimum of the feasible set and a nondecreasing objective picks it.
Costs zero binaries; the textbook big-M encoding of `max` would cost one per
bus-hour.

**2. McCormick for `w = x·z`.** Only lower bounds are needed (coefficient
`ρ·d ≥ 0`, minimising):

```
w_it ≥ x_it − xmax_it (1 − z_it)
w_it ≥ 0
```

Exact for binary `z` **provided `x ≤ xmax`**. If that were violated, `w` would go
positive at `z = 0` and the model would charge outage cost to a bus that was
never shed — so the bound is a *correctness* condition, not a tightness knob.

`xmax` is the never-served trajectory `Σ_{k<t} a^{t−1−k} φ_k` (`thermal.state_bound`),
which dominates every schedule by induction and is *attained*, hence tight. The
closed form `max_t(T_eff) − T_set` is also valid — the house can never exceed the
hottest effective outdoor temperature — but runs ~2× loose over a real diurnal
profile, which weakens the LP relaxation.

### Reporting caveat

After a bus's final shed hour, `x` stops entering the objective and the solver may
leave it above its lower bound. Schedule and objective are unaffected, but the raw
trajectory can be inflated. **`solve()` re-simulates `x` from the optimal `z`** with
the exact thermostat and reports that; `RBResult.state_slack` exposes the gap as a
diagnostic. Always use `result.x`, not `result.x_milp`.

---

## 4. Data flow

```
RTS-GMLC CSVs ──► rts_data.load_rts() ──► System
  bus/branch/gen                            B, A, b_line, f_max, ref_idx
  Load/PV/RTPV/WIND/Hydro timeseries         gen_bus_idx, p_nom, gen_cost
                                             load (nb,T), p_max_t (ng,T)

weather.synthetic_weather(system) ──► Weather(t_out, t_eq) ──► t_eff
                                                  │
ThermalParams.build(n_bus) ──► a, ḡ, t_set, x_thr │
                                        │         │
                                        ▼         ▼
                          thermal.thermal_forcing(t_eff, t_set, a) ──► φ (nd,T)
                          thermal.state_bound(φ, params)           ──► xmax (nd,T)
                                        │
RBConfig ──────────────────────────────►│
  hvac_share, v̄, ρ, v_other,            ▼
  scarcity_mw, mip_gap        RollingBlackout ──► MILP ──► RBResult
                                                            z, x, shed MWh,
                                                            degree_hours, costs
```

**Load split (v0):** `d^HVAC = hvac_share × load`, `d^other = (1−hvac_share) × load
+ scarcity_mw`. Both are `cp.Parameter`, so the problem compiles once and
`set_window()` re-points it per day.

Nodal load follows RTS-GMLC's own README: regional series × each bus's share of
its area's peak. **Consequence:** every bus in an area has an identical normalised
shape. Only 51 of 73 buses carry load; the other 22 get no `z` and no thermal
state, dropping 22·T useless binaries.

---

## 5. Parameters

| symbol | code | meaning | value / source |
|---|---|---|---|
| `a` | `ThermalParams.a` | fraction of excess surviving one hour | `exp(−1/τ)` = 0.936, τ = 15 h (Wang TTC median) |
| `φ_t` | `thermal_forcing()` | °C/h gained with AC off | `(1−a)(T_out + T_eq − T_set)` |
| `ḡ` | `ThermalParams.gbar` | °C/h the AC removes | `1.15 × φ` at design temperature ≈ 2.3 |
| `T_set` | `ThermalParams.t_set` | cooling setpoint | 24 °C (22 °C pre-cooled) |
| `x_thr` | `ThermalParams.x_thr` | **evaluation** threshold | 4 °C = Wang's 28 °C |
| `v̄`, `ρ`, `v_other` | `RBConfig` | VoLL intercept, slope, non-HVAC VoLL | 1000, 250, 1000 |

**Only the ratios `ρ/v̄` and `v_other/v̄` matter.** During scarcity VoLL
(~$1,000/MWh) dwarfs generation cost (~$30–80/MWh), so scaling all three leaves
the schedule unchanged. Sensitivity analysis is a 2-D sweep, not 3-D.

**`ḡ` is the one parameter Wang cannot supply** — she models free-float only, with
no HVAC. Sizing it from design conditions makes rebound headroom `ḡ/φ_t` *shrink*
as it gets hotter (2.2× at 28 °C, 1.05× at 45 °C): least recovery capability
exactly when it is most needed. A constant multiplicative headroom gets that
backwards.

**Validation:** the parameter set reproduces Wang's headline without tuning —
28 °C reached at **2.31 h** un-notified, **3.35 h** pre-cooled (Wang: "no more
than two hours", "expand … by about an hour"). Both are assertions in
`tests/test_thermal.py`.

### On the removed knee

Earlier versions had a knee inside the VoLL (`v = v̄ + ρ·max(0, x − x_thr)`). It is
gone; VoLL is now `v̄ + ρ·x` throughout. The knee made the dynamic term
**unfirable**: the premium rides on `u = d·z` and so is only payable while
de-energised, but at τ = 15 h a house needs ~2.3 h of *continuous* outage to reach
28 °C while the optimiser rotates in 1–3 h slices. Measured: 0 of 130 shed
bus-hours ever reached a 4 °C knee, and 100% of the degree-hours accrued *after*
restoration, while recovering on full power.

`x_thr` survives as the **evaluation** threshold only — it is what
`RBResult.degree_hours` measures. Wang's threshold is the right thing to measure
and the wrong thing to put inside the VoLL.

---

## 6. Using the 20 years of data from `e2e-downscale`

`/Users/charlesgulian/Desktop/Projects/e2e-downscale` holds time-synchronised load,
solar, wind and weather for the same 73-bus system. **This is the single biggest
available upgrade to result quality** — it replaces both the synthetic temperature
and RTS-GMLC's synthetic 2020 load.

### Coverage (verified)

| dataset | path pattern | buses | years |
|---|---|---|---|
| load profiles | `load-data/profiles/bus{N}/NSRDB_load-profile_bus{N}_{yr}.csv` | 73 | 2000–2020 |
| weather inputs | `load-data/inputs/bus{N}/NSRDB_weather-inputs_bus{N}_{yr}.csv` | 73 | 1998–2023 |
| solar profiles | `solar-data/profiles/bus{N}/NSRDB_profile_bus{N}_{yr}.csv` | 17 | 1998–2023 |
| wind profiles | `wind-data/profiles/bus{N}/WTK-LED_profile_bus{N}-{yr}.csv` | 4 | 2001–2020 |
| ERA5 | `weather-data/processed/{N}/bus{N}-instant-{yr}.csv` | 73 | 1982–2024 |

**Complete overlap across all four: 2001–2020 — exactly 20 years.**

The 17 solar buses are precisely the 17 PV+RTPV generator buses, and the 4 wind
buses are precisely the 4 WIND generator buses. Nothing is missing.

### Why this also fixes the timezone problem

Load, solar, wind and temperature all come from the **same NSRDB / WTK-LED pull at
the same bus coordinates**, so they are synchronised by construction. The earlier
concern — that RTS-GMLC's unlabelled local clock might be offset from a UTC weather
pull, silently shifting the phase between temperature, load and solar — dissolves
for everything except hydro.

### What a loader must handle

1. **Load profiles are normalised.** Multiply by `bus.csv` `MW Load`. Verified: this
   gives a system peak of 8,329–8,608 MW and 38.1–39.2 TWh/yr, against RTS-GMLC
   native 8,192 MW / 37.66 TWh. Close enough to be a drop-in.
2. **Wind profiles exceed 1.0** (max 1.025 at bus 122). Multiplying by nameplate
   would breach `PMax`. Clip, or rescale — decide deliberately and document it.
3. **Solar is `p_mp`**, per-unit, max ≈ 0.94. Multiply by nameplate.
4. **Hydro has no multi-year data.** Reuse RTS-GMLC's 2020 profile every year — fine,
   as agreed. But note the hour-count mismatch: RTS-GMLC 2020 has **8,784** h
   (leap), and target years vary — 2004 has 8,784, 2006 has 8,760, and **2020 in
   `e2e-downscale` has only 8,772**, i.e. it is *not* a complete leap year. Align by
   timestamp, not by position, or the tail of every year will be silently offset.
5. **Filename inconsistency:** wind uses `bus{N}-{yr}` (hyphen); load and solar use
   `bus{N}_{yr}` (underscore).

### Suggested shape

Add `scripts/e2e_data.py` exposing `load_rts_year(year, e2e_root, …) -> System`,
returning the **same `System` dataclass** so `dcopf.py` and `rolling_blackout.py`
need no changes. Temperature and GHI for the thermal model come from the same
`load-data/inputs` files, so `weather.py` gains an `nsrdb_weather(system, year)`
alongside `synthetic_weather` — again same `Weather` container, no downstream change.

### What 20 years buys

- **Real per-bus temperature.** All 51 buses currently share one diurnal shape with
  a ±2 °C latitude offset and a uniform τ, making them near-interchangeable. That
  flatters the rotation result *and* causes the MILP symmetry that stalls the last
  0.03 % of gap. Real data fixes both.
- **Genuine heat-wave episode selection** — rank days by coincident temperature and
  load across 20 years instead of inventing one.
- **Train/test splits** for fitting `ρ` (or a pole bank) against a downstream
  evaluator, without overfitting to a single year's weather.
- **Note:** the load profiles are themselves XGBoost forecasts trained on EIA
  Southwest demand (`data-scripts/load_data.py`), so weather dependence is already
  baked in. When demand-ninja later splits HVAC out, it will be splitting an
  ML-derived load, not RTS-GMLC's synthetic shape.

---

## 7. Tests

```
tests/test_thermal.py          14   physics; no solver, <1 s
tests/test_dcopf.py            13   data layer + LP
tests/test_rolling_blackout.py  7   MILP correctness (3 solves, ~60 s)
```

The load-bearing one is **`test_milp_objective_equals_true_nonlinear_cost`**: it
recomputes the true bilinear cost from the re-simulated thermostat and matches it
against the solver's objective. Both relaxations are conditional, and that single
test is what certifies both.

Comparative results (static vs dynamic VoLL, etc.) are deliberately **not** tests —
they are the object of study, not invariants, and each costs several MILP solves.

---

## 8. Known limitations and open decisions

**Scope limits, deliberate:**

- **HVAC load is exogenous.** Rebound affects the thermal state but not the power
  balance — no cold-load-pickup feedback on adequacy. Making it endogenous would
  hand the operator implicit thermostat control, which is exactly the capability a
  rolling blackout exists *because* the operator lacks.
- **One `z` per bus.** A feeder outage de-energises HVAC and non-HVAC together;
  they differ in price, not in switching.
- **CSP excluded (200 MW).** Its file is `Natural_Inflow` — solar *thermal* input,
  not electrical output.
- **No unit commitment**, so `p_min = 0`. Imposing RTS-GMLC's PMin without
  commitment binaries would force must-run output and distort scarcity.
- **Hydro is must-take** at its profile. Mean availability is 22.1 MW against a
  50 MW cap, so energy-budgeting it instead would free materially more peak
  capability — worth revisiting.

**Open:**

- `RBConfig.time_limit` is **wall-clock**, so a non-converged solve can give
  different degree-hours run to run. Gurobi's `WorkLimit` + fixed `Seed` is the
  deterministic replacement if reproducibility matters.
- The MILP is strongly symmetric: it reaches ~0.03 % gap in seconds then grinds
  hundreds of thousands of nodes. `mip_gap = 1e-3` is the practical setting, but
  **degree-hours is far more gap-sensitive than the objective** (4.0 → 9.0 across
  1e-3 → 1e-2 while the objective moves < 1 %). Do not loosen past 1e-3.
- `hvac_share = 0.30` flat makes `d^HVAC` and `d^other` identically shaped, so the
  two-VoLL split changes magnitudes but not timing. Real timing effects need
  demand-ninja HVAC profiles.

**Do not use** the $118–153 M/yr production-cost figure from
`notebooks/Research Notes.ipynb` as a validation target here. It is arithmetically
unreachable on RTS-GMLC native data: annual load is 37.66 TWh, VRE supplies at most
13.05 TWh and hydro 4.08 TWh, so ≥ 20.7 TWh must come from thermal — a floor of
$166 M/yr even if all of it were the $8.10/MWh nuclear unit, and ~$534 M/yr at the
coal/CC band those units actually occupy. A correct DC-OPF here annualises to
~$477 M. That figure came from a different system: an EIA-trained load model plus
NSRDB/WTK-LED profiles and a +600 MW PV / +600 MW storage pad.

---

## 9. References

- Wang, Hong & Li (2021). *Informing the planning of rotating power outages in heat
  waves through data analytics of connected smart thermostats for residential
  buildings.* Environ. Res. Lett. **16** 074003.
  → `../optimal-rolling-blackout-archive/papers/`
- Staffell, Pfenninger & Johnson (2023). *A global model of hourly space heating and
  cooling demand at multiple spatial scales.* Nature Energy. → demand.ninja, for the
  future HVAC/non-HVAC load split.
- NREL RTS-GMLC → `RTS-GMLC-master/`. Wind and solar derive from WWSIS-2 (WRF
  reanalysis, 2004–2006 vintage); load is a synthetic RTS-96 shape and is **not**
  weather-derived — which is why no single "weather year" synchronises the native
  data, and why `e2e-downscale` is the better foundation.
