# Calibrating `COOLING_SHARE`

**Result: `COOLING_SHARE = 0.50`**, set in `data-scripts/other_data.py`.

Reproduce every number here with:

```bash
python analysis/cooling_share_calibration.py
```

Last run 2026-09-29.

---

## 1. What the parameter does

The 20-year `baseline` load profiles carry no end-use breakdown. To give cooling
its own load type — and therefore its own VoLL, eventually a dynamic one — we
split each bus's load:

```
cooling(t) = COOLING_SHARE x bus_peak x cooling_profile(t)      demand.ninja
other(t)   = baseline(t) - cooling(t)                            residual
```

`COOLING_SHARE` is cooling's **own peak** as a fraction of the bus's `MW Load`
from `bus.csv`. Because the subtraction is exact and hourly, `cooling + other`
reproduces `baseline` at every hour: the system load shape, the system peak and
the reserve margin are all unchanged by the decomposition. Only the split is new.

Note this is *not* the same as cooling's share of **coincident** system peak,
which is the statistic usually quoted. Cooling peaks at 17:00 local on its own
hottest day; the system peaks at 17:00 on a different day. A share of 0.50 gives
43.2% of coincident peak.

## 2. Evidence

### 2a. Change-point regression on observed demand

Fit to 2019 hourly data, where `T` is the mean NSRDB dry-bulb across all 73 RTS
bus locations:

```
D_t = base + slope x max(0, T_t - T0) + hour-of-day effects
```

`T0` by grid search on SSE. Hour dummies absorb the diurnal pattern, so `slope`
measures temperature response rather than the correlation between time of day
and temperature. The temperature-sensitive share at peak is
`slope x max(0, T_peak - T0) / D_peak`.

| series | `T0` | MW/°C | R² | peak MW | T@peak | **at peak** | **annual** |
|---|---|---|---|---|---|---|---|
| EIA AZPS | 19.0 | 157.6 | 0.78 | 9993 | 30.0 | 17.3% | 18.1% |
| EIA SRP | 18.0 | 159.3 | 0.83 | 7347 | 40.4 | **48.5%** | 20.0% |
| EIA SW (region) | 18.5 | 498.3 | 0.82 | 23822 | 41.5 | **48.1%** | **17.1%** |
| our baseline load | 18.0 | 172.6 | 0.86 | 8319 | 40.4 | **46.4%** | **17.1%** |

Two things to take from this.

**Our baseline profiles reproduce observed temperature sensitivity.** 46.4% vs
48.1% at peak and 17.1% vs 17.1% of annual energy, against the EIA "SW" region.
That is an independent validation of the baseline load model itself, which had
not previously been checked. It is not a coincidence — those profiles were
XGBoost-trained on EIA SW demand (`data-scripts/load_data.py`) — but it does
confirm the training transferred.

**The empirically implied share is ~46–48% of coincident peak**, well above the
30% initially assumed.

### 2b. Sweep over candidate shares

Over the full 2001–2020 record:

| share | neg. buses | cooling @ peak | cooling / yr | other Jul/Jan |
|---|---|---|---|---|
| 0.35 | 0 | 30.3% | 10.8% | 1.23 |
| 0.40 | 0 | 34.6% | 12.3% | 1.17 |
| 0.45 | 0 | 38.9% | 13.9% | 1.12 |
| **0.50** | **0** | **43.2%** | **15.4%** | **1.07** |
| 0.55 | 0 | 47.6% | 16.9% | 1.02 |
| 0.60 | 0 | 51.9% | 18.5% | 0.97 |
| 0.65 | 0 | 56.2% | 20.0% | 0.92 |

**Feasibility does not bind.** No bus produces a negative residual at any share
up to 0.65 — cooling and baseline are correlated at 0.93 and cooling's peak sits
below baseline everywhere. `other_data.py` raises on a negative residual rather
than clipping, so this would fail loudly.

**The third column is independent corroboration.** `other Jul/Jan` is the ratio
of the residual's mean July load to its mean January load. A genuine non-thermal
load should be roughly seasonally flat. Nothing in the construction targets this
— the share was calibrated against peak and energy statistics — so the residual
approaching 1.0 near the same value is a separate line of evidence.

## 3. Why 0.50 rather than 0.55

0.55 matches the regression almost exactly (47.6% vs 46–48% at peak, 16.9% vs
17.1% of energy). 0.50 was chosen anyway, for three reasons.

**The regression is an upper bound.** It attributes *all* temperature-correlated
load to cooling. Commercial and industrial load tracks temperature for reasons
other than air conditioning — longer operating hours, refrigeration, ventilation
— so the true air-conditioning share is below the temperature-sensitive share.

**A perfectly flat residual would be suspicious.** At 0.55 the Jul/Jan ratio is
1.02. Genuine non-cooling seasonal effects exist, so a residual carrying a mild
7% summer excess (1.07, at 0.50) is more plausible than one carrying none.

**It is conservative for the study's central claim.** The objective is

```
gen cost + sum v_bar x d_cooling x z + sum rho x d_cooling x w + sum v_other x d_other x z
                                        ^ the dynamic term
```

Cooling and other always sum to baseline, so a larger `COOLING_SHARE` moves MW
from the flat-VoLL bucket into the dynamically-priced one, giving the thermal
state more weight in the optimiser's decision. A smaller share weakens the
dynamic signal. If "dynamic VoLL rotates better than static" holds at 0.50, it
holds a fortiori at the true share — whereas overstating cooling would invite
the objection that the effect was manufactured.

Three caveats on that last point, to avoid overclaiming:

- The **discomfort metric is unaffected**. Degree-hours above 28 °C derive from
  the thermal state, which depends on weather, AC capacity and the schedule —
  not on how many MW were labelled cooling. A smaller share does not understate
  how hot buildings get, only the optimiser's incentive to prevent it.
- **Adequacy is unaffected**: cooling + other = baseline at every hour.
- It is **mildly understated for any absolute cooling statistic**. Do not quote
  our cooling MWh as an estimate of regional air-conditioning energy.

The defensible range is roughly **0.45–0.55**. Sensitivity within that band is
worth reporting.

## 4. Limitations

- **One year of demand data.** The EIA file covers 2019 only, despite its name.
  More years would tighten the estimate and show interannual variation.
- **AZPS is anomalous** — its 2019 maximum lands on 30 May at 30 °C, implausible
  for Arizona, with the weakest fit (R² 0.78). Excluded from the conclusion.
  SRP, SW and our baseline agree closely with each other.
- **Area-mean temperature** across the 73 RTS bus locations is a proxy for each
  balancing authority's service territory, which it does not exactly match.
- **A single balance point per series.** A five-parameter change-point model
  with both heating and cooling segments would fit the shoulder seasons better,
  though it would barely change the summer-peak estimate.
- **Share is constant across buses.** Every bus gets the same fraction, so no
  bus is more or less air-conditioning-intensive than another. Real variation in
  building stock and end-use mix is not represented.

## 5. Data and code

| | |
|---|---|
| `analysis/cooling_share_calibration.py` | this analysis, re-runnable |
| `data/load-data/historical/EIA_hourly_load_2016_2019.csv` | observed demand (2019), UTC |
| `data/load-data/inputs/bus*/NSRDB_weather-inputs_*.csv` | NSRDB temperature, UTC |
| `data-scripts/other_data.py` | where `COOLING_SHARE` is set and applied |
| `data-scripts/cooling_data.py` | demand.ninja cooling profiles |

**Timezone**: EIA demand and NSRDB inputs are both UTC. Local is UTC−7 (Arizona
MST, no DST), used only for reporting peak hours.
