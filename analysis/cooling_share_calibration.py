"""
Calibrate COOLING_SHARE: what fraction of each bus's peak load is cooling?

Reproduces the analysis behind the value set in data-scripts/other_data.py.
Run it to regenerate every number quoted in docs/cooling-share-calibration.md:

    python analysis/cooling_share_calibration.py

Three independent pieces of evidence:

  1. A change-point regression of observed EIA hourly demand on area-mean
     temperature, giving the temperature-sensitive share of coincident peak and
     of annual energy for real Southwest balancing authorities.

  2. The same regression applied to *our* baseline load profiles, to check they
     reproduce the observed temperature sensitivity. (They should: the profiles
     were XGBoost-trained on EIA "SW" demand.)

  3. A sweep over COOLING_SHARE, reporting for each value the resulting cooling
     share of peak, cooling share of annual energy, and the seasonal flatness of
     the "other" residual -- the last being independent corroboration, since
     nothing in the construction targets it.

Timezone: EIA demand and the NSRDB inputs are both stored in UTC. Local time is
UTC-7 (Arizona MST, no DST) and is used only for reporting peak hours.
"""

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "load-data" / "inputs"
PROFILES = ROOT / "data" / "load-data" / "profiles"
EIA_FILE = ROOT / "data" / "load-data" / "historical" / "EIA_hourly_load_2016_2019.csv"

REGRESSION_YEAR = 2019           # the year present in the EIA file
SWEEP_YEARS = range(2001, 2021)  # window with complete profile coverage
LOCAL_OFFSET = -7                # UTC -> Arizona MST
BALANCE_GRID = np.arange(10, 26, 0.5)   # candidate balance-point temperatures


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def area_mean_temperature(year=REGRESSION_YEAR):
    """Mean NSRDB dry-bulb across all buses, UTC."""
    series = []
    for bus_dir in sorted(INPUTS.glob("bus*")):
        f = bus_dir / f"NSRDB_weather-inputs_{bus_dir.name}_{year}.csv"
        if f.exists():
            series.append(pd.read_csv(f, index_col=0, parse_dates=True)["Temperature"])
    return pd.concat(series, axis=1).mean(axis=1)


def eia_demand(respondent, year=REGRESSION_YEAR):
    """Observed hourly demand (MW) for one balancing authority, UTC."""
    df = pd.read_csv(EIA_FILE, usecols=["respondent", "type", "value", "datetime"])
    df = df[(df.respondent == respondent) & (df.type == "D")]
    s = pd.Series(df["value"].astype(float).values, index=pd.to_datetime(df["datetime"]))
    s = s[~s.index.duplicated()].sort_index()
    return s[s.index.year == year]


def bus_peaks(system_config="RTS-ORB"):
    df = pd.read_csv(ROOT / "system-config" / system_config / "bus.csv")
    return {int(r["Bus ID"]): r["MW Load"] for _, r in df.iterrows() if r["MW Load"] > 0}


def read_profile(bus, pattern, years):
    parts = []
    for year in years:
        f = PROFILES / f"bus{bus}" / pattern.format(b=bus, y=year)
        if f.exists():
            s = pd.read_csv(f, index_col=0).squeeze()
            s.index = pd.to_datetime(s.index)
            parts.append(s)
    return pd.concat(parts).sort_index()


# ----------------------------------------------------------------------
# 1 & 2. change-point regression
# ----------------------------------------------------------------------

def changepoint_fit(demand, temperature):
    """Fit  D_t = base + slope * max(0, T_t - T0) + hour-of-day effects.

    T0 is chosen by grid search on sum of squared errors. Hour dummies absorb
    the diurnal pattern so `slope` picks up temperature response rather than
    the correlation between time of day and temperature.

    Returns the balance point, slope (MW/degC), R^2, the temperature-sensitive
    share of demand at the peak hour, and its share of annual energy.
    """
    df = pd.concat([demand.rename("D"), temperature.rename("T")], axis=1).dropna()
    hours = pd.get_dummies(df.index.hour, prefix="h", drop_first=True)
    hours.index = df.index
    hours = hours.astype(float)

    best = None
    for t0 in BALANCE_GRID:
        X = pd.concat(
            [pd.Series(1.0, index=df.index, name="const"),
             np.maximum(0.0, df["T"] - t0).rename("cdd"),
             hours],
            axis=1,
        ).values
        beta, *_ = np.linalg.lstsq(X, df["D"].values, rcond=None)
        sse = float(((df["D"].values - X @ beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, t0, beta)

    sse, t0, beta = best
    slope = float(beta[1])
    peak_time = df["D"].idxmax()
    sensitive_at_peak = slope * max(0.0, df.loc[peak_time, "T"] - t0)
    sensitive_annual = float((slope * np.maximum(0.0, df["T"] - t0)).sum())

    return {
        "balance_point_C": t0,
        "slope_MW_per_C": slope,
        "r2": 1 - sse / float(((df["D"] - df["D"].mean()) ** 2).sum()),
        "peak_MW": float(df.loc[peak_time, "D"]),
        "temp_at_peak_C": float(df.loc[peak_time, "T"]),
        "share_of_peak_pct": 100 * sensitive_at_peak / float(df.loc[peak_time, "D"]),
        "share_of_energy_pct": 100 * sensitive_annual / float(df["D"].sum()),
        "peak_time_local": peak_time + pd.Timedelta(hours=LOCAL_OFFSET),
    }


def our_baseline_load(year=REGRESSION_YEAR):
    """Our baseline system load (MW), i.e. bus peak x normalized profile."""
    return sum(read_profile(bus, "NSRDB_load-profile_bus{b}_{y}.csv", [year]) * peak
               for bus, peak in bus_peaks().items())


# ----------------------------------------------------------------------
# 3. sweep
# ----------------------------------------------------------------------

def sweep(shares=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65), years=SWEEP_YEARS):
    """For each candidate share, the three diagnostics that drove the choice."""
    peaks = bus_peaks()
    baseline = {b: read_profile(b, "NSRDB_load-profile_bus{b}_{y}.csv", years) * p
                for b, p in peaks.items()}
    cooling_pu = {b: read_profile(b, "NSRDB_cooling-load-profile_bus{b}_{y}.csv", years)
                  for b in peaks}

    total_baseline = sum(baseline.values())
    peak_time = total_baseline.idxmax()

    rows = []
    for share in shares:
        negative_buses, worst = 0, 0.0
        total_cooling = total_other = None
        for bus, peak in peaks.items():
            cool = cooling_pu[bus] * (share * peak)
            base, cool = baseline[bus].align(cool, join="inner")
            other = base - cool
            if other.min() < 0:
                negative_buses += 1
                worst = min(worst, float(other.min()))
            total_cooling = cool if total_cooling is None else total_cooling + cool
            total_other = other if total_other is None else total_other + other

        july = total_other[total_other.index.month == 7].mean()
        january = total_other[total_other.index.month == 1].mean()
        rows.append({
            "share": share,
            "negative_buses": negative_buses,
            "worst_residual_MW": worst,
            "cooling_at_peak_pct": 100 * total_cooling[peak_time] / total_baseline[peak_time],
            "cooling_of_energy_pct": 100 * total_cooling.sum() / total_baseline.sum(),
            "other_peak_MW": float(total_other.max()),
            "other_july_over_january": float(july / january),
        })
    return pd.DataFrame(rows), peak_time


# ----------------------------------------------------------------------

def main():
    temperature = area_mean_temperature()
    print(f"Area-mean temperature {REGRESSION_YEAR}: {len(temperature)} h, "
          f"{temperature.min():.1f} to {temperature.max():.1f} C\n")

    print("1 & 2. CHANGE-POINT REGRESSION (demand ~ temperature + hour effects)")
    print(f"{'series':<24} {'T0':>5} {'MW/C':>7} {'R2':>5} {'peak MW':>8} "
          f"{'T@pk':>6} {'@peak':>7} {'annual':>7}  peak (local)")
    targets = {}
    for label, series in [("EIA AZPS", eia_demand("AZPS")),
                          ("EIA SRP", eia_demand("SRP")),
                          ("EIA SW (region)", eia_demand("SW")),
                          ("our baseline load", our_baseline_load())]:
        r = changepoint_fit(series, temperature)
        targets[label] = r
        print(f"{label:<24} {r['balance_point_C']:5.1f} {r['slope_MW_per_C']:7.1f} "
              f"{r['r2']:5.2f} {r['peak_MW']:8.0f} {r['temp_at_peak_C']:6.1f} "
              f"{r['share_of_peak_pct']:6.1f}% {r['share_of_energy_pct']:6.1f}%  "
              f"{r['peak_time_local']}")
    print("\n  NOTE AZPS's 2019 maximum falls on 30 May at 30 C, which is implausible for")
    print("  Arizona and gives a weak fit. Treat that row as suspect; SRP, SW and our")
    print("  baseline agree closely with one another.")
    print("\n  The regression is an UPPER bound on cooling: it attributes all")
    print("  temperature-correlated load to cooling, and some commercial/industrial")
    print("  load tracks temperature for other reasons.\n")

    print(f"3. SWEEP OVER COOLING_SHARE ({SWEEP_YEARS.start}-{SWEEP_YEARS.stop - 1})")
    table, peak_time = sweep()
    print(f"   system peak at {peak_time + pd.Timedelta(hours=LOCAL_OFFSET)} local\n")
    print(f"{'share':>6} {'neg buses':>10} {'worst MW':>9} {'cool@peak':>10} "
          f"{'cool/yr':>8} {'other peak':>11} {'other Jul/Jan':>14}")
    for _, r in table.iterrows():
        print(f"{r['share']:6.2f} {int(r['negative_buses']):10d} {r['worst_residual_MW']:9.1f} "
              f"{r['cooling_at_peak_pct']:9.1f}% {r['cooling_of_energy_pct']:7.1f}% "
              f"{r['other_peak_MW']:11.0f} {r['other_july_over_january']:14.2f}")

    sw = targets["EIA SW (region)"]
    ours = targets["our baseline load"]
    print(f"\n   regression targets: {sw['share_of_peak_pct']:.1f}% of peak (EIA SW), "
          f"{ours['share_of_peak_pct']:.1f}% (our baseline); "
          f"{sw['share_of_energy_pct']:.1f}% / {ours['share_of_energy_pct']:.1f}% of energy")
    print("   'other Jul/Jan' of 1.00 would mean a seasonally flat residual.")
    print("\n   CHOSEN: 0.50 -- deliberately below the regression, which is an upper")
    print("   bound, and leaving the residual mildly summer-peaking (1.07) as genuine")
    print("   non-cooling seasonal load would be.")
    return table, targets


if __name__ == "__main__":
    main()
