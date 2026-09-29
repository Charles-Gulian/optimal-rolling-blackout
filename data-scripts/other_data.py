"""
Derive the "other" (non-cooling) load profiles by subtracting cooling from the
baseline load, and record the peaks both types need in load.csv.

    other(t) = baseline(t) - cooling(t)        at every hour, per bus

Because the subtraction is exact and hourly, cooling + other reproduces the
baseline series exactly, so the system load shape and system peak are unchanged
by the decomposition. Only the split between the two types is new.

Peaks
-----
    cooling peak = COOLING_SHARE * bus.csv MW Load
    other peak   = the residual's own maximum over the year range

Both profiles are then normalised by their own 20-year maximum, so `Peak MW` in
load.csv means "this load's peak MW over 2001-2020". That is the convention
going forward.

NOTE the existing "baseline" profiles use a different convention (normalised by
each bus's 2006 peak, scaled by bus.csv MW Load). They are left as they are;
they are only an input here, not an output.

Timezone: every profile read and written here is on the UTC clock, matching the
NSRDB-derived inputs. No conversion happens in this script.
"""

import pathlib

import pandas as pd

curr_dir = pathlib.Path(__file__).resolve().parents[1]
load_data_dir = curr_dir / "data" / "load-data"

YEARS = range(2001, 2021)
# Cooling peak as a fraction of the bus's bus.csv peak.
#
# Calibrated against a change-point regression (demand on area-mean temperature,
# with hour-of-day effects) of 2019 EIA hourly demand. The temperature-sensitive
# component is ~48% of coincident peak and ~17% of annual energy for the EIA "SW"
# region, and our baseline profiles reproduce that almost exactly (46.4% / 17.1%),
# which is unsurprising since they were XGBoost-trained on SW demand.
#
# That regression is an UPPER bound: it attributes all temperature-correlated load
# to cooling, and some commercial/industrial load tracks temperature for other
# reasons. 0.55 would match it exactly; 0.50 is deliberately conservative.
#
#   share  cool@peak  cool/yr  other Jul/Jan   (sweep over 2001-2020)
#    0.35     30.3%     10.8%       1.23
#    0.50     43.2%     15.4%       1.07     <- chosen
#    0.55     47.6%     16.9%       1.02
#    0.65     56.2%     20.0%       0.92
#
# The Jul/Jan column is the seasonal flatness of the residual: a genuine
# non-thermal load should be roughly flat across seasons, and nothing in the
# construction forces it, so it is independent corroboration. Feasibility is not
# binding -- no bus goes negative even at 0.65.
COOLING_SHARE = 0.50


def read_profile(bus, pattern, years):
    """Concatenate one profile type across years, indexed in UTC."""
    parts = []
    for year in years:
        f = load_data_dir / "profiles" / f"bus{bus}" / pattern.format(b=bus, y=year)
        if not f.exists():
            continue
        s = pd.read_csv(f, index_col=0).squeeze()
        s.index = pd.to_datetime(s.index)
        parts.append(s)
    if not parts:
        raise FileNotFoundError(f"bus{bus}: no profiles matching {pattern}")
    return pd.concat(parts).sort_index()


def create_other_profiles(system_config="RTS-GMLC", years=YEARS, share=COOLING_SHARE):
    df_bus = pd.read_csv(curr_dir / "system-config" / system_config / "bus.csv")
    buses = {int(r["Bus ID"]): r["MW Load"] for _, r in df_bus.iterrows() if r["MW Load"] > 0}

    peak_rows = []
    for bus, bus_peak in buses.items():
        baseline = read_profile(bus, "NSRDB_load-profile_bus{b}_{y}.csv", years) * bus_peak
        cooling_peak = share * bus_peak
        cooling = read_profile(bus, "NSRDB_cooling-load-profile_bus{b}_{y}.csv", years) * cooling_peak

        baseline, cooling = baseline.align(cooling, join="inner")
        other = baseline - cooling
        if other.min() < 0:
            raise ValueError(
                f"bus{bus}: cooling exceeds baseline by {-other.min():.1f} MW at "
                f"{other.idxmin()} -- COOLING_SHARE of {share} is too high for this bus"
            )

        other_peak = other.max()
        normalised = (other / other_peak).rename("profile")
        normalised.index.name = "datetime"

        bus_dir = load_data_dir / "profiles" / f"bus{bus}"
        for year in years:
            chunk = normalised[normalised.index.year == year]
            if len(chunk):
                chunk.to_csv(bus_dir / f"NSRDB_other-load-profile_bus{bus}_{year}.csv")

        peak_rows.append({"Bus ID": bus, "Load Type": "cooling", "Peak MW": cooling_peak})
        peak_rows.append({"Bus ID": bus, "Load Type": "other", "Peak MW": other_peak})

    df_peaks = pd.DataFrame(peak_rows)
    out = load_data_dir / "load_peaks.csv"
    df_peaks.to_csv(out, index=False)

    for load_type, grp in df_peaks.groupby("Load Type"):
        print(f"  {load_type:8} {len(grp):3} buses, total peak {grp['Peak MW'].sum():7.1f} MW")
    print(f"  wrote {out.relative_to(curr_dir)}")
    return df_peaks


if __name__ == "__main__":
    create_other_profiles()
