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
# Cooling peak as a fraction of the bus's bus.csv peak. Set so that cooling
# lands at ~30% of the *coincident* system peak, which is how the "AC is ~30% of
# summer peak" statistic is normally quoted. Cooling's own peak (17:00 local)
# falls a little before and on a different day from the system peak, so a share
# of 0.30 here would give only ~26% coincident; 0.35 gives ~30%.
COOLING_SHARE = 0.35


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
