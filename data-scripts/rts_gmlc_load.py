"""
Build native RTS-GMLC load profiles.

    python data-scripts/rts_gmlc_load.py

Writes data/load-data/profiles/bus{N}/RTS-GMLC_load-profile_bus{N}_2020.csv
as a fraction of each bus's peak, for every bus carrying load. Native load is
regional, so it is disaggregated by each bus's share of its area's static
load (bus.csv "MW Load"). Timestamps are tz-naive.
"""

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TS_DIR = ROOT / "RTS-GMLC-master" / "RTS_Data" / "timeseries_data_files"
OUT_DIR = ROOT / "data" / "load-data" / "profiles"

SOURCE_YEAR = 2020          # the only year RTS-GMLC ships; 8784 h (leap)
EXPECTED_HOURS = 8784
STAMP = "RTS-GMLC"


def rts_datetime(df):
    """Year/Month/Day/Period -> tz-naive hourly index. Period is 1-indexed."""
    return (pd.to_datetime(df[["Year", "Month", "Day"]])
            + pd.to_timedelta(df["Period"] - 1, unit="h"))


def area_load():
    """Hourly load per area, MW, indexed by tz-naive timestamp."""
    df = pd.read_csv(TS_DIR / "Load" / "DAY_AHEAD_regional_Load.csv")
    index = rts_datetime(df)
    areas = df.drop(columns=["Year", "Month", "Day", "Period"])
    areas.index = index
    areas.columns = [int(c) for c in areas.columns]
    if len(areas) != EXPECTED_HOURS:
        raise ValueError(f"expected {EXPECTED_HOURS} hourly rows, got {len(areas)}")
    return areas


def bus_peaks(config="RTS-GMLC"):
    """Per-bus peak MW implied by the native area profiles.

    Returns a DataFrame indexed by Bus ID with columns Area, Share, Peak MW,
    for buses carrying load. Asserts the result matches bus.csv "MW Load".
    """
    bus = pd.read_csv(ROOT / "system-config" / config / "bus.csv")
    bus = bus[bus["MW Load"] > 0][["Bus ID", "Area", "MW Load"]].copy()

    areas = area_load()
    missing = set(bus["Area"]) - set(areas.columns)
    if missing:
        raise ValueError(f"bus.csv areas {sorted(missing)} absent from the load file")

    bus["Share"] = bus["MW Load"] / bus.groupby("Area")["MW Load"].transform("sum")
    bus["Peak MW"] = bus["Share"] * bus["Area"].map(areas.max())

    drift = (bus["Peak MW"] - bus["MW Load"]).abs().max()
    if drift > 1e-6:
        raise ValueError(
            f"{config}: derived bus peaks disagree with bus.csv 'MW Load' by up "
            f"to {drift:.4g} MW. Either the area profiles or the static loads "
            f"have been changed independently of the other.")
    return bus.set_index("Bus ID")


def main(config="RTS-GMLC"):
    areas = area_load()
    peaks = bus_peaks(config)

    for bus_id, row in peaks.iterrows():
        shape = areas[row["Area"]] / areas[row["Area"]].max()
        out = OUT_DIR / f"bus{bus_id}"
        out.mkdir(parents=True, exist_ok=True)
        shape.rename("profile").rename_axis("datetime").to_csv(
            out / f"{STAMP}_load-profile_bus{bus_id}_{SOURCE_YEAR}.csv")

    total = areas.sum(axis=1)
    print(f"{len(peaks)} load buses across areas "
          f"{sorted(int(a) for a in peaks['Area'].unique())}, "
          f"{len(areas)} hours {areas.index[0]} .. {areas.index[-1]}")
    # Peak MW sums to the NON-coincident total by construction, since each
    # area's shares sum to 1. The coincident peak is the lower, real one.
    print(f"sum of bus Peak MW {peaks['Peak MW'].sum():,.1f} MW "
          f"(= sum of area peaks, non-coincident)")
    print(f"coincident system peak {total.max():,.1f} MW at {total.idxmax()}")
    print(f"wrote {STAMP}_load-profile_bus*_{SOURCE_YEAR}.csv under {OUT_DIR}")


if __name__ == "__main__":
    main()
