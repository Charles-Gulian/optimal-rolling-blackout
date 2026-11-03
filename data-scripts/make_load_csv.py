"""
Generate load.csv for a system config.

    python data-scripts/make_load_csv.py

Writes system-config/<name>/load.csv with one row per bus:
Load UID, Bus ID, Load Type, Peak MW. Peaks come from
rts_gmlc_load.bus_peaks(), so run rts_gmlc_load.py first.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rts_gmlc_load import bus_peaks

curr_dir = pathlib.Path(__file__).resolve().parents[1]
config_dir = curr_dir / "system-config"

# Uniform multiplier on every load's peak, to make RTS-ORB scarce enough to
# shed. STALE: 1.10 was calibrated against the old NSRDB load, not RTS-GMLC.
# Re-tune before trusting any blackout-frequency result.
PEAK_SCALING = {"RTS-GMLC": 1.0, "RTS-ORB": 1.10, "RTS-ORB-HR": 1.10}

LOAD_TYPE = "baseline"


def make_load_csv(system_config="RTS-GMLC", peak_scaling=None):
    if peak_scaling is None:
        peak_scaling = PEAK_SCALING.get(system_config, 1.0)

    peaks = bus_peaks(system_config)

    df_load = pd.DataFrame({
        "Load UID": [f"{bus}_{LOAD_TYPE}" for bus in peaks.index],
        "Bus ID": peaks.index,
        "Load Type": LOAD_TYPE,
        # Rounded only to keep the CSV readable: the derived peak carries
        # float noise from share x area-max (108.00000000000001), and 1e-6 MW
        # is a milliwatt.
        "Peak MW": (peaks["Peak MW"].to_numpy() * peak_scaling).round(6),
    }).sort_values("Bus ID").reset_index(drop=True)

    out = config_dir / system_config / "load.csv"
    df_load.to_csv(out, index=False)
    print(f"  {out.relative_to(curr_dir)}: {len(df_load)} loads, "
          f"{df_load['Peak MW'].sum():,.1f} MW total peak "
          f"(scaling {peak_scaling:g})")
    return df_load


if __name__ == "__main__":
    # Regenerate every config we know a scaling for, so one invocation leaves
    # all of system-config/ consistent with this file.
    for config in PEAK_SCALING:
        make_load_csv(config)
