"""
Generate load.csv for a system config.

RTS-GMLC has no load input file -- demand is implied by bus.csv's MW Load
column, which gives one load per bus and no way to say what kind it is. This
writes an explicit table instead:

    system-config/<name>/load.csv
    Load UID, Bus ID, Load Type, Peak MW, VOLL

One row per (bus, load type). Loads become declared rather than inferred, so
adding a load type is a matter of adding rows, and each type carries its own
peak and its own VOLL.

Peaks come from data/load-data/load_peaks.csv, written by other_data.py, which
holds each load's own maximum over 2001-2020. Run the profile scripts first:

    python data-scripts/cooling_data.py     # cooling profiles
    python data-scripts/other_data.py       # other profiles + load_peaks.csv
    python data-scripts/make_load_csv.py    # this

Pass load_types=("baseline",) to fall back to the original single-type config,
whose peak comes straight from bus.csv.
"""

import pathlib

import pandas as pd

curr_dir = pathlib.Path(__file__).resolve().parents[1]
config_dir = curr_dir / "system-config"
peaks_file = curr_dir / "data" / "load-data" / "load_peaks.csv"

# $/MWh of unserved energy, per load type. Placeholders: the whole point of
# splitting cooling out is that it will be priced differently, and eventually
# dynamically, but that model is not built yet.
VOLL = {"baseline": 1e4, "cooling": 1e4, "other": 1e4}

# Uniform multiplier applied to every load's peak, per system config.
#
# RTS-ORB is deliberately made scarce. As built the system is adequate (~9.3 GW
# dispatchable against an ~8.7 GW peak) and never sheds, so the rolling-blackout
# model has nothing to schedule. Scaling load induces the scarcity the study
# needs. Calibrated in analysis/scarcity_calibration.py by counting days with
# any unserved energy across July-August, 2011-2020, DC-OPF with continuous
# shedding:
#
#   factor  blackout days/yr   unserved MWh   years with none   worst year
#     1.12          0.5                 639          7  of 10       3
#     1.14          1.3               1,910          6              7
#     1.15   <- chosen (interpolates to ~2/yr)
#     1.16          3.0               5,275          2             13
#     1.18          5.2              12,519          1             19
#     1.20          8.8              27,187          0             30
#
# NOTE this is a steep curve -- a 2% load change roughly doubles the blackout
# count -- so the factor is a consequential assumption, not a detail. Results
# should carry a sensitivity band over roughly 1.14-1.18.
#
# RTS-GMLC stays unscaled so it remains a faithful representation of the
# published test system.
PEAK_SCALING = {"RTS-GMLC": 1.0, "RTS-ORB": 1.15}


def make_load_csv(system_config="RTS-GMLC", load_types=("cooling", "other"),
                  peak_scaling=None):
    src = config_dir / system_config
    if peak_scaling is None:
        peak_scaling = PEAK_SCALING.get(system_config, 1.0)
    df_bus = pd.read_csv(src / "bus.csv")
    bus_peaks = {int(r["Bus ID"]): r["MW Load"] for _, r in df_bus.iterrows() if r["MW Load"] > 0}

    if set(load_types) == {"baseline"}:
        peaks = {(bus, "baseline"): peak for bus, peak in bus_peaks.items()}
    else:
        if not peaks_file.exists():
            raise FileNotFoundError(f"{peaks_file} missing -- run other_data.py first")
        df_peaks = pd.read_csv(peaks_file)
        peaks = {(int(r["Bus ID"]), r["Load Type"]): r["Peak MW"] for _, r in df_peaks.iterrows()}

    rows = []
    for bus in sorted(bus_peaks):
        for load_type in load_types:
            peak = peaks.get((bus, load_type))
            if peak is None:
                raise KeyError(f"no peak for bus {bus} / {load_type}")
            rows.append({
                "Load UID": f"{bus}_{load_type}",
                "Bus ID": bus,
                "Load Type": load_type,
                "Peak MW": peak * peak_scaling,
                "VOLL": VOLL[load_type],
            })

    df_load = pd.DataFrame(rows)
    out = src / "load.csv"
    df_load.to_csv(out, index=False)
    print(f"  {out.relative_to(curr_dir)}: {len(df_load)} rows, "
          f"peak scaling {peak_scaling:g}")
    for load_type, grp in df_load.groupby("Load Type"):
        print(f"    {load_type:8} {len(grp):3} loads, {grp["Peak MW"].sum():7.1f} MW total peak")
    return df_load


if __name__ == "__main__":
    make_load_csv()
