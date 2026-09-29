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


def make_load_csv(system_config="RTS-GMLC", load_types=("cooling", "other")):
    src = config_dir / system_config
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
                "Peak MW": peak,
                "VOLL": VOLL[load_type],
            })

    df_load = pd.DataFrame(rows)
    out = src / "load.csv"
    df_load.to_csv(out, index=False)
    print(f"  {out.relative_to(curr_dir)}: {len(df_load)} rows")
    for load_type, grp in df_load.groupby("Load Type"):
        print(f"    {load_type:8} {len(grp):3} loads, {grp["Peak MW"].sum():7.1f} MW total peak")
    return df_load


if __name__ == "__main__":
    make_load_csv()
