"""
Reformat RTS-GMLC's hydro timeseries into the same per-bus profile layout the
NSRDB/WTK-LED profiles use.

RTS-GMLC ships hydro as one wide file in MW, one column per unit, stamped 2020:

    RTS-GMLC-master/RTS_Data/timeseries_data_files/Hydro/DAY_AHEAD_hydro.csv

This writes it out as:

    data/hydro-data/profiles/bus{N}/RTS-GMLC_profile_bus{N}_2020.csv

with per-unit values (divided by the 50 MW unit nameplate), matching how solar
and wind profiles are stored so HydroResource can read them the same way.

Two facts that make a per-bus file well defined, both asserted below:
  * every HYDRO/ROR unit is 50 MW
  * units at the same bus have identical profiles

2020 is the only year available. Every simulated year re-uses it; the re-map
onto another calendar happens at read time in HydroResource.get_gen_profile.
"""

import pathlib

import numpy as np
import pandas as pd

curr_dir = pathlib.Path(__file__).resolve().parents[1]
rts_dir = curr_dir / "RTS-GMLC-master" / "RTS_Data"
hydro_data_dir = curr_dir / "data" / "hydro-data"

HYDRO_TYPES = ("HYDRO", "ROR")
SOURCE_YEAR = 2020


def create_hydro_profiles():
    df_gen = pd.read_csv(rts_dir / "SourceData" / "gen.csv")
    df_hydro = df_gen[df_gen["Unit Type"].isin(HYDRO_TYPES)]

    ts_path = rts_dir / "timeseries_data_files" / "Hydro" / "DAY_AHEAD_hydro.csv"
    df_ts = pd.read_csv(ts_path)
    stamp = (pd.to_datetime(dict(year=df_ts.Year, month=df_ts.Month, day=df_ts.Day))
             + pd.to_timedelta(df_ts.Period - 1, unit="h"))

    for bus, units in df_hydro.groupby("Bus ID"):
        uids = list(units["GEN UID"])
        nameplates = units["PMax MW"].unique()
        assert len(nameplates) == 1, f"bus {bus}: mixed nameplates {nameplates}"
        nameplate = float(nameplates[0])

        # Units at a bus share one profile, so check before collapsing them.
        for uid in uids[1:]:
            assert np.allclose(df_ts[uids[0]], df_ts[uid]), \
                f"bus {bus}: {uid} differs from {uids[0]}"

        profile = pd.Series(df_ts[uids[0]].to_numpy(dtype=float) / nameplate,
                            index=stamp, name="profile")
        profile.index.name = "datetime"
        assert profile.max() <= 1.0 + 1e-9, f"bus {bus}: profile exceeds nameplate"

        bus_dir = hydro_data_dir / "profiles" / f"bus{bus}"
        bus_dir.mkdir(exist_ok=True, parents=True)
        fname = f"RTS-GMLC_profile_bus{bus}_{SOURCE_YEAR}.csv"
        profile.to_csv(bus_dir / fname)
        print(f"  bus{bus}: {len(profile)} h, {len(uids)} units x {nameplate:.0f} MW -> {fname}")


if __name__ == "__main__":
    create_hydro_profiles()
