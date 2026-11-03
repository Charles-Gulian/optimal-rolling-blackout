"""
Build native RTS-GMLC solar, wind and hydro profiles.

    python data-scripts/rts_gmlc_renewables.py --configs RTS-GMLC RTS-ORB

Writes data/{solar,wind,hydro}-data/profiles/bus{N}/
RTS-GMLC_profile_{GEN UID}_2020.csv, per generator, as a fraction of
nameplate. Units a config adds that RTS-GMLC never had borrow the profile of
a same-bus, same-type neighbour.

CSP is skipped: its file is natural inflow to storage, not output, and peaks
at 1.96x nameplate. system.py's unit-type whitelist excludes it anyway.
"""

import pathlib

import pandas as pd

from rts_gmlc_load import EXPECTED_HOURS, ROOT, SOURCE_YEAR, STAMP, TS_DIR, rts_datetime

# Unit Type -> (data/ subtree, timeseries file). HYDRO and ROR share a file.
UNIT_TYPE_SOURCES = {
    "PV":    ("solar", "PV/DAY_AHEAD_pv.csv"),
    "RTPV":  ("solar", "RTPV/DAY_AHEAD_rtpv.csv"),
    "WIND":  ("wind",  "WIND/DAY_AHEAD_wind.csv"),
    "HYDRO": ("hydro", "Hydro/DAY_AHEAD_hydro.csv"),
    "ROR":   ("hydro", "Hydro/DAY_AHEAD_hydro.csv"),
}

_cache = {}


def timeseries(relpath):
    """Hourly MW per generator from one RTS-GMLC file, tz-naive index."""
    if relpath not in _cache:
        df = pd.read_csv(TS_DIR / relpath)
        index = rts_datetime(df)
        df = df.drop(columns=["Year", "Month", "Day", "Period"])
        df.index = index
        if len(df) != EXPECTED_HOURS:
            raise ValueError(f"{relpath}: expected {EXPECTED_HOURS} rows, got {len(df)}")
        _cache[relpath] = df
    return _cache[relpath]


def check_nameplates(configs=("RTS-GMLC", "RTS-ORB", "RTS-ORB-HR")):
    """Fail if any config disagrees about a shared unit's nameplate."""
    ref = None
    for config in configs:
        path = ROOT / "system-config" / config / "gen.csv"
        if not path.exists():
            continue
        g = pd.read_csv(path).set_index("GEN UID")
        g = g.loc[g["Unit Type"].isin(UNIT_TYPE_SOURCES), "PMax MW"]
        if ref is None:
            ref = g
            continue
        shared = ref.index.intersection(g.index)
        drift = (ref[shared] - g[shared]).abs().max()
        if drift > 1e-9:
            raise ValueError(
                f"{config} disagrees with the reference config about shared "
                f"nameplates by up to {drift:.4g} MW; per-unit profiles can no "
                f"longer live in a single shared data/ tree.")


def find_twin(uid, row, gen, native):
    """The native unit a synthetic unit borrows its shape from.

    Matches on same bus and same Unit Type: PV and RTPV are both solar but
    have different shapes, so borrowing across them would change technology.
    """
    siblings = gen.index[(gen["Bus ID"] == row["Bus ID"])
                         & (gen["Unit Type"] == row["Unit Type"])
                         & gen.index.isin(native)]
    if not len(siblings):
        raise KeyError(
            f"{uid} ({row['Unit Type']} at bus {row['Bus ID']}) has no native "
            f"series and no native {row['Unit Type']} unit at its bus to borrow "
            f"from. Either drop it from gen.csv or choose a donor explicitly.")
    return sorted(siblings)[0]


def main(config="RTS-GMLC"):
    check_nameplates()

    gen = pd.read_csv(ROOT / "system-config" / config / "gen.csv").set_index("GEN UID")
    gen = gen.loc[gen["Unit Type"].isin(UNIT_TYPE_SOURCES)]

    written, borrowed = {}, []
    for uid, row in gen.iterrows():
        subtree, relpath = UNIT_TYPE_SOURCES[row["Unit Type"]]
        nameplate = float(row["PMax MW"])
        if nameplate <= 0:
            raise ValueError(f"{uid}: PMax MW is {nameplate}, cannot normalise")

        native = timeseries(relpath)
        source_uid = uid
        if uid not in native.columns:
            source_uid = find_twin(uid, row, gen, native.columns)
            borrowed.append((uid, source_uid))
        # Normalise by the DONOR's nameplate, so the result is a per-unit
        # shape; the borrower's own nameplate is applied at dispatch time.
        series = native[source_uid] / float(gen.loc[source_uid, "PMax MW"])
        if series.max() > 1.001:
            raise ValueError(f"{uid}: profile peaks at {series.max():.3f} of "
                             f"nameplate; is this file really MW of output?")

        out = ROOT / "data" / f"{subtree}-data" / "profiles" / f"bus{row['Bus ID']}"
        out.mkdir(parents=True, exist_ok=True)
        series.rename("profile").rename_axis("datetime").to_csv(
            out / f"{STAMP}_profile_{uid}_{SOURCE_YEAR}.csv")
        written.setdefault(subtree, []).append((uid, nameplate, series.mean()))

    print(f"--- {config}")
    for subtree, rows in sorted(written.items()):
        mw = sum(n for _, n, _ in rows)
        cf = sum(n * m for _, n, m in rows) / mw
        print(f"{subtree:6} {len(rows):3} units  {mw:8,.1f} MW nameplate  "
              f"capacity factor {cf:.3f}")
    for uid, donor in borrowed:
        print(f"  borrowed shape: {uid:12} <- {donor}")
    print(f"wrote {STAMP}_profile_<GEN UID>_{SOURCE_YEAR}.csv under data/*-data/profiles/bus*/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--configs", nargs="+", default=["RTS-GMLC"])
    for config in p.parse_args().configs:
        main(config)
