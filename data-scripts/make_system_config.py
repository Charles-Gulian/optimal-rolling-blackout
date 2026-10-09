"""
Derive our system configs from the published RTS-GMLC data.

system-config/RTS-GMLC/ holds the upstream files unchanged. Everything else is
generated from it here, so each departure from the published test system is
recorded in one place rather than hidden in a hand-edited CSV.

    python data-scripts/make_system_config.py

Run make_load_csv.py afterwards to regenerate load.csv for the derived configs.
"""

import pathlib
import shutil

import pandas as pd

curr_dir = pathlib.Path(__file__).resolve().parents[1]
config_dir = curr_dir / "system-config"

BASE = "RTS-GMLC"
CONFIG_FILES = ("bus.csv", "branch.csv", "gen.csv", "storage.csv")

# --------------------------------------------------------------------------
# RTS-ORB: the base case for this study
# --------------------------------------------------------------------------
# Modifications to the published gen.csv, each with its reason.
#
# 1. NUCLEAR forced outage rate 0.12 -> 0.02.
#    RTS-GMLC's 12% is far above reality: US nuclear unplanned capability loss
#    factor runs 1-2%. At 0.12 the single 400 MW baseload unit would be down
#    roughly 44 days a year, manufacturing scarcity that has nothing to do with
#    the heat waves this study is about. MTTR stays at 150 h, so the unit still
#    fails in realistic multi-day blocks -- about 1.2 outages a year at ~6 days
#    -- just far less often.
#
# 2. A "Resource ID" column is added. Outage draws are seeded by Cantor pairing
#    a unit's ID with the year, so the ID must identify the same physical unit
#    across every derived config. Assigned once here, from sorted GEN UID, and
#    inherited unchanged by everything derived from RTS-ORB -- so a unit that
#    exists in both RTS-ORB and RTS-ORB-HR sees an identical outage history and
#    the reliability gap between the two configs is purely structural.
NUCLEAR_FOR = 0.02


def make_rts_orb():
    src, dst = config_dir / BASE, config_dir / "RTS-ORB"
    dst.mkdir(parents=True, exist_ok=True)
    for f in CONFIG_FILES:
        shutil.copy2(src / f, dst / f)

    gen = pd.read_csv(dst / "gen.csv")
    nuclear = gen["Unit Type"] == "NUCLEAR"
    before = gen.loc[nuclear, "FOR"].unique()
    gen.loc[nuclear, "FOR"] = NUCLEAR_FOR

    order = {uid: i for i, uid in enumerate(sorted(gen["GEN UID"]))}
    gen["Resource ID"] = gen["GEN UID"].map(order)
    gen.to_csv(dst / "gen.csv", index=False)

    print(f"  RTS-ORB: copied {len(CONFIG_FILES)} files from {BASE}")
    print(f"    NUCLEAR FOR {before} -> {NUCLEAR_FOR} ({int(nuclear.sum())} unit)")
    print(f"    Resource ID 0-{len(gen) - 1} assigned to {len(gen)} units")
    return dst





# --------------------------------------------------------------------------
# RTS-ORB-HR: high-renewables variant
# --------------------------------------------------------------------------
# Built from RTS-ORB, so it inherits the nuclear FOR fix. Three kinds of change:
#
# 1. New wind, solar and storage, sited only at buses that ALREADY host the
#    same technology. VariableResource.get_gen_profile reads profiles per BUS,
#    not per unit, so an addition at an existing bus needs no new profile data
#    and inherits that bus's resource quality. Storage carries no profile, so
#    it can go anywhere.
# 2. Solar sites chosen to rebalance areas. Existing solar is 498/138/2080 MW
#    across areas 1/2/3, with 994 MW at bus 313 alone; piling on more area-3
#    solar would deepen an export constraint rather than add usable capacity.
#    The split here is +600/+300/+600. Buses with solar but no load (312, 324)
#    and the already-saturated 313 are skipped.
# 3. Coal STEAM retired in merit order, most expensive first, to bring the
#    reliability back in line with RTS-ORB. The two largest coal blocks (123,
#    505 MW and 223, 660 MW) are last on the ladder despite mid-range costs:
#    retiring either moves >0.5 GW at one bus, which would make the binding
#    constraint transmission rather than resource adequacy and muddy the
#    interpretation of every blackout event downstream.

NEW_WIND = {122: 125.0, 303: 125.0, 309: 125.0, 317: 125.0}
NEW_SOLAR = {101: 150.0, 103: 150.0, 113: 150.0, 118: 150.0,      # area 1
             213: 150.0, 215: 150.0,                              # area 2
             308: 150.0, 310: 150.0, 314: 150.0, 319: 150.0}      # area 3
NEW_STORAGE = {313: 100.0,                                        # existing storage bus
               115: 100.0, 118: 100.0, 218: 100.0, 318: 100.0,    # 4 largest load buses
               103: 100.0, 113: 100.0, 215: 100.0, 310: 100.0, 319: 100.0}  # new-solar buses
STORAGE_TEMPLATE = "313_STORAGE_1"   # cloned for buses with no storage today

# (bus, fuel) in retirement order, with cumulative MW in the comment.
RETIREMENT_LADDER = [
    (101, "Coal"),   # 152 MW @ $28.1/MWh   cum   152
    (116, "Coal"),   # 155 MW @ $28.0       cum   307
    (201, "Coal"),   #  76 MW @ $27.5       cum   383
    (202, "Coal"),   # 152 MW @ $25.0       cum   535
    (316, "Coal"),   # 155 MW @ $25.0       cum   690
    (102, "Coal"),   # 152 MW @ $24.5       cum   842
    (115, "Coal"),   # 155 MW @ $24.2       cum   997
    (216, "Coal"),   # 155 MW @ $23.0       cum  1152
    (123, "Coal"),   # 505 MW @ $24.4       cum  1657
    (223, "Coal"),   # 660 MW @ $23.2       cum  2317
]


def _add_units(gen, additions, unit_type, source_types=None, template_uid=None):
    """Append one new unit per bus in `additions`, cloning an existing row.

    Cloning rather than building a row from scratch keeps every column we do
    not model (emissions, reactive limits, inertia) consistent with upstream.
    The clone source is a unit at the same bus drawn from `source_types`
    (default: `unit_type` alone), so quality-linked fields come from the right
    place; `template_uid` supplies a fallback for buses that do not host any
    of those types yet. Buses 118, 213 and 308 host only rooftop PV, so RTPV is
    an acceptable source for a new PV unit -- solar profiles are read per bus,
    shared by PV and RTPV, so the distinction is bookkeeping only here.
    """
    source_types = source_types or (unit_type,)
    # New units take IDs past the highest inherited one, so they never collide
    # with a unit that exists in the parent config.
    next_id = int(gen["Resource ID"].max()) + 1
    rows = []
    for bus, mw in sorted(additions.items()):
        same = gen[(gen["Bus ID"] == bus) & (gen["Unit Type"] == unit_type)]
        source = gen[(gen["Bus ID"] == bus) & (gen["Unit Type"].isin(source_types))]
        if len(source):
            row = source.iloc[0].copy()
        elif template_uid is not None:
            row = gen[gen["GEN UID"] == template_uid].iloc[0].copy()
            row["Bus ID"] = bus
        else:
            raise KeyError(f"no {'/'.join(source_types)} at bus {bus} and no template given")

        gen_id = len(same) + 1
        row["GEN UID"] = f"{bus}_{unit_type}_{gen_id}"
        row["Gen ID"] = gen_id
        row["Resource ID"] = next_id
        next_id += 1
        # Force the type fields: the clone source may be a different type.
        row["Unit Type"] = row["Unit Group"] = unit_type
        if len(same):
            row["Category"] = same.iloc[0]["Category"]
        elif unit_type == "PV":
            row["Category"] = "Solar PV"
        for col in ("PMax MW", "Ramp Rate MW/Min", "Base MVA"):
            row[col] = mw
        if unit_type == "STORAGE":
            row["Pump Load MW"] = mw
        rows.append(row)

    return pd.concat([gen, pd.DataFrame(rows)], ignore_index=True)


def _retire(gen, rungs):
    """Drop the first `rungs` entries of the retirement ladder.

    Safe to delete the row outright because outage seeds key off the explicit
    Resource ID, not a unit's position in the fleet -- every surviving unit
    keeps its own draw, so two rungs differ only by the units retired between
    them (common random numbers).

    tune_orb_hr.py instead zeroes nameplate_capacity to switch rungs, since a
    compiled CVXPY problem has a fixed variable set. The two are equivalent:
    a retired unit contributes nothing either way, and a zeroed unit's own
    outage draw is irrelevant because it cannot generate.
    """
    retired = 0.0
    for bus, fuel in RETIREMENT_LADDER[:rungs]:
        mask = (gen["Bus ID"] == bus) & (gen["Unit Type"] == "STEAM") & (gen["Fuel"] == fuel)
        if not mask.any():
            raise KeyError(f"no {fuel} STEAM at bus {bus}")
        retired += gen.loc[mask, "PMax MW"].sum()
        gen = gen[~mask]
    return gen.reset_index(drop=True), retired


def make_rts_orb_hr(rungs=0):
    """Build RTS-ORB-HR from RTS-ORB. `rungs` = coal retirement depth."""
    src, dst = config_dir / "RTS-ORB", config_dir / "RTS-ORB-HR"
    dst.mkdir(parents=True, exist_ok=True)
    for f in CONFIG_FILES:
        shutil.copy2(src / f, dst / f)

    gen = pd.read_csv(src / "gen.csv")
    before = gen.groupby("Unit Type")["PMax MW"].sum()

    gen = _add_units(gen, NEW_WIND, "WIND")
    gen = _add_units(gen, NEW_SOLAR, "PV", source_types=("PV", "RTPV"))
    gen = _add_units(gen, NEW_STORAGE, "STORAGE", template_uid=STORAGE_TEMPLATE)
    gen, retired = _retire(gen, rungs)
    gen.to_csv(dst / "gen.csv", index=False)

    after = gen.groupby("Unit Type")["PMax MW"].sum()
    print(f"  RTS-ORB-HR: derived from RTS-ORB, retirement rung {rungs}")
    print(f"    +{sum(NEW_WIND.values()):.0f} MW wind at {len(NEW_WIND)} buses, "
          f"+{sum(NEW_SOLAR.values()):.0f} MW solar at {len(NEW_SOLAR)}, "
          f"+{sum(NEW_STORAGE.values()):.0f} MW storage at {len(NEW_STORAGE)}")
    print(f"    -{retired:.0f} MW coal STEAM ")
    for ut in ("PV", "WIND", "STORAGE", "STEAM"):
        print(f"      {ut:8} {before.get(ut, 0):7.1f} -> {after.get(ut, 0):7.1f} MW")
    return dst


if __name__ == "__main__":
    import sys
    make_rts_orb()
    make_rts_orb_hr(rungs=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
