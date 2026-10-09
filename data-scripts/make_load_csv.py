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

import numpy as np
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
PEAK_SCALING = {"RTS-GMLC": 1.0, "RTS-ORB": 1.10, "RTS-ORB-HR": 1.10}

# --------------------------------------------------------------------------
# Dynamic VOLL: Wang et al. (2021) ERL 16 074003 1R-1C thermal model.
# --------------------------------------------------------------------------
# Only cooling load carries these; "other" load gets blanks and keeps the flat
# VOLL above. See scripts/thermal.py for the derivation of each quantity.
#
# From Wang -- properties of the building, independent of thermostat choice:
#   Tau Hr        Thermal time constant RC. Wang Fig. 5/6 puts cooling-season
#                 medians at 14.2-17.3 h across Californian cities.
#   T Eq degC     Equivalent temperature of solar + internal heat gains, added
#                 to dry-bulb to get T_eff (Wang Fig. 6 median).
#
# Ours -- the valuation, which Wang does not address:
#   T Set degC    Cooling setpoint, 21 degC (69.8 F).
#   Rho $/MWh/degC  How fast VOLL rises above setpoint. Derived from a stated
#                 willingness to pay ~$5 per degF of indoor excess (so $50 at
#                 80 F, $100 at 90 F against a 70 F setpoint), divided by the
#                 ~2 kWh an AC would have drawn over that hour:
#                     $5/degF / 2 kWh = $2500/MWh/degF = $4500/MWh/degC
#                 rounded to 5000. NOTE the cost term is rho * d_t * w_t, so
#                 the premium scales with the actual cooling demand that hour,
#                 not the 2 kW assumed here -- it matches the intended WTP at
#                 the design peak and runs below it on milder evenings.
#
# Discomfort starts AT the setpoint, with no separate threshold. Wang's 28 degC
# overheating limit is a health threshold; this is a revealed preference -- if
# you were indifferent to being warmer you would have set the thermostat
# higher. That also makes the model simpler (no hinge variable) and means a
# single-hour outage registers, where a 4 degC knee needed three consecutive
# hours before anything did.
WANG_TAU_HR = 15.0
WANG_T_EQ = 12.0
T_SET = 21.0                           # degC, i.e. 69.8 F
RHO = 5000.0
SIZING_MARGIN = 1.15
# ASHRAE 0.4% cooling design condition = 99.6th percentile of hourly dry-bulb.
DESIGN_QUANTILE = 0.996
WEATHER_YEARS = range(2001, 2021)

weather_dir = curr_dir / "data" / "load-data" / "inputs"


def design_temperature(bus, years=WEATHER_YEARS, quantile=DESIGN_QUANTILE):
    """Bus-specific ASHRAE 0.4% cooling design temperature, degC.

    Sizing every AC off one number would undersize equipment at the desert
    buses and oversize it at the cooler ones, which directly distorts how fast
    each bus overheats during an outage. The NSRDB pull already carries the
    dry-bulb series, so use it.

    TIMEZONE: these inputs are on the UTC clock, like the load profiles. A
    whole-year percentile is clock-invariant anyway, but do not add a local
    shift here -- that is the bug that put the cooling peak at 09:00 local.
    """
    temps = []
    for year in years:
        f = weather_dir / f"bus{bus}" / f"NSRDB_weather-inputs_bus{bus}_{year}.csv"
        temps.append(pd.read_csv(f, usecols=["Temperature"])["Temperature"])
    return float(pd.concat(temps).quantile(quantile))


def thermal_params(bus):
    """Wang parameters for the cooling load at one bus."""
    a = float(np.exp(-1.0 / WANG_TAU_HR))
    t_design = design_temperature(bus)
    gbar = SIZING_MARGIN * (1.0 - a) * (t_design + WANG_T_EQ - T_SET)
    return {
        "Tau Hr": WANG_TAU_HR,
        "T Set degC": T_SET,
        "T Eq degC": WANG_T_EQ,
        "Gbar degC/Hr": round(gbar, 4),
        "Rho $/MWh/degC": RHO,
    }



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
                **(thermal_params(bus) if load_type == "cooling" else {}),
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
    # Regenerate every config we know a scaling for, so one invocation leaves
    # all of system-config/ consistent with this file.
    for config in PEAK_SCALING:
        make_load_csv(config)
