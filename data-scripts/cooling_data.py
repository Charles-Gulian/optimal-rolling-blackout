"""
Generate per-bus cooling load profiles with demand.ninja.

Reads the NSRDB weather inputs already on disk and writes normalized cooling
profiles in the same layout as the load/solar/wind profiles:

    data/load-data/profiles/bus{N}/NSRDB_cooling-load-profile_bus{N}_{year}.csv

Only *cooling* is written. Heating is left to fold into the "other" load type
later: the dynamic-VOLL thermal state is one-sided (overheating), so a load
type carrying both would be priced by a cooling-oriented state that means
nothing in January. Heating and cooling never overlap in an hour, so splitting
them cannot double count.

Parameters
----------
Defaults below follow Staffell's documented ranges (see example_1.r in
github.com/iain-staffell/demand_ninja), with two deliberate departures for the
Desert Southwest:

  humidity_discomfort = 0.0  (default 0.05, published range 0.00-0.10)
      BAIT scales discomfort by (1 + (q - q_typical)*h) where
      q_typical = exp(1.1 + 0.06*T) is a *global average* humidity for that
      temperature. This region is far drier: at 40 C we measure 6.9 g/kg
      against a setpoint of 33. With h = 0.05 the multiplier goes negative, so
      BAIT falls as temperature rises -- on the hottest day of 2010 at bus 101,
      BAIT reads 24.0 C instead of 35.6 C, erasing two thirds of the cooling
      signal exactly when it matters. Zero removes the adjustment entirely.

  cooling_threshold = 18.0   (default 20, published range 17-23)
      Low balance point for a hot region with near-universal air conditioning.

base_power is 0: this profile is cooling only. heating_power/cooling_power set
magnitudes that the normalisation below divides out, so they do not affect the
result -- only the shape does.

Normalisation
-------------
Each bus is normalised by its own maximum across the whole year range, so the
profile is per-unit of that bus's multi-year peak cooling demand and `Peak MW`
in load.csv is that peak in MW.

NOTE this differs from the existing "baseline" profiles, which were normalised
by each bus's 2006 peak. Peaks across load types are therefore not directly
comparable until we harmonise the two conventions.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import demand_ninja  # noqa: E402

curr_dir = pathlib.Path(__file__).resolve().parents[1]
load_data_dir = curr_dir / "data" / "load-data"

YEARS = range(2001, 2021)  # the window with complete load/solar/wind coverage

# TIMEZONE. The NSRDB inputs are stored in UTC (data-scripts/solar_data.py
# requests "utc": "true"), and so are the baseline/solar profiles. demand.ninja
# assumes LOCAL time: it applies its diurnal profile by index.hour and resamples
# to daily. Feeding it UTC put the cooling peak at 16:00 UTC = 09:00 local, a
# 7-hour error. So convert to local before calling demand(), then convert the
# result back to UTC for storage so every profile shares one clock.
LOCAL_UTC_OFFSET = pd.Timedelta(hours=-7)  # Arizona MST, no DST

BAIT_PARAMS = dict(
    smoothing=0.50,             # default; published range 0.20-0.80
    solar_gains=0.012,          # default; published range 0.004-0.020
    wind_chill=-0.20,           # default; published range -0.05 to -0.35
    humidity_discomfort=0.0,    # see module docstring
)
ENERGY_PARAMS = dict(
    heating_threshold=14.0,     # default; published range 11-17
    cooling_threshold=18.0,     # see module docstring
    base_power=0.0,             # cooling only
    heating_power=0.0,          # not written; skips the HDD branch entirely
    cooling_power=1.0,          # magnitude is normalised out below
)


def specific_humidity(temperature, relative_humidity, pressure_hpa=1013.25):
    """Relative humidity (%) -> specific humidity (g/kg), via Magnus.

    demand.ninja wants g of water per kg of air; NSRDB reports RH. Standard sea
    level pressure is assumed -- the RTS-GMLC footprint sits at 300-1500 m, so
    this is a few percent off, which is immaterial while humidity_discomfort is
    zero but would want revisiting if it were not.
    """
    es = 6.112 * np.exp(17.67 * temperature / (temperature + 243.5))  # hPa
    e = relative_humidity / 100.0 * es
    return 622.0 * e / (pressure_hpa - e)


def read_weather_inputs(bus, year):
    """Our NSRDB inputs -> the four columns demand.ninja requires."""
    fname = load_data_dir / "inputs" / f"bus{bus}" / f"NSRDB_weather-inputs_bus{bus}_{year}.csv"
    df = pd.read_csv(fname, index_col=[0])
    df.index = pd.to_datetime(df.index)
    return pd.DataFrame(
        {
            "temperature": df["Temperature"],
            "humidity": specific_humidity(df["Temperature"].values,
                                          df["Relative Humidity"].values),
            "radiation_global_horizontal": df["GHI"],
            "wind_speed_2m": df["Wind Speed"],  # NSRDB wind speed is already at 2 m
        },
        index=df.index,
    )


def cooling_demand(bus, year):
    """Hourly cooling demand for one bus-year, returned on the UTC clock.

    Reads UTC, shifts to local for demand.ninja, shifts the answer back. The
    shift has to happen before the call, not after: demand.ninja groups into
    days and applies an hour-of-day profile, so both operations need the local
    calendar.
    """
    weather = read_weather_inputs(bus, year)          # UTC
    weather.index = weather.index + LOCAL_UTC_OFFSET  # -> local
    result = demand_ninja.demand(weather, **BAIT_PARAMS, **ENERGY_PARAMS)
    cooling = result["cooling_demand"]
    cooling.index = cooling.index - LOCAL_UTC_OFFSET  # -> back to UTC
    return cooling


def load_buses(system_config="RTS-GMLC"):
    """Buses carrying non-zero load. Cooling coverage matches baseline coverage."""
    df_bus = pd.read_csv(curr_dir / "system-config" / system_config / "bus.csv")
    return sorted(int(r["Bus ID"]) for _, r in df_bus.iterrows() if r["MW Load"] > 0)


def create_cooling_profiles(buses=None, years=YEARS):
    if buses is None:
        buses = load_buses()

    for bus in buses:
        # Two passes: build every year first so the normalisation is over the
        # whole record rather than per year.
        by_year = {}
        for year in years:
            try:
                by_year[year] = cooling_demand(bus, year)
            except FileNotFoundError:
                continue
        if not by_year:
            print(f"  bus{bus}: no weather inputs, skipped")
            continue

        peak = max(s.max() for s in by_year.values())
        if peak <= 0:
            print(f"  bus{bus}: no cooling demand, skipped")
            continue

        bus_dir = load_data_dir / "profiles" / f"bus{bus}"
        bus_dir.mkdir(exist_ok=True, parents=True)
        for year, series in by_year.items():
            profile = (series / peak).rename("profile")
            profile.index.name = "datetime"
            profile.to_csv(bus_dir / f"NSRDB_cooling-load-profile_bus{bus}_{year}.csv")

        print(f"  bus{bus}: {len(by_year)} years, peak demand {peak:.3f} "
              f"(normalised to 1.0), mean load factor {np.mean([s.mean() for s in by_year.values()])/peak:.3f}")


if __name__ == "__main__":
    create_cooling_profiles()
