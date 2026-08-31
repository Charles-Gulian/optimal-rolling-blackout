import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import shutil
import os
import requests
import time

import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, SingleAxisTrackerMount
from pvlib.modelchain import ModelChain

curr_dir = pathlib.Path(".")
data_dir = curr_dir / "RTS-GMLC-master" / "RTS_Data"
solar_data_dir = curr_dir / "solar-data"

# Read in bus geodata and generation data for IEEE RTS GMLC (Reliability Test System)
df_bus = pd.read_csv(data_dir / "SourceData" / "bus.csv", index_col=[0])
df_geodata = df_bus[["lat", "lng"]]
df_gen_full = pd.read_csv(data_dir / "SourceData" / "gen.csv", index_col=[0])
solar_geodata = df_geodata.loc[df_gen_full.loc[df_gen_full["Unit Type"].isin(["PV", "RTPV"]), "Bus ID"]]

def download_solar_data():
    # Read in API key
    with open("nrel-api-key.txt", "r") as f:
        API_KEY = f.readlines()[0].strip()

    # Build request

    url = (
        "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download"
        ".csv"
        f"?api_key={API_KEY}"
    )

    for bus_id, row in df_geodata.loc[[211, 307]].iterrows():

        print(f"Retrieving solar data for bus {bus_id}...")

        # Create directory for solar data for all years
        bus_solar_data_dir = solar_data_dir / "inputs" / f"bus{bus_id}"
        bus_solar_data_dir.mkdir(exist_ok=True, parents=True)

        # Extract bus geodata for request
        lat, lon = row.lat, row.lng
        point_wkt = f"POINT({lon} {lat})"

        for year in np.arange(1998, 2024):
            print(f"Year: {year}")
            # CSV filename
            fname = bus_solar_data_dir / f"NSRDB_weather-inputs_bus{bus_id}_{year}.csv"
            if fname.exists():
                print(f"File {fname} exists - skipping.")
                continue
            else:
                payload = {
                    "wkt": point_wkt,
                    "names": str(year),
                    "interval": 30,
                    "utc": "true",
                    "leap_day": "true",
                    "attributes": "ghi,dni,dhi,air_temperature,wind_speed,relative_humidity,surface_albedo",
                    "email": "charlesgulian@berkeley.edu",
                    "full_name": "Charles Gulian",
                    "affiliation": "UC Berkeley IEOR",
                    "reason": "Research",
                    "mailing_list": "false"
                }

                # Send request
                print("Submitting NSRDB data request to NREL...")
                response = requests.post(url, data=payload, timeout=120)

                if response.status_code == 200:
                    print("Request submitted successfully.")

                    # Save CSV file
                    with open(fname, "wb") as f:
                        f.write(response.content)

                    # Re-open CSV with pandas and clean
                    df_solar = pd.read_csv(fname, skiprows=2)  # Remove header
                    df_solar.columns = [c.strip() for c in df_solar.columns]  # Clean columns names
                    df_solar["datetime"] = pd.to_datetime(
                        df_solar[["Year", "Month", "Day", "Hour", "Minute"]])  # Create datetime column
                    df_solar = df_solar[
                        ["datetime", "GHI", "DNI", "DHI", "Temperature", "Wind Speed", "Relative Humidity",
                         "Surface Albedo"]].set_index("datetime")  # Restrict to columns of interest + change index
                    df_solar.index = pd.to_datetime(df_solar.index)  # Convert to datetime index
                    df_solar = df_solar.resample("h").asfreq()  # Re-sample to hourly frequency

                    # Save file again
                    df_solar.to_csv(fname)

                    del response
                    del df_solar
                    time.sleep(2)

                else:
                    print(f"Request failed with status {response.status_code}")
                    print(response.text)
                    continue

        time.sleep(2)
        print("Done.")

def create_solar_profiles():

    # Create solar profiles

    for bus, row in solar_geodata.iterrows():
        lat, lon = row.lat, row.lng

        # Get location
        site = pvlib.location.Location(lat, lon, "UTC")

        # Model generic parameters
        azimuth = 180
        module_params = {"pdc0": 1, "gamma_pdc": -0.0033}
        inverter_params = {"pdc0": 1, "pac0": 1}

        # Model single-axis tracking
        mount = SingleAxisTrackerMount(
            axis_tilt=0,
            axis_azimuth=0,
            max_angle=60,
            backtrack=True,
            gcr=0.33,
            racking_model="open_rack",
        )

        # Create array
        array = Array(
            mount=mount,
            module_parameters=module_params,
            module_type="glass_polymer",
        )

        # Create system
        system = PVSystem(
            arrays=array,
            inverter_parameters=inverter_params,
        )

        # Create model chain
        mc = ModelChain(
            system,
            site,
            aoi_model='no_loss',
            spectral_model='no_loss',
            temperature_model=None,
            losses_model='no_loss'
        )

        # Create profiles
        for year in np.arange(1998, 2024):
            # Read in weather data
            bus_solar_data_dir = solar_data_dir / "inputs" / f"bus{bus}"
            fname = bus_solar_data_dir / f"NSRDB_weather-inputs_bus{bus}_{year}.csv"
            df_weather = pd.read_csv(fname, index_col=[0])
            df_weather.index = pd.to_datetime(df_weather.index)
            df_weather_mc = df_weather.rename(columns={
                "GHI": "ghi",
                "DNI": "dni",
                "DHI": "dhi",
                "Temperature": "temp_air"
            })

            # Run the model with your measured weather data
            mc.run_model(df_weather_mc)

            # Convert and save
            solar_mw = mc.results.ac
            bus_solar_profiles_dir = solar_data_dir / "profiles" / f"bus{bus}"
            bus_solar_profiles_dir.mkdir(exist_ok=True, parents=True)
            solar_mw.to_csv(bus_solar_profiles_dir / f"NSRDB_profile_bus{bus}_{year}.csv")