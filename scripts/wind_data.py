import numpy as np
import pandas as pd
import pathlib
import shutil
import os
import requests
from windpowerlib import WindTurbine, ModelChain

curr_dir = pathlib.Path(".")
data_dir = curr_dir / "RTS-GMLC-master" / "RTS_Data"
wind_data_dir = curr_dir / "wind-data"

# Read in bus geodata and generation data for IEEE RTS GMLC (Reliability Test System)
df_bus = pd.read_csv(data_dir / "SourceData" / "bus.csv", index_col=[0])
df_geodata = df_bus[["lat", "lng"]]
df_gen_full = pd.read_csv(data_dir / "SourceData" / "gen.csv", index_col=[0])

# Get wind geodata
wind_geodata = df_geodata.loc[df_gen_full.loc[df_gen_full["Unit Type"] == "WIND", "Bus ID"]]

def download_wind_data():

    # Read in API key
    with open("nrel-api-key.txt", "r") as f:
        API_KEY = f.readlines()[0].strip()

    # Build request

    url = (
        "https://developer.nrel.gov/api/wind-toolkit/v2/wind/"
        "wtk-led-climate-v1-0-0-download.csv"
        f"?api_key={API_KEY}"
    )

    for bus_id, row in wind_geodata.iterrows():

        print(f"Retrieving wind data for bus {bus_id}...")

        # Create directory for wind data for all years
        bus_wind_data_dir = wind_data_dir / "inputs" / f"bus{bus_id}"
        bus_wind_data_dir.mkdir(exist_ok=True, parents=True)

        # Extract bus geodata for request
        lat, lon = row.lat, row.lng
        point_wkt = f"POINT({lon} {lat})"

        for year in np.arange(2001, 2021):
            print(f"Year: {year}")
            payload = {
                "wkt": point_wkt,
                "names": str(year),
                "interval": 60,
                "utc": "true",
                "leap_day": "true",
                "attributes": "windspeed_100m,winddirection_100m",
                "email": "charlesgulian@berkeley.edu",
                "full_name": "Charles Gulian",
                "affiliation": "UC Berkeley IEOR",
                "reason": "Research",
                "mailing_list": "false"
            }

            # Send request
            print("Submitting WTK data request to NREL...")
            response = requests.post(url, data=payload, timeout=60)

            if response.status_code == 200:
                print("Request submitted successfully.")

                # Save CSV file
                fname = bus_wind_data_dir / f"WTK-LED_weather-inputs_bus{bus_id}_{year}.csv"
                with open(fname, "wb") as f:
                    f.write(response.content)

                # Re-open CSV with pandas and clean
                df_wind = pd.read_csv(fname, skiprows=1)  # Remove header
                df_wind.columns = [c.strip() for c in df_wind.columns]  # Clean columns names
                df_wind["datetime"] = pd.to_datetime(
                    df_wind[["Year", "Month", "Day", "Hour", "Minute"]])  # Create datetime column
                df_wind = df_wind[["datetime", "wind speed at 100m (m/s)", "wind direction at 100m (deg)"]].set_index(
                    "datetime")  # Restrict to columns of interest + change index
                df_wind.index = pd.to_datetime(df_wind.index)  # Convert to datetime index

                # Save file again
                df_wind.to_csv(fname)

                del response
                del df_wind

            else:
                print(f"Request failed with status {response.status_code}")
                print(response.text)
                break
        print("Done.")


def create_wind_profiles():

    # Create wind profiles

    buses = list(wind_geodata.index)
    for bus in buses:
        bus_data_files = sorted(list((wind_data_dir / "inputs" / f"bus{bus}").glob("*")))
        for f in bus_data_files:
            # Read in / re-format data
            df = pd.read_csv(f, index_col=[0])[["wind speed at 100m (m/s)"]]
            df.index = pd.to_datetime(df.index)
            df.columns = pd.MultiIndex.from_tuples([('wind_speed', 100)])
            # Get normalized power output
            turbine = WindTurbine(
                hub_height=100,
                turbine_type='V112/3000'
            )
            mc = ModelChain(turbine)
            mc.run_model(df)
            df_power = mc.power_output / turbine.nominal_power
            # Save normalized profile
            df_power.name = "profile"
            (data_dir / "profiles" / f"bus{bus}").mkdir(exist_ok=True, parents=True)

            df_power.to_csv(data_dir / "profiles" / f"bus{bus}" / f"{f.stem.replace("weather-inputs", "profile")}.csv")