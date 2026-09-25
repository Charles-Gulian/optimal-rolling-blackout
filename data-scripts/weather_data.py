import numpy as np
import pandas as pd
import pathlib
import shutil
import os

import cdsapi
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import zipfile
import xarray as xr

curr_dir = pathlib.Path(".")
processed_dir = curr_dir / "weather-data" / "processed"

# Read in bus/line geodata for IEEE RTS GMLC (Reliability Test System)
df_bus = pd.read_csv(curr_dir / "RTS-GMLC-master" / "RTS_Data" / "SourceData" / "bus.csv", index_col=[0])
df_geodata = df_bus[["lat", "lng"]]
def download_weather_data():

    # Get study area
    lat_min = df_geodata["lat"].min() - 0.25
    lat_max = df_geodata["lat"].max() + 0.25
    lon_min = df_geodata["lng"].min() - 0.25
    lon_max = df_geodata["lng"].max() + 0.25
    area = [lat_max, lon_min, lat_min, lon_max]

    # Initialize the CDS API client
    c = cdsapi.Client()

    # Define the dataset and request parameters
    dataset = "reanalysis-era5-single-levels"

    # Define chunks
    months = [str(m).zfill(2) for m in np.arange(12) + 1]
    chunks = [months[0:4], months[4:8], months[8:12]]

    # Define coverage years
    years = np.arange(1982, 2025)
    for y in years:
        for j, chunk in enumerate(chunks):
            print(f"Downloading data for {y} - chunk {j + 1} {chunk}...")
            # Formulate request
            request = {
                "product_type": "reanalysis",
                "variable": ["2m_temperature", "clear_sky_direct_solar_radiation_at_surface",
                             "surface_solar_radiation_downwards", "total_cloud_cover", "100m_u_component_of_wind",
                             "100m_v_component_of_wind"],
                "year": [str(y)],
                "month": [str(m).zfill(2) for m in chunk],
                "day": [str(d + 1).zfill(2) for d in np.arange(31)],
                "time": [f"{str(h).zfill(2)}:00" for h in range(24)],
                "area": area,  # [North, West, South, East] # IEEE RTS GMLC Test System coordinates
                "format": "netcdf"  # File format (NetCDF or GRIB)
            }
            # Retrieve data
            fname = f"weather-data/raw/weather-data-{y}-chunk{j + 1}.nc"
            if not os.path.exists(fname):
                c.retrieve(dataset, request).download(fname)
                print("Done.")
                time.sleep(10)
            else:
                print(f"File {fname} already exists, skipping.")

    print("Data download complete!")

def process_weather_data():
    # Extract data from zip files
    extract_to = curr_dir / "weather-data" / "extracted-data"
    zip_files = sorted(list((curr_dir / "weather-data" / "raw").glob("*.nc")))
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            zip_ref.extractall(extract_to)
            for file_name in extracted_files:
                file_path = extract_to / file_name
                data_type = file_path.stem.split("-")[-1]
                file_path.rename(
                    curr_dir / "weather-data" / f"{data_type}-data" / (zip_path.stem + "-" + data_type + ".nc"))
    print("Extraction complete.")

def create_weather_profiles(ftype):

    # Open instant data files and sort chronologically
    assert ftype in ["accum", "instant"]  # accum or instant
    data_files = sorted(list((curr_dir / "weather-data" / f"{ftype}-data").glob("*.nc")))
    ds = xr.open_mfdataset(
        data_files,
        combine="by_coords",
        engine="netcdf4",
        chunks={"time": 1000},
    )

    # Create weather profiles

    for bus in df_geodata.index:
        # Get geodata
        lat, lng = df_geodata.loc[bus, ["lat", "lng"]]

        # Get subset of data for this location
        ds_bus = ds.sel(
            latitude=lat,
            longitude=lng,
            method="nearest"
        )

        # Get years
        years = pd.to_datetime(ds_bus.valid_time.values).year
        years = sorted(set(years))

        # Loop over years
        for year in years:
            # Get data for this year
            ds_bus_year = ds_bus.sel(valid_time=str(year))

            # Convert to DataFrame
            df_bus_year = ds_bus_year.to_dataframe().reset_index()
            df_bus_year = df_bus_year.set_index("valid_time")

            # Save to processed data
            (processed_dir / str(bus)).mkdir(exist_ok=True, parents=True)
            df_bus_year.to_csv(processed_dir / str(bus) / f"ERA5_{ftype}-profile_bus{bus}_{year}.csv")