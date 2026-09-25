import datetime

import numpy as np
import pandas as pd
import pathlib
import warnings
warnings.simplefilter("ignore", FutureWarning)

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import XGBModel

curr_dir = pathlib.Path(".")
data_dir = curr_dir / "RTS-GMLC-master" / "RTS_Data"
load_data_dir = curr_dir / "load-data"

# Years for training data
training_years = [2019, 2020, 2021, 2022, 2023]

# Buses for training data
training_buses = [111, 211, 311]  # Subset of buses to train on

def get_targets(years):
    # Read in historical regional demand
    df = pd.read_csv(load_data_dir / "historical" / "EIA_hourly_load_2019_2025.csv")
    df_load = df.loc[df["type-name"] == "Demand"]
    df_load = df_load.pivot(index=["datetime"], columns=["respondent"], values=["value"])
    df_load.columns = df_load.columns.droplevel(0)
    df_load.index = pd.to_datetime(df_load.index)

    # Get subset of data for given years
    df_sw = df_load["SW"] # Southwest U.S. regional demand from EIA website
    df_sw = df_sw.loc[df_sw.index.year.isin(years)]

    return df_sw

def get_features(years, buses):
    # Get NSRDB weather data for buses (and additional features)
    df_features_buses = {}
    for bus in buses:
        df_weather_concat = pd.DataFrame()
        for year in years:
            fname = load_data_dir / "inputs" / f"bus{bus}" / f"NSRDB_weather-inputs_bus{bus}_{year}.csv"
            if not fname.exists():
                continue
            df_weather = pd.read_csv(fname, index_col=["datetime"])
            df_weather.index = pd.to_datetime(df_weather.index)

            # Get subset of columns needed for load calculation
            df_weather = df_weather[["Temperature", "Relative Humidity", "GHI", "Wind Speed"]]
            df_weather.columns = ["temperature", "humidity", "radiation_global_horizontal", "wind_speed_2m"]
            df_weather_concat = pd.concat([df_weather_concat, df_weather])
        df_weather = df_weather_concat.copy()

        # Create features
        df_features = df_weather.copy()

        # Calendar features
        df_features["hour_angle_sin"] = np.sin(2 * np.pi * df_features.index.hour / 24)
        df_features["hour_angle_cos"] = np.cos(2 * np.pi * df_features.index.hour / 24)
        df_features["year_angle_sin"] = np.sin(2 * np.pi * df_features.index.dayofyear / 365)
        df_features["year_angle_cos"] = np.cos(2 * np.pi * df_features.index.dayofyear / 365)

        # Other weather features
        df_features["temperature^2"] = df_features["temperature"] ** 2
        df_features["temperature^3"] = df_features["temperature"] ** 3
        df_features["temperature-humidity"] = df_features["temperature"] * df_features["humidity"]
        df_features["temperature-radiation"] = df_features["temperature"] * df_features["radiation_global_horizontal"]

        # Save features for bus
        df_features_buses[bus] = df_features

    return df_features_buses

class LoadModel:

    def __init__(self):
        self.scaler_d = None
        self.model = None

    @staticmethod
    def features_cov_func(df_features):
        return TimeSeries.from_dataframe(
            df_features,
            value_cols=[
                "hour_angle_sin",
                "hour_angle_cos",
                "year_angle_sin",
                "year_angle_cos",
                "temperature",
                "temperature^2",
                "temperature^3",
                "humidity",
                "temperature-humidity",
                "wind_speed_2m",
                "radiation_global_horizontal",
                "temperature-radiation",
            ],
        )

    def train_model(self, buses, years, save=False):

        # Get training data
        df_features_buses, df_sw = get_features(years, buses), get_targets(years)

        # Demand and features
        regional_demand = TimeSeries.from_series(df_sw)

        # Create lists of time series
        demand_list = [regional_demand.copy() for _ in buses]
        features_cov_list = [self.features_cov_func(df_features_buses[bus]) for bus in buses]

        # Create weights
        df_weight = (1 + 4 * (df_sw > df_sw.quantile(0.95)))
        weights = TimeSeries.from_series(df_weight)

        # Normalize data
        scalers_d = []
        scalers_f = []
        demand_scaled_list = []
        features_scaled_list = []

        # Scale data
        for demand, features in zip(demand_list, features_cov_list):
            sd = Scaler()
            sf = Scaler()
            demand_scaled_list.append(sd.fit_transform(demand))
            features_scaled_list.append(sf.fit_transform(features))
            scalers_d.append(sd)
            scalers_f.append(sf)

        self.scaler_d = scalers_d[0] # Demand scaler

        # Train/validation split
        split_after = pd.Timestamp("2022-12-31 23:00:00")

        # Define lists
        train_list, val_list = [], []
        features_train_list, features_val_list = [], []
        for d, f in zip(demand_scaled_list, features_scaled_list):
            # Split data
            train_d, val_d = d.split_after(split_after)
            train_f, val_f = f.split_after(split_after)
            # Save to lists
            train_list.append(train_d)
            val_list.append(val_d)
            features_train_list.append(train_f)
            features_val_list.append(val_f)

        # Split weights
        weights_train, _ = weights.split_after(split_after)
        weights_train_list = [weights_train for _ in buses]

        # XGBoost Model (ChatGPT - modify this part)
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        model = XGBModel(
            lags=None,
            lags_future_covariates=(48, 12),
            output_chunk_length=1,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            likelihood="quantile",
            quantiles=quantiles,
        )

        # Train
        model.fit(
            series=train_list,
            future_covariates=features_train_list,
            sample_weight=weights_train_list,
        )

        self.model = model

        if save:
            # Save model
            model_path = load_data_dir / "models" / f"XGBoost_demand_probabilistic_model_{str(datetime.datetime.now().date())}"
            model.save(str(model_path))

    @staticmethod
    def markov_uniform(T):
        # Define Markov process for sampling different quantiles
        rho = 0.9  # correlation parameter
        u = np.zeros(T)
        u[0] = np.random.rand()
        for t in range(1, T):
            v = np.random.rand()
            u[t] = rho * u[t-1] + (1 - rho) * v
        return u

    def get_load_profiles(self, buses, years):

        # Get features
        df_features_buses = get_features(years, buses)

        # Create lists of time series
        features_cov_dict = {bus: self.features_cov_func(df_features_buses[bus]) for bus in buses}

        # Normalize data
        features_scaled_dict = {}

        # Scale data
        for bus, features in features_cov_dict.items():
            sf = Scaler()
            features_scaled_dict[bus] = sf.fit_transform(features)

        # Create load seed (placeholder with structural requirements)
        start_forecast = pd.Timestamp(f"{min(years)}-01-01 00:00:00")
        end_forecast = pd.Timestamp(f"{max(years)}-12-31 11:59:59")
        load_seed = TimeSeries.from_times_and_values(
            times=pd.date_range(
                start=start_forecast - pd.Timedelta(hours=48),
                end=end_forecast,
                freq="h"
            ),
            values=np.zeros((int((end_forecast - start_forecast) / pd.Timedelta(hours=1)) + 49, 1)),
        )

        bus_load_profiles = {}
        for bus in buses:
            # Generate historical forecasts over the validation period
            prob_forecast = self.model.historical_forecasts(
                series=load_seed,
                future_covariates=features_scaled_dict[bus],
                forecast_horizon=1,
                start=pd.Timestamp(f"{min(years)}-01-01 00:00:00"),
                stride=1,
                retrain=False,
                verbose=True,
                last_points_only=True,
                predict_likelihood_parameters=True,
            )

            # Re-scale
            prob_forecast = self.scaler_d.inverse_transform(prob_forecast)

            # Convert to dataframe
            prob_forecast = prob_forecast.to_dataframe()

            # Sample quantiles of the probabilistic forecast randomly
            probs = np.cumsum([0.05, 0.2, 0.5, 0.2, 0.05])
            quantile_cols = prob_forecast.columns
            np.random.seed(0)
            u_arr = self.markov_uniform(len(prob_forecast))

            # Map uniform random number to quantile column
            def select_quantile(u):
                for q_col, prob in zip(quantile_cols, probs):
                    if u <= prob:
                        return q_col

            # Vectorized mapping
            selected_quantiles = np.vectorize(select_quantile)(u_arr)

            # Sample the forecast

            # Melt the DataFrame so each row is (timestamp, quantile, value)
            prob_forecast_melted = prob_forecast.reset_index().melt(
                id_vars="index", var_name="quantile", value_name="value"
            )

            # Create a DataFrame mapping each timestamp to the randomly selected quantile
            selection_df = pd.DataFrame({
                "index": prob_forecast.index,
                "quantile": selected_quantiles
            })

            # Merge to select the corresponding forecast values
            forecast = selection_df.merge(
                prob_forecast_melted,
                on=["index", "quantile"],
                how="left"
            ).set_index("index")["value"]

            # Convert back to series
            #forecast = TimeSeries.from_series(forecast)

            # Save as bus load profile
            bus_load_profiles[bus] = forecast

        return bus_load_profiles


# def create_load_profiles():
#
#     ### Create load profiles; normalize to 2006 load data
#
#     # Read in list of buses
#     df_bus = pd.read_csv(curr_dir / "RTS-GMLC-master" / "RTS_Data" / "SourceData" / "bus.csv", index_col=[0])
#
#     for bus in df_bus.index:
#         print(bus)
#         skip_bus = False
#         years = np.arange(1998, 2024)
#         df_in_concat = pd.DataFrame()
#         for year in years:
#             fname = load_data_dir / "inputs" / f"bus{bus}" / f"NSRDB_weather-inputs_bus{bus}_{year}.csv"
#             if not fname.exists():
#                 skip_bus = True
#                 continue
#             df_in = pd.read_csv(fname, index_col=["datetime"])
#             df_in.index = pd.to_datetime(df_in.index)
#
#             # Get subset of columns needed for load calculation in demand ninja
#             df_in = df_in[["Temperature", "Relative Humidity", "GHI", "Wind Speed"]]
#             df_in.columns = ["temperature", "humidity", "radiation_global_horizontal", "wind_speed_2m"]
#             df_in_concat = pd.concat([df_in_concat, df_in])
#         df_in = df_in_concat.copy()
#
#         if skip_bus:
#             print("Skipping bus", bus)
#             continue
#
#         # Calculate demand with demand ninja
#         df_out = demand(
#             df_in,
#             base_power=10,
#             heating_power=0.8,
#             cooling_power=1.2,
#             cooling_threshold=16.0,
#             heating_threshold=24.0,
#             humidity_discomfort=0.003,
#             solar_gains=0.01,
#             smoothing=0.5,
#         )["total_demand"]
#
#         # Normalize by peak load in 2006
#         peak_2006 = df_out.loc[df_out.index.year == 2006].max()
#         df_out /= peak_2006
#
#         # Save data
#         # Profiles directory
#         bus_dir = load_data_dir / "profiles" / f"bus{bus}"
#         bus_dir.mkdir(exist_ok=True, parents=True)
#         for year in years:
#             df_out_year = df_out.loc[df_out.index.year == year]
#             df_out_year.to_csv(bus_dir / f"NSRDB_load-profile_bus{bus}_{year}.csv")