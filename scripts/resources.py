import numpy as np
import pandas as pd
import cvxpy as cp

from component import Component, T
from outages import forced_outage_profile, outage_seed


class Resource(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack basic resource attributes from data
        self.node_ID = data["Bus ID"]  # Bus ID
        self.unit_type = data["Unit Type"]  # Unit type
        self.pmax = data["PMax MW"]  # Nameplate capacity (MW)
        self.VOM = data["VOM"]  # Variable O&M costs ($/MWh)

        self.nameplate_capacity = data["PMax MW"]  # pmax is overwritten by a
        # cp.Parameter once the model is built, so keep the constant separately

        # Forced outages. FOR = 0 means the unit is never forcibly out -- true
        # upstream for PV, RTPV, WIND and STORAGE, whose availability is already
        # carried by their profiles.
        self.FOR = float(data.get("FOR", 0.0) or 0.0)
        self.MTTR = float(data.get("MTTR Hr", 0.0) or 0.0)
        self.unit_index = None      # set by System, for reproducible seeding
        self.outage_profile = None

        # Initialize node
        self.node = None

    @property
    def has_forced_outages(self):
        return self.FOR > 0.0 and self.MTTR > 0.0

    def get_outage_profile(self, year, scenario_seed=0):
        """Sample this unit's hourly availability for `year` (1 = up, 0 = out)."""
        if not self.has_forced_outages:
            self.outage_profile = None
            return
        index = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
        rng = np.random.default_rng(outage_seed(self.unit_index, year, scenario_seed))
        self.outage_profile = pd.Series(
            forced_outage_profile(self.FOR, self.MTTR, len(index), rng).astype(float),
            index=index,
        )

    def availability(self, date: pd.Timestamp):
        """(T,) availability over the solve window; all ones if never out."""
        if self.outage_profile is None:
            return np.ones(T)
        return self.outage_profile.loc[date:date + pd.Timedelta(hours=T - 1)].to_numpy()

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def link_node(self, nodes: dict):
        # Link node to resource
        self.node = nodes[self.node_ID]
        # Link resource to node
        nodes[self.node_ID].resources.append(self)

    def create_variables(self):
        self.p_out = cp.Variable(T, nonneg=True)

    def write_constraints(self):
        return [self.p_out <= self.pmax]


class ThermalResource(Resource):

    def __init__(self, name, data):
        super().__init__(name, data)

        # Unpack thermal resource attributes from data

        # Cost attributes
        self.fuel_price = data["Fuel Price $/MMBTU"]
        self.heat_rate = data["HR_avg_0"] / 1000  # BTU/kWh --> MMBTU/MWh
        self.variable_costs = self.fuel_price * self.heat_rate + self.VOM

        # Operational attributes
        self.pmin = data["PMin MW"]  # Minimum output (for unit commitment
        self.ramp_rate = 60 * data["Ramp Rate MW/Min"]

    def create_parameters(self):
        # Time-varying so forced outages can zero it out. Without outages this
        # is just the nameplate repeated.
        self.pmax = cp.Parameter(T, nonneg=True)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.pmax.value = self.nameplate_capacity * self.availability(date)


class VariableResource(Resource):

    def __init__(self, name, data):
        super().__init__(name, data)

        # Resource type
        if self.unit_type in ["PV", "RTPV"]:
            self.resource_type = "solar"
        elif self.unit_type in ["WIND"]:
            self.resource_type = "wind"
        elif self.unit_type in ["HYDRO", "ROR"]:
            self.resource_type = "hydro"
        else:
            print(f"Unknown variable resource type: {self.unit_type}")
            self.resource_type = None

        # Cost attributes
        self.variable_costs = self.VOM

        # Operational attributes
        self.nameplate_capacity = data["PMax MW"]

        # Initialize generation profile
        self.gen_profile = None

    def get_gen_profile(self, year):
        # Infer resource type
        if self.resource_type == "solar":
            data_source = "NSRDB"
        elif self.resource_type == "wind":
            data_source = "WTK-LED"
        ts_data_dir = self.system.base_dir / f"{self.resource_type}-data" / "profiles"
        ts_data_path = ts_data_dir / f"bus{self.node_ID}" / f"{data_source}_profile_bus{self.node_ID}_{year}.csv"
        df_profile = pd.read_csv(ts_data_path, index_col=[0])
        df_profile.index = pd.to_datetime(df_profile.index)
        self.gen_profile = df_profile.squeeze()

    def create_parameters(self):
        self.pmax = cp.Parameter(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        profile = self.gen_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values
        self.pmax.value = self.nameplate_capacity * profile * self.availability(date)


class HydroResource(VariableResource):
    """Hydro and run-of-river.

    Same dispatch behaviour as wind/solar -- zero marginal cost, capped hourly by
    a per-unit profile -- so it inherits the parameter/variable machinery. It
    differs only in the calendar: there is no multi-year hydro data, so every
    simulated year re-uses the single RTS-GMLC profile, which is stamped 2020.

    Profiles are built by data-scripts/hydro_data.py into the same per-bus
    layout the NSRDB/WTK-LED profiles use.
    """

    SOURCE_YEAR = 2020

    def get_gen_profile(self, year):
        ts_data_dir = self.system.base_dir / "hydro-data" / "profiles"
        fname = f"RTS-GMLC_profile_bus{self.node_ID}_{self.SOURCE_YEAR}.csv"
        df_profile = pd.read_csv(ts_data_dir / f"bus{self.node_ID}" / fname, index_col=[0])
        df_profile.index = pd.to_datetime(df_profile.index)
        source = df_profile.squeeze()

        # Re-map the source calendar onto the requested year by (month, day,
        # hour) rather than by position, so seasonality stays aligned. The
        # source year is a leap year, so 29 Feb is there when the target needs
        # it and goes unused when it does not.
        source.index = pd.MultiIndex.from_arrays(
            [source.index.month, source.index.day, source.index.hour]
        )
        source = source[~source.index.duplicated()]

        target = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
        key = pd.MultiIndex.from_arrays([target.month, target.day, target.hour])
        values = source.reindex(key).to_numpy()
        if np.isnan(values).any():
            raise ValueError(f"{self.name}: hydro profile has gaps when re-mapped to {year}")

        self.gen_profile = pd.Series(values, index=target)


class StorageResource(Resource):

    def __init__(self, name, data, duration=4.0):
        super().__init__(name, data)

        # Cost attributes
        self.variable_costs = self.VOM

        # Operational attributes
        self.duration = duration
        self.max_SOC = self.pmax * self.duration
        self.efficiency = data["Storage Roundtrip Efficiency"] / 100

    def create_variables(self):
        self.charge = cp.Variable(T, nonneg=True)
        self.discharge = cp.Variable(T, nonneg=True)
        self.SOC = cp.Variable(T, nonneg=True)

    def create_expressions(self):
        self.p_out = self.discharge - self.charge

    def write_constraints(self):
        constraints = [
            self.charge <= self.pmax,
            self.discharge <= self.pmax,
            self.SOC <= self.max_SOC
        ]
        constraints += [
            self.SOC[t] == self.SOC[np.mod(t - 1, T)] + self.efficiency * self.charge[np.mod(t - 1, T)]
            - self.discharge[np.mod(t - 1, T)]
            for t in range(T)
        ]
        return constraints
