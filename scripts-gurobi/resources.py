import numpy as np
import pandas as pd
import gurobipy as gp

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

        self.nameplate_capacity = data["PMax MW"]  # pmax is overwritten by the
        # per-hour array once the model is built, so keep the constant separately

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
        self.p_out = self.model.addMVar(T, lb=0.0, name=f"p_out[{self.name}]")

    def write_constraints(self):
        # The pmax cap is carried by p_out's upper bound, set each solve.
        pass


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
        self.pmax = np.zeros(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.pmax = self.nameplate_capacity * self.availability(date)
        self.p_out.UB = self.pmax


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

    PROFILE_SOURCE = "RTS-GMLC"

    def get_gen_profile(self, year):
        """Read this generator's per-unit hourly profile, keyed by GEN UID."""
        ts_data_dir = (self.system.base_dir / f"{self.resource_type}-data"
                       / "profiles" / f"bus{self.node_ID}")
        fname = f"{self.PROFILE_SOURCE}_profile_{self.name}_{year}.csv"
        df_profile = pd.read_csv(ts_data_dir / fname, index_col=[0])
        df_profile.index = pd.to_datetime(df_profile.index)
        self.gen_profile = df_profile.squeeze()

    def create_parameters(self):
        self.pmax = np.zeros(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        profile = self.gen_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values
        self.pmax = self.nameplate_capacity * profile * self.availability(date)
        self.p_out.UB = self.pmax


class HydroResource(VariableResource):
    """Hydro and run-of-river. Dispatches as wind and solar do."""

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
        self.charge = self.model.addMVar(
            T, lb=0.0, ub=self.pmax, name=f"charge[{self.name}]")
        self.discharge = self.model.addMVar(
            T, lb=0.0, ub=self.pmax, name=f"discharge[{self.name}]")
        # Free, so the SOC limits can be explicit rows whose duals are readable.
        self.SOC = self.model.addMVar(
            T, lb=-gp.GRB.INFINITY, name=f"SOC[{self.name}]")

    def create_expressions(self):
        self.p_out = self.discharge - self.charge

    def write_constraints(self):
        m = self.model
        self.SOC_min_constraint = m.addConstr(
            self.SOC >= 0.0, name=f"SOC_min[{self.name}]")
        self.SOC_max_constraint = m.addConstr(
            self.SOC <= self.max_SOC, name=f"SOC_max[{self.name}]")
        soc, chg, dis = self.SOC.tolist(), self.charge.tolist(), self.discharge.tolist()
        self.soc_balance_constraints = [
            m.addConstr(soc[t] == soc[(t - 1) % T] + self.efficiency * chg[(t - 1) % T]
                        - dis[(t - 1) % T], name=f"SOC_bal[{self.name},{t}]")
            for t in range(T)
        ]

    @property
    def SOC_min_duals(self):
        return self.SOC_min_constraint.Pi

    @property
    def SOC_max_duals(self):
        return self.SOC_max_constraint.Pi
