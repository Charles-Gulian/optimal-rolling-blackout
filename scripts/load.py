import pandas as pd
import cvxpy as cp

from component import Component, T


class Load(Component):
    """Electricity demand of one type at one node.

    A node carries one Load per load type. Today there is a single "baseline"
    type; the eventual split into "hvac" and "other" is the reason demand,
    unserved energy and VOLL live here rather than on Node -- the two types are
    priced differently, which is the whole point of the dynamic VOLL model.

    The blackout decision stays on Node for now, since a feeder outage
    de-energises every load at the bus at once. Moving it here would model
    load-level control, which is a separate question.
    """

    # Profile filename per load type, under load-data/profiles/bus{bus}/.
    PROFILE_PATTERNS = {
        "baseline": "NSRDB_load-profile_bus{bus}_{year}.csv",
        "hvac": "NSRDB_hvac-load-profile_bus{bus}_{year}.csv",
        "other": "NSRDB_other-load-profile_bus{bus}_{year}.csv",
    }

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack load attributes from data
        self.node_ID = data["Bus ID"]
        self.load_type = data["Load Type"]
        self.peak_load = data["Peak MW"]  # MW, this load type's own peak
        self.VOLL = data["VOLL"]  # $/MWh of unserved energy

        # Initialize node, profile
        self.node = None
        self.load_profile = None

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def link_node(self, nodes: dict):
        # Link node to load
        self.node = nodes[self.node_ID]
        # Link load to node
        nodes[self.node_ID].loads.append(self)

    def get_load_profile(self, year):
        fname = self.PROFILE_PATTERNS[self.load_type].format(bus=self.node_ID, year=year)
        ts_data_path = self.system.base_dir / "load-data" / "profiles" / f"bus{self.node_ID}" / fname
        df_load = pd.read_csv(ts_data_path, index_col=[0])
        df_load.index = pd.to_datetime(df_load.index)
        self.load_profile = df_load.squeeze()

    def create_parameters(self):
        self.demand = cp.Parameter(T, nonneg=True)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.demand.value = self.peak_load * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values

    def create_variables(self):
        self.unserved_energy = cp.Variable(T, nonneg=True)

    def write_constraints(self):
        return [self.unserved_energy <= self.demand]  # Limit unserved energy
