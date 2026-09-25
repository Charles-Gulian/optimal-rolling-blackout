import numpy as np
import pandas as pd
import cvxpy as cp

from component import Component, T


class Node(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack node attributes from data
        self.peak_load = data["MW Load"]
        self.node_type = data["Bus Type"]
        self.lat = data["lat"]
        self.long = data["lng"]
        self.area = data["Area"]
        self.sub_area = data["Sub Area"]
        self.zone = data["Zone"]

        # Set other, generic attributes
        self.VOLL = 1e4  # $/MWh of unserved energy
        self.theta_max = np.deg2rad(30)  # 30 degree maximum voltage angle

        # Initialize resources
        self.resources = []

        # Initialize in/out lines
        self.in_lines = []
        self.out_lines = []

        # Initialize load
        self.load_profile = None

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def get_load_profile(self, year):
        data_source = "NSRDB"
        ts_data_dir = self.system.base_dir / "load-data" / "profiles"
        ts_data_path = ts_data_dir / f"bus{self.name}" / f"{data_source}_load-profile_bus{self.name}_{year}.csv"
        df_load = pd.read_csv(ts_data_path, index_col=[0])
        df_load.index = pd.to_datetime(df_load.index)
        self.load_profile = df_load.squeeze()

    def create_parameters(self):
        self.load = cp.Parameter(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.load.value = self.peak_load * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values

    def create_variables(self):
        self.unserved_energy = cp.Variable(T, nonneg=True)
        self.theta = cp.Variable(T)

    def write_constraints(self):
        constraints = [
            cp.sum([r.p_out for r in self.resources])  # Nodal generation
            + cp.sum([l.flow for l in self.in_lines])  # + line inflows
            - cp.sum([l.flow for l in self.out_lines])  # - line outflows
            == self.load - self.unserved_energy,  # = load minus unserved energy (power balance constraint)
            self.unserved_energy <= self.load,  # Limit unserved energy
            self.theta <= self.theta_max, self.theta >= -self.theta_max  # Voltage angle limits
        ]
        self.power_balance_constraint = constraints[0]
        if self.node_type == "Ref":
            constraints += [self.theta == 0.0]
        return constraints
