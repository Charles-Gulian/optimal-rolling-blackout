import numpy as np
import pandas as pd
import cvxpy as cp

from component import Component, T

# Flat value of lost load, $/MWh. Placeholder.
DEFAULT_VOLL = 1e4

# Discomfort dynamics: x[t+1] = (1 - GAMMA) x[t] + load_shed[t], and the
# marginal cost of load shed rises by BETA per unit of x.
GAMMA = 0.3
BETA = DEFAULT_VOLL


class Load(Component):
    """Electricity demand of one type at one node, with flat VoLL."""

    PROFILE_SOURCE = "RTS-GMLC"
    PROFILE_PATTERN = "{source}_load-profile_bus{bus}_{year}.csv"

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack load attributes from data
        self.node_ID = data["Bus ID"]
        self.load_type = data["Load Type"]
        self.peak_load = data["Peak MW"]
        self.static_voll = DEFAULT_VOLL

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
        """Fraction-of-peak hourly shape for this load."""
        fname = self.PROFILE_PATTERN.format(
            source=self.PROFILE_SOURCE, bus=self.node_ID, year=year)
        ts_data_path = (self.system.base_dir / "load-data" / "profiles"
                        / f"bus{self.node_ID}" / fname)
        df_load = pd.read_csv(ts_data_path, index_col=[0])
        df_load.index = pd.to_datetime(df_load.index)
        self.load_profile = df_load.squeeze()

    def create_parameters(self):
        self.demand = cp.Parameter(T, nonneg=True)
        self.voll = cp.Parameter(T, nonneg=True)
        # Min-H state and costate, frozen at the current iterate each pass.
        self.x = cp.Parameter(T + 1, nonneg=True)
        self.lam = cp.Parameter(T + 1)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.demand.value = self.peak_load * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values
        self.voll.value = np.full(T, self.static_voll) # Temporary
        self.set_costate(np.zeros(T + 1), np.zeros(T + 1))

    def set_costate(self, x, lam):
        """Freeze the state and costate trajectories for the next min-H solve."""
        self.x.value = x
        self.lam.value = lam

    def create_variables(self):
        self.load_shed = cp.Variable(T, nonneg=True)
        self.unserved_energy = cp.Variable(T, nonneg=True)

    def write_constraints(self):
        return [
            self.unserved_energy == cp.multiply(self.demand, self.load_shed),
            self.load_shed <= 1,  # Limit load shed fraction 0 <= z <= 1
        ]

    def cost_expression(self):
        return cp.sum(cp.multiply(self.voll, self.unserved_energy))

    def hamiltonian_expression(self):
        """Per-load min-H objective: static cost + state cost + costate term."""
        return (cp.sum(cp.multiply(self.voll, self.unserved_energy))
                + cp.sum(cp.multiply(BETA * self.x[:-1], self.unserved_energy))
                + cp.sum(cp.multiply(self.lam[1:], self.load_shed)))
