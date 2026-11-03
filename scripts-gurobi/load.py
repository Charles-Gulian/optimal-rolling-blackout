import numpy as np
import pandas as pd
import gurobipy as gp

from component import Component, T

# Flat value of lost load, $/MWh. Placeholder.
DEFAULT_VOLL = 1e4


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
        # Plain arrays, rewritten each solve. gurobipy has no Parameter object:
        # data reaches the model through bounds, RHS and coefficients instead.
        self.demand = np.zeros(T)
        self.voll = np.full(T, self.static_voll)

    def create_variables(self):
        self.load_shed = self.model.addMVar(
            T, lb=0.0, ub=1.0, name=f"shed[{self.name}]")
        self.unserved_energy = self.model.addMVar(
            T, lb=0.0, name=f"ue[{self.name}]")

    def write_constraints(self):
        # unserved_energy == demand * load_shed. demand is a coefficient on
        # load_shed, so it is rewritten with chgCoeff each solve; the -1.0 here
        # is a placeholder that reserves the matrix entry.
        self.ue_definition = self.model.addConstr(
            self.unserved_energy - 1.0 * self.load_shed == np.zeros(T),
            name=f"ue_def[{self.name}]")
        self._ue_rows = self.ue_definition.tolist()
        self._shed_vars = self.load_shed.tolist()

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.demand = (self.peak_load
                       * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values)
        self.voll = np.full(T, self.static_voll)  # Temporary
        self.unserved_energy.UB = self.demand
        for t in range(T):
            self.model.chgCoeff(self._ue_rows[t], self._shed_vars[t], -self.demand[t])

    def cost_expression(self):
        return gp.quicksum(self.voll[t] * self.unserved_energy[t] for t in range(T))
