import numpy as np
import pandas as pd
import gurobipy as gp

from component import Component, T

class Load(Component):
    """Electricity demand of one type at one node."""

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

        # Initialize node, profile, voll
        self.node = None
        self.load_profile = None
        self.voll = VoLL(self) # Static by default

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
        ts_data_path = (self.system.base_dir / "load-data" / "profiles" / f"bus{self.node_ID}" / fname)
        df_load = pd.read_csv(ts_data_path, index_col=[0])
        df_load.index = pd.to_datetime(df_load.index)
        self.load_profile = df_load.squeeze()

    def create_parameters(self):
        self.demand = np.zeros(T)
        self.voll.create_parameters()

    def create_variables(self):
        self.load_shed = self.model.addMVar(T, lb=0.0, ub=1.0, name=f"shed[{self.name}]")
        self.unserved_energy = self.model.addMVar(T, lb=0.0, name=f"ue[{self.name}]")
        self.voll.create_variables()

    def write_constraints(self):
        self.ue_constraint = self.model.addConstr(
            self.unserved_energy - self.demand * self.load_shed == 0,
            name=f"ue_constraint[{self.name}]"
        )
        self.ue_constraint_rows = self.ue_constraint.tolist()
        self.load_shed_vars = self.load_shed.tolist()
        self.voll.write_constraints()

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.demand = (self.peak_load * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values)
        for t in range(T):
            self.model.chgCoeff(self.ue_constraint_rows[t], self.load_shed_vars[t], -self.demand[t])
        self.voll.update_timeseries_parameters()

    def cost_expression(self):
        return gp.quicksum(self.voll.value[t] * self.unserved_energy[t] for t in range(T))

class VoLL:
    """VoLL model."""
    STATIC_VOLL = 2.2e4 # $/MWh
    GAMMA = 0.6 # Discount rate
    BETA = 1.2e4 # Sensitivity

    def __init__(self, load):
        super().__init__()
        self.name = f"{load.name}_voll"
        self.dynamic = False
        self.beta = self.BETA
        self.load = load

    def create_parameters(self):
        self.static_voll = self.STATIC_VOLL * np.ones(T)
    def create_variables(self):
        if not self.dynamic:
            return
        self.dynamic_voll = self.load.model.addMVar(
            T, lb=0.0, name=f"voll[{self.name}]")

    def write_constraints(self):
        if not self.dynamic:
            return
        # voll[t+1] == (1 - gamma) * voll[t] + load_shed[t], for t = 0 .. T-2
        self.voll_model = self.load.model.addConstr(
            self.dynamic_voll[1:] == (1 - self.GAMMA) * self.dynamic_voll[:-1]
            + self.load.load_shed[:-1],
            name=f"voll_model[{self.name}]"
        )
        self.voll_initial = self.load.model.addConstr(
            self.dynamic_voll[0] == 0.0,
            name=f"voll_initial[{self.name}]"
        )

    def update_timeseries_parameters(self):
        pass

    @property
    def value(self):
        if self.dynamic:
            return self.static_voll + self.beta * self.dynamic_voll
        else:
            return self.static_voll