import numpy as np
import pandas as pd
import gurobipy as gp

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
        self.theta_max = np.deg2rad(30)  # 30 degree maximum voltage angle

        # Initialize resources
        self.resources = []

        # Initialize loads (one per load type at this node)
        self.loads = []

        # Initialize in/out lines
        self.in_lines = []
        self.out_lines = []

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def create_variables(self):
        # Reference bus is pinned by its bounds rather than a constraint row.
        bound = 0.0 if self.node_type == "Ref" else self.theta_max
        self.theta = self.model.addMVar(
            T, lb=-bound, ub=bound, name=f"theta[{self.name}]")

    def write_constraints(self):
        lhs = gp.MLinExpr.zeros(T)
        lhs += gp.quicksum(r.p_out for r in self.resources)        # Nodal generation
        lhs += gp.quicksum(l.flow for l in self.in_lines)          # + line inflows
        lhs -= gp.quicksum(l.flow for l in self.out_lines)         # - line outflows
        lhs += gp.quicksum(ld.unserved_energy for ld in self.loads)  # + unserved
        # ... == nodal demand, which is data and so lives in the RHS.
        self.power_balance_constraint = self.model.addConstr(
            lhs == np.zeros(T), name=f"balance[{self.name}]")

    def update_timeseries_parameters(self, date):
        demand = sum((ld.demand for ld in self.loads), start=np.zeros(T))
        self.power_balance_constraint.RHS = demand

    @property
    def power_balance_duals(self):
        return self.power_balance_constraint.Pi
