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
        self.theta = cp.Variable(T)

    def write_constraints(self):
        constraints = [
            cp.sum([r.p_out for r in self.resources])  # Nodal generation
            + cp.sum([l.flow for l in self.in_lines])  # + line inflows
            - cp.sum([l.flow for l in self.out_lines])  # - line outflows
            == cp.sum([ld.demand - ld.unserved_energy for ld in self.loads]),  # = served load (power balance constraint)
            self.theta <= self.theta_max, self.theta >= -self.theta_max  # Voltage angle limits
        ]
        self.power_balance_constraint = constraints[0]
        if self.node_type == "Ref":
            constraints += [self.theta == 0.0]
        return constraints

    @property
    def power_balance_duals(self):
        return self.power_balance_constraint.dual_value