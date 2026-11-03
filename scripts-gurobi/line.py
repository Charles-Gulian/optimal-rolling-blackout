import pandas as pd

from component import Component, T


class Line(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack line attributes from data
        self.from_node_ID = data["From Bus"]
        self.to_node_ID = data["To Bus"]
        self.max_flow = data["LTE Rating"]  # Long-term flow limit; alternatively can use continuous/short-term limits
        self.X = data["X"]  # Line susceptance # MVA base = 100?

        # Initialize other parameters
        self.baseMVA = 100.0

        # Initialize from node / to node
        self.from_node = None
        self.to_node = None

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def link_nodes(self, nodes: dict):
        # Link nodes to line
        self.from_node = nodes[self.from_node_ID]
        self.to_node = nodes[self.to_node_ID]
        # Link line to nodes
        nodes[self.from_node_ID].out_lines.append(self)
        nodes[self.to_node_ID].in_lines.append(self)

    def create_variables(self):
        # Thermal limits as variable bounds rather than constraint rows.
        self.flow = self.model.addMVar(
            T, lb=-self.max_flow, ub=self.max_flow, name=f"flow[{self.name}]")

    def write_constraints(self):
        self.dc_flow_constraint = self.model.addConstr(
            self.flow == (1 / self.X) * self.baseMVA
            * (self.from_node.theta - self.to_node.theta),
            name=f"dc_flow[{self.name}]")
