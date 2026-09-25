import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import pathlib

from component import T
from node import Node
from line import Line
from resource import ThermalResource, VariableResource, HydroResource, StorageResource


class System:
    def __init__(self, base_dir, system_dir, system_config):

        # Save base directory, system directory
        self.base_dir = base_dir
        self.system_dir = system_dir
        self.system_config = system_config

        # Current date for OPF dispatch results (placeholder)
        self.opf_date = None

        # Read in data for IEEE RTS GMLC (Reliability Test System)
        df_bus = pd.read_csv(system_dir / system_config / "bus.csv", index_col=[0])
        df_line = pd.read_csv(system_dir / system_config / "branch.csv", index_col=[0])
        df_gen = pd.read_csv(system_dir / system_config / "gen.csv", index_col=[0])

        # Select unit types
        unit_types = ["CT", "STEAM", "CC", "NUCLEAR", "PV", "RTPV", "WIND", "HYDRO", "ROR", "STORAGE"]
        # Select columns
        columns = ["Bus ID", "Unit Type", "Category", "Fuel", "Fuel Price $/MMBTU", "HR_avg_0", "VOM", "PMax MW", "PMin MW", "Ramp Rate MW/Min", "FOR", "Storage Roundtrip Efficiency"]
        # Get final resource input data
        df_gen = df_gen.loc[df_gen["Unit Type"].isin(unit_types), columns]

        # 1. Instantiate nodes
        self.nodes = {}
        for node in df_bus.index:
            self.nodes[node] = Node.from_series(df_bus.loc[node])

        # 2. Instantiate lines
        self.lines = {}
        for line in df_line.index:
            self.lines[line] = Line.from_series(df_line.loc[line])

        # 3. Instantiate resources
        thermal_resource_types = ["CT", "STEAM", "CC", "NUCLEAR"]
        variable_resource_types = ["PV", "RTPV", "WIND"]
        hydro_resource_types = ["HYDRO", "ROR"]
        storage_resource_types = ["STORAGE"]
        self.thermal_resources = {}
        self.variable_resources = {}
        self.hydro_resources = {}
        self.storage_resources = {}
        for resource in df_gen.index:
            unit_type = df_gen.loc[resource, "Unit Type"]
            if unit_type in thermal_resource_types:
                self.thermal_resources[resource] = ThermalResource.from_series(df_gen.loc[resource])
            elif unit_type in variable_resource_types:
                self.variable_resources[resource] = VariableResource.from_series(df_gen.loc[resource])
            elif unit_type in hydro_resource_types:
                self.hydro_resources[resource] = HydroResource.from_series(df_gen.loc[resource])
            elif unit_type in storage_resource_types:
                self.storage_resources[resource] = StorageResource.from_series(df_gen.loc[resource])

        # 4. Link various components
        for obj in self.components:
            obj.link_system(self) # Link component to system
        for resource in self.resources.values():
            resource.link_node(self.nodes)  # Link resource to node
        for line in self.lines.values():
            line.link_nodes(self.nodes)  # Link line to nodes

    @property
    def resources(self):
        # Create dictionary of all resources
        return self.thermal_resources | self.variable_resources | self.hydro_resources | self.storage_resources

    @property
    def components(self):
        # Create list of components
        return list(self.nodes.values()) + list(self.lines.values()) + list(self.resources.values())

    def read_timeseries(self, year):
        for node in self.nodes.values():
            node.get_load_profile(year)
        for resource in self.variable_resources.values():
            resource.get_gen_profile(year)
        for resource in self.hydro_resources.values():
            resource.get_gen_profile(year)

    def write_opf(self):
        # Create CVXPY model

        ### Parameters, Variables, Expressions ###

        # Create parameters, variables, and expressions for all objects
        for obj in self.components:
            obj.create_parameters()
            obj.create_variables()
            obj.create_expressions()

        ### Constraints ###

        # Write constraints for all system components
        self.constraints = []
        for obj in self.components:
            self.constraints += obj.write_constraints()

        ### Objective Function ###

        # Get total system variable costs
        self.total_variable_costs = cp.sum(cp.sum([cp.multiply(r.variable_costs, r.p_out) for r in self.resources.values()]))
        self.total_unserved_energy_costs = cp.sum(cp.sum([n.VOLL * n.unserved_energy for n in self.nodes.values()]))
        self.total_cost = self.total_variable_costs + self.total_unserved_energy_costs # + startup costs + ... etc.
        self.objective = cp.Minimize(self.total_cost)

        ### Problem ###
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve_opf(self, date):
        # Solve CVXPY model
        self.opf_date = date

        # Update time series parameters
        for obj in self.components:
            obj.update_timeseries_parameters(date)

        # Solve model
        result = self.prob.solve(solver=cp.GUROBI)

        return result

    @property
    def dispatch_results(self):
        # Collect dispatch results
        df_dispatch = pd.DataFrame()
        df_dispatch["load"] = sum(node.load.value for node in self.nodes.values())
        df_dispatch["thermal"] = sum(resource.p_out.value for resource in self.thermal_resources.values())
        df_dispatch["solar"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "solar")
        df_dispatch["wind"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "wind")
        df_dispatch["hydro"] = sum(resource.p_out.value for resource in self.hydro_resources.values())
        df_dispatch["storage charge"] = sum(resource.charge.value for resource in self.storage_resources.values())
        df_dispatch["storage discharge"] = sum(resource.discharge.value for resource in self.storage_resources.values())
        df_dispatch["unserved energy"] = sum(node.unserved_energy.value for node in self.nodes.values())

        # Correct index
        df_dispatch.index = pd.date_range(start=self.opf_date, periods=24, freq="h", tz="UTC")

        return df_dispatch

    @staticmethod
    def dispatch_plot(df_dispatch):
        # Copy dataframe
        df_dispatch = df_dispatch.copy()

        # Correct index
        df_dispatch = pd.concat([df_dispatch.iloc[8:], df_dispatch.iloc[0:8]])
        df_dispatch = df_dispatch.reset_index()

        plt.figure(figsize=(9, 5))
        time = df_dispatch.index
        plt.plot(
            time,
            df_dispatch["load"],
            color="black",
            linewidth=3,
            label="load",
        )
        plt.plot(
            time,
            df_dispatch["load"] + df_dispatch["storage charge"],
            color="black",
            linewidth=2,
            linestyle="--",
            label="load + storage charge",
        )
        plt.stackplot(
            time,
            df_dispatch["solar"], df_dispatch["wind"], df_dispatch["hydro"], df_dispatch["thermal"],
            df_dispatch["storage discharge"], df_dispatch["unserved energy"],
            labels=("solar", "wind", "hydro", "thermal", "storage discharge", "unserved energy"),
            colors=("gold", "skyblue", "steelblue", "lightgrey", "purple", "red"),
        )
        plt.legend(fontsize=9)
        plt.axis([0, 23, 0, None])
        plt.xlabel("Time (h)")
        plt.ylabel("MW")
        plt.show()

    def simulate_year(self, year):

        # Read in time series data for  year
        self.read_timeseries(year)

        # Create CVXPY OPF model
        self.write_opf()

        # Store results
        df_results = pd.DataFrame(
            index=pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59", freq="d"),
            columns=["Cost", "Unserved Energy"],
        )
        df_LMP = pd.DataFrame(
            index=pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59", freq="h"),
            columns=list(self.nodes.keys()),
        )

        # Simulate each day
        for date in df_results.index:
            # Re-solve model
            self.solve_opf(date)

            # Save results
            df_results.loc[date, "Cost"] = self.total_variable_costs.value
            df_results.loc[date, "Unserved Energy"] = cp.sum(
                cp.sum([n.unserved_energy for n in self.nodes.values()])).value
            for n in self.nodes.keys():
                df_LMP.loc[date:date + pd.Timedelta(hours=T - 1), n] = -self.nodes[n].power_balance_constraint.dual_value

        return df_results, df_LMP
