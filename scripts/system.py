import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import pathlib

from component import T

UNSERVED_TOL = 1e-3  # MWh; ignore LP/MILP numerical dust
from node import Node
from line import Line
from load import Load
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
        df_load = pd.read_csv(system_dir / system_config / "load.csv", index_col=[0])

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

        # 2. Instantiate loads (one row per bus and load type in load.csv)
        self.loads = {}
        for load in df_load.index:
            self.loads[load] = Load.from_series(df_load.loc[load])

        # 3. Instantiate lines
        self.lines = {}
        for line in df_line.index:
            self.lines[line] = Line.from_series(df_line.loc[line])

        # 4. Instantiate resources
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

        # 5. Link various components
        for obj in self.components:
            obj.link_system(self) # Link component to system
        for load in self.loads.values():
            load.link_node(self.nodes)  # Link load to node
        for resource in self.resources.values():
            resource.link_node(self.nodes)  # Link resource to node
        for line in self.lines.values():
            line.link_nodes(self.nodes)  # Link line to nodes

    @property
    def has_blackouts(self):
        return any(node.blackout_enabled for node in self.nodes.values())

    def enable_blackouts(self, nodes=None):
        """Make load shedding an all-or-nothing binary decision per node.

        Defaults to nodes that actually carry load. The other 22 buses would
        otherwise get binaries that are entirely unconstrained -- unserved
        energy is zero there regardless -- which only enlarges the search tree.

        Call before write_opf(); the problem is rebuilt from the components, so
        toggling and rebuilding switches modes without re-reading any data.
        """
        if nodes is None:
            nodes = [name for name, node in self.nodes.items() if node.loads]
        for name in nodes:
            self.nodes[name].blackout_enabled = True

    def disable_blackouts(self):
        for node in self.nodes.values():
            node.blackout_enabled = False
            node.blackout = None

    @property
    def resources(self):
        # Create dictionary of all resources
        return self.thermal_resources | self.variable_resources | self.hydro_resources | self.storage_resources

    @property
    def components(self):
        # Create list of components
        return list(self.nodes.values()) + list(self.lines.values()) + list(self.loads.values()) + list(self.resources.values())

    @property
    def load_profile(self):
        """Hourly system load (MW) for the year read in by read_timeseries.

        Each load's normalized profile scaled by its own peak, summed across
        every load at every bus. bus.csv's MW Load column plays no part.
        """
        if any(load.load_profile is None for load in self.loads.values()):
            raise RuntimeError("call read_timeseries(year) before reading load_profile")
        return sum(load.peak_load * load.load_profile for load in self.loads.values())

    @property
    def peak_load(self):
        """Coincident system peak (MW).

        The maximum of the summed profiles, not the sum of individual peaks --
        loads peak at different hours, so the latter would overstate it.
        """
        return self.load_profile.max()

    def read_timeseries(self, year):
        for load in self.loads.values():
            load.get_load_profile(year)
        for resource in self.variable_resources.values():
            resource.get_gen_profile(year)
        for resource in self.hydro_resources.values():
            resource.get_gen_profile(year)

    def available_days(self, year, periods=T):
        """Dates in `year` with a full `periods`-hour window in every profile.

        Profile coverage is not always complete: the 2020 NSRDB profiles end at
        11:00 on 31 December, so that day has only 12 hours and would raise a
        dimension error on the demand Parameter. Iterate over this rather than a
        raw date_range.
        """
        index = None
        for load in self.loads.values():
            index = load.load_profile.index if index is None \
                else index.intersection(load.load_profile.index)
        for resource in list(self.variable_resources.values()) + list(self.hydro_resources.values()):
            index = index.intersection(resource.gen_profile.index)

        available = set(index)
        offsets = [pd.Timedelta(hours=h) for h in range(periods)]
        return [day for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                if all(day + off in available for off in offsets)]

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
        self.total_unserved_energy_costs = cp.sum(cp.sum([l.VOLL * l.unserved_energy for l in self.loads.values()]))
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

    # ------------------------------------------------------------------
    # Results reporting for the current solve
    #
    # All timestamps are on the UTC clock, matching the profiles. solve_opf
    # slices T hours forward from midnight UTC, so a "day" runs 17:00 to 16:00
    # local (UTC-7). That is fine for counting and costing, but outage *timing*
    # should be converted before interpretation.
    # ------------------------------------------------------------------

    @property
    def opf_index(self):
        """Timestamps of the current solve window, UTC, tz-naive."""
        return pd.date_range(start=self.opf_date, periods=T, freq="h")

    @property
    def unserved_energy_profile(self):
        """(T, n_loads) MW of unserved energy, one column per load."""
        return pd.DataFrame(
            {name: load.unserved_energy.value for name, load in self.loads.items()},
            index=self.opf_index,
        )

    @property
    def blackout_schedule(self):
        """(T, n_nodes with blackouts) binary schedule, or None in LP mode.

        Columns are bus IDs, since the decision lives on the node and drives
        every load at it.
        """
        if not self.has_blackouts:
            return None
        return pd.DataFrame(
            {name: np.round(node.blackout.value).astype(int)
             for name, node in self.nodes.items() if node.blackout_enabled},
            index=self.opf_index,
        )

    @property
    def solve_summary(self):
        """One row of results for the current solve."""
        ue = self.unserved_energy_profile
        by_type = {}
        for name, load in self.loads.items():
            by_type[load.load_type] = by_type.get(load.load_type, 0.0) + float(ue[name].sum())
        hourly = ue.sum(axis=1)
        return {
            "date": pd.Timestamp(self.opf_date),
            "gen_cost": float(self.total_variable_costs.value),
            "unserved_cost": float(self.total_unserved_energy_costs.value),
            "total_cost": float(self.prob.value),
            "load_MWh": float(sum(l.demand.value.sum() for l in self.loads.values())),
            "unserved_MWh": float(hourly.sum()),
            "unserved_hours": int((hourly > UNSERVED_TOL).sum()),
            "buses_affected": int((ue.sum(axis=0) > UNSERVED_TOL).sum()),
            **{f"unserved_{k}_MWh": v for k, v in by_type.items()},
        }

    @property
    def dispatch_results(self):
        # Collect dispatch results
        df_dispatch = pd.DataFrame()
        df_dispatch["load"] = sum(load.demand.value for load in self.loads.values())
        df_dispatch["thermal"] = sum(resource.p_out.value for resource in self.thermal_resources.values())
        df_dispatch["solar"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "solar")
        df_dispatch["wind"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "wind")
        df_dispatch["hydro"] = sum(resource.p_out.value for resource in self.hydro_resources.values())
        df_dispatch["storage charge"] = sum(resource.charge.value for resource in self.storage_resources.values())
        df_dispatch["storage discharge"] = sum(resource.discharge.value for resource in self.storage_resources.values())
        df_dispatch["unserved energy"] = sum(load.unserved_energy.value for load in self.loads.values())

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
                cp.sum([l.unserved_energy for l in self.loads.values()])).value
            # Gurobi returns no duals for a MIP, so there are no LMPs once
            # blackout decisions are binary.
            if not self.has_blackouts:
                for n in self.nodes.keys():
                    df_LMP.loc[date:date + pd.Timedelta(hours=T - 1), n] = -self.nodes[n].power_balance_constraint.dual_value

        return df_results, df_LMP
