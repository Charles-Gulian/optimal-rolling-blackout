import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
import pathlib

from component import T

UNSERVED_TOL = 1e-3  # MWh; ignore LP/MILP numerical dust

# Timestamps are tz-naive throughout: RTS-GMLC profiles are on a plain local
# clock, so nothing is shifted anywhere.
from node import Node
from line import Line
from load import Load
from resources import ThermalResource, VariableResource, HydroResource, StorageResource


class System:
    def __init__(self, base_dir, system_dir, system_config,
                 forced_outages=True, scenario_seed=0):

        # Save base directory, system directory
        self.base_dir = base_dir
        self.system_dir = system_dir
        self.system_config = system_config

        # Forced outages
        self.forced_outages = forced_outages

        # Scenario random seed
        self.scenario_seed = scenario_seed

        # Solver options
        self.solver_options = {}

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
        columns = ["Bus ID", "Unit Type", "Category", "Fuel", "Fuel Price $/MMBTU", "HR_avg_0", "VOM", "PMax MW", "PMin MW", "Ramp Rate MW/Min", "FOR", "MTTR Hr", "Storage Roundtrip Efficiency"]
        # Configs we derive ourselves carry an explicit Resource ID; the
        # pristine upstream RTS-GMLC does not.
        if "Resource ID" in df_gen.columns:
            columns = columns + ["Resource ID"]
        # Get final resource input data
        df_gen = df_gen.loc[df_gen["Unit Type"].isin(unit_types), columns]

        # 1. Instantiate nodes
        self.nodes = {}
        for node in df_bus.index:
            self.nodes[node] = Node.from_series(df_bus.loc[node])

        # 2. Instantiate loads (one row per bus in load.csv).
        #
        # Loads are priced at the flat VOLL that Load picks up from load.csv,
        # or its default when that column is absent. The per-policy
        # voll_<model>.csv files that used to be merged in here, and the
        # CoolingLoad subclass that consumed their thermal parameters, went
        # with the old policy comparison; pricing is being rebuilt.
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

        # 5. Unit indices for reproducible outage seeding. gen.csv carries an explicit Resource ID
        if "Resource ID" in df_gen.columns:
            for name in self.resources:
                self.resources[name].unit_index = int(df_gen.loc[name, "Resource ID"])
        else:
            for index, name in enumerate(sorted(self.resources)):
                self.resources[name].unit_index = index

        # 6. Link various components
        for obj in self.components:
            obj.link_system(self) # Link component to system
        for load in self.loads.values():
            load.link_node(self.nodes)  # Link load to node
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
        # Loads first, then nodes: a Node's power-balance RHS is the sum of its
        # loads' demand, so the loads must have updated theirs by then.
        return (list(self.loads.values()) + list(self.resources.values())
                + list(self.lines.values()) + list(self.nodes.values()))

    @property
    def load_profile(self):
        """Hourly system load (MW) for the year read in by read_timeseries."""
        if any(load.load_profile is None for load in self.loads.values()):
            raise RuntimeError("call read_timeseries(year) before reading load_profile")
        return sum(load.peak_load * load.load_profile for load in self.loads.values())

    @property
    def peak_load(self):
        """Coincident system peak (MW)."""
        return self.load_profile.max()

    def read_timeseries(self, year):
        for load in self.loads.values():
            load.get_load_profile(year)
        for resource in self.variable_resources.values():
            resource.get_gen_profile(year)
        for resource in self.hydro_resources.values():
            resource.get_gen_profile(year)
        for resource in self.resources.values():
            if self.forced_outages:
                resource.get_outage_profile(year, self.scenario_seed)
            else:
                resource.outage_profile = None

    def available_days(self, year, periods=T):
        """Window starts in `year` with a full `periods`-hour span in every profile."""
        index = None
        for load in self.loads.values():
            index = load.load_profile.index if index is None \
                else index.intersection(load.load_profile.index)
        for resource in list(self.variable_resources.values()) + list(self.hydro_resources.values()):
            index = index.intersection(resource.gen_profile.index)

        available = set(index)
        offsets = [pd.Timedelta(hours=h) for h in range(periods)]
        starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        return [day for day in starts
                if all(day + off in available for off in offsets)]

    def write_opf(self):
        # Create the gurobipy model first: variables are created on it.
        self.model = gp.Model(f"opf[{self.system_config}]")
        self.model.Params.OutputFlag = 0
        for key, value in self.solver_options.items():
            self.model.setParam(key, value)

        ### Parameters, Variables, Expressions ###

        # Create parameters, variables, and expressions for all objects
        for obj in self.components:
            obj.create_parameters()
            obj.create_variables()
        self.model.update()
        for obj in self.components:
            obj.create_expressions()

        ### Constraints ###

        # Write constraints for all system components
        for obj in self.components:
            obj.write_constraints()

        ### Objective Function ###

        self.total_unserved_energy_costs = gp.quicksum(
            l.cost_expression() for l in self.loads.values())
        self.model.setObjective(self.total_unserved_energy_costs, gp.GRB.MINIMIZE)
        self.model.update()

    def solve_opf(self, date):
        self.opf_date = date

        # Push this day's data into bounds, RHS and coefficients
        for obj in self.components:
            obj.update_timeseries_parameters(date)
        self.model.update()

        self.model.optimize()
        return self.model.ObjVal

    def solve_mpols(self, date):

        # Solve multi-period optimal load shedding with dynamic unserved energy costs using min-H method
        pass

    # ------------------------------------------------------------------
    # Results reporting for the current solve
    # ------------------------------------------------------------------

    @property
    def opf_index(self):
        """Timestamps of the current solve window, UTC, tz-naive."""
        return pd.date_range(start=self.opf_date, periods=T, freq="h")

    @property
    def unserved_energy_profile(self):
        """(T, n_loads) MW of unserved energy, one column per load."""
        return pd.DataFrame(
            {name: load.unserved_energy.X for name, load in self.loads.items()},
            index=self.opf_index,
        )

    @property
    def load_shed_profile(self):
        """(T, n_loads) MW of unserved energy, one column per load."""
        return pd.DataFrame(
            {name: load.load_shed.X for name, load in self.loads.items()},
            index=self.opf_index,
        )

    @property
    def lmp_profile(self):
        """(T, n_nodes) locational marginal price, one column per bus."""
        return pd.DataFrame(
            {name: np.abs(node.power_balance_duals) for name, node in self.nodes.items()},
            index=self.opf_index,
        )

    @property
    def ue_event_type(self):
        """Characterize unserved energy event type"""
        event_type = {
            0: "no-unserved-energy",
            1: "capacity-limited",
            2: "storage-limited",
            3: "energy-limited",
        }

        lmp_profile = self.lmp_profile.max(axis=1)
        if lmp_profile.max() < 1e-2:
            return event_type[0] # No unserved energy

        SOC_min_duals = pd.DataFrame(
            {name: np.abs(storage.SOC_min_duals) for name, storage in self.storage_resources.items()},
            index=self.opf_index,
        ).max(axis=1)

        SOC_max_duals = pd.DataFrame(
            {name: np.abs(storage.SOC_max_duals) for name, storage in self.storage_resources.items()},
            index=self.opf_index,
        ).max(axis=1)

        if SOC_max_duals.max() < 1e-2 and SOC_min_duals.max() < 1e-2:
            return event_type[1] # Capacity-limited
        elif SOC_max_duals.max() > 1e-2 and SOC_min_duals.max() > 1e-2:
            return event_type[2] # Storage-limited
        elif SOC_max_duals.max() < 1e-2 and SOC_min_duals.max() > 1e-2:
            return event_type[3] # Energy-limited


    @property
    def dispatch_results(self):
        # Collect dispatch results
        df_dispatch = pd.DataFrame()
        df_dispatch["load"] = sum(load.demand for load in self.loads.values())
        df_dispatch["thermal"] = sum(resource.p_out.X for resource in self.thermal_resources.values())
        df_dispatch["solar"] = sum(resource.p_out.X for resource in self.variable_resources.values() if resource.resource_type == "solar")
        df_dispatch["wind"] = sum(resource.p_out.X for resource in self.variable_resources.values() if resource.resource_type == "wind")
        df_dispatch["hydro"] = sum(resource.p_out.getValue() for resource in self.hydro_resources.values())
        df_dispatch["storage charge"] = sum(resource.charge.X for resource in self.storage_resources.values())
        df_dispatch["storage discharge"] = sum(resource.discharge.X for resource in self.storage_resources.values())
        df_dispatch["unserved energy"] = sum(load.unserved_energy.X for load in self.loads.values())

        # Correct index
        df_dispatch.index = pd.date_range(start=self.opf_date, periods=24, freq="h")

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
