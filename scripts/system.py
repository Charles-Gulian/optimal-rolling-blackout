import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import pathlib

from component import T

UNSERVED_TOL = 1e-3  # MWh; ignore LP/MILP numerical dust

# All profiles are stored on the UTC clock. The system sits in Arizona (MST,
# no DST), so local = UTC - 7. A solve window starts at LOCAL midnight, i.e.
# 07:00 UTC, so one problem covers one whole local day.
#
# Why this matters: shedding concentrates 17:00-21:00 local. Starting windows
# at 00:00 UTC (17:00 local) cut straight through that block, so a single
# real outage running e.g. 15:00-20:00 local was split across two independent
# problems -- each solved with the thermal state reset to setpoint. That
# understated discomfort on exactly the longest outages, which are the ones
# the dynamic VOLL model turns on.
LOCAL_UTC_OFFSET = pd.Timedelta(hours=-7)
from node import Node
from line import Line
from load import CoolingLoad, Load
from resources import ThermalResource, VariableResource, HydroResource, StorageResource


class System:
    def __init__(self, base_dir, system_dir, system_config,
                 forced_outages=True, scenario_seed=0, voll_model="static"):

        # Save base directory, system directory
        self.base_dir = base_dir
        self.system_dir = system_dir
        self.system_config = system_config

        # Forced outages. Sampled per unit-year from a seed, so a given
        # scenario_seed always reproduces the same draw; change it for an
        # independent Monte Carlo replication.
        self.forced_outages = forced_outages
        self.scenario_seed = scenario_seed

        # Extra kwargs passed to every prob.solve(). Set Threads=1 when
        # running a process pool: Gurobi defaults to one thread per core, so
        # N workers each grabbing N threads oversubscribe badly -- measured
        # 0.159 s/day solo against 0.222 s/day with only 3 workers on 8 cores.
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

        # 2. Instantiate loads (one row per bus and load type in load.csv).
        #    Cooling gets its own class because it carries the Wang thermal
        #    state; every other type is a plain flat-VOLL Load.
        load_classes = {"cooling": CoolingLoad}
        self.loads = {}
        for load in df_load.index:
            cls = load_classes.get(df_load.loc[load, "Load Type"], Load)
            self.loads[load] = cls.from_series(df_load.loc[load])
        if voll_model != "static":
            self.set_voll_model(voll_model)

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

        # 5. Unit indices for reproducible outage seeding. gen.csv carries an
        #    explicit Resource ID, assigned once when the config is derived and
        #    inherited unchanged by configs derived from it, so the same
        #    physical unit draws the same outages in every config it appears in
        #    (common random numbers) and reliability differences between
        #    configs are purely structural.
        #
        #    Sorted position will NOT do: it is a unit's rank in the name list,
        #    so inserting or dropping one unit slides every unit after it into
        #    a different RNG stream.
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

    VOLL_MODELS = ("static", "exogenous", "dynamic")

    @property
    def voll_model(self):
        """The VOLL model in force, or "mixed" if loads disagree."""
        models = {load.voll_model for load in self.loads.values()
                  if isinstance(load, CoolingLoad)}
        if len(models) == 1:
            return models.pop()
        return "mixed" if models else "static"

    def set_voll_model(self, model):
        """Choose how cooling load prices unserved energy.

            static      v_t = VOLL                                flat
            exogenous   v_t = VOLL + rho c phi_t                  weather only
            dynamic     v_t = VOLL + rho (kappa x_t + c phi_t)     + outage history

        Applies only to CoolingLoad instances carrying a full set of Wang
        parameters; every other load keeps its flat VOLL, so this is a no-op
        for "other" load.

        "dynamic" needs binary blackouts, since the McCormick envelope for
        w = x z is exact only for binary z. "exogenous" does not -- it has no
        bilinear term -- but it does need the blackout variable to price
        z_t, so call enable_blackouts() for both.

        Like enable_blackouts, call before write_opf(): the problem is rebuilt
        from the components, so changing this afterwards has no effect.
        """
        if model not in self.VOLL_MODELS:
            raise ValueError(f"voll_model must be one of {self.VOLL_MODELS}, got {model!r}")
        for load in self.loads.values():
            if isinstance(load, CoolingLoad) and load.has_thermal_model:
                load.voll_model = model

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
            if load.voll_model != "static" and load.needs_thermal:
                load.get_temperature_profile(year)   # CoolingLoad only
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
        """Window starts in `year` with a full `periods`-hour span in every profile.

        Returns UTC timestamps at LOCAL midnight (07:00 UTC), so each window
        covers one local calendar day, 00:00-23:00 local.

        Profile coverage is not always complete, and the coverage test here
        handles two cases at once. The 2020 NSRDB profiles end at 11:00 on 31
        December, so that day is short. And because local midnight is 07:00
        UTC, the last local day of any year needs 6 hours from the following
        year's file, which is not loaded -- so it drops out too. That costs
        one day per year (20 of 7305, 0.3%) and is why the year totals are a
        day short of the calendar.
        """
        index = None
        for load in self.loads.values():
            index = load.load_profile.index if index is None \
                else index.intersection(load.load_profile.index)
        for resource in list(self.variable_resources.values()) + list(self.hydro_resources.values()):
            index = index.intersection(resource.gen_profile.index)

        available = set(index)
        offsets = [pd.Timedelta(hours=h) for h in range(periods)]
        # Local midnight on local date D == D 00:00 UTC minus the offset.
        starts = (pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                  - LOCAL_UTC_OFFSET)
        return [day for day in starts
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
        # Each Load prices its own unserved energy: flat VOLL, or the
        # McCormick-linearised dynamic VOLL when it has a thermal model and
        # the toggle is on. Built after the constraints loop, since the
        # dynamic form needs the McCormick variables to exist.
        self.total_unserved_energy_costs = cp.sum([l.cost_expression() for l in self.loads.values()])
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
        result = self.prob.solve(solver=cp.GUROBI, **self.solver_options)

        return result

    # ------------------------------------------------------------------
    # Results reporting for the current solve
    #
    # Timestamps are on the UTC clock, matching the profiles. solve_opf slices
    # T hours forward from LOCAL midnight (07:00 UTC), so a "day" is one whole
    # local day. `date` in solve_summary is the local calendar date; opf_index
    # stays in UTC, so convert with LOCAL_UTC_OFFSET before reading timing off
    # it.
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
            # Two views of the same instant, because they are used for
            # different things and conflating them has bitten us once already.
            #
            # window_start is where the solve actually begins, in UTC. It is
            # what you pass back to update_timeseries_parameters() to re-solve
            # this day -- the profiles are sliced forward from it.
            #
            # date is the local calendar day, for labelling results. opf_date
            # is local midnight expressed in UTC, so shifting back by the
            # offset and truncating gives the local day the outage belongs to.
            "window_start": pd.Timestamp(self.opf_date),
            "date": pd.Timestamp(self.opf_date + LOCAL_UTC_OFFSET).normalize(),
            "gen_cost": float(self.total_variable_costs.value),
            "unserved_cost": float(self.total_unserved_energy_costs.value),
            "total_cost": float(self.prob.value),
            "load_MWh": float(sum(l.demand.value.sum() for l in self.loads.values())),
            "unserved_MWh": float(hourly.sum()),
            "unserved_hours": int((hourly > UNSERVED_TOL).sum()),
            # Distinct BUSES, not loads. `ue` has one column per load and
            # every bus carries two (cooling and other), so counting columns
            # double-counts -- which it silently did until 2026-10-08.
            "buses_affected": len({self.loads[name].node_ID
                                   for name in ue.columns
                                   if ue[name].sum() > UNSERVED_TOL}),
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
