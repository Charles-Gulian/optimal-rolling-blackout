import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import pathlib

from component import T

UNSERVED_TOL = 1e-3  # MWh; ignore LP/MILP numerical dust

# Timestamps are tz-naive throughout: RTS-GMLC profiles are on a plain local
# clock, so nothing is shifted anywhere.
from node import Node
from line import Line
from load import Load, GAMMA, BETA
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
        # Create list of components
        return list(self.nodes.values()) + list(self.lines.values()) + list(self.loads.values()) + list(self.resources.values())

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
        self.total_unserved_energy_costs = cp.sum([l.cost_expression() for l in self.loads.values()])
        self.objective = cp.Minimize(self.total_unserved_energy_costs)

        # Min-H objective: same variables and constraints, state and costate
        # entering as parameters. Used by solve_mpols.
        self.total_hamiltonian = cp.sum([l.hamiltonian_expression() for l in self.loads.values()])

        ### Problem ###
        self.prob = cp.Problem(self.objective, self.constraints)
        self.prob_hamiltonian = cp.Problem(cp.Minimize(self.total_hamiltonian), self.constraints)

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
    # Multi-period optimal load shedding, min-H / MSA method
    # ------------------------------------------------------------------

    @property
    def load_order(self):
        """Load names in a fixed order; rows of every (n_loads, T) array below."""
        return list(self.loads.keys())

    def _stack(self, attr):
        return np.vstack([getattr(self.loads[n], attr).value for n in self.load_order])

    @property
    def shed_matrix(self):
        """(n_loads, T) load shed fractions currently held by the variables."""
        return self._stack("load_shed")

    def x_eval(self, z):
        """Forward state: x[:, t+1] = (1 - GAMMA) x[:, t] + z[:, t], x[:, 0] = 0."""
        x = np.zeros((z.shape[0], T + 1))
        for t in range(T):
            x[:, t + 1] = (1 - GAMMA) * x[:, t] + z[:, t]
        return x

    def lam_eval(self, z):
        """Backward costate: lam[:, t] = BETA u[:, t] + (1 - GAMMA) lam[:, t+1], lam[:, T] = 0."""
        u = self._stack("demand") * z
        lam = np.zeros((z.shape[0], T + 1))
        for t in reversed(range(T)):
            lam[:, t] = BETA * u[:, t] + (1 - GAMMA) * lam[:, t + 1]
        return lam

    def J_eval(self, z):
        """True objective: sum over buses and hours of (voll + BETA x) * unserved energy."""
        x = self.x_eval(z)
        return float(np.sum((self._stack("voll") + BETA * x[:, :-1])
                            * self._stack("demand") * z))

    def H_eval(self, z):
        """Hamiltonian at z, including the constant that the LP objective drops."""
        x, lam = self.x_eval(z), self.lam_eval(z)
        u = self._stack("demand") * z
        return float(np.sum((self._stack("voll") + BETA * x[:, :-1]) * u)
                     + np.sum(lam[:, 1:] * z)
                     + (1 - GAMMA) * np.sum(lam[:, 1:] * x[:, :-1]))

    def set_costate(self, z):
        """Freeze x and lam from `z` into every load's parameters."""
        x, lam = self.x_eval(z), self.lam_eval(z)
        for i, name in enumerate(self.load_order):
            self.loads[name].set_costate(x[i], lam[i])

    def set_shed(self, z):
        """Write a shed matrix back into the load variables."""
        for i, name in enumerate(self.load_order):
            self.loads[name].load_shed.value = z[i]

    def exact_line_search(self, z_hat, z):
        """Step in [0, 1] minimising J along z_hat -> z. J is quadratic there."""
        g = z - z_hat
        phi0, phi12, phi1 = (self.J_eval(z_hat), self.J_eval(z_hat + 0.5 * g),
                             self.J_eval(z_hat + g))
        a = 2 * phi1 + 2 * phi0 - 4 * phi12
        b = phi1 - phi0 - a
        if a > 1e-12:
            return float(np.clip(-b / (2 * a), 0.0, 1.0))
        return 0.0 if phi0 <= phi1 else 1.0

    def solve_mpols(self, date, max_iter=25, tol=1e-6, verbose=False):
        """Min-H solve of multi-period load shedding with dynamic unserved energy costs.

        Returns the final objective; the iteration log is left in self.msa_history
        and the load variables hold the final iterate.
        """
        self.opf_date = date
        for obj in self.components:
            obj.update_timeseries_parameters(date)

        # Static-VoLL solution as the initial iterate.
        self.prob.solve(solver=cp.GUROBI, **self.solver_options)
        z_hat = self.shed_matrix

        rows, status = [], "max_iter"
        for k in range(max_iter):
            self.set_costate(z_hat)
            self.prob_hamiltonian.solve(solver=cp.GUROBI, **self.solver_options)
            z_star = self.shed_matrix
            step = float(np.abs(z_star - z_hat).max())
            alpha = self.exact_line_search(z_hat, z_star)
            rows.append({"iter": k, "H": self.H_eval(z_hat), "J": self.J_eval(z_hat),
                         "alpha": alpha, "step": step})
            if verbose:
                print(f"  [{k:3}] H {rows[-1]['H']:16,.3f}  J {rows[-1]['J']:16,.3f}"
                      f"  alpha {alpha:7.4f}  max|dz| {step:.3e}", flush=True)
            if step < tol:
                status = "converged"
                break
            if alpha < tol:
                # The min-H solution is not an improving direction: stalled, not solved.
                status = "stalled"
                break
            z_hat = (1 - alpha) * z_hat + alpha * z_star

        self.set_shed(z_hat)
        self.msa_history = pd.DataFrame(rows)
        self.msa_status = status
        return self.J_eval(z_hat)

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
            {name: load.unserved_energy.value for name, load in self.loads.items()},
            index=self.opf_index,
        )

    @property
    def load_shed_profile(self):
        """(T, n_loads) MW of unserved energy, one column per load."""
        return pd.DataFrame(
            {name: load.load_shed.value for name, load in self.loads.items()},
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
        df_dispatch["load"] = sum(load.demand.value for load in self.loads.values())
        df_dispatch["thermal"] = sum(resource.p_out.value for resource in self.thermal_resources.values())
        df_dispatch["solar"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "solar")
        df_dispatch["wind"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "wind")
        df_dispatch["hydro"] = sum(resource.p_out.value for resource in self.hydro_resources.values())
        df_dispatch["storage charge"] = sum(resource.charge.value for resource in self.storage_resources.values())
        df_dispatch["storage discharge"] = sum(resource.discharge.value for resource in self.storage_resources.values())
        df_dispatch["unserved energy"] = sum(load.unserved_energy.value for load in self.loads.values())

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
            for n in self.nodes.keys():
                df_LMP.loc[date:date + pd.Timedelta(hours=T - 1), n] = -self.nodes[n].power_balance_constraint.dual_value

        return df_results, df_LMP
