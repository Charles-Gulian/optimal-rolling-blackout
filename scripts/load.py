import numpy as np
import pandas as pd
import cvxpy as cp

import thermal
from component import Component, T


class Load(Component):
    """Electricity demand of one type at one node, priced at a flat VOLL.

    A node carries one Load per load type, because the types are priced
    differently -- which is the whole point of the dynamic VOLL model. Demand,
    unserved energy and VOLL therefore live here rather than on Node.

    The blackout decision stays on Node, since a feeder outage de-energises
    every load at the bus at once. Moving it here would model load-level
    control, which is a separate question.

    Subclasses may price unserved energy on something richer than a constant
    by overriding `cost_expression`; see CoolingLoad.
    """

    # Profile filename per load type, under load-data/profiles/bus{bus}/.
    PROFILE_PATTERNS = {
        "baseline": "NSRDB_load-profile_bus{bus}_{year}.csv",
        "cooling": "NSRDB_cooling-load-profile_bus{bus}_{year}.csv",
        "other": "NSRDB_other-load-profile_bus{bus}_{year}.csv",
    }

    # Overridden per instance by subclasses that support it. Declared here so
    # System can read it off any load without an isinstance check.
    voll_model = "static"

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack load attributes from data
        self.node_ID = data["Bus ID"]
        self.load_type = data["Load Type"]
        self.peak_load = data["Peak MW"]  # MW, this load type's own peak
        self.VOLL = data["VOLL"]  # $/MWh of unserved energy

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
        fname = self.PROFILE_PATTERNS[self.load_type].format(bus=self.node_ID, year=year)
        ts_data_path = self.system.base_dir / "load-data" / "profiles" / f"bus{self.node_ID}" / fname
        df_load = pd.read_csv(ts_data_path, index_col=[0])
        df_load.index = pd.to_datetime(df_load.index)
        self.load_profile = df_load.squeeze()

    @property
    def needs_thermal(self):
        """True when this load's VOLL model needs phi (hence temperature)."""
        return False

    def create_parameters(self):
        self.demand = cp.Parameter(T, nonneg=True)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.demand.value = self.peak_load * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values

    def create_variables(self):
        self.unserved_energy = cp.Variable(T, nonneg=True)

    def write_constraints(self):
        return [self.unserved_energy <= self.demand]  # Limit unserved energy

    def cost_expression(self):
        """Unserved-energy cost. Called by System when assembling the objective."""
        return cp.sum(self.VOLL * self.unserved_energy)


class CoolingLoad(Load):
    """Cooling demand, priced under one of three VOLL models.

    All three price the same quantity -- energy not served, valued at the
    temperature excess the household experiences over the hour it goes without
    -- and differ only in which arguments v is allowed to see:

        static      v_t = VOLL                                   flat
        exogenous   v_t = VOLL + rho (c phi_t)                   weather only
        dynamic     v_t = VOLL + rho (kappa x_t + c phi_t)       + outage history

    x_t is the excess the household INHERITS entering hour t, carried by the
    Wang et al. (2021) ERL 16 074003 1R-1C state. Shedding a bus that has
    already been dark therefore costs more than shedding a cool one, which is
    what makes rolling the blackout worth something -- and the exogenous model
    is blind to exactly that, so the gap between the two isolates the value of
    knowing history.

    Discomfort starts AT the setpoint, with no separate threshold: choosing a
    setpoint is itself a statement that you would rather not be warmer. Wang's
    28 degC overheating limit is a health threshold and a different quantity.

    scripts/thermal.py owns the dynamics; this class only builds the
    optimisation around them. data-scripts/make_load_csv.py documents where
    each parameter comes from.
    """

    def __init__(self, name, data):
        super().__init__(name, data)

        self.tau = self._opt(data, "Tau Hr")           # h, thermal time constant
        self.gbar = self._opt(data, "Gbar degC/Hr")    # degC/h, rated cooling
        self.t_set = self._opt(data, "T Set degC")     # degC, cooling setpoint
        self.t_eq = self._opt(data, "T Eq degC")       # degC, solar+internal gains
        self.rho = self._opt(data, "Rho $/MWh/degC")   # $/MWh per degC of excess

        self.voll_model = "static"   # System.set_voll_model() changes this
        self.temperature = None      # degC dry-bulb, UTC clock

    @staticmethod
    def _opt(data, key):
        """Read an optional numeric column; None when absent or blank."""
        if key not in data:
            return None
        value = data[key]
        return None if pd.isna(value) else float(value)

    @property
    def has_thermal_model(self):
        """True when load.csv supplied every Wang parameter for this load."""
        return all(getattr(self, attr) is not None for attr in
                   ("tau", "gbar", "t_set", "t_eq", "rho"))

    @property
    def needs_thermal(self):
        """True when this load's VOLL model needs phi (hence temperature)."""
        return self.voll_model in ("exogenous", "dynamic") and self.has_thermal_model

    @property
    def a(self):
        """Retention exp(-1/tau): fraction of the gap to T_eff surviving 1 h."""
        return float(thermal.retention(self.tau))

    @property
    def kappa(self):
        """Weight on inherited stress in the hour-average. See thermal.py.

        The memory term: 1 means the house keeps everything it had, 0 means it
        forgets within the hour, at which point dynamic VOLL collapses into
        the exogenous model. 0.9674 at tau = 15 h.
        """
        return float(thermal.hour_average_weights(self.tau)[0])

    @property
    def c(self):
        """Weight on this hour's forcing in the hour-average. See thermal.py.

        0.5056 at tau = 15 h, which is why phi_t/2 is a good approximation
        here and a poor one for a leaky house.
        """
        return float(thermal.hour_average_weights(self.tau)[1])

    def get_temperature_profile(self, year):
        """Hourly dry-bulb at this bus, degC, on the UTC clock.

        Same NSRDB pull that cooling_data.py consumes. TIMEZONE: stored in UTC
        like every other profile, so it needs NO shift -- phi lines up with
        demand hour for hour. cooling_data.py shifts to local only because
        demand.ninja applies a diurnal profile by index.hour; nothing here
        does, so shifting would reintroduce the 7-hour error.
        """
        path = (self.system.base_dir / "load-data" / "inputs" / f"bus{self.node_ID}"
                / f"NSRDB_weather-inputs_bus{self.node_ID}_{year}.csv")
        df = pd.read_csv(path, index_col=[0], usecols=[0, 4])
        df.index = pd.to_datetime(df.index)
        self.temperature = df.squeeze()

    def create_parameters(self):
        super().create_parameters()
        if self.needs_thermal:
            # phi_t: degC/h the house heats with the AC off.
            self.phi = cp.Parameter(T, nonneg=True)
            # d_t * phi_t, precomputed. The cost needs this product, and
            # multiplying two Parameters together is not DPP -- CVXPY would
            # recompile on every solve, defeating the compile-once design.
            self.d_phi = cp.Parameter(T, nonneg=True)
        if self.voll_model == "dynamic":
            # Tightest valid upper bound on the state, for the McCormick
            # envelope. Time-varying, so much tighter than a constant.
            self.x_max = cp.Parameter(T, nonneg=True)

    def create_variables(self):
        super().create_variables()
        if self.voll_model == "dynamic":
            self.x = cp.Variable(T, nonneg=True)   # excess entering each hour
            self.w = cp.Variable(T, nonneg=True)   # McCormick: w = x * z

    def update_timeseries_parameters(self, date: pd.Timestamp):
        super().update_timeseries_parameters(date)
        if not self.needs_thermal:
            return
        t_out = self.temperature.loc[date:date + pd.Timedelta(hours=T - 1)].values
        self.phi.value = thermal.forcing(t_out, self.t_eq, self.t_set, self.a)
        self.d_phi.value = self.demand.value * self.phi.value
        if self.voll_model == "dynamic":
            self.x_max.value = thermal.state_bound(self.phi.value, self.a)

    def write_constraints(self):
        constraints = super().write_constraints()
        # Only the dynamic model carries state. The exogenous model prices
        # phi_t alone, which is a Parameter, so it needs no variables, no
        # dynamics and no McCormick envelope -- it stays a plain MILP.
        if self.voll_model != "dynamic":
            return constraints

        z = self.node.blackout   # (T,) binary, 1 = bus de-energised
        if z is None:
            raise RuntimeError(
                f"{self.name}: dynamic VOLL needs the bus blackout binary. "
                f"Call System.enable_blackouts() before write_opf().")

        # 1. Thermal state, HOUR-BEGINNING convention:
        #        x_{t+1} = max(0, a x_t + phi_t - gbar (1 - z_t))
        #    so x_t is what the household inherits entering hour t, fixed by
        #    history up to t-1, and the cost below prices this hour's demand
        #    at that inherited stress. Charging x_{t+1} would multiply this
        #    hour's demand by next hour's stress.
        #
        #    Relaxed to >=, tight because the objective is nondecreasing in x
        #    whenever rho >= 0, so the minimiser drives every x_t onto its
        #    floor. z enters ADDITIVELY because the AC is rate-limited;
        #    assuming instantaneous recovery would give
        #    x_{t+1} = z_t (a x_t + phi_t), bilinear and far harder.
        constraints += [self.x[0] == 0.0]   # at setpoint entering the day
        constraints += [
            self.x[1:] >= self.a * self.x[:-1] + self.phi[:-1]
            - self.gbar * (1 - z[:-1])
        ]

        # 2. McCormick envelope for w = x * z, exact for binary z given
        #    0 <= x <= x_max. The lower bounds are what price the outage; the
        #    upper bounds are slack at the optimum but keep w equal to the true
        #    product in the returned solution, so w is safe to report.
        constraints += [
            self.w >= self.x - cp.multiply(self.x_max, 1 - z),
            self.w <= self.x,
            self.w <= cp.multiply(self.x_max, z),
        ]
        return constraints

    def thermal_trajectory(self):
        """(T,) exact hour-beginning excess, rolled forward from the solved z.

        Use this for reporting, NOT self.x. The `>=` relaxation is tight only
        where the objective actually sees x; elsewhere the solver is
        indifferent and x floats anywhere above its floor -- we have measured
        it 15 degC high. That costs nothing (the objective is nondecreasing in
        x, so slack never buys anything) but makes self.x useless as a
        diagnostic. Once z is known the recursion is exact and trivial.
        """
        if not self.needs_thermal:
            return None
        return thermal.simulate(self.phi.value,
                                np.round(self.node.blackout.value),
                                self.a, self.gbar)

    def cost_expression(self):
        """Unserved-energy cost under this load's VOLL model.

        xbar_t = kappa x_t + c phi_t is the exact time-average of the excess
        over one unserved hour (see thermal.hour_average_weights). It splits
        into stress INHERITED from earlier outages and stress GENERATED this
        hour by the weather. The exogenous model keeps only the second, so it
        prices every hour as if the household entered it at setpoint --
        correct for the first hour of any outage, progressively too cheap
        after that.

        Expanding v_t * u_t with u_t = d_t z_t, which Node enforces:

            VOLL u_t  +  rho kappa d_t w_t  +  rho c (d_t phi_t) z_t

        with w_t = x_t z_t from the McCormick envelope. Both products are
        Parameter-times-Variable, so the objective is affine and the problem
        stays a MILP. d_t phi_t is precomputed into one Parameter because a
        product of two Parameters is not DPP.
        """
        cost = super().cost_expression()
        if self.voll_model == "static":
            return cost
        if self.voll_model not in ("exogenous", "dynamic"):
            raise ValueError(f"{self.name}: unknown voll_model {self.voll_model!r}")

        # Flow term: present in both exogenous and dynamic.
        cost = cost + self.rho * self.c * cp.sum(
            cp.multiply(self.d_phi, self.node.blackout))
        # State term: dynamic only.
        if self.voll_model == "dynamic":
            cost = cost + self.rho * self.kappa * cp.sum(
                cp.multiply(self.demand, self.w))
        return cost
