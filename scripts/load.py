import numpy as np
import pandas as pd
import cvxpy as cp

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
    dynamic_voll = False

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
    """Cooling demand, optionally priced at a VOLL that rises as the house heats.

    With `dynamic_voll` off this behaves exactly like a plain Load. With it on,
    the load carries the Wang et al. (2021) ERL 16 074003 1R-1C thermal state
    and prices unserved energy at

        v_t = VOLL + rho * x_t

    where x_t is indoor temperature excess over setpoint, floored at zero by
    the thermostat. Shedding a bus that has already been dark for hours
    therefore costs more than shedding a cool one, which is what makes rolling
    the blackout worth something.

    Discomfort starts AT the setpoint: there is no separate threshold, because
    choosing a setpoint is itself a statement that you would rather not be
    warmer. Wang's 28 degC overheating limit is a health threshold and a
    different quantity. Dropping it also removes a variable -- with a knee at
    zero the hinge max(0, x - x_thr) collapses to x.

    See scripts/thermal.py for the derivation of the dynamics, and
    data-scripts/make_load_csv.py for where each parameter comes from.
    """

    def __init__(self, name, data):
        super().__init__(name, data)

        self.tau = self._opt(data, "Tau Hr")           # h, thermal time constant
        self.gbar = self._opt(data, "Gbar degC/Hr")    # degC/h, rated cooling
        self.t_set = self._opt(data, "T Set degC")     # degC, cooling setpoint
        self.t_eq = self._opt(data, "T Eq degC")       # degC, solar+internal gains
        self.rho = self._opt(data, "Rho $/MWh/degC")   # $/MWh per degC of excess

        self.dynamic_voll = False   # System.enable_dynamic_voll() flips this
        self.temperature = None     # degC dry-bulb, UTC clock

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
    def a(self):
        """Retention exp(-1/tau): fraction of the gap to T_eff surviving 1 h."""
        return float(np.exp(-1.0 / self.tau))

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
        if self.dynamic_voll:
            # phi_t: degC/h the house heats with the AC off.
            self.phi = cp.Parameter(T, nonneg=True)
            # x_max,t: tightest valid upper bound on the state, needed by the
            # McCormick envelope. Time-varying, so the envelope is much
            # tighter than a horizon-wide constant would give.
            self.x_max = cp.Parameter(T, nonneg=True)

    def create_variables(self):
        super().create_variables()
        if self.dynamic_voll:
            self.x = cp.Variable(T, nonneg=True)   # indoor excess over setpoint
            self.w = cp.Variable(T, nonneg=True)   # McCormick: w = x * z

    def update_timeseries_parameters(self, date: pd.Timestamp):
        super().update_timeseries_parameters(date)
        if not self.dynamic_voll:
            return
        # phi_t = (1-a)(T_eff,t - T_set), T_eff = dry-bulb + T_eq.
        # Clipped at zero: below setpoint there is no cooling season and the
        # model does not apply. Keeping negative forcing would let the house
        # drift below setpoint -- which the thermostat prevents in reality,
        # and which would break the monotonicity that licenses the inequality
        # form of the dynamics.
        t_out = self.temperature.loc[date:date + pd.Timedelta(hours=T - 1)].values
        self.phi.value = np.maximum(
            0.0, (1.0 - self.a) * (t_out + self.t_eq - self.t_set))
        self.x_max.value = self._state_bound(self.phi.value)

    def _state_bound(self, phi):
        """(T,) upper bound on the state x_t, for the McCormick envelope.

        Cooling never raises x and a >= 0, so the never-served trajectory
        (g == 0 throughout) dominates every schedule by induction:

            x_max,t = sum_{k<t} a^(t-1-k) phi_k

        That is the exact never-served path, so the bound is attained, not
        merely valid -- which matters twice over. Too loose and the LP
        relaxation weakens; too tight and `w >= x - x_max (1-z)` would go
        positive at z = 0, charging outage cost to a bus that never went dark.
        """
        a, x, out = self.a, 0.0, np.zeros(T)
        for t in range(T):
            out[t] = x
            x = a * x + phi[t]
        return out

    def write_constraints(self):
        constraints = super().write_constraints()
        if not self.dynamic_voll:
            return constraints

        z = self.node.blackout   # (T,) binary, 1 = bus de-energised
        if z is None:
            raise RuntimeError(
                f"{self.name}: dynamic VOLL needs the bus blackout binary. "
                f"Call System.enable_blackouts() before write_opf().")

        # 1. Thermal state. Exact dynamics are
        #        x_{t+1} = max(0, a x_t + phi_t - gbar (1 - z_t)),
        #    relaxed to >=. The relaxation is tight because the objective is
        #    nondecreasing in x (via s, then w) whenever rho >= 0, so the
        #    minimiser drives every x_t down onto its floor. z enters
        #    ADDITIVELY, not multiplicatively, because the AC is rate-limited;
        #    assuming instantaneous recovery to setpoint would give
        #    x_{t+1} = z_t (a x_t + phi_t), bilinear and far harder.
        #    x_0 = 0: each day starts at setpoint. Overnight recovery is fast
        #    relative to a day, so intra-day carryover is the part that counts.
        constraints += [self.x[0] >= self.phi[0] - self.gbar * (1 - z[0])]
        constraints += [
            self.x[1:] >= self.a * self.x[:-1] + self.phi[1:]
            - self.gbar * (1 - z[1:])
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
        """(T,) exact indoor excess over setpoint, rolled forward from solved z.

        Use this for reporting, NOT self.x. The `>=` relaxation is tight only
        where the objective actually sees x, i.e. on hours feeding a shed hour
        whose discomfort clears the knee. Everywhere else the solver is
        indifferent and x floats anywhere above its floor -- we have measured
        it 15 degC high. That costs nothing (the objective is nondecreasing in
        x, so slack never buys anything) but makes self.x useless as a
        diagnostic. Once z is known the exact bang-bang recursion is trivial,
        so there is no reason to report the relaxed value.
        """
        if not self.dynamic_voll:
            return None
        z = np.round(self.node.blackout.value)
        a, g, phi = self.a, self.gbar, self.phi.value
        x, out = 0.0, np.zeros(T)
        for t in range(T):
            drift = (a * x if t else 0.0) + phi[t]
            # Powered: cool toward setpoint, capped at rated capacity.
            x = max(0.0, drift - (0.0 if z[t] > 0.5 else g))
            out[t] = x
        return out

    def cost_expression(self):
        """Unserved-energy cost, flat or thermal-state dependent.

        Static:   VOLL * u_t
        Dynamic:  (VOLL + rho * x_t) * u_t
                = VOLL * u_t + rho * d_t * x_t * z_t
                = VOLL * u_t + rho * d_t * w_t

        The middle step uses u_t = d_t * z_t, which Node already enforces, and
        the last replaces the only bilinear term with its McCormick surrogate.
        d_t is a Parameter, so rho * d_t * w_t is affine in w and the problem
        stays a MILP.
        """
        cost = super().cost_expression()
        if self.dynamic_voll:
            cost = cost + cp.sum(self.rho * cp.multiply(self.demand, self.w))
        return cost
