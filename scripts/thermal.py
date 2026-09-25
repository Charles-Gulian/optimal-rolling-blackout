"""
Building thermal model, derived from Wang et al. (2021) ERL 16 074003.

Chain (see the derivation notes):

    Wang eq (1):   C dT_in/dt = (T_out - T_in)/R + T_eq/R + Q_HVAC
    divide by C,   tau := RC,  T_eff := T_out + T_eq
                   dT_in/dt = (T_eff - T_in)/tau + Q_HVAC/C
    solve exactly over one hour, a := exp(-1/tau):
                   T_in,t+1 = a T_in,t + (1-a) T_eff,t - g_t
    shift x := T_in - T_set,  phi_t := (1-a)(T_eff,t - T_set):
                   x_{t+1} = a x_t + phi_t - g_t
    thermostat (g is a *controller*, not a decision):
                   x_{t+1} = max(0, a x_t + phi_t - gbar (1 - z_t))

``g`` is defined as the number of degC the HVAC removes over the hour, i.e.
``g := -(1-a) tau Q_HVAC / C``. That single change of variable retires both the
thermal capacitance C and the equipment COP from the model -- neither is
obtainable from Wang, and neither is ever needed.

Why the state equation is linear in z
-------------------------------------
``z`` enters *additively*, because the air conditioner is rate-limited. Had we
assumed the AC restores setpoint instantaneously we would have
``x_{t+1} = z_t (a x_t + phi_t)``, which is bilinear in (x, z) and much harder.
Rate-limiting is both the more realistic assumption and the one that keeps the
optimisation linear.

Interpretation trap
-------------------
The coefficient on ``Q_HVAC/C`` in the exact discretisation is ``(1-a) tau``,
not ``dt`` -- 0.967 h rather than 1 h at tau = 15 h, because cooling delivered
early in the hour partially leaks back out before the hour ends. Because we
define ``gbar`` directly in degC/h that factor is already absorbed and never
appears. But anyone deriving ``gbar`` from a nameplate capacity in kW must carry
``(1-a) tau / C`` through the conversion or they will oversize the AC by ~3%.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Wang Fig. 5/6: cooling-season TTC medians run 14.2-17.3 h across Californian
# cities, with a within-city IQR of roughly 12-18 h.
WANG_TTC_HOURS = 15.0
# Wang Fig. 6: median equivalent temperature of solar + internal heat gains.
WANG_T_EQ = 12.0
# Wang section 3.2: assumed indoor temperature at outage onset, un-notified.
WANG_T_SET = 24.0
WANG_T_SET_PRECOOLED = 22.0
# Wang section 4.1: overheating thresholds (NOAA heat index / CIBSE).
WANG_THRESHOLD_C = 28.0
WANG_THRESHOLD_SEVERE_C = 32.0


def retention(tau_hours: float | np.ndarray, dt_hours: float = 1.0) -> np.ndarray:
    """``a = exp(-dt/tau)`` -- fraction of the gap to T_eff surviving one step.

    Exact discretisation rather than forward Euler (which gives ``1 - dt/tau``).
    The difference is small at tau = 15 h (0.9355 vs 0.9333, ~3% over 12 h) but
    the exact form costs one exp() at setup and is unconditionally stable for
    every tau > 0, whereas Euler requires dt < tau. Wang's TTC distribution has
    a lower whisker near 5 h, so fitting tau per household could produce short
    draws; this removes that failure mode entirely.
    """
    return np.exp(-dt_hours / np.asarray(tau_hours, dtype=float))


@dataclasses.dataclass
class ThermalParams:
    """Per-bus thermal parameters, all in degC and dimensionless."""

    a: np.ndarray        # (nb,) retention, exp(-1/tau)
    gbar: np.ndarray     # (nb,) rated cooling capability, degC/h
    t_set: float         # cooling setpoint, degC
    x_thr: float         # discomfort knee above setpoint, degC

    @classmethod
    def build(
        cls,
        n_bus: int,
        tau_hours: float | np.ndarray = WANG_TTC_HOURS,
        t_set: float = WANG_T_SET,
        threshold_c: float = WANG_THRESHOLD_C,
        t_design: float = 43.0,
        t_eq_design: float = WANG_T_EQ,
        sizing_margin: float = 1.15,
    ) -> "ThermalParams":
        """Construct parameters, sizing the AC from design conditions.

        ``gbar`` is the one parameter Wang cannot supply -- she models free-float
        only, with no HVAC at all. We size it the way equipment is actually
        sized: it must just hold setpoint at the design outdoor temperature,
        times a margin.

            gbar = m (1-a) (T_design + T_eq - T_set)

        With real data ``t_design`` should be the 99.6th percentile of hourly
        dry-bulb at each bus, which is exactly the ASHRAE 0.4% cooling design
        condition -- so it comes from the NSRDB pull rather than a lookup table.

        Note this makes rebound headroom *shrink* as it gets hotter: the ratio
        gbar/phi_t falls toward 1 on the worst days, so there is nearly no
        recovery capability precisely when it is most needed. A constant
        multiplicative headroom would get that backwards.
        """
        a = np.broadcast_to(retention(tau_hours), (n_bus,)).astype(float).copy()
        gbar = sizing_margin * (1.0 - a) * (t_design + t_eq_design - t_set)
        return cls(a=a, gbar=gbar, t_set=t_set, x_thr=threshold_c - t_set)


def thermal_forcing(t_eff: np.ndarray, t_set: float, a: np.ndarray) -> np.ndarray:
    """``phi_t = (1-a)(T_eff,t - T_set)`` -- degC/h of heating with the AC off.

    Clipped at zero: when it is cooler outside than the setpoint there is no
    cooling load and the cooling-season model does not apply. Retaining negative
    forcing would let the house drift below setpoint, which the thermostat
    prevents in reality and which would break the monotonicity argument that
    licenses the inequality form of the dynamics.
    """
    return np.maximum(0.0, (1.0 - a)[:, None] * (t_eff - t_set))


def simulate(phi: np.ndarray, z: np.ndarray, params: ThermalParams,
             x0: np.ndarray | None = None) -> np.ndarray:
    """Roll the exact bang-bang thermostat forward. Ground truth for the MILP.

    Parameters
    ----------
    phi : (nb, T) thermal forcing
    z   : (nb, T) blackout indicator, 1 = de-energised

    Returns
    -------
    (nb, T) indoor excess over setpoint at the *start* of each hour.
    """
    nb, T = phi.shape
    a, gbar = params.a, params.gbar
    x = np.zeros(nb) if x0 is None else np.asarray(x0, dtype=float).copy()
    out = np.zeros((nb, T))
    for t in range(T):
        out[:, t] = x
        drift = a * x + phi[:, t]
        # Powered: cool toward setpoint, capped at rated capacity. Shed: nothing.
        g = np.where(z[:, t] > 0.5, 0.0, np.minimum(gbar, np.maximum(0.0, drift)))
        x = np.maximum(0.0, drift - g)
    return out


def state_bound(phi: np.ndarray, params: ThermalParams) -> np.ndarray:
    """(nb, T) tightest valid upper bound on x, for the McCormick relaxation.

    Cooling never raises x and ``a >= 0``, so the never-served trajectory
    (g == 0 throughout) dominates every schedule by induction. Unrolling it,

        x_max,t = sum_{k<t} a^{t-1-k} phi_k

    which is the exact never-served path and therefore attained, not merely
    valid. The looser closed form ``max_t(T_eff) - T_set`` follows by bounding
    phi_k and letting the geometric sum telescope against the (1-a) prefactor;
    it is horizon-independent and physically obvious (the house cannot get
    hotter than the hottest effective outdoor temperature), but it can be ~80%
    loose over a 24 h window, which weakens the LP relaxation.

    Validity matters, not just tightness: if x could exceed this bound, the
    McCormick lower bound ``w >= x - x_max (1-z)`` would become positive at
    z = 0, charging outage cost to a bus that was never blacked out.
    """
    nb, T = phi.shape
    a = params.a
    out = np.zeros((nb, T))
    x = np.zeros(nb)
    for t in range(T):
        out[:, t] = x
        x = a * x + phi[:, t]
    return out


def hours_to_indoor(target_c: float, t_out: float, t_eq: float = WANG_T_EQ,
                    t_set: float = WANG_T_SET, tau_hours: float = WANG_TTC_HOURS
                    ) -> float:
    """Free-float time to reach ``target_c`` indoors, in hours (continuous time).

    Inverts ``T_in(t) = T_eff - (T_eff - T_set) exp(-t/tau)``. Used to check the
    parameter set against Wang's headline result.
    """
    t_eff = t_out + t_eq
    if target_c >= t_eff:
        return float("inf")
    return -tau_hours * np.log((t_eff - target_c) / (t_eff - t_set))
