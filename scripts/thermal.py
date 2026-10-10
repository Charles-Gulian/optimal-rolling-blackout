"""
Building thermal model, derived from Wang et al. (2021) ERL 16 074003.

THE single implementation of the dynamics. Both the optimiser (CoolingLoad,
for its McCormick bound and its reported trajectory) and the evaluator call
into here. They used to carry separate copies, which silently diverged by a
factor of two on the hour convention -- hence the one-implementation rule.

Chain (see the derivation notes):

    Wang eq (1):   C dT_in/dt = (T_out - T_in)/R + T_eq/R + Q_HVAC
    divide by C,   tau := RC,  T_eff := T_out + T_eq
                   dT_in/dt = (T_eff - T_in)/tau + Q_HVAC/C
    solve exactly over one step, a := exp(-dt/tau):
                   T_in,t+1 = a T_in,t + (1-a) T_eff,t - g_t
    shift x := T_in - T_set,  phi_t := (1-a)(T_eff,t - T_set):
                   x_{t+1} = a x_t + phi_t - g_t
    thermostat (g is a *controller*, not a decision):
                   x_{t+1} = max(0, a x_t + phi_t - gbar (1 - z_t))

``g`` is defined as the number of degC the HVAC removes over the step, i.e.
``g := -(1-a) tau Q_HVAC / C``. That single change of variable retires both the
thermal capacitance C and the equipment COP from the model -- neither is
obtainable from Wang, and neither is ever needed.

HOUR-BEGINNING CONVENTION
-------------------------
``x_t`` is the excess the household INHERITS entering step t, fixed by history
up to t-1. ``simulate`` returns that series. The cost is v(x_t) * u_t: this
step's unserved energy priced at the stress level the household is actually
sitting at when the step starts. Pricing ``x_{t+1}`` instead would multiply
this step's demand by next step's stress.

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
early in the step partially leaks back out before it ends. Because we define
``gbar`` directly in degC/h that factor is already absorbed and never appears.
But anyone deriving ``gbar`` from a nameplate capacity in kW must carry
``(1-a) tau / C`` through the conversion or they will oversize the AC by ~3%.
"""

from __future__ import annotations

import numpy as np

# Wang Fig. 5/6: cooling-season TTC medians run 14.2-17.3 h across Californian
# cities, with a within-city IQR of roughly 12-18 h.
WANG_TTC_HOURS = 15.0
# Wang Fig. 6: median equivalent temperature of solar + internal heat gains.
WANG_T_EQ = 12.0
# Wang section 4.1: overheating thresholds (NOAA heat index / CIBSE). Recorded
# for reference only -- the VOLL model prices discomfort from the setpoint up,
# with no separate threshold, since choosing a setpoint is itself a statement
# that you would rather not be warmer.
WANG_THRESHOLD_C = 28.0
WANG_THRESHOLD_SEVERE_C = 32.0


def retention(tau_hours, dt_hours: float = 1.0):
    """``a = exp(-dt/tau)`` -- fraction of the gap to T_eff surviving one step.

    Exact discretisation rather than forward Euler (which gives ``1 - dt/tau``).
    The difference is small at tau = 15 h (0.9355 vs 0.9333) but the exact form
    costs one exp() at setup and is unconditionally stable for every tau > 0,
    whereas Euler requires dt < tau.
    """
    return np.exp(-dt_hours / np.asarray(tau_hours, dtype=float))


def hour_average_weights(tau_hours, dt_hours: float = 1.0):
    """``(kappa, c)`` for the exact mean excess over one unserved step.

    Integrating the continuous solution across a step during which the AC is
    off, starting from ``x_t``:

        xbar_t = kappa x_t + c phi_t,
        kappa  = (tau/dt)(1-a),      c = 1/(1-a) - tau/dt

    That mean is what the household actually experiences, so it is what the
    cost should price -- not either endpoint. ``kappa`` is the memory: it tends
    to 1 as tau/dt grows (the house keeps everything it had) and to 0 as tau/dt
    shrinks (it forgets within the step, at which point state-dependent VOLL
    collapses into the exogenous model). ``c`` runs from 1/2 at large tau/dt,
    where the house heats near-linearly so the mean is the midpoint, to 1 at
    small tau/dt, where it equilibrates immediately and sits at T_eff
    throughout.

    At tau = 15 h and dt = 1 h: kappa = 0.9674, c = 0.5056. The endpoint
    average (x_t + x_{t+1})/2 gives 0.9678 and 0.5000 -- fine here, and wrong
    by 14% at tau = 1 h.
    """
    a = retention(tau_hours, dt_hours)
    n = np.asarray(tau_hours, dtype=float) / dt_hours
    return n * (1.0 - a), 1.0 / (1.0 - a) - n


def forcing(t_out, t_eq: float, t_set: float, a):
    """``phi_t = (1-a)(T_out + T_eq - T_set)``, degC of heating per step, AC off.

    Clipped at zero: below setpoint there is no cooling season and the model
    does not apply. Retaining negative forcing would let the house drift below
    setpoint, which the thermostat prevents in reality and which would break
    the monotonicity argument licensing the inequality form of the dynamics.
    """
    return np.maximum(0.0, (1.0 - np.asarray(a)) * (np.asarray(t_out) + t_eq - t_set))


def simulate(phi, z, a, gbar, x0=0.0):
    """Roll the exact bang-bang thermostat forward. Ground truth for the MILP.

    Accepts ``(T,)`` for one load or ``(nb, T)`` for many; ``a`` and ``gbar``
    broadcast against the leading axis.

    Returns the HOUR-BEGINNING series: element t is the excess inherited
    entering step t, so element 0 is ``x0`` and the last step's shedding
    decision never appears.
    """
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    zz = np.atleast_2d(np.asarray(z, dtype=float))
    nb, n = phi.shape
    a = np.broadcast_to(np.asarray(a, dtype=float).reshape(-1, 1), (nb, 1))
    g = np.broadcast_to(np.asarray(gbar, dtype=float).reshape(-1, 1), (nb, 1))
    x = np.full((nb,), float(x0)) if np.isscalar(x0) else np.asarray(x0, dtype=float).copy()
    out = np.zeros((nb, n))
    for t in range(n):
        out[:, t] = x
        # Powered: cool toward setpoint, capped at rated capacity. Shed: nothing.
        cool = np.where(zz[:, t] > 0.5, 0.0, g[:, 0])
        x = np.maximum(0.0, a[:, 0] * x + phi[:, t] - cool)
    return out[0] if np.asarray(phi).shape[0] == 1 and np.ndim(z) == 1 else out


def state_bound(phi, a):
    """Tightest valid upper bound on x, for the McCormick relaxation.

    Cooling never raises x and ``a >= 0``, so the never-served trajectory
    (g == 0 throughout) dominates every schedule by induction. Unrolling it,

        x_max,t = sum_{k<t} a^(t-1-k) phi_k

    which is the exact never-served path and therefore attained, not merely
    valid. The looser closed form ``max_t(T_eff) - T_set`` follows by bounding
    phi_k and letting the geometric sum telescope against the (1-a) prefactor;
    it is horizon-independent and physically obvious, but it can be ~80% loose
    over a 24 h window, which weakens the LP relaxation.

    Validity matters, not just tightness: if x could exceed this bound, the
    McCormick lower bound ``w >= x - x_max (1-z)`` would become positive at
    z = 0, charging outage cost to a bus that was never blacked out.
    """
    phi_2d = np.atleast_2d(np.asarray(phi, dtype=float))
    return simulate(phi_2d, np.ones_like(phi_2d), a, 0.0,
                    x0=np.zeros(phi_2d.shape[0]))[0 if np.ndim(phi) == 1 else slice(None)]


def hours_to_indoor(target_c: float, t_out: float, t_eq: float = WANG_T_EQ,
                    t_set: float = 21.0, tau_hours: float = WANG_TTC_HOURS) -> float:
    """Free-float time to reach ``target_c`` indoors, in hours (continuous time).

    Inverts ``T_in(t) = T_eff - (T_eff - T_set) exp(-t/tau)``. Used to check the
    parameter set against Wang's headline result.
    """
    t_eff = t_out + t_eq
    if target_c >= t_eff:
        return float("inf")
    return -tau_hours * np.log((t_eff - target_c) / (t_eff - t_set))
