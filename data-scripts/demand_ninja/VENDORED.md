# Vendored demand-ninja

Source: https://github.com/renewables-ninja/demand-ninja (BSD-3-Clause, see LICENSE)

Cite: Iain Staffell, Stefan Pfenninger and Nathan Johnson (2023). *A global model
of hourly space heating and cooling demand at multiple spatial scales.* Nature
Energy 8, 1328-1344. https://doi.org/10.1038/s41560-023-01341-5

Vendored rather than installed because it is not on PyPI and its requirements.txt
pins `pandas >= 1.4, < 1.5` (this project runs pandas 2.2.3).

## Local changes

1. **Humidity unit fix** (`core.py`, `_bait`). Upstream computes
   `(humidity / 1000) - setpoint_H`, converting the input to kg/kg while
   `setpoint_H = exp(1.1 + 0.06*T)` remains in g/kg. The setpoint then dominates
   by ~1000x, so the humidity *input* barely affects the result and every
   location is treated as maximally dry. Staffell's own R reference
   implementation (https://github.com/iain-staffell/demand_ninja,
   `ninja_temperature_index`) compares g/kg to g/kg with no division. Removed
   the `/1000` to match.

   Note this project sets `humidity_discomfort = 0` for the Desert Southwest
   regardless: the region is far drier than the global average the setpoint
   encodes (6.9 g/kg measured at 40 C against a setpoint of 33), so a non-zero
   coefficient drives the discomfort multiplier negative and BAIT *falls* as
   temperature rises. The fix matters only if that knob is ever turned up.

2. **pandas deprecations**: `lag[0]` -> `lag.iloc[0]` (util.py), `"12H"` ->
   `"12h"` (core.py). No behavioural change.
