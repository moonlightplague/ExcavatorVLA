#!/usr/bin/env python3
"""Inspect the authoritative protocol-v2 bridge embedded in run_simulation.py.

The former version of this file started a second server on port 5555 and ran a
separate camera/control loop.  That loop used raw DOF order, captured cameras
continuously, and integrated velocity only once per camera sweep.  It must not
run beside the integrated bridge.

Run this file from Isaac Sim's Script Editor only as a bridge-status check.  The
actual server is created by ``run_simulation.py``.
"""

import builtins
import json


runtime = getattr(builtins, "_EXCAVATOR_BRIDGE_RUNTIME", None)
if not isinstance(runtime, dict):
    raise RuntimeError(
        "The authoritative ExcavatorVLA bridge is not running. Start the scene "
        "with run_simulation.py; this diagnostic no longer starts a second "
        "unsafe bridge server."
    )

print("[pbridge] authoritative bridge status:", flush=True)
print(json.dumps(runtime, indent=2, sort_keys=True), flush=True)
