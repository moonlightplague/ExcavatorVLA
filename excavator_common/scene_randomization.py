"""Pure helpers for reproducible, spatially uniform scene randomization."""

import hashlib
import json
import math


def stable_scene_seed(run_identity, attempt_index, base_seed=410700, worker_identity=""):
    payload = json.dumps(
        {
            "schema": "excavator_scene_seed_v2",
            "base_seed": int(base_seed),
            "run_identity": str(run_identity or ""),
            "worker_identity": str(worker_identity or ""),
            "attempt_index": int(attempt_index),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def area_uniform_radius(unit_sample, radius_range):
    """Map a uniform [0, 1] sample to a radius uniform over annular area."""
    r0, r1 = float(radius_range[0]), float(radius_range[1])
    lo, hi = min(r0, r1), max(r0, r1)
    if lo < 0.0:
        raise ValueError("radius range must be non-negative")
    u = min(1.0, max(0.0, float(unit_sample)))
    return math.sqrt(lo * lo + u * (hi * hi - lo * lo))
