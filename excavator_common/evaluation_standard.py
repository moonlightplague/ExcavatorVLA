"""Versioned excavation scoring rules, shared by simulator, evaluator and reports."""
from dataclasses import dataclass
import math


EVALUATION_STANDARD = 'excavator_success_v2'
TRUCK_COUNT_METHOD = 'current_bed_local_box_excluding_bucket_v3'
BUCKET_OPEN_RAD = -0.8
COMPLETION_OBSERVATIONS = 3


@dataclass
class CompletionTracker:
    """Require threshold payload AND an open bucket on consecutive action observations.

    Increasing payload is welcome. The initial observation is never submitted.
    Success ends the rollout immediately, so completion cannot later be revoked.
    """
    minimum: int
    consecutive: int = 0

    def __post_init__(self):
        if type(self.minimum) is not int or self.minimum < 1:
            raise ValueError('Minimum payload must be a positive integer')

    def observe(self, payload, bucket_position):
        if type(payload) is not int or payload < 0 or not math.isfinite(bucket_position):
            raise ValueError('Invalid payload or bucket position')
        qualifies = payload >= self.minimum and bucket_position >= BUCKET_OPEN_RAD
        self.consecutive = self.consecutive + 1 if qualifies else 0
        return self.consecutive >= COMPLETION_OBSERVATIONS


def execution_profile(fresh_simulator, render_every_physics_step, headless, reference=False):
    return ('standard' if fresh_simulator and render_every_physics_step and headless and not reference
            else 'diagnostic')


def canonical_joint_limits(articulation, indices, np_module):
    """Read legacy or current Isaac public APIs; never silently disable clipping."""
    reader = getattr(articulation, 'get_dof_limits', None)
    if callable(reader):
        raw = np_module.asarray(reader(), dtype=np_module.float32)
    else:
        properties = articulation.dof_properties
        raw = np_module.column_stack((properties['lower'], properties['upper'])).astype(np_module.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError('Invalid articulation joint-limit shape')
    limits = raw[indices].copy()
    if (limits.shape != (4, 2) or np_module.isnan(limits).any()
            or np_module.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError('Invalid canonical joint limits')
    return limits
