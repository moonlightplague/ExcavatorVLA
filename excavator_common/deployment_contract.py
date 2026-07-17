"""Pure-Python helpers for the simulator/SmolVLA deployment contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from excavator_common import vla_observation_contract


PROTOCOL_VERSION = 2
PHYSICS_HZ = 60.0
CANONICAL_DOF_NAMES = ("swing", "boom", "arm", "bucket")
STATE_NAMES_18D = (
    "base_x",
    "base_y",
    "base_yaw",
    "swing",
    "boom",
    "arm",
    "bucket",
    "bucket_load_estimate",
    "bucket_tip_x",
    "bucket_tip_y",
    "bucket_tip_z",
    "bucket_load_x",
    "bucket_load_y",
    "bucket_load_z",
    "swing_measured_effort",
    "boom_measured_effort",
    "arm_measured_effort",
    "bucket_measured_effort",
)
ACTION_NAMES_4D = tuple(f"{name}_cmd_velocity" for name in CANONICAL_DOF_NAMES)
CAMERA_KEYS = tuple(f"observation.images.{index}" for index in range(3))
OBSERVATION_SCHEMA_LEGACY_18D = "legacy_18d_state"
OBSERVATION_SCHEMA_27D_PLUS_EFFORT = "state_27d_plus_effort_4d"
OBSERVATION_SCHEMA_28D_PLUS_EFFORT = "state_28d_plus_effort_4d"
STATE_NAMES_28D = tuple(vla_observation_contract.STATE_NAMES_28D)
STATE_NAMES_27D = tuple(
    name for name in STATE_NAMES_28D if name != "phase_index"
)
EFFORT_NAMES_4D = tuple(vla_observation_contract.EFFORT_NAMES_4D)


def _name_tokens(value):
    text = str(value).strip().lower()
    for separator in ("-", ".", ":", "/", " "):
        text = text.replace(separator, "_")
    return tuple(token for token in text.split("_") if token)


def resolve_canonical_dof_indices(raw_dof_names):
    """Resolve canonical joints strictly, rejecting missing or ambiguous names."""
    raw_names = [str(name) for name in raw_dof_names]
    if not raw_names:
        raise ValueError("robot.dof_names is empty")

    resolved = []
    for canonical in CANONICAL_DOF_NAMES:
        matches = []
        for index, raw_name in enumerate(raw_names):
            tokens = _name_tokens(raw_name)
            if canonical in tokens or str(raw_name).strip().lower() == canonical:
                matches.append(index)
        if not matches:
            raise ValueError(
                f"Missing canonical DOF {canonical!r}; raw_dof_names={raw_names}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"Ambiguous canonical DOF {canonical!r}; matches={matches}; "
                f"raw_dof_names={raw_names}"
            )
        resolved.append(matches[0])

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Canonical DOF mapping is not one-to-one: {resolved}")
    return tuple(resolved)


def canonical_values(raw_values, canonical_to_raw):
    values = list(raw_values)
    indices = tuple(int(index) for index in canonical_to_raw)
    if len(indices) != len(CANONICAL_DOF_NAMES):
        raise ValueError(f"Expected four canonical indices, got {indices}")
    if not values or max(indices) >= len(values):
        raise ValueError(
            f"Raw vector length {len(values)} cannot satisfy mapping {indices}"
        )
    return [values[index] for index in indices]


def validate_training_fps(value):
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0.0 or fps > PHYSICS_HZ:
        raise ValueError(f"Training FPS must be in (0, {PHYSICS_HZ:g}], got {value!r}")
    return fps


def load_training_fps(metadata_path):
    path = Path(metadata_path).expanduser()
    if path.is_dir():
        path = path / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "fps" not in payload:
        raise ValueError(f"Dataset metadata has no fps field: {path}")
    return validate_training_fps(payload["fps"]), path.resolve()


def sha256_files(paths):
    digest = hashlib.sha256()
    found = 0
    for value in paths:
        path = Path(value)
        if not path.is_file():
            continue
        found += 1
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest() if found else ""


def build_client_contract(
    training_fps,
    normalization_hash="",
    observation_schema=OBSERVATION_SCHEMA_LEGACY_18D,
    observation_context=None,
):
    schema = str(observation_schema or OBSERVATION_SCHEMA_LEGACY_18D)
    if schema == OBSERVATION_SCHEMA_LEGACY_18D:
        state_names = STATE_NAMES_18D
        effort_names = ()
    elif schema == OBSERVATION_SCHEMA_27D_PLUS_EFFORT:
        state_names = STATE_NAMES_27D
        effort_names = EFFORT_NAMES_4D
    elif schema == OBSERVATION_SCHEMA_28D_PLUS_EFFORT:
        state_names = STATE_NAMES_28D
        effort_names = EFFORT_NAMES_4D
    else:
        raise ValueError(f"Unsupported observation schema: {schema!r}")
    context = dict(observation_context or {})
    if schema == OBSERVATION_SCHEMA_28D_PLUS_EFFORT:
        has_external_phase = (
            context.get("phase_index") is not None
            or context.get("phase_name") is not None
            or context.get("phase") is not None
        )
        context.setdefault(
            "phase_mode",
            "external" if has_external_phase else "auto",
        )
    return {
        "type": "handshake",
        "protocol_version": PROTOCOL_VERSION,
        "training_fps": validate_training_fps(training_fps),
        "observation_schema": schema,
        "state_names": list(state_names),
        "effort_names": list(effort_names),
        "action_names": list(ACTION_NAMES_4D),
        "action_units": "rad/s",
        "camera_keys": list(CAMERA_KEYS),
        "camera_shape": [1, 3, 256, 256],
        "normalization_hash": str(normalization_hash or ""),
        "observation_context": context,
    }


def validate_client_contract(contract):
    if not isinstance(contract, dict) or contract.get("type") != "handshake":
        raise ValueError("First bridge message must be a handshake")
    if int(contract.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError(
            f"Protocol mismatch: expected {PROTOCOL_VERSION}, "
            f"got {contract.get('protocol_version')!r}"
        )
    schema = str(contract.get("observation_schema") or "").strip()
    state_names = tuple(contract.get("state_names", ()))
    effort_names = tuple(contract.get("effort_names", ()))
    if not schema:
        schema = (
            OBSERVATION_SCHEMA_LEGACY_18D
            if state_names == STATE_NAMES_18D
            else OBSERVATION_SCHEMA_27D_PLUS_EFFORT
            if state_names == STATE_NAMES_27D
            else OBSERVATION_SCHEMA_28D_PLUS_EFFORT
            if state_names == STATE_NAMES_28D
            else ""
        )
    if schema == OBSERVATION_SCHEMA_LEGACY_18D:
        if state_names != STATE_NAMES_18D or effort_names:
            raise ValueError("Legacy checkpoint/runtime state schema mismatch")
    elif schema == OBSERVATION_SCHEMA_27D_PLUS_EFFORT:
        if state_names != STATE_NAMES_27D:
            raise ValueError("27D checkpoint/runtime state schema mismatch")
        if effort_names != EFFORT_NAMES_4D:
            raise ValueError("4D effort checkpoint/runtime schema mismatch")
    elif schema == OBSERVATION_SCHEMA_28D_PLUS_EFFORT:
        if state_names != STATE_NAMES_28D:
            raise ValueError("28D checkpoint/runtime state schema mismatch")
        if effort_names != EFFORT_NAMES_4D:
            raise ValueError("4D effort checkpoint/runtime schema mismatch")
        context = contract.get("observation_context")
        if not isinstance(context, dict):
            raise ValueError("28D deployment requires an observation_context object")
        required = (
            "dig_target_xyz",
            "unload_landing_xyz",
            "task_text",
        )
        missing = [name for name in required if context.get(name) is None]
        if missing:
            raise ValueError(f"28D deployment context is missing: {missing}")
        phase_mode = str(context.get("phase_mode") or "").strip().lower()
        if phase_mode not in ("auto", "external"):
            raise ValueError(
                "28D deployment phase_mode must be 'auto' or 'external'"
            )
        has_external_phase = (
            context.get("phase_index") is not None
            or context.get("phase_name") is not None
            or context.get("phase") is not None
        )
        if phase_mode == "external" and not has_external_phase:
            raise ValueError(
                "28D external phase_mode requires phase_index or phase_name"
            )
    else:
        raise ValueError(f"Unsupported observation schema: {schema!r}")
    if tuple(contract.get("action_names", ())) != ACTION_NAMES_4D:
        raise ValueError("Checkpoint/runtime action schema mismatch")
    if str(contract.get("action_units", "")) != "rad/s":
        raise ValueError("Bridge action units must be rad/s")
    if tuple(contract.get("camera_keys", ())) != CAMERA_KEYS:
        raise ValueError("Checkpoint/runtime camera-key mismatch")
    camera_shape = [int(value) for value in contract.get("camera_shape", ())]
    if camera_shape != [1, 3, 256, 256]:
        raise ValueError(f"Unexpected camera tensor shape: {camera_shape}")
    result = dict(contract)
    result["observation_schema"] = schema
    result["observation_context"] = dict(contract.get("observation_context") or {})
    result["training_fps"] = validate_training_fps(contract.get("training_fps"))
    return result


class PhysicsTickScheduler:
    """Distribute 60 Hz ticks across arbitrary policy FPS without drift."""

    def __init__(self, policy_fps, physics_hz=PHYSICS_HZ):
        self.physics_hz = float(physics_hz)
        self.policy_fps = validate_training_fps(policy_fps)
        self._ticks_exact = self.physics_hz / self.policy_fps
        self._remainder = 0.0

    def next_ticks(self):
        target = self._ticks_exact + self._remainder
        ticks = max(1, int(math.floor(target + 1.0e-9)))
        self._remainder = target - ticks
        return ticks

