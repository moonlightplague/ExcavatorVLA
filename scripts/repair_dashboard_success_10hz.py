#!/usr/bin/env python3
"""One-time migration of legacy dashboard success episodes to uniform 10 Hz.

The migration preserves every camera/state row and all image files. It changes
only the episode time axis and time-derived fields. Measured effort is retained
because force/torque is not a time derivative.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Iterable, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_POOL = REPO_ROOT / "excavator_auto_dataset" / ".dashboard_success"
INDEX_NAMES = (
    "episodes.jsonl",
    "successful_episodes.jsonl",
    "trainable_episodes.jsonl",
)
CAMERA_KEYS = (
    "observation.images.0",
    "observation.images.1",
    "observation.images.2",
)
MIGRATION_VERSION = "dashboard_success_uniform_10hz_v1"
RAW_STATE_NAMES_FALLBACK = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair legacy .dashboard_success trajectories to an exact 10 Hz timeline."
    )
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag the command is a read-only audit.",
    )
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_dataset_tools():
    module_path = REPO_ROOT / "excavator_dataset_tools.py"
    repo_root_text = str(REPO_ROOT)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    spec = importlib.util.spec_from_file_location("excavator_dataset_tools_10hz_repair", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared dataset tools: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path, default=None):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def backup_file(path: Path, pool_dir: Path, backup_root: Optional[Path]) -> None:
    if backup_root is None or not path.is_file():
        return
    relative = path.resolve().relative_to(pool_dir.resolve())
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(path, destination)
    except OSError:
        shutil.copy2(path, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_10hz(report: dict, target_hz: float) -> bool:
    if int(report.get("sample_count", 0) or 0) < 2:
        return True
    expected_dt = 1.0 / float(target_hz)
    tolerance = max(1.0e-6, expected_dt * 1.0e-4)
    return (
        abs(float(report.get("median_dt_s", 0.0) or 0.0) - expected_dt) <= tolerance
        and abs(float(report.get("min_dt_s", 0.0) or 0.0) - expected_dt) <= tolerance
        and abs(float(report.get("max_dt_s", 0.0) or 0.0) - expected_dt) <= tolerance
    )


def camera_sequence(rows: List[dict], tools) -> List[tuple]:
    return [
        tuple(str(tools.sample_image_value(row, key) or "") for key in CAMERA_KEYS)
        for row in rows
    ]


def rebuild_vla_states(rows: List[dict], meta: dict, episode_row: dict, raw_state_names: List[str], tools) -> None:
    previous_fill: Optional[float] = None
    previous_rate: Optional[float] = None
    previous_t: Optional[float] = None
    for index, sample in enumerate(rows):
        legacy = tools._legacy_base_state_from_sample(sample, raw_state_names)
        particles = tools._sample_bucket_load_particles(sample, legacy)
        if particles is None:
            raise ValueError(f"sample {index}: missing bucket load particles")
        fill = tools.vla_observation_contract.bucket_fill_fraction(float(particles))
        sample_t = float(sample.get("t", index / 10.0) or 0.0)
        dt_sample = 0.0 if previous_t is None else max(1.0e-6, sample_t - previous_t)
        fill_rate = tools.vla_observation_contract.causal_bucket_fill_rate(
            fill,
            previous_fill,
            previous_rate,
            dt_sample,
        )
        state, reason = tools.build_lerobot_state_28d(
            sample,
            meta,
            episode_row,
            raw_state_names,
            fill,
            fill_rate,
        )
        if state is None:
            raise ValueError(f"sample {index}: cannot rebuild 28D state: {reason}")
        sample["observation.state"] = state
        sample["obs.state"] = state
        sample["observation.stage_current_id"] = int(
            tools.canonical_lerobot_phase_index(sample.get("phase") or sample.get("label"))
        )
        previous_fill = float(state[26])
        previous_rate = float(state[27])
        previous_t = sample_t


def retime_auxiliary_rows(
    rows: List[dict],
    migrated_trajectory: List[dict],
    source_time_scale: float,
) -> List[dict]:
    if not rows:
        return []
    timestamp_base = float(rows[0].get("timestamp", 0.0) or 0.0)
    simulation_base = float(rows[0].get("timestamp.simulation", 0.0) or 0.0)
    source_t_base = float(rows[0].get("t_episode", rows[0].get("t", 0.0)) or 0.0)
    scale = max(1.0e-9, float(source_time_scale or 1.0))
    out: List[dict] = []
    for index, source in enumerate(rows):
        row = dict(source)
        if len(rows) == len(migrated_trajectory):
            trajectory_row = migrated_trajectory[index]
            new_t = float(trajectory_row.get("t", 0.0) or 0.0)
            new_timestamp = float(
                trajectory_row.get("timestamp", timestamp_base + new_t) or 0.0
            )
            new_simulation = float(
                trajectory_row.get(
                    "timestamp.simulation",
                    simulation_base + new_t,
                )
                or 0.0
            )
        else:
            source_t = float(
                source.get("t_episode", source.get("t", source_t_base))
                or source_t_base
            )
            new_t = max(0.0, source_t - source_t_base) / scale
            new_timestamp = timestamp_base + new_t
            new_simulation = simulation_base + new_t
        row.setdefault("timestamp.raw", source.get("timestamp"))
        row.setdefault("timestamp.simulation.raw", source.get("timestamp.simulation"))
        row.setdefault("t_episode.raw", source.get("t_episode"))
        row["timestamp"] = new_timestamp
        row["timestamp.simulation"] = new_simulation
        row["t_episode"] = new_t
        row["timestamp.source"] = "legacy_uniform_10hz_repair"
        out.append(row)
    return out


def invalidate_existing_export(pool_dir: Path, backup_root: Optional[Path]) -> None:
    manifest_path = pool_dir / "lerobot_v3" / "manifest.json"
    if not manifest_path.is_file():
        return
    backup_file(manifest_path, pool_dir, backup_root)
    manifest = read_json(manifest_path, default={}) or {}
    if not isinstance(manifest, dict):
        return
    manifest["vla_training_ready"] = False
    manifest["standard_lerobot_ready"] = False
    manifest["pool_invalidated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["pool_invalidated_reason"] = "legacy_10hz_repair_requires_full_reexport"
    atomic_write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    target_hz = float(args.target_hz)
    if not math.isfinite(target_hz) or target_hz <= 0.0:
        print("ERROR: --target-hz must be positive", file=sys.stderr)
        return 2
    pool_dir = args.pool_dir.expanduser().resolve()
    episodes_root = pool_dir / "episodes"
    if not episodes_root.is_dir():
        print(f"ERROR: success pool not found: {pool_dir}", file=sys.stderr)
        return 2

    tools = load_dataset_tools()
    run_meta = read_json(pool_dir / "run_meta.json", default={}) or {}
    raw_state_names = list(run_meta.get("state_names") or RAW_STATE_NAMES_FALLBACK)
    index_rows_by_path: Dict[Path, List[dict]] = {}
    episode_row_by_dir: Dict[str, dict] = {}
    for index_name in INDEX_NAMES:
        index_path = pool_dir / index_name
        if not index_path.is_file():
            continue
        rows = read_jsonl(index_path)
        index_rows_by_path[index_path] = rows
        for row in rows:
            episode_dir = tools.episode_dir_from_row(row, run_dir=pool_dir)
            if episode_dir:
                episode_row_by_dir.setdefault(Path(episode_dir).name, row)

    episode_dirs = sorted(path for path in episodes_root.iterdir() if path.is_dir())
    if int(args.limit or 0) > 0:
        episode_dirs = episode_dirs[: int(args.limit)]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = None
    if args.apply and not args.no_backup:
        backup_root = pool_dir / ".10hz_repair_backup" / stamp

    repaired: List[dict] = []
    already_uniform = 0
    errors: List[dict] = []
    print(f"Pool: {pool_dir}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Target: {target_hz:.4f} Hz")
    print(f"Episodes: {len(episode_dirs)}")

    for ordinal, episode_dir in enumerate(episode_dirs, 1):
        trajectory_path = episode_dir / "trajectory.jsonl"
        meta_path = episode_dir / "meta.json"
        sand_path = episode_dir / "sand_metrics.jsonl"
        try:
            trajectory = read_jsonl(trajectory_path)
            source_report = tools.trajectory_sampling_report(trajectory)
            if uniform_10hz(source_report, target_hz):
                already_uniform += 1
                continue
            if not trajectory:
                raise ValueError("trajectory is empty")
            before_cameras = camera_sequence(trajectory, tools)
            if any(not all(values) for values in before_cameras):
                raise ValueError("one or more trajectory rows are missing a camera path")
            meta = read_json(meta_path, default={}) or {}
            episode_row = dict(episode_row_by_dir.get(episode_dir.name, {}))
            transformed = tools.apply_export_time_policy_to_trajectory(
                trajectory,
                policy={"time_mode": "uniform_fps"},
                base_fps=target_hz,
                state_names=raw_state_names,
            )
            migrated = list(transformed.get("samples") or [])
            rebuild_vla_states(migrated, meta, episode_row, raw_state_names, tools)
            after_report = tools.trajectory_sampling_report(migrated)
            if not uniform_10hz(after_report, target_hz):
                raise ValueError(f"post-migration cadence invalid: {after_report}")
            if len(migrated) != len(trajectory):
                raise ValueError("frame count changed")
            if camera_sequence(migrated, tools) != before_cameras:
                raise ValueError("camera path sequence changed")

            source_hash = sha256_file(trajectory_path)
            migration = {
                "version": MIGRATION_VERSION,
                "target_hz": target_hz,
                "sample_count": len(migrated),
                "source_sampling": source_report,
                "result_sampling": after_report,
                "source_time_scale": float(transformed.get("source_time_scale") or 1.0),
                "raw_duration_s": float(transformed.get("raw_duration_s") or 0.0),
                "timeline_duration_s": float(transformed.get("duration_s") or 0.0),
                "media_duration_s": float(transformed.get("media_duration_s") or 0.0),
                "source_sha256": source_hash,
                "measured_effort_policy": "preserved_unscaled_force_torque",
                "frame_and_camera_policy": "same_count_same_paths",
            }

            if args.apply:
                backup_file(trajectory_path, pool_dir, backup_root)
                backup_file(meta_path, pool_dir, backup_root)
                if sand_path.is_file():
                    backup_file(sand_path, pool_dir, backup_root)
                atomic_write_jsonl(trajectory_path, migrated)
                if sand_path.is_file():
                    atomic_write_jsonl(
                        sand_path,
                        retime_auxiliary_rows(
                            read_jsonl(sand_path),
                            migrated,
                            float(transformed.get("source_time_scale") or 1.0),
                        ),
                    )
                meta["dataset_sampling_policy"] = {
                    **dict(meta.get("dataset_sampling_policy") or {}),
                    "clock": "simulation",
                    "target_hz": target_hz,
                    "sample_interval_s": 1.0 / target_hz,
                    "camera_target_hz": int(round(target_hz)),
                    "migration": MIGRATION_VERSION,
                }
                meta["legacy_10hz_repair"] = migration
                atomic_write_json(meta_path, meta)
                migration["result_sha256"] = sha256_file(trajectory_path)

            repaired.append({"episode_dir": episode_dir.name, **migration})
            if len(repaired) <= 5 or len(repaired) % 50 == 0:
                print(
                    f"[{ordinal}/{len(episode_dirs)}] {episode_dir.name}: "
                    f"{float(source_report.get('median_hz', 0.0) or 0.0):.3f} -> {target_hz:.3f} Hz"
                )
        except Exception as exc:
            error = {
                "episode_dir": episode_dir.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"ERROR {episode_dir.name}: {error['error']}", file=sys.stderr)

    repaired_by_dir = {row["episode_dir"]: row for row in repaired}
    if args.apply and repaired_by_dir:
        for index_path, rows in index_rows_by_path.items():
            changed = False
            for row in rows:
                episode_dir = tools.episode_dir_from_row(row, run_dir=pool_dir)
                migration = repaired_by_dir.get(Path(episode_dir).name if episode_dir else "")
                if not migration:
                    continue
                row["sampling_hz"] = target_hz
                row["sampling_interval_s"] = 1.0 / target_hz
                row["legacy_10hz_repair"] = {
                    "version": MIGRATION_VERSION,
                    "source_sampling_hz": float(
                        (migration.get("source_sampling") or {}).get("median_hz", 0.0) or 0.0
                    ),
                    "source_time_scale": float(migration.get("source_time_scale", 1.0) or 1.0),
                }
                changed = True
            if changed:
                backup_file(index_path, pool_dir, backup_root)
                atomic_write_jsonl(index_path, rows)
        camera_config_path = pool_dir / "camera_config.json"
        camera_config = read_json(camera_config_path, default={}) or {}
        if isinstance(camera_config, dict):
            backup_file(camera_config_path, pool_dir, backup_root)
            camera_config["frequency"] = target_hz
            camera_config["sample_stride"] = 1
            camera_config["source_sampling_contract"] = "original_uniform_10hz_v1"
            atomic_write_json(camera_config_path, camera_config)
        run_meta_path = pool_dir / "run_meta.json"
        pool_run_meta = read_json(run_meta_path, default={}) or {}
        if isinstance(pool_run_meta, dict):
            backup_file(run_meta_path, pool_dir, backup_root)
            pool_run_meta["dataset_hz"] = target_hz
            pool_run_meta["source_sampling_contract"] = "original_uniform_10hz_v1"
            pool_run_meta["legacy_10hz_repair_version"] = MIGRATION_VERSION
            atomic_write_json(run_meta_path, pool_run_meta)
        invalidate_existing_export(pool_dir, backup_root)

    manifest = {
        "schema": MIGRATION_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "pool_dir": str(pool_dir),
        "target_hz": target_hz,
        "episodes_scanned": len(episode_dirs),
        "already_uniform": already_uniform,
        "repair_count": len(repaired),
        "error_count": len(errors),
        "backup_dir": str(backup_root) if backup_root is not None else "",
        "repaired": repaired,
        "errors": errors,
    }
    manifest_path = pool_dir / f"repair_10hz_{'applied' if args.apply else 'dry_run'}.json"
    if args.apply:
        atomic_write_json(manifest_path, manifest)
    print(
        f"Summary: already_uniform={already_uniform} repair={len(repaired)} "
        f"errors={len(errors)} manifest={manifest_path if args.apply else 'not written'}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
