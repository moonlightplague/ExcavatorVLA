"""Export one finalized repeat episode, then retain only metrics in raw storage.

Run with the policy Python, after the simulator's asynchronous writer is flushed.
The blocking lock provides backpressure across workers sharing a publication.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys


KEEP_FILES = {"meta.json", "score.json", "repeat_metrics.json", "repeat_cycles.jsonl", "export.json"}
META_FIELDS = {
    "episode_id", "episode_index", "success", "status", "failure_reason", "warning_reason",
    "repeat_baseline", "repeat_cycles", "repeat_returns_completed", "scoops_completed",
    "all_scoops_success", "source_sampling_audit", "score_summary", "final_metrics",
    "repeat_metrics", "created_at", "finished_at", "duration_wall_s", "duration_simulation_s",
    "repeat_episode_quality", "scoop_results",
}
POLICY_ENV_KEYS = ("PATH", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD")


def policy_environment():
    """Restore paths captured by the shell before Isaac's setup modifies them."""
    environment = os.environ.copy()
    saved = environment.pop("REPEAT_POLICY_ENV", "")
    if saved:
        original = json.loads(saved)
        for key in POLICY_ENV_KEYS:
            if original.get(key) is None:
                environment.pop(key, None)
            else:
                environment[key] = original[key]
    else:
        # Direct baseline invocations have no shell snapshot. Remove only
        # Isaac's additions, preserving caller paths outside the installation.
        isaac = environment.get("ISAAC_PATH", "")
        if isaac:
            root = Path(isaac).resolve()
            for key in POLICY_ENV_KEYS:
                if key not in environment:
                    continue
                entries = re.split(r"[\s:]+", environment[key]) if key == 'LD_PRELOAD' else environment[key].split(os.pathsep)
                kept = [value for value in entries if value and
                        Path(value).resolve() != root and root not in Path(value).resolve().parents]
                if kept:
                    environment[key] = os.pathsep.join(kept)
                else:
                    environment.pop(key, None)
    for key in ("ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH", "RESOURCE_NAME", "PYTHONEXE"):
        environment.pop(key, None)
    return environment


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def export_lock(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode: unlinking it would allow waiters on different inodes.
    with (destination.parent / f".{destination.name}.repeat-export.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def exporter():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ExcavatorVLA-isaac"))
    from excavator_dataset_tools import export_lerobot_dataset
    return export_lerobot_dataset


def finish_episode(run_dir, episode_dir, export_dir, export_function=None):
    run = Path(run_dir).resolve(strict=True)
    episode = Path(episode_dir)
    if episode.is_symlink() or episode.resolve(strict=True).parent != run or not re.fullmatch(r"episode_\d+", episode.name):
        raise ValueError("Refusing cleanup outside a direct episode directory of this run")
    episode = episode.resolve()
    meta = json.loads((episode / "meta.json").read_text())
    metrics = json.loads((episode / "repeat_metrics.json").read_text())
    if metrics != meta.get("repeat_metrics"):
        raise ValueError("Episode metrics and metadata disagree; raw files retained")
    indexed = [r for r in read_jsonl(run / "repeat_metrics.jsonl") if r.get("episode_id") == meta.get("episode_id")]
    if len(indexed) != 1 or {k: v for k, v in indexed[0].items() if k not in ("episode_id", "episode_index")} != metrics:
        raise ValueError("Run metric index is missing or inconsistent; raw files retained")
    destination = Path(export_dir).resolve()
    if destination == run or run in destination.parents or destination in episode.parents:
        raise ValueError("Export destination overlaps raw storage")
    receipt_path = episode / "export.json"
    receipt = None
    if meta.get("status") == "trainable" and meta.get("success"):
        with export_lock(destination):
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if not (receipt.get("validated") and receipt.get("episode_id") == meta["episode_id"]
                        and receipt.get("export_dir") == str(destination)):
                    raise ValueError("Existing export receipt does not match this episode/destination")
                info = json.loads((destination / "meta/info.json").read_text())
                if info["total_episodes"] <= receipt["export_episode_index"]:
                    raise ValueError("Export receipt is absent from the destination")
            else:
                trainable = read_jsonl(run / "trainable_episodes.jsonl")
                offsets = [i for i, row in enumerate(trainable) if row.get("episode_id") == meta["episode_id"]]
                if len(offsets) != 1:
                    raise ValueError("Episode must appear exactly once in the trainable index")
                info_path = destination / "meta/info.json"
                previous = json.loads(info_path.read_text())["total_episodes"] if info_path.exists() else 0
                result = (export_function or exporter())(
                    str(run), output_dir=str(destination), split="trainable", append=True,
                    episode_offset=offsets[0], limit_episodes=1,
                    state_schema="current-v4", effort_policy="exclude",
                    require_standard=True, require_vla=True,
                )
                if not (result.get("vla_training_ready") and result.get("validation", {}).get("ok")
                        and result.get("total_episodes") == previous + 1):
                    raise RuntimeError("Export did not validate exactly one appended episode; raw files retained")
                receipt = {
                    "schema_version": "repeat_episode_export_v1", "episode_id": meta["episode_id"],
                    "raw_episode_index": meta["episode_index"], "export_dir": str(destination),
                    "export_episode_index": previous, "validated": True, "raw_pruned": False,
                }
                # Persist the commit receipt BEFORE any irreversible raw cleanup.
                write_json(receipt_path, receipt)
    elif meta.get("status") not in ("rejected", "failed", "diagnostic") or meta.get("success"):
        raise ValueError("Episode has no recognized final status; raw files retained")

    compact = {key: value for key, value in meta.items() if key in META_FIELDS}
    compact["raw_storage"] = {"policy": "metrics_only", "exported": receipt is not None}
    write_json(episode / "meta.json", compact)
    for path in episode.iterdir():
        if path.name in KEEP_FILES:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    if receipt is not None:
        receipt["raw_pruned"] = True
        write_json(receipt_path, receipt)
    print(f"[REPEAT STORAGE] episode={meta['episode_id']} exported={receipt is not None} raw=metrics_only", flush=True)
    return receipt


def verify_run(run_dir, expected_successes, export_dir):
    run = Path(run_dir)
    rows = read_jsonl(run / "trainable_episodes.jsonl")
    if len(rows) != expected_successes:
        raise ValueError("Trainable index does not match the worker success count")
    for row in rows:
        path = run / f"episode_{int(row['episode_index']):06d}" / "export.json"
        receipt = json.loads(path.read_text())
        if not (receipt.get("validated") and receipt.get("raw_pruned")
                and receipt.get("episode_id") == row["episode_id"]
                and receipt.get("export_dir") == str(Path(export_dir).resolve())):
            raise ValueError(f"Missing validated export/cleanup receipt: {path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--verify-successes", type=int)
    args = parser.parse_args()
    if args.verify_successes is not None:
        verify_run(args.run_dir, args.verify_successes, args.export_dir)
    elif args.episode_dir is not None:
        finish_episode(args.run_dir, args.episode_dir, args.export_dir)
    else:
        parser.error("--episode-dir or --verify-successes is required")


if __name__ == "__main__":
    main()
