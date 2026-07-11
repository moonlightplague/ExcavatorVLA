#!/usr/bin/env python3
"""Rewrite legacy .dashboard_success task prompts with the shared VLA builder.

This script uses only the Python standard library plus the repository's
excavator_dataset_tools.py. It is safe to run with regular Python, Isaac Sim
Python, or /isaac-sim/python.sh on Windows and Linux.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_POOL = REPO_ROOT / "excavator_auto_dataset" / ".dashboard_success"
INDEX_NAMES = ("episodes.jsonl", "successful_episodes.jsonl", "trainable_episodes.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite legacy dashboard_success task prompts using the shared VLA exporter prompt builder."
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=DEFAULT_POOL,
        help=f"Path to .dashboard_success (default: {DEFAULT_POOL})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag the script performs a read-only dry run.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy changed text files into .prompt_backups before replacement.",
    )
    return parser.parse_args()


def load_dataset_tools():
    module_path = REPO_ROOT / "excavator_dataset_tools.py"
    if not module_path.is_file():
        raise RuntimeError(f"shared exporter module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("excavator_dataset_tools_prompt_rewrite", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared exporter module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {type(exc).__name__}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def first_jsonl_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {type(exc).__name__}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            return row
    raise ValueError(f"trajectory has no samples: {path}")


def validate_jsonl_and_count_task_changes(path: Path, prompt: str) -> Tuple[int, int]:
    rows = 0
    changes = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {type(exc).__name__}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows += 1
            if str(row.get("task") or "") != prompt:
                changes += 1
    if rows <= 0:
        raise ValueError(f"trajectory has no samples: {path}")
    return rows, changes


def generic_prompt_inputs(tools, sample: dict, meta: dict, episode_row: Optional[dict] = None):
    generic = str(tools.GENERIC_LEROBOT_TASK_TEXT)
    sample_for_builder = copy.deepcopy(sample)
    meta_for_builder = copy.deepcopy(meta)
    row_for_builder = copy.deepcopy(episode_row or {})
    for value in (sample_for_builder, meta_for_builder, row_for_builder):
        value["task"] = generic
        value["dataset_task_text"] = generic
    return sample_for_builder, meta_for_builder, row_for_builder


def normalized_leaf(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def row_episode_dir_name(row: dict) -> str:
    for key in ("transferred_episode_dir", "trajectory", "meta", "score_path"):
        leaf = normalized_leaf(row.get(key))
        if key != "transferred_episode_dir" and leaf:
            leaf = normalized_leaf(str(row.get(key) or "").replace("\\", "/").rsplit("/", 1)[0])
        if leaf:
            return leaf
    return ""


def backup_file(path: Path, pool_dir: Path, backup_dir: Optional[Path]) -> None:
    if backup_dir is None:
        return
    try:
        relative = path.resolve().relative_to(pool_dir.resolve())
    except Exception:
        relative = Path(path.name)
    destination = backup_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def atomic_write_json(path: Path, value: dict) -> None:
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


def rewritten_trajectory_rows(path: Path, prompt: str) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            row["task"] = prompt
            yield row


def main() -> int:
    args = parse_args()
    pool_dir = args.pool_dir.expanduser().resolve()
    episodes_dir = pool_dir / "episodes"
    if not pool_dir.is_dir() or not episodes_dir.is_dir():
        print(f"ERROR: dashboard success pool not found: {pool_dir}", file=sys.stderr)
        return 2

    tools = load_dataset_tools()
    prompt_version = str(tools.LEROBOT_TASK_PROMPT_VERSION)
    episode_dirs = sorted(path for path in episodes_dir.iterdir() if path.is_dir())
    if not episode_dirs:
        print(f"ERROR: no episode directories found under {episodes_dir}", file=sys.stderr)
        return 2

    index_rows_by_path: Dict[Path, List[dict]] = {}
    row_by_dir: Dict[str, dict] = {}
    for name in INDEX_NAMES:
        path = pool_dir / name
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        index_rows_by_path[path] = rows
        for row in rows:
            directory_name = row_episode_dir_name(row)
            if directory_name:
                row_by_dir.setdefault(directory_name, row)

    plans: List[dict] = []
    prompt_by_dir: Dict[str, str] = {}
    prompt_by_source_id: Dict[str, str] = {}
    total_rows = 0
    total_row_changes = 0
    errors: List[str] = []

    print(f"Pool: {pool_dir}")
    print(f"Prompt version: {prompt_version}")
    print(f"Episodes found: {len(episode_dirs)}")

    for index, episode_dir in enumerate(episode_dirs, 1):
        meta_path = episode_dir / "meta.json"
        trajectory_path = episode_dir / "trajectory.jsonl"
        try:
            if not meta_path.is_file() or not trajectory_path.is_file():
                raise ValueError("missing meta.json or trajectory.jsonl")
            meta = read_json(meta_path)
            sample = first_jsonl_row(trajectory_path)
            episode_row = row_by_dir.get(episode_dir.name, {})
            sample_input, meta_input, row_input = generic_prompt_inputs(tools, sample, meta, episode_row)
            state_names = meta.get("state_names")
            prompt = str(
                tools.build_episode_task_text(
                    sample_input,
                    meta_input,
                    episode_row=row_input,
                    state_names=state_names,
                )
                or ""
            ).strip()
            if not prompt or tools.is_generic_lerobot_task_text(prompt):
                raise ValueError("shared builder could not derive a scene-relative prompt")
            row_count, row_changes = validate_jsonl_and_count_task_changes(trajectory_path, prompt)
            meta_changed = any(
                str(meta.get(key) or "") != expected
                for key, expected in (
                    ("task", prompt),
                    ("dataset_task_text", prompt),
                    ("task_prompt_version", prompt_version),
                )
            )
            plans.append(
                {
                    "episode_dir": episode_dir,
                    "meta_path": meta_path,
                    "trajectory_path": trajectory_path,
                    "meta": meta,
                    "prompt": prompt,
                    "row_count": row_count,
                    "row_changes": row_changes,
                    "meta_changed": meta_changed,
                }
            )
            prompt_by_dir[episode_dir.name] = prompt
            episode_id = str(meta.get("episode_id") or "")
            if episode_id:
                prompt_by_source_id[episode_id] = prompt
            total_rows += row_count
            total_row_changes += row_changes
            if index <= 5:
                print(f"  {episode_dir.name}: {prompt}")
            elif index == 6:
                print("  ...")
            if index % 10 == 0 or index == len(episode_dirs):
                print(f"Validated episodes: {index}/{len(episode_dirs)}")
        except Exception as exc:
            errors.append(f"{episode_dir}: {type(exc).__name__}: {exc}")

    index_change_counts: Dict[Path, int] = {}
    for path, rows in index_rows_by_path.items():
        changed = 0
        for row in rows:
            prompt = prompt_by_dir.get(row_episode_dir_name(row))
            if not prompt:
                prompt = prompt_by_source_id.get(str(row.get("source_episode_id") or ""))
            if not prompt:
                continue
            if (
                str(row.get("task") or "") != prompt
                or str(row.get("dataset_task_text") or "") != prompt
                or str(row.get("task_prompt_version") or "") != prompt_version
            ):
                changed += 1
        index_change_counts[path] = changed

    if errors:
        print("ERROR: validation failed; no files were changed.", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2

    changed_episodes = sum(1 for plan in plans if plan["row_changes"] or plan["meta_changed"])
    changed_index_rows = sum(index_change_counts.values())
    print(
        "Summary: "
        f"episodes={len(plans)}, changed_episodes={changed_episodes}, "
        f"trajectory_rows={total_rows}, changed_trajectory_rows={total_row_changes}, "
        f"changed_index_rows={changed_index_rows}"
    )

    if not args.apply:
        print("Dry run complete. Re-run with --apply to write changes.")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = None if args.no_backup else pool_dir / ".prompt_backups" / timestamp
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    for index, plan in enumerate(plans, 1):
        prompt = str(plan["prompt"])
        meta_path = Path(plan["meta_path"])
        trajectory_path = Path(plan["trajectory_path"])
        if plan["meta_changed"]:
            backup_file(meta_path, pool_dir, backup_dir)
            meta = dict(plan["meta"])
            meta["task"] = prompt
            meta["dataset_task_text"] = prompt
            meta["task_prompt_version"] = prompt_version
            atomic_write_json(meta_path, meta)
        if int(plan["row_changes"]) > 0:
            backup_file(trajectory_path, pool_dir, backup_dir)
            atomic_write_jsonl(trajectory_path, rewritten_trajectory_rows(trajectory_path, prompt))
        if index % 10 == 0 or index == len(plans):
            print(f"Updated episodes: {index}/{len(plans)}")

    for path, rows in index_rows_by_path.items():
        if index_change_counts.get(path, 0) <= 0:
            continue
        updated_rows = []
        for row in rows:
            updated = dict(row)
            prompt = prompt_by_dir.get(row_episode_dir_name(updated))
            if not prompt:
                prompt = prompt_by_source_id.get(str(updated.get("source_episode_id") or ""))
            if prompt:
                updated["task"] = prompt
                updated["dataset_task_text"] = prompt
                updated["task_prompt_version"] = prompt_version
            updated_rows.append(updated)
        backup_file(path, pool_dir, backup_dir)
        atomic_write_jsonl(path, updated_rows)

    run_meta_path = pool_dir / "run_meta.json"
    if run_meta_path.is_file():
        backup_file(run_meta_path, pool_dir, backup_dir)
        run_meta = read_json(run_meta_path)
        run_meta["task_prompt_version"] = prompt_version
        run_meta["task_prompt_rewritten_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write_json(run_meta_path, run_meta)

    print("Prompt rewrite complete.")
    if backup_dir is not None:
        print(f"Backup: {backup_dir}")
    print("Re-run the dashboard VLA export to rebuild tasks/data/meta. Existing MP4 files remain reusable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
