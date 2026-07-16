# Code Redundancy And Error Analysis

> Historical static audit (2026-07-01). File counts, paths, runtime size,
> temporary-code findings, and unresolved-item claims have not been revalidated
> against the current working tree. Do not treat this report as a current defect
> list without checking the referenced source first.

This report reviews the current Python scripts in the ExcavatorVLA workspace for redundant, invalid, dead, or error-prone code. It is based on static analysis only; Isaac Sim runtime behavior was not executed.

## Scope And Method

- Files scanned: 24 Python files.
- Syntax check: all Python files passed `ast.parse()` and in-memory `compile()`.
- External lint tools: `ruff`, `flake8`, `pyflakes`, and `vulture` are not installed in the current environment, so this report uses built-in Python AST and text scans.
- Static limits: this project uses Isaac Sim dynamic reloads, Omni UI callbacks, global state dictionaries, and async task names. Some "possibly unused" findings may be false positives.

## Executive Summary

The codebase does not currently show syntax-level invalid Python. The main risks are not parser errors; they are maintainability and runtime-environment risks:

- Several scripts contain hardcoded machine-specific paths under `/root`, `/isaac-sim`, and `/root/gpufree-data`.
- `run_excavator_standalone.py` still contains active temporary test code that forcibly relocates the robot after reset.
- Backup scripts and bridge/policy clients duplicate substantial code.
- The main runtime is very large: `scripts/excavator_app/excavator_runtime.py` has about 30.5k lines and 854 functions/methods.
- Silent exception swallowing is common, especially in the main runtime.
- A few no-op functions look like incomplete hooks or stale placeholders.
- Some runtime modules perform large side effects at import time, which is intentional for Isaac Sim Script Editor reloads but fragile.

## Confirmed Issues Or High-Confidence Risks

### 1. Hardcoded Machine-Specific Paths

The following paths make scripts non-portable outside the author's original machine/container layout.

| File | Line | Hardcoded path or default | Risk |
| --- | ---: | --- | --- |
| `run_excavator_standalone.py` | 303-304 | `/root/Documents/trae_projects/vla_test/assets/glb/no-brand_dump_truck.*` | Dump truck will not load on this workspace; converter may silently skip the intended truck asset. |
| `archive/standalone/run_excavator_standalone_backup_camera_bug.py` | 152-153 | `/root/Documents/trae_projects/vla_test/assets/glb/no-brand_dump_truck.*` | Same issue in backup launcher. |
| `scripts/diagnose_scene.py` | 9 | `/isaac-sim/ExcavatorVLA/assets/usd/excavator_scene.usd` | Diagnostic fails unless repo is mounted exactly at `/isaac-sim/ExcavatorVLA`. |
| `scripts/simple_diagnose.py` | 4 | `/isaac-sim/ExcavatorVLA/assets/usd/excavator_scene.usd` | Same issue in simple diagnostic. |
| `scripts/bridge_test/smolvla_client.py` | 114 | `/root/gpufree-data/checkpoints/smolvla_base` | Policy client default will fail in most environments. |
| `scripts/bridge_test/archive/smolvla_client_backup_raw_action.py` | 114 | `/root/gpufree-data/checkpoints/smolvla_base` | Same issue in backup client. |
| `scripts/bridge_test/archive/smolvla_client_guided_adapter_backup.py` | 218 | `/root/gpufree-data/checkpoints/smolvla_base` | Same issue in guided backup client. |
| `scripts/bridge_test/smolvla_policy_client.py` | 230, 234 | `/root/gpufree-data/...` | Checkpoint and VLM defaults are environment-specific. |

Recommended cleanup:

- Resolve asset paths from `PROJECT_DIR` or `excavator_config.json`.
- Prefer checked-in assets such as `assets/fbx/truck/truck.usd` over untracked `/root/Documents/...` GLB paths.
- Keep model/checkpoint paths as CLI args, but make defaults empty or environment-variable driven.

### 2. Active Temporary Robot Pose Override

`run_excavator_standalone.py:360-375` contains a block marked `TEMP TEST` that always moves the robot after `world.reset()`:

- Position: `[-9.2, 6.7, 1.243]`
- Orientation: `[1.0, 0.0, 0.0, 0.0]`

This is not just debug text; it changes runtime behavior every standalone launch. If the standalone script is meant to be a general launcher, this should be gated by a CLI flag or config setting.

Recommended cleanup:

- Add a flag such as `--force-test-pose`.
- Or move the pose into `excavator_config.json`.
- Or remove this block if the promoted `main_zsp.py` runtime now owns scene placement.

### 3. Backup Scripts Are Redundant And Easy To Drift

Static similarity checks found near-duplicate or intentionally retained backup scripts:

| Pair | Similarity | Comment |
| --- | ---: | --- |
| `scripts/bridge_test/smolvla_client.py` vs `scripts/bridge_test/archive/smolvla_client_backup_raw_action.py` | 0.993 | Almost identical. This is the strongest redundancy candidate. |
| `scripts/bridge_test/smolvla_client.py` vs `scripts/bridge_test/archive/smolvla_client_guided_adapter_backup.py` | 0.514 | Shared protocol/model-loading code plus adapter differences. |
| `scripts/bridge_test/smolvla_client.py` vs `scripts/bridge_test/smolvla_policy_client.py` | 0.531 | Shared client protocol code plus checkpoint/language logic. |
| `run_excavator_standalone.py` vs `archive/standalone/run_excavator_standalone_backup_camera_bug.py` | 0.419 | Backup launcher shares bridge/server flow but keeps older camera behavior. |

Recommended cleanup:

- Move backup scripts to `scripts/archive/` if they must be kept.
- Extract common bridge helpers into one module, for example `scripts/bridge_test/bridge_protocol.py`.
- Extract SmolVLA observation/action conversion into one module used by all policy clients.

### 4. Duplicated Helper Functions Across Scripts

Exact AST-normalized duplicates were found in bridge and runtime helper code.

Examples:

| Function | Duplicated in |
| --- | --- |
| `read_json`, `write_json`, `make_action_from_command` | `run_excavator_standalone.py`, `archive/standalone/run_excavator_standalone_backup_camera_bug.py`, `scripts/bridge_test/gui_tcp_bridge_server.py` |
| `recv_exact`, `read_json`, `write_json`, `decode_rgb` | SmolVLA client variants |
| `send_json`, `recv_json`, `recv_exact` | `external_client_policy.py`, `gui_client.py` |
| `sdf_path`, `ui_short_text`, `simulation_timeline_is_playing`, polygon helpers | `excavator_runtime.py`, `sand_site_runtime.py` |

This is not immediately wrong, but bug fixes to protocol framing, payload decoding, or geometry helpers must be repeated manually.

Recommended cleanup:

- Introduce a shared bridge protocol module.
- Introduce a shared USD/geometry helper module only after confirming both runtime modules can import it safely in Isaac Sim reload mode.

### 5. Silent Exception Swallowing

AST scan found many `except ...: pass` blocks:

| File | Silent except count |
| --- | ---: |
| `scripts/excavator_app/excavator_runtime.py` | 142 |
| `scripts/excavator_app/sand_site_runtime.py` | 25 |
| `excavator_dataset_tools.py` | 8 |
| Other scripts | 1-3 each |

Some are reasonable cleanup guards around UI/window shutdown, optional APIs, or best-effort diagnostics. However, at this volume, real runtime failures can disappear without logs.

Recommended cleanup:

- Keep silent guards only around known harmless cleanup paths.
- For state-changing operations, replace `pass` with throttled logging through `info_print()` or `update_status()`.
- In dataset export and bridge code, include operation name and path in warnings.

### 6. No-Op Or Placeholder Functions

The following functions have bodies that only return `None`:

| File | Line | Function | Assessment |
| --- | ---: | --- | --- |
| `excavator_dataset_tools.py` | 2578 | `Handler.log_message()` | Intentional: suppresses HTTP server logs. Keep. |
| `scripts/excavator_app/excavator_runtime.py` | 15571 | `notify_sand_site_tool_sample(stage_name)` | Looks like an incomplete hook or stale placeholder. |
| `scripts/excavator_app/excavator_runtime.py` | 17219 | `ensure_trace_dig_plan_current()` | Looks like an incomplete hook or stale placeholder. |

Recommended cleanup:

- If the runtime hooks are intentionally disabled, add a short comment explaining why.
- If they are obsolete, remove them and any call sites.
- If they are planned features, raise a visible warning or implement the intended behavior.

### 7. Import-Time Side Effects

Two runtime files do substantial work at import time:

| File | Import-time behavior |
| --- | --- |
| `scripts/excavator_app/sand_site_runtime.py` | Calls `apply_sand_fidelity_to_particle_globals()`, `build_sand_site()`, and `build_ui()` at module load. |
| `scripts/excavator_app/excavator_runtime.py` | Registers `main_loop` with `register_async_task("main_loop", main(), replace=True)` at module load. |

This pattern is likely intentional for Isaac Sim Script Editor reloads, but it makes normal import, static testing, and dependency analysis risky.

Recommended cleanup:

- Keep Script Editor behavior, but consider a small explicit entry function such as `start_runtime()` for tests and tools.
- Document import-time side effects near the bottom of each module.

### 8. Runtime Asset Mutation From Launcher

`main_zsp.py` can create or refresh `assets/zsp/configuration/*.usd` aliases and can normalize USD sublayer paths during launch.

Relevant functions:

- `_ensure_zsp_configuration_aliases()`
- `_normalize_zsp_layer_paths()`

This may be useful for compatibility, but it means launching the runtime can mutate files under version control.

Recommended cleanup:

- Make mutation explicit with a log and config flag.
- Prefer a one-time repair script if the USD asset layout is stable.
- Add generated/alias files to a documented asset policy.

### 9. Backup Server Scope Is Confusing

`archive/standalone/run_excavator_standalone_backup_camera_bug.py:324-329` defines `_server = None` inside `main()`, then nested `start_bridge_server()` declares `global _server`.

This works as a module-global assignment, but the local `_server = None` is unused and misleading. The promoted `run_excavator_standalone.py` uses `nonlocal _server`, which is clearer.

Recommended cleanup:

- Change the backup script to match the promoted launcher, or remove/archive the backup script.

### 10. README / Config Default Mismatch

`README.md` shows an example config with `"model_source": "zsp"`, while the current `excavator_config.json` uses `"model_source": "original"`.

This is not a code error, but it can confuse operators because the documented promoted entry discusses ZSP first while the active config loads the original scene.

Recommended cleanup:

- Add a short note in README that the repository default currently uses `original`.
- Or switch config to `zsp` if that is now the intended default.

## Static Dead-Code Candidates

The following top-level functions were referenced only by their own definition in a simple whole-repo text scan. Treat this as a candidate list only; dynamic UI callbacks, future imports, or external scripts may call some of these.

Higher-value candidates to review:

| File | Line | Symbol |
| --- | ---: | --- |
| `scripts/excavator_app/excavator_runtime.py` | 17219 | `ensure_trace_dig_plan_current` |
| `scripts/excavator_app/excavator_runtime.py` | 6267 | `dataset_writer_shutdown` |
| `scripts/excavator_app/excavator_runtime.py` | 8081 | `clear_bucket_load_volume_caches` |
| `scripts/excavator_app/excavator_runtime.py` | 13052 | `set_unload_models_from_xyz` |
| `scripts/excavator_app/excavator_runtime.py` | 19440 | `path_obstacle_check_debug_profile_data` |
| `scripts/excavator_app/excavator_runtime.py` | 19630 | `path_segment_check_debug_profile_data` |
| `scripts/excavator_app/excavator_runtime.py` | 27224 | `get_or_build_dig_plan` |
| `scripts/excavator_app/sand_site_runtime.py` | 1688 | `sand_make_sphere` |
| `scripts/excavator_app/sand_site_runtime.py` | 1790 | `sand_make_cylinder` |
| `scripts/excavator_app/sand_site_runtime.py` | 1803 | `sand_make_polygon_prism` |
| `scripts/excavator_app/bootstrap.py` | 56 | `run_sand_site_only` |

Dataset utility functions such as `iter_episodes()`, `load_episode_bundle()`, and `write_jsonl()` also appear unreferenced internally, but these are plausible public helper APIs and should not be removed without checking external usage.

## What Does Not Look Broken

- No Python syntax errors were detected by AST parse or in-memory compile.
- No clear high-confidence undefined local/global name errors were found in promoted scripts.
- `main.py` is intentionally small and valid as a compatibility wrapper.
- The embedded dashboard's `log_message()` no-op is intentional, not dead code.
- Many backup scripts are not invalid by themselves; the issue is redundancy and drift risk.

## Recommended Cleanup Order

1. Gate or remove the active temporary robot pose override in `run_excavator_standalone.py`.
2. Replace hardcoded `/root` and `/isaac-sim` paths with config, CLI args, or project-relative paths.
3. Archive or delete near-duplicate backup scripts, starting with `smolvla_client_backup_raw_action.py`.
4. Extract shared bridge protocol helpers used by standalone server, Script Editor server, GUI client, and policy clients.
5. Add comments or remove the two no-op runtime hooks.
6. Reduce silent `except: pass` in state-changing code paths.
7. Document or isolate import-time side effects in the runtime modules.
8. Split `excavator_runtime.py` gradually by ownership area: dataset IO, bridge/protocol, USD geometry, planning, execution, and UI.

