# Script Function Memo Index

Generated for the ExcavatorVLA workspace. This is a code-navigation memo, not a behavioral specification.

## Scope

- Project root: `/Users/fora/Documents/450/ExcavatorVLA`
- Python files indexed: 24
- Functions and methods indexed: 1246
- Classes indexed: 5
- Async functions indexed: 55
- Line numbers are from the current local workspace at generation time.
- Function purpose text is intentionally concise. When a function has a docstring, the first docstring sentence is used; otherwise the purpose is inferred from the name and local module role.
- Working tree note at generation time: `?? SCRIPT_FUNCTION_MEMO_INDEX.md`

## Entry Points

- `main.py` - Forwards to main_zsp.py.
- `main_zsp.py` - Run inside Isaac Sim Script Editor or from repo root to open the selected stage and bootstrap runtime.
- `run_excavator_standalone.py` - Run with Isaac Sim Python to launch the original scene plus TCP bridge.
- `archive/standalone/run_excavator_standalone_backup_camera_bug.py` - Backup standalone launcher retained for camera bug comparison.
- `excavator_dataset_tools.py` - CLI for summary, analysis, plots, LeRobot export, and dashboard.
- `run_dataset_dashboard.py` - CLI shortcut for the dashboard server.
- `scripts/diagnose_scene.py` - CLI diagnostic for USD scene content.
- `scripts/simple_diagnose.py` - Simple import/path diagnostic script.
- `scripts/bridge_test/gui_client.py` - External pygame teleoperation GUI client.
- `scripts/bridge_test/gui_tcp_bridge_server.py` - Paste/run inside Isaac Sim Script Editor to start a bridge.
- `scripts/bridge_test/external_client_policy.py` - External policy socket smoke client.
- `scripts/bridge_test/smolvla_client.py` - External SmolVLA policy client.
- `scripts/bridge_test/archive/smolvla_client_backup_raw_action.py` - Backup SmolVLA raw action client.
- `scripts/bridge_test/archive/smolvla_client_guided_adapter_backup.py` - Backup guided adapter client.
- `scripts/bridge_test/smolvla_policy_client.py` - SmolVLA checkpoint/language-token policy client.

## Thread And Async Memo

- `scripts/excavator_app/excavator_runtime.py:1816` `register_async_task()` - Registers named coroutines on the Isaac Kit asyncio loop and stores them in STATE["async_tasks"].
- `scripts/excavator_app/excavator_runtime.py:6193` `dataset_writer_ensure()` - Creates queue.Queue plus daemon thread named excavator_dataset_writer for async dataset file writes.
- `scripts/excavator_app/excavator_runtime.py:10960` `auto_collect_export_lerobot_v3_on_finish()` - Runs the LeRobot export through asyncio.to_thread() or run_in_executor() so the Kit loop is not blocked.
- `scripts/excavator_app/excavator_runtime.py:30391` `main()` - Long-running Isaac Kit async main loop for UI sync, follow mode, trace drawing, freeze checks, and task dispatch.
- `scripts/excavator_app/sand_site_runtime.py:1985` `step_updates()` - Async helper that advances Isaac Kit frames through next_update_async().
- `scripts/excavator_app/sand_site_runtime.py:2146` `reset_sand_surface_stably()` - Scheduled through asyncio.ensure_future() by the sand UI reset callback.
- `run_excavator_standalone.py:462` `main().command_queue / response_queue` - Queue handoff between async TCP handlers and the synchronous Isaac Sim stepping loop.
- `run_excavator_standalone.py:531` `main().start_bridge_server()` - Starts asyncio.start_server() and serves the standalone TCP bridge forever.
- `archive/standalone/run_excavator_standalone_backup_camera_bug.py:327` `main().start_bridge_server()` - Backup TCP bridge server using the same async server pattern.
- `scripts/bridge_test/gui_tcp_bridge_server.py:162` `main()` - Async Script Editor bridge initializer; final line schedules it with asyncio.ensure_future(main()).
- `excavator_dataset_tools.py:2620` `serve_dashboard().ThreadingServer` - Uses socketserver.ThreadingMixIn with daemon_threads=True for dashboard HTTP requests.
- `scripts/bridge_test/gui_client.py:257` `main()` - Runs the blocking pygame event/render loop and calls socket requests from the GUI loop.

## Module Inventory

| File | Lines | Classes | Functions | Async | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| `excavator_dataset_tools.py` | 2710 | 2 | 81 | 0 | Dataset inspection, plotting, dashboard, and LeRobot v3 export utilities. |
| `main.py` | 6 | 0 | 0 | 0 | Compatibility entry that forwards execution to main_zsp.py. |
| `main_zsp.py` | 358 | 0 | 22 | 0 | Isaac Sim Script Editor entry for selecting the USD source, normalizing ZSP layers, and bootstrapping the excavator runtime. |
| `run_dataset_dashboard.py` | 17 | 0 | 1 | 0 | Thin CLI wrapper around excavator_dataset_tools.serve_dashboard(). |
| `run_excavator_standalone.py` | 626 | 0 | 11 | 4 | Standalone Isaac Sim launcher with viewport-based RGB capture and a TCP bridge server. |
| `archive/standalone/run_excavator_standalone_backup_camera_bug.py` | 406 | 0 | 7 | 4 | Backup standalone launcher that uses the older camera API path kept for regression reference. |
| `scripts/bridge_test/external_client_policy.py` | 115 | 0 | 5 | 0 | Small socket client that sends simple external policy commands to the bridge. |
| `scripts/bridge_test/gui_client.py` | 301 | 2 | 19 | 0 | Pygame teleoperation client for the TCP bridge with multi-camera display. |
| `scripts/bridge_test/gui_tcp_bridge_server.py` | 212 | 0 | 6 | 4 | Script Editor TCP bridge server using Isaac Camera.get_rgb(). |
| `scripts/bridge_test/smolvla_client.py` | 222 | 0 | 8 | 0 | SmolVLA socket client that converts bridge observations into model inputs and velocity commands. |
| `scripts/bridge_test/archive/smolvla_client_backup_raw_action.py` | 219 | 0 | 8 | 0 | Backup SmolVLA client variant preserving raw action conversion behavior. |
| `scripts/bridge_test/archive/smolvla_client_guided_adapter_backup.py` | 424 | 0 | 11 | 0 | Backup SmolVLA client with guided digging/action adapter logic. |
| `scripts/bridge_test/smolvla_policy_client.py` | 347 | 0 | 11 | 0 | SmolVLA policy client with checkpoint path patching and language token handling. |
| `scripts/diagnose_scene.py` | 110 | 0 | 1 | 0 | USD scene diagnostic script for checking scene content without SimulationApp. |
| `scripts/excavator_app/__init__.py` | 6 | 0 | 0 | 0 | Package marker for excavator_app runtime modules. |
| `scripts/excavator_app/auto_dataset_collect.py` | 455 | 0 | 9 | 1 | Automatic dataset planning loop helpers and planning failure classification. |
| `scripts/excavator_app/bootstrap.py` | 58 | 0 | 4 | 0 | Runtime reload bootstrap used from Isaac Sim to load sand and excavator modules. |
| `scripts/excavator_app/excavator_runtime.py` | 30560 | 1 | 854 | 37 | Main excavator runtime: UI, state, IK, planning, collisions, execution, auto collection, and export orchestration. |
| `scripts/excavator_app/ik_calculation.py` | 157 | 0 | 3 | 0 | IK and path metric helpers used by the excavator planner. |
| `scripts/excavator_app/ik_movement.py` | 134 | 0 | 2 | 2 | Async motion execution helpers for planned dig and unload stages. |
| `scripts/excavator_app/joint_space_planner.py` | 507 | 0 | 20 | 0 | Joint-space route planner with sampling, connection, collision checks, shortcutting, and smoothing. |
| `scripts/excavator_app/sand_site_runtime.py` | 3803 | 0 | 157 | 3 | Sand site runtime: PhysX particle sand, sandbox/unload bin geometry, controls, and stable resets. |
| `scripts/excavator_app/trace_showing.py` | 164 | 0 | 6 | 0 | Trace cache helpers for planned and active bucket path visualization. |
| `scripts/simple_diagnose.py` | 25 | 0 | 0 | 0 | Minimal import/path diagnostic script for Isaac Sim and USD scene assets. |

## Function Index

### `excavator_dataset_tools.py`

- Module role: Dataset inspection, plotting, dashboard, and LeRobot v3 export utilities.
- Primary imports: `json`, `math`, `os`, `re`, `shutil`, `time`, `collections`, `statistics`, `typing`, `urllib.parse`
- Runtime note: Has __main__ CLI/script guard.
- Runtime note: Defines a threaded HTTP server.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 46 | function | `def read_json(path, default=None)` | Reads json for JSON payload. |
| 54 | function | `def read_jsonl(path)` | Reads jsonl for JSON payload. |
| 67 | function | `def read_jsonl_limited(path, limit=10)` | Reads jsonl limited for JSON payload. |
| 82 | function | `def latest_run(dataset_root)` | Handles latest run behavior. |
| 96 | function | `def index_path(run_dir, split='trainable')` | Handles index path behavior. |
| 100 | function | `def load_index(run_dir, split='trainable')` | Loads index. |
| 104 | function | `def iter_episodes(run_dir, split='trainable')` | Iterates episodes for dataset episode. |
| 109 | function | `def load_trajectory(row_or_path)` | Loads trajectory. |
| 117 | function | `def load_episode_bundle(row)` | Loads episode bundle for dataset episode. |
| 126 | function | `def summarize_run(run_dir)` | Summarizes run. |
| 152 | function | `def split_reason(reason)` | Handles split reason behavior. |
| 159 | function | `def classify_reason(reason)` | Classifies reason. |
| 184 | function | `def numeric_stats(values)` | Handles numeric stats behavior. |
| 204 | function | `def top_counter(counter, limit=10)` | Handles top counter behavior. |
| 208 | function | `def compact_episode(row)` | Handles dataset episode behavior for compact episode. |
| 225 | function | `def analyze_events_for_rows(rows, limit_rows=None)` | Analyzes events for rows. |
| 279 | function | `def analyze_debug_timeline(run_dir, max_lines=None)` | Analyzes debug timeline. |
| 317 | function | `def inspect_trajectory_schema(run_dir, max_episodes=5, max_rows_per_episode=8)` | Handles inspect trajectory schema behavior. |
| 382 | function | `def analyze_run(run_dir, include_timeline=True)` | Analyzes run. |
| 452 | function | `def compact_analysis(report)` | Handles compact analysis behavior. |
| 482 | function | `def svg_escape(value)` | Handles svg escape behavior. |
| 492 | function | `def ensure_dir(path)` | Ensures or creates the required runtime state for dir. |
| 498 | function | `def write_text(path, text)` | Writes text. |
| 506 | function | `def safe_float_value(value, default=None)` | Handles safe float value behavior. |
| 516 | function | `def counter_rows_to_pairs(rows, limit=12)` | Handles counter rows to pairs behavior. |
| 533 | function | `def write_svg_bar_chart(path, title, rows, width=980, bar_height=28, left=300)` | Writes svg bar chart. |
| 574 | function | `def write_svg_histogram(path, title, values, bins=12)` | Writes svg histogram. |
| 604 | function | `def write_svg_scatter(path, title, points, x_label, y_label, width=880, height=560)` | Writes svg scatter. |
| 645 | function | `write_svg_scatter.sx` `def sx(value)` | Handles sx behavior. |
| 648 | function | `write_svg_scatter.sy` `def sy(value)` | Handles sy behavior. |
| 690 | function | `def write_html_report(path, compact, plot_files)` | Writes html report. |
| 790 | function | `def generate_plots(run_dir, output_dir=None)` | Handles generate plots behavior. |
| 903 | function | `def print_plots(run_dir, output_dir=None)` | Prints plots. |
| 908 | function | `def trajectory_columns(trajectory)` | Handles trajectory columns behavior. |
| 939 | function | `def write_json(path, data)` | Writes json for JSON payload. |
| 943 | function | `def write_jsonl(path, rows)` | Writes jsonl for JSON payload. |
| 953 | function | `def vector_or_none(value, length=None)` | Handles vector or none behavior. |
| 973 | function | `def row_path_value(row, key)` | Handles row path value behavior. |
| 978 | function | `def sample_image_value(sample, canonical_key)` | Samples image value. |
| 986 | function | `def episode_dir_from_row(row)` | Handles dataset episode behavior for episode dir from row. |
| 994 | function | `def resolve_episode_file(episode_dir, value)` | Resolves episode file for dataset episode. |
| 1003 | function | `def relpath_posix(path, base)` | Handles relpath posix behavior. |
| 1007 | function | `def safe_copy_file(src, dst)` | Handles safe copy file behavior. |
| 1015 | function | `def infer_export_fps(run_dir, explicit_fps=None)` | Infers export fps. |
| 1028 | function | `def lerobot_task_text(sample, episode_meta)` | Handles LeRobot export behavior for lerobot task text. |
| 1040 | function | `def try_write_parquet(rows, path)` | Handles try write parquet behavior. |
| 1057 | function | `def try_write_dataframe_parquet(df, path, index=False)` | Handles try write dataframe parquet behavior. |
| 1066 | function | `def resize_rgb_frame(frame, target_size=None)` | Handles RGB image payload behavior for resize rgb frame. |
| 1090 | function | `def try_encode_mp4_imageio(image_paths, output_path, fps, target_size=None)` | Handles try encode mp4 imageio behavior. |
| 1113 | function | `def try_encode_mp4_cv2(image_paths, output_path, fps, target_size=None)` | Handles try encode mp4 cv2 behavior. |
| 1150 | function | `def encode_mp4(image_paths, output_path, fps, target_size=None)` | Encodes mp4. |
| 1171 | function | `def copy_image_fallback(image_paths, export_dir, image_key, rows)` | Handles copy image fallback behavior. |
| 1195 | function | `def merge_image_storage(current, new_value)` | Handles merge image storage behavior. |
| 1205 | function | `def vector_stats_for_rows(rows, key, dim)` | Handles vector stats for rows behavior. |
| 1239 | function | `def scalar_stats_for_rows(rows, key)` | Handles scalar stats for rows behavior. |
| 1266 | function | `def visual_identity_stats()` | Handles visual identity stats behavior. |
| 1276 | function | `def build_lerobot_v3_stats(rows, state_dim, action_dim, effort_dim=None, image_keys=None)` | Builds lerobot v3 stats for LeRobot export. |
| 1296 | function | `def lerobot_v3_required_paths(export_dir, image_keys)` | Handles LeRobot export behavior for lerobot v3 required paths. |
| 1308 | function | `def validate_lerobot_v3_export(export_dir, image_keys)` | Validates lerobot v3 export for LeRobot export. |
| 1376 | function | `def collect_lerobot_rows(run_dir, split='trainable', limit_episodes=None)` | Collects lerobot rows for LeRobot export. |
| 1548 | function | `def export_lerobot_dataset(run_dir, output_dir=None, split='trainable', fps=None, limit_episodes=None, overwrite=False, require_standard=False, require_vla=False)` | Exports lerobot dataset. |
| 1815 | function | `def print_lerobot_export(run_dir, output_dir=None, split='trainable', fps=None, limit_episodes=None, overwrite=False, require_standard=False, require_vla=False)` | Prints lerobot export for LeRobot export. |
| 1842 | function | `def nested_dict_value(data, path, default=None)` | Handles nested dict value behavior. |
| 1851 | function | `def vector_xy(value)` | Handles vector xy behavior. |
| 1858 | function | `def vector_xyz(value)` | Handles vector xyz behavior. |
| 1865 | function | `def dashboard_scene_from_episode(row)` | Handles dataset dashboard behavior for dashboard scene from episode. |
| 1916 | function | `def dashboard_episode_summary(row)` | Handles dataset dashboard behavior for dashboard episode summary. |
| 1939 | function | `def list_dashboard_runs(dataset_root, limit=80)` | Handles dataset dashboard behavior for list dashboard runs. |
| 1965 | function | `def dashboard_run_payload(run_dir)` | Handles dataset dashboard behavior for dashboard run payload. |
| 1983 | function | `def downsample_indices(count, max_points)` | Handles downsample indices behavior. |
| 2001 | function | `def contiguous_stage_spans(samples, t0)` | Handles contiguous stage spans behavior. |
| 2025 | function | `def radians_vector_to_degrees(value, length=4)` | Handles radians vector to degrees behavior. |
| 2038 | function | `def dashboard_episode_payload(run_dir, episode_index, max_points=1800)` | Handles dataset dashboard behavior for dashboard episode payload. |
| 2108 | function | `def dashboard_html()` | Handles dataset dashboard behavior for dashboard html. |
| 2565 | function | `def serve_dashboard(dataset_root='excavator_auto_dataset', host='127.0.0.1', port=8765, open_browser=False)` | Handles dataset dashboard behavior for serve dashboard. |
| 2577 | class | `class Handler(http.server.BaseHTTPRequestHandler)` | Handles Handler behavior. |
| 2578 | function | `serve_dashboard.Handler.log_message` `def log_message(self, fmt, *args)` | HTTP handler method used by the embedded dashboard server. |
| 2581 | function | `serve_dashboard.Handler.send_bytes` `def send_bytes(self, data, content_type='application/json', status=200)` | Sends bytes. |
| 2589 | function | `serve_dashboard.Handler.send_json` `def send_json(self, data, status=200)` | Sends json for JSON payload. |
| 2592 | function | `serve_dashboard.Handler.do_GET` `def do_GET(self)` | HTTP handler method used by the embedded dashboard server. |
| 2620 | class | `class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer)` | Handles ThreadingServer behavior. |
| 2641 | function | `def print_summary(run_dir)` | Prints summary. |
| 2646 | function | `def print_analysis(run_dir, include_timeline=True, compact=False)` | Prints analysis. |

### `main.py`

- Module role: Compatibility entry that forwards execution to main_zsp.py.
- Primary imports: `os`, `runpy`
- Runtime note: Runs main_zsp.py at import/script execution time.
- No functions or classes are defined in this file.

### `main_zsp.py`

- Module role: Isaac Sim Script Editor entry for selecting the USD source, normalizing ZSP layers, and bootstrapping the excavator runtime.
- Primary imports: `os`, `json`, `shutil`, `sys`, `urllib.parse`, `excavator_app.bootstrap`
- Runtime note: Opens the selected USD stage and runs bootstrap at module execution time.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 15 | function | `def _load_config()` | Loads config. |
| 46 | function | `def _config_text(key, default='')` | Handles config text behavior. |
| 51 | function | `def _truthy_model_source(value)` | Handles truthy model source behavior. |
| 61 | function | `def _script_dir()` | Handles script dir behavior. |
| 68 | function | `def _looks_like_project_root(root)` | Handles inverse kinematics behavior for looks like project root. |
| 75 | function | `def _import_roots(root)` | Handles import roots behavior. |
| 85 | function | `def _zsp_dir(root)` | Handles zsp dir behavior. |
| 89 | function | `def _zsp_stage_path(root)` | Handles zsp stage path behavior. |
| 93 | function | `def _original_scene_path(root)` | Handles original scene path behavior. |
| 97 | function | `def _selected_stage_path(root)` | Handles selected stage path behavior. |
| 101 | function | `def _selected_model_label()` | Handles selected model label behavior. |
| 105 | function | `def _url_to_path(value)` | Handles url to path behavior. |
| 119 | function | `def _current_stage_file()` | Handles current stage file behavior. |
| 142 | function | `def _parents(path, max_depth=6)` | Handles parents behavior. |
| 158 | function | `def _stage_related_roots()` | Handles stage related roots behavior. |
| 175 | function | `def _known_repo_roots()` | Handles known repo roots behavior. |
| 184 | function | `def _candidate_roots()` | Handles candidate roots behavior. |
| 203 | function | `def ensure_project_root()` | Ensures or creates the required runtime state for project root. |
| 220 | function | `def _normalize_zsp_layer_paths(project_root)` | Normalizes zsp layer paths. |
| 271 | function | `def _ensure_zsp_configuration_aliases(project_root)` | Ensures or creates the required runtime state for zsp configuration aliases. |
| 308 | function | `def _pump_kit_updates(frame_count=5)` | Handles pump kit updates behavior. |
| 319 | function | `def _open_selected_stage(project_root)` | Handles open selected stage behavior. |

### `run_dataset_dashboard.py`

- Module role: Thin CLI wrapper around excavator_dataset_tools.serve_dashboard().
- Primary imports: `argparse`, `excavator_dataset_tools`
- Runtime note: Has __main__ CLI/script guard.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 6 | function | `def main()` | Entry point or long-running loop for this script/module. |

### `run_excavator_standalone.py`

- Module role: Standalone Isaac Sim launcher with viewport-based RGB capture and a TCP bridge server.
- Primary imports: `argparse`, `sys`, `os`, `ctypes`
- Runtime note: Has __main__ CLI/script guard.
- Runtime note: Schedules coroutine(s) with asyncio.ensure_future().
- Runtime note: Starts an asyncio TCP server.
- Runtime note: Creates or depends on Isaac Sim SimulationApp.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 62 | function | `def capsule_to_numpy_rgba(capsule, buffer_size, width, height, np_module)` | Convert viewport capture PyCapsule buffer to a numpy RGBA array. |
| 89 | function | `def cleanup_viewport_capture_helpers(force=False)` | Handles viewport capture behavior for cleanup viewport capture helpers. |
| 99 | function | `def capture_rgb_from_viewport(viewport, world, simulation_app, capture_viewport_to_buffer, np_module, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, wait_frames=CAP...)` | Capture RGB image from the active viewport. |
| 115 | function | `capture_rgb_from_viewport.on_capture` `def on_capture(capsule, buffer_size, w, h, fmt)` | Handles on capture behavior. |
| 161 | function | `def capture_rgb_from_cameras(viewport, camera_paths, world, simulation_app, capture_viewport_to_buffer, np_module, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, wa...)` | Handles camera capture behavior for capture rgb from cameras. |
| 188 | function | `def main()` | Main entry point for standalone launch. |
| 465 | async function | `main.read_json` `async def read_json(reader)` | Reads json for JSON payload. |
| 471 | async function | `main.write_json` `async def write_json(writer, obj)` | Writes json for JSON payload. |
| 477 | function | `main.make_action_from_command` `def make_action_from_command(cmd)` | Creates action from command for articulation action. |
| 502 | async function | `main.handle_client` `async def handle_client(reader, writer)` | Handles handle client behavior. |
| 531 | async function | `main.start_bridge_server` `async def start_bridge_server()` | Starts bridge server for TCP bridge. |

### `archive/standalone/run_excavator_standalone_backup_camera_bug.py`

- Module role: Backup standalone launcher that uses the older camera API path kept for regression reference.
- Primary imports: `argparse`, `sys`, `os`
- Runtime note: Has __main__ CLI/script guard.
- Runtime note: Schedules coroutine(s) with asyncio.ensure_future().
- Runtime note: Starts an asyncio TCP server.
- Runtime note: Creates or depends on Isaac Sim SimulationApp.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 34 | function | `def main()` | Main entry point for standalone launch. |
| 207 | function | `main.ensure_camera_prim` `def ensure_camera_prim(stage, camera_path)` | Ensures or creates the required runtime state for camera prim for camera capture. |
| 261 | async function | `main.read_json` `async def read_json(reader)` | Reads json for JSON payload. |
| 267 | async function | `main.write_json` `async def write_json(writer, obj)` | Writes json for JSON payload. |
| 273 | function | `main.make_action_from_command` `def make_action_from_command(cmd)` | Creates action from command for articulation action. |
| 295 | async function | `main.handle_client` `async def handle_client(reader, writer)` | Handles handle client behavior. |
| 327 | async function | `main.start_bridge_server` `async def start_bridge_server()` | Starts bridge server for TCP bridge. |

### `scripts/bridge_test/external_client_policy.py`

- Module role: Small socket client that sends simple external policy commands to the bridge.
- Primary imports: `argparse`, `base64`, `json`, `socket`, `struct`, `time`, `zlib`, `numpy`
- Runtime note: Has __main__ CLI/script guard.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 25 | function | `def send_json(sock, obj)` | Sends json for JSON payload. |
| 31 | function | `def recv_exact(sock, n)` | Receives exact. |
| 43 | function | `def recv_json(sock)` | Receives json for JSON payload. |
| 50 | function | `def decode_rgb(reply)` | Decodes rgb for RGB image payload. |
| 56 | function | `def main()` | Entry point or long-running loop for this script/module. |

### `scripts/bridge_test/gui_client.py`

- Module role: Pygame teleoperation client for the TCP bridge with multi-camera display.
- Primary imports: `argparse`, `base64`, `json`, `socket`, `struct`, `zlib`, `numpy`, `pygame`
- Runtime note: Has __main__ CLI/script guard.
- Runtime note: Runs a pygame GUI/event loop.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 48 | function | `def send_json(sock, obj)` | Sends json for JSON payload. |
| 54 | function | `def recv_exact(sock, n)` | Receives exact. |
| 66 | function | `def recv_json(sock)` | Receives json for JSON payload. |
| 73 | function | `def decode_camera_rgb(camera_payload)` | Decodes camera rgb for camera capture. |
| 79 | function | `def decode_rgb(reply)` | Decodes rgb for RGB image payload. |
| 83 | function | `def decode_camera_images(reply)` | Decodes camera images for camera capture. |
| 94 | function | `def clamp_joint(name, value)` | Handles joint state or command behavior for clamp joint. |
| 99 | class | `class Button` | Handles Button behavior. |
| 100 | function | `Button.__init__` `def __init__(self, rect, label, callback)` | Handles init behavior. |
| 106 | function | `Button.handle` `def handle(self, event)` | Pygame GUI object method for user interaction, drawing, or socket lifecycle. |
| 115 | function | `Button.draw` `def draw(self, surf, font)` | Pygame GUI object method for user interaction, drawing, or socket lifecycle. |
| 122 | class | `class GuiClient` | Handles Omni UI controls behavior for GuiClient. |
| 123 | function | `GuiClient.__init__` `def __init__(self, host, port, ticks)` | Handles init behavior. |
| 136 | function | `GuiClient._build_buttons` `def _build_buttons(self)` | Builds buttons for Omni UI controls. |
| 150 | function | `GuiClient.nudge` `def nudge(self, idx, delta)` | Handles nudge behavior. |
| 154 | function | `GuiClient.connect` `def connect(self)` | Pygame GUI object method for user interaction, drawing, or socket lifecycle. |
| 165 | function | `GuiClient.close` `def close(self)` | Pygame GUI object method for user interaction, drawing, or socket lifecycle. |
| 173 | function | `GuiClient.step_sim` `def step_sim(self)` | Handles step sim behavior. |
| 199 | function | `GuiClient.handle_key` `def handle_key(self, key)` | Handles handle key behavior. |
| 214 | function | `GuiClient.draw` `def draw(self, surf, font, small_font)` | Pygame GUI object method for user interaction, drawing, or socket lifecycle. |
| 257 | function | `def main()` | Entry point or long-running loop for this script/module. |

### `scripts/bridge_test/gui_tcp_bridge_server.py`

- Module role: Script Editor TCP bridge server using Isaac Camera.get_rgb().
- Primary imports: `asyncio`, `base64`, `json`, `struct`, `zlib`, `numpy`, `omni.usd`, `omni.kit.app`, `pxr`, `isaacsim.core.api.world`, `isaacsim.core.prims`, `isaacsim.core.utils.types`, `isaacsim.sensors.camera`
- Runtime note: Schedules coroutine(s) with asyncio.ensure_future().
- Runtime note: Starts an asyncio TCP server.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 46 | function | `def ensure_camera_prim(stage, camera_path)` | Ensures or creates the required runtime state for camera prim for camera capture. |
| 66 | async function | `async def read_json(reader)` | Reads json for JSON payload. |
| 73 | async function | `async def write_json(writer, obj)` | Writes json for JSON payload. |
| 80 | function | `def make_action_from_command(cmd)` | Creates action from command for articulation action. |
| 105 | async function | `async def handle_client(reader, writer)` | Handles handle client behavior. |
| 162 | async function | `async def main()` | Entry point or long-running loop for this script/module. |

### `scripts/bridge_test/smolvla_client.py`

- Module role: SmolVLA socket client that converts bridge observations into model inputs and velocity commands.
- Primary imports: `argparse`, `base64`, `json`, `socket`, `struct`, `time`, `zlib`, `numpy`, `torch`, `torch.nn.functional`, `transformers`, `lerobot.policies.smolvla.modeling_smolvla`
- Runtime note: Has __main__ CLI/script guard.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 17 | function | `def read_json(sock)` | Reads json for JSON payload. |
| 24 | function | `def write_json(sock, obj)` | Writes json for JSON payload. |
| 30 | function | `def recv_exact(sock, n)` | Receives exact. |
| 42 | function | `def decode_rgb(reply)` | Decodes rgb for RGB image payload. |
| 59 | function | `def rgb_to_tensor(rgb, device)` | Input rgb: H x W x 3, uint8, range 0..255. |
| 71 | function | `def make_state(reply, device)` | SmolVLA expects state shape (1, 6). |
| 86 | function | `def action_to_joint_velocities(action, num_joints, max_vel=0.1)` | Convert SmolVLA 6D action to excavator joint velocity command. |
| 110 | function | `def main()` | Entry point or long-running loop for this script/module. |

### `scripts/bridge_test/archive/smolvla_client_backup_raw_action.py`

- Module role: Backup SmolVLA client variant preserving raw action conversion behavior.
- Primary imports: `argparse`, `base64`, `json`, `socket`, `struct`, `time`, `zlib`, `numpy`, `torch`, `torch.nn.functional`, `transformers`, `lerobot.policies.smolvla.modeling_smolvla`
- Runtime note: Has __main__ CLI/script guard.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 17 | function | `def read_json(sock)` | Reads json for JSON payload. |
| 24 | function | `def write_json(sock, obj)` | Writes json for JSON payload. |
| 30 | function | `def recv_exact(sock, n)` | Receives exact. |
| 42 | function | `def decode_rgb(reply)` | Decodes rgb for RGB image payload. |
| 59 | function | `def rgb_to_tensor(rgb, device)` | Input rgb: H x W x 3, uint8, range 0..255. |
| 71 | function | `def make_state(reply, device)` | SmolVLA expects state shape (1, 6). |
| 86 | function | `def action_to_joint_velocities(action, num_joints, max_vel=0.1)` | Convert SmolVLA 6D action to excavator joint velocity command. |
| 110 | function | `def main()` | Entry point or long-running loop for this script/module. |

### `scripts/bridge_test/archive/smolvla_client_guided_adapter_backup.py`

- Module role: Backup SmolVLA client with guided digging/action adapter logic.
- Primary imports: `argparse`, `base64`, `json`, `socket`, `struct`, `time`, `zlib`, `numpy`, `torch`, `torch.nn.functional`, `transformers`, `lerobot.policies.smolvla.modeling_smolvla`
- Runtime note: Has __main__ CLI/script guard.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 17 | function | `def recv_exact(sock, n)` | Receives exact. |
| 31 | function | `def read_json(sock)` | Reads json for JSON payload. |
| 38 | function | `def write_json(sock, obj)` | Writes json for JSON payload. |
| 43 | function | `def decode_rgb(reply)` | Decodes rgb for RGB image payload. |
| 60 | function | `def rgb_to_tensor(rgb, device)` | Input: rgb: H x W x 3, uint8, range 0..255 Output: 1 x 3 x 256 x 256, float32, range 0..1. |
| 75 | function | `def make_state(reply, device)` | SmolVLA checkpoint expects observation.state shape = (1, 6). |
| 92 | function | `def extract_action_np(action)` | Handles articulation action behavior for extract action np. |
| 101 | function | `def raw_action_to_joint_velocities(action, num_joints, max_vel)` | Pure zero-shot mapping: SmolVLA 6D action -> first num_joints joint velocities This usually causes only small random shaking for excavator, because pretrained SmolVLA was not tr. |
| 119 | function | `def model_guided_digging_velocities(action, step, num_joints, max_vel, cycle_len=120)` | Model-guided excavator action adapter. |
| 191 | function | `def position_target_from_velocity(reply, vel, ticks, scale=3.0)` | Optional position-control mode. |
| 212 | function | `def main()` | Entry point or long-running loop for this script/module. |

### `scripts/bridge_test/smolvla_policy_client.py`

- Module role: SmolVLA policy client with checkpoint path patching and language token handling.
- Primary imports: `argparse`, `base64`, `json`, `socket`, `struct`, `time`, `zlib`, `pathlib`, `numpy`, `torch`, `torch.nn.functional`, `transformers`, `lerobot.policies.smolvla.modeling_smolvla`
- Runtime note: Has __main__ CLI/script guard.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 18 | function | `def recv_exact(sock, n)` | Receives exact. |
| 30 | function | `def read_json(sock)` | Reads json for JSON payload. |
| 37 | function | `def write_json(sock, obj)` | Writes json for JSON payload. |
| 43 | function | `def decode_rgb(reply)` | Decode RGB image from bridge reply. |
| 68 | function | `def rgb_to_tensor(rgb, device)` | Input: rgb: H x W x 3, uint8, range 0..255 Output: tensor: 1 x 3 x 256 x 256, float32, range 0..1. |
| 83 | function | `def make_state(reply, device)` | New LeRobot dataset uses 14D observation.state: base_x base_y base_yaw swing boom arm bucket bucket_load_estimate bucket_tip_x bucket_tip_y bucket_tip_z bucket_load_x bucket_loa. |
| 121 | function | `def action_to_joint_velocities(action, num_joints, max_vel=1.0)` | Model action order: action[0] = swing action[1] = boom action[2] = arm action[3] = bucket Bridge joint_velocities order: vel[0] = bucket vel[1] = arm vel[2] = boom vel[3] = swing. |
| 156 | function | `def patch_checkpoint_paths(ckpt_dir, vlm_dir)` | Make the checkpoint fully local/offline. |
| 176 | function | `patch_checkpoint_paths.replace_obj` `def replace_obj(obj)` | Handles replace obj behavior. |
| 198 | function | `def load_language_tokens(task, vlm_dir, device)` | Create language token tensors manually. |
| 222 | function | `def main()` | Entry point or long-running loop for this script/module. |

### `scripts/diagnose_scene.py`

- Module role: USD scene diagnostic script for checking scene content without SimulationApp.
- Primary imports: `os`, `sys`
- Runtime note: Has __main__ CLI/script guard.
- Runtime note: Creates or depends on Isaac Sim SimulationApp.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 11 | function | `def diagnose_scene()` | Handles diagnose scene behavior. |

### `scripts/excavator_app/__init__.py`

- Module role: Package marker for excavator_app runtime modules.
- No functions or classes are defined in this file.

### `scripts/excavator_app/auto_dataset_collect.py`

- Module role: Automatic dataset planning loop helpers and planning failure classification.
- Primary imports: `copy`

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 24 | function | `def _snapshot_plan_state(rt)` | Handles snapshot plan state behavior. |
| 28 | function | `def _restore_plan_state(rt, snapshot)` | Handles restore plan state behavior. |
| 33 | function | `def _clear_executable_plan_state(rt)` | Clears executable plan state. |
| 52 | function | `def _group_targets_by_ring(rows)` | Handles group targets by ring behavior. |
| 70 | function | `def _planning_failure_signature(row)` | Handles planning failure signature behavior. |
| 88 | function | `def _is_route_budget_signature(signature)` | Returns whether route budget signature for motion route. |
| 92 | async function | `async def find_plan(rt, attempt_index)` | Finds plan. |
| 160 | function | `find_plan._return_best_ring_success` `def _return_best_ring_success(reason)` | Handles return best ring success behavior. |
| 451 | function | `def update_prepare_failure_streak(rt, current_count)` | Updates prepare failure streak. |

### `scripts/excavator_app/bootstrap.py`

- Module role: Runtime reload bootstrap used from Isaac Sim to load sand and excavator modules.
- Primary imports: `importlib`, `os`, `sys`

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 9 | function | `def ensure_project_root(extra_root=None)` | Ensures or creates the required runtime state for project root. |
| 33 | function | `def reload_runtime(module_name)` | Handles reload runtime behavior. |
| 41 | function | `def run_excavator_with_sand()` | Handles sand site or particle sand behavior for run excavator with sand. |
| 56 | function | `def run_sand_site_only()` | Handles sand site or particle sand behavior for run sand site only. |

### `scripts/excavator_app/excavator_runtime.py`

- Module role: Main excavator runtime: UI, state, IK, planning, collisions, execution, auto collection, and export orchestration.
- Primary imports: `asyncio`, `copy`, `hashlib`, `importlib`, `importlib.util`, `json`, `math`, `os`, `queue`, `subprocess`, `sys`, `threading`, `time`, `traceback`, `numpy`, `builtins`, `omni.usd`, `omni.kit.app`
- Runtime note: Schedules coroutine(s) with asyncio.ensure_future().
- Runtime note: Starts Python Thread worker(s).
- Runtime note: Registers main_loop task at import time.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 57 | function | `def _load_optional_joint_space_planner()` | Loads optional joint space planner for joint state or command. |
| 101 | function | `def project_path(*parts)` | Handles project path behavior. |
| 455 | function | `def runtime_module()` | Handles runtime module behavior. |
| 753 | function | `def _resample_closed_profile_2d(control_points, target_count)` | Handles resample closed profile 2d behavior. |
| 1072 | function | `def sdf_path(path)` | Handles sdf path behavior. |
| 1080 | function | `def get_prim(path)` | Returns or resolves prim. |
| 1084 | async function | `async def step_updates(n=1)` | Handles step updates behavior. |
| 1090 | function | `def deg_to_rad(x)` | Handles deg to rad behavior. |
| 1094 | function | `def rad_to_deg(x)` | Handles rad to deg behavior. |
| 1098 | function | `def safe_float(x, default=0.0)` | Handles safe float behavior. |
| 1221 | function | `def normalize_log_mode(mode)` | Normalizes log mode. |
| 1226 | function | `def current_log_mode()` | Handles current log mode behavior. |
| 1232 | function | `def debug_diagnostics_enabled()` | Handles debug diagnostics enabled behavior. |
| 1236 | function | `def debug_profile_enabled()` | Handles debug profile enabled behavior. |
| 1240 | function | `def non_quiet_diagnostics_enabled()` | Handles Omni UI controls behavior for non quiet diagnostics enabled. |
| 1244 | function | `def log_text_from_args(args, kwargs=None)` | Handles log text from args behavior. |
| 1251 | function | `def log_matches_any(text, tokens)` | Handles log matches any behavior. |
| 1255 | function | `def log_signature(text)` | Handles log signature behavior. |
| 1261 | function | `def should_emit_log(text, force=False)` | Handles should emit log behavior. |
| 1273 | function | `def info_print(*args, **kwargs)` | Handles info print behavior. |
| 1280 | function | `def set_log_mode(mode, announce=True)` | Sets or updates log mode. |
| 1291 | function | `def toggle_debug_log()` | Toggles debug log. |
| 1296 | function | `def toggle_quiet_log()` | Toggles quiet log for Omni UI controls. |
| 1301 | function | `def cycle_log_mode()` | Handles cycle log mode behavior. |
| 1310 | function | `def print_log_state()` | Prints log state. |
| 1332 | function | `def disable_usd_audio_extension()` | Handles disable usd audio extension behavior. |
| 1345 | function | `def cached_real_joint_positions()` | Handles joint state or command behavior for cached real joint positions. |
| 1358 | function | `def simulation_timeline_is_playing()` | Handles simulation timeline is playing behavior. |
| 1370 | function | `def ensure_timeline_playing(label='')` | Ensures or creates the required runtime state for timeline playing. |
| 1388 | function | `def timeline_allows_background_work()` | Handles timeline allows background work behavior. |
| 1392 | function | `def handle_timeline_stop_if_needed(label='')` | Handles handle timeline stop if needed behavior. |
| 1421 | function | `def object_physics_view_state(obj, depth=0)` | Handles object physics view state behavior. |
| 1446 | function | `def robot_joint_read_ready()` | Handles joint state or command behavior for robot joint read ready. |
| 1461 | function | `def robot_articulation_action_ready()` | Handles articulation action behavior for robot articulation action ready. |
| 1474 | function | `def articulation_action_ready_detail()` | Handles articulation action behavior for articulation action ready detail. |
| 1491 | function | `def format_action_ready_detail(detail)` | Formats action ready detail for articulation action. |
| 1503 | async function | `async def recover_articulation_action_channel(label='action_channel_recover')` | Handles articulation action behavior for recover articulation action channel. |
| 1549 | async function | `async def wait_for_articulation_action_ready(label='action', min_stable_frames=ACTION_READY_MIN_STABLE_FRAMES, max_frames=ACTION_READY_MAX_WAIT_FRAMES, record_failure=...)` | Waits for for articulation action ready. |
| 1582 | function | `def get_real_joint_positions()` | Returns or resolves real joint positions for joint state or command. |
| 1613 | function | `def update_q_cmd_from_real()` | Updates q cmd from real. |
| 1628 | function | `def planner_joint_positions()` | Handles joint state or command behavior for planner joint positions. |
| 1632 | function | `def planner_end_effector_pos(end_effector='mid', q=None)` | Handles planner state behavior for planner end effector pos. |
| 1650 | function | `def hold_manual_ui_sync()` | Handles Omni UI controls behavior for hold manual ui sync. |
| 1654 | function | `def manual_ui_sync_is_held()` | Handles Omni UI controls behavior for manual ui sync is held. |
| 1658 | function | `def get_speed_multiplier()` | Returns or resolves speed multiplier. |
| 1665 | function | `def get_manual_speed_multiplier()` | Returns or resolves manual speed multiplier. |
| 1670 | function | `def ui_short_text(text, max_chars=UI_STATUS_MAX_CHARS)` | Handles Omni UI controls behavior for ui short text. |
| 1677 | function | `def apply_speed_to_physx_joint_limits()` | Applies speed to physx joint limits for joint state or command. |
| 1683 | function | `def update_status(text, force=False)` | Updates status. |
| 1711 | function | `def cancel_active_task(reason='manual override')` | Cancels active task. |
| 1724 | function | `def start_task(name)` | Starts task. |
| 1738 | function | `def task_alive(task_id)` | Handles task alive behavior. |
| 1745 | function | `def invalidate_active_task(reason='')` | Handles invalidate active task behavior. |
| 1754 | function | `def motion_cancel_requested(task_id=None)` | Handles motion execution behavior for motion cancel requested. |
| 1766 | function | `def cancel_registered_task(name, reason='')` | Cancels registered task. |
| 1787 | function | `def cancel_registered_tasks(reason='', keep=None)` | Cancels registered tasks. |
| 1795 | function | `def _registered_task_done(name, task)` | Handles registered task done behavior. |
| 1816 | function | `def register_async_task(name, coro_or_task, replace=True)` | Registers async task. |
| 1829 | function | `def find_robot_paths()` | Finds robot paths. |
| 1838 | function | `def clear_xform(prim)` | Clears xform for USD transforms. |
| 1844 | function | `def set_xform(prim, translate=None, scale=None)` | Sets or updates xform for USD transforms. |
| 1852 | function | `def set_translate_preserve_xform_ops(prim_or_path, translate)` | Sets or updates translate preserve xform ops for USD transforms. |
| 1873 | function | `def _xform_op_type(op)` | Handles USD transforms behavior for xform op type. |
| 1880 | function | `def _is_xform_rotate_op(op)` | Returns whether xform rotate op for USD transforms. |
| 1896 | function | `def _set_rotation_op_z(op, yaw_deg)` | Sets or updates rotation op z. |
| 1925 | function | `def set_translate_rotate_z_preserve_xform_ops(prim_or_path, translate=None, yaw_deg=None)` | Sets or updates translate rotate z preserve xform ops for USD transforms. |
| 1973 | function | `def get_prim_local_yaw_z_deg(path, default=0.0)` | Returns or resolves prim local yaw z deg. |
| 2004 | function | `def set_color(prim, color)` | Sets or updates color. |
| 2013 | function | `def set_prim_visibility(prim_or_path, visible)` | Sets or updates prim visibility. |
| 2026 | function | `def bucket_load_volume_hidden_root_path()` | Handles bucket load volume hidden root path behavior. |
| 2031 | function | `def is_bucket_load_volume_runtime_path(path)` | Returns whether bucket load volume runtime path. |
| 2037 | function | `def set_dataset_camera_prims_invisible()` | Sets or updates dataset camera prims invisible. |
| 2057 | function | `def restore_excavator_control_visibility()` | Handles restore excavator control visibility behavior. |
| 2080 | function | `def subtree_has_mesh(prim)` | Handles subtree has mesh behavior. |
| 2093 | function | `def collect_excavator_display_roots()` | Collects excavator display roots. |
| 2127 | function | `def apply_excavator_render_mode(render_on=None, force_status=True)` | Applies excavator render mode. |
| 2163 | function | `def debug_profile_record(label, elapsed_ms, data=None, threshold_ms=None, error='')` | Handles debug profile record behavior. |
| 2239 | function | `def increment_runtime_counter(name, amount=1, total_name=None)` | Handles increment runtime counter behavior. |
| 2250 | function | `def debug_profile_span(label, start_time, threshold_ms=5.0, data=None)` | Handles debug profile span behavior. |
| 2258 | function | `def debug_profiled(label=None, threshold_ms=None)` | Handles debug profiled behavior. |
| 2259 | function | `debug_profiled.decorate` `def decorate(fn)` | Handles decorate behavior. |
| 2263 | async function | `debug_profiled.decorate.async_wrapper` `async def async_wrapper(*args, **kwargs)` | Handles async wrapper behavior. |
| 2292 | function | `debug_profiled.decorate.wrapper` `def wrapper(*args, **kwargs)` | Handles wrapper behavior. |
| 2323 | function | `def toggle_excavator_render_mode()` | Toggles excavator render mode. |
| 2327 | function | `def debug_visuals_enabled()` | Handles debug visuals enabled behavior. |
| 2331 | function | `def debug_visual_root_paths()` | Handles debug visual root paths behavior. |
| 2365 | function | `def apply_debug_visuals_visibility(visible=None, force_status=True)` | Applies debug visuals visibility. |
| 2398 | function | `def toggle_debug_visuals_from_ui()` | Toggles debug visuals from ui for Omni UI controls. |
| 2402 | function | `def set_prim_attr(prim, name, value, type_name=None)` | Sets or updates prim attr. |
| 2423 | function | `def with_root_edit_target(fn)` | Handles with root edit target behavior. |
| 2438 | function | `def get_physx_schema()` | Returns or resolves physx schema for PhysX settings. |
| 2452 | function | `def apply_physx_api_by_names(prim, names)` | Applies physx api by names for PhysX settings. |
| 2470 | function | `def apply_physx_scene_api(prim)` | Applies physx scene api for PhysX settings. |
| 2486 | function | `def set_schema_attr(obj, names, value)` | Sets or updates schema attr. |
| 2498 | function | `def enable_physx_gpu_runtime_settings()` | Handles PhysX settings behavior for enable physx gpu runtime settings. |
| 2533 | function | `def configure_world_gpu_physics(world, label='')` | Configures world gpu physics. |
| 2550 | function | `configure_world_gpu_physics.call_bool` `def call_bool(names, value=True)` | Handles call bool behavior. |
| 2562 | function | `configure_world_gpu_physics.call_value` `def call_value(names, value)` | Handles call value behavior. |
| 2599 | function | `def ensure_particle_gpu_physics_scene(label='')` | Ensures or creates the required runtime state for particle gpu physics scene for PhysX particle sand. |
| 2663 | function | `def rebind_sand_particles_to_physics_scene(label='')` | Handles sand site or particle sand behavior for rebind sand particles to physics scene. |
| 2693 | function | `def get_bbox(path)` | Returns or resolves bbox for bounding boxes. |
| 2709 | function | `def bbox_center(path)` | Handles bounding boxes behavior for bbox center. |
| 2719 | function | `def bbox_min_z(path)` | Handles bounding boxes behavior for bbox min z. |
| 2730 | function | `def bbox_values_are_valid(mn, mx)` | Handles bounding boxes behavior for bbox values are valid. |
| 2747 | function | `def bbox_min_max(path)` | Handles bounding boxes behavior for bbox min max. |
| 2761 | function | `def bbox_center_size(path)` | Handles bounding boxes behavior for bbox center size. |
| 2770 | function | `def selected_prim_paths()` | Handles selected prim paths behavior. |
| 2782 | function | `def first_selected_prim_path()` | Handles first selected prim path behavior. |
| 2787 | function | `def mesh_world_xy_points_under(prim)` | Handles mesh world xy points under behavior. |
| 2809 | function | `def mesh_points_under_in_root_local(root_prim)` | Handles mesh points under in root local behavior. |
| 2837 | function | `def transform_root_local_points_to_world(root_prim, points)` | Handles transform root local points to world behavior. |
| 2852 | function | `def unload_mesh_local_frame_footprint(root_prim, shrink_d)` | Handles unload target or dump sequence behavior for unload mesh local frame footprint. |
| 2900 | function | `def convex_hull_xy(points)` | Handles convex hull xy behavior. |
| 2909 | function | `convex_hull_xy.cross` `def cross(o, a, b)` | Handles cross behavior. |
| 2926 | function | `def polygon_signed_area(poly)` | Handles polygon signed area behavior. |
| 2935 | function | `def polygon_centroid_xy(poly)` | Handles polygon centroid xy behavior. |
| 2953 | function | `def shrink_convex_polygon_xy(poly, shrink_d)` | Handles shrink convex polygon xy behavior. |
| 2963 | function | `shrink_convex_polygon_xy.inside` `def inside(pt, a, b)` | Handles inside behavior. |
| 2968 | function | `shrink_convex_polygon_xy.intersect` `def intersect(s, ept, a, b)` | Handles intersect behavior. |
| 3003 | function | `def visual_polygon_xy(poly, max_vertices)` | Handles visual polygon xy behavior. |
| 3013 | function | `def circle_polygon_xy(cx, cy, radius, segments=20)` | Handles circle polygon xy behavior. |
| 3023 | function | `def prim_collision_enabled(prim)` | Handles collision state behavior for prim collision enabled. |
| 3036 | function | `def disable_collision(prim, label='', log=True)` | Handles collision state behavior for disable collision. |
| 3049 | function | `def enforce_bucket_cut_volume_hidden(label='', force_log=False)` | Handles enforce bucket cut volume hidden behavior. |
| 3095 | function | `def iter_prim_subtree(root_prim)` | Iterates prim subtree. |
| 3105 | function | `def nearest_rigid_ancestor_path(prim)` | Handles nearest rigid ancestor path behavior. |
| 3121 | function | `def collision_bbox_for_subtree(root_path)` | Handles collision state behavior for collision bbox for subtree. |
| 3150 | function | `def compact_path(path, max_len=120)` | Handles compact path behavior. |
| 3159 | function | `def collision_bbox_lowest_contributors(root_path, limit=3)` | Handles collision state behavior for collision bbox lowest contributors. |
| 3197 | function | `def collision_lowest_detail(link_name, limit=2)` | Handles collision state behavior for collision lowest detail. |
| 3214 | function | `def robot_support_bbox()` | Handles bounding boxes behavior for robot support bbox. |
| 3221 | function | `def robot_full_collision_bbox()` | Handles collision state behavior for robot full collision bbox. |
| 3227 | function | `def sync_articulation_pose_from_usd(label='')` | Synchronizes articulation pose from usd. |
| 3257 | function | `def get_prim_translation(path)` | Returns or resolves prim translation. |
| 3267 | function | `def make_cube(path, translate, scale, color, collision=True)` | Creates cube. |
| 3278 | function | `def make_box(path, translate, scale, color, collision=False, opacity=None)` | Creates box. |
| 3287 | function | `def make_sphere(path, translate, radius, color, collision=False)` | Creates sphere. |
| 3300 | function | `def set_opacity(prim, opacity)` | Sets or updates opacity. |
| 3307 | function | `def make_cylinder(path, translate, radius, height, color, collision=False, opacity=None)` | Creates cylinder. |
| 3323 | function | `def make_polygon_prism(path, polygon_xy, z_min, z_max, color, opacity=None)` | Creates polygon prism. |
| 3356 | function | `def make_polygon_wire_column(path, polygon_xy, z_min, z_max, color, width=None)` | Creates polygon wire column. |
| 3417 | function | `def bucket_collision_mesh_candidates()` | Handles collision state behavior for bucket collision mesh candidates. |
| 3468 | function | `def configure_robot_bucket_collision_mesh(prim)` | Configures robot bucket collision mesh for collision state. |
| 3469 | function | `configure_robot_bucket_collision_mesh.do_configure` `def do_configure()` | Handles do configure behavior. |
| 3542 | function | `def deinstance_bucket_mesh_scopes()` | Handles deinstance bucket mesh scopes behavior. |
| 3566 | function | `def configure_robot_bucket_sand_collision(label='')` | Configures robot bucket sand collision for sand site or particle sand. |
| 3587 | function | `def set_target_color(color)` | Sets or updates target color. |
| 3595 | function | `def get_sand_site_api()` | Returns or resolves sand site api for sand site or particle sand. |
| 3600 | function | `def sand_site_active()` | Handles sand site or particle sand behavior for sand site active. |
| 3605 | function | `def api_array(api, key, default, shape=None)` | Handles api array behavior. |
| 3624 | function | `def task_scene_context()` | Handles task scene context behavior. |
| 3751 | function | `def unload_bin_dump_point(height_delta=0.0, xy_offset=None, ctx=None)` | Handles unload target or dump sequence behavior for unload bin dump point. |
| 3783 | function | `def unload_bin_landing_point(height_delta=0.06, xy_offset=None, ctx=None)` | Handles unload target or dump sequence behavior for unload bin landing point. |
| 3809 | function | `def unload_landing_point_from_release(point=None)` | Handles unload target or dump sequence behavior for unload landing point from release. |
| 3820 | function | `def unload_context_polygon_xy(ctx, safe_margin=0.0)` | Handles unload target or dump sequence behavior for unload context polygon xy. |
| 3841 | function | `def point_to_segment_projection_xy(point_xy, a_xy, b_xy)` | Handles point to segment projection xy behavior. |
| 3854 | function | `def closest_point_on_polygon_boundary_xy(point_xy, poly)` | Handles closest point on polygon boundary xy behavior. |
| 3868 | function | `def unload_xy_inside_context(ctx, xy, safe_margin=0.0, outside_margin=0.0)` | Handles unload target or dump sequence behavior for unload xy inside context. |
| 3880 | function | `def unload_xy_overflow_context(ctx, xy, safe_margin=0.0)` | Handles unload target or dump sequence behavior for unload xy overflow context. |
| 3898 | function | `def clamp_unload_xy_to_context(ctx, xy, safe_margin=0.0)` | Handles unload target or dump sequence behavior for clamp unload xy to context. |
| 3914 | function | `def choose_unload_landing_point_for_flat_fill()` | Chooses unload landing point for flat fill for unload target or dump sequence. |
| 3988 | function | `def log_unload_context(label, target_xyz=None, unload_point=None)` | Handles unload target or dump sequence behavior for log unload context. |
| 4028 | function | `def ensure_unload_marker(point=None, label='')` | Ensures or creates the required runtime state for unload marker for unload target or dump sequence. |
| 4059 | function | `def update_sphere_marker(path, point, radius, color)` | Updates sphere marker. |
| 4087 | function | `def hide_debug_prim(path)` | Handles hide debug prim behavior. |
| 4098 | function | `def update_debug_line(path, points, color, width=0.035)` | Updates debug line. |
| 4143 | function | `def box_corners_from_min_max(mn, mx)` | Handles box corners from min max behavior. |
| 4154 | function | `def update_debug_segments(path, segments, color, width=0.025, visible=None)` | Updates debug segments. |
| 4203 | function | `def update_debug_rect_loop(path, center_xy, half_xy, z, color, width=0.025)` | Updates debug rect loop. |
| 4221 | function | `def update_debug_polygon_loop(path, polygon_xy, z, color, width=0.025)` | Updates debug polygon loop. |
| 4236 | function | `def draw_unload_dump_debug(label, landing_target=None, release_target=None, q_seed=None, q_seed_dump=None, best_drop=None, reason='', best_reachable_point=None,...)` | Draws unload dump debug for unload target or dump sequence. |
| 4400 | function | `def manual_unload_radius()` | Handles unload target or dump sequence behavior for manual unload radius. |
| 4407 | function | `def manual_unload_mesh_shrink_d()` | Handles unload target or dump sequence behavior for manual unload mesh shrink d. |
| 4414 | function | `def ensure_unload_range_column(point=None, label='')` | Ensures or creates the required runtime state for unload range column for unload target or dump sequence. |
| 4544 | function | `def bucket_point_world(name, q=None, reference_q=None)` | Handles bucket point world behavior. |
| 4558 | function | `def unload_alignment_report(q=None, effector='pour', reference_q=None)` | Handles unload target or dump sequence behavior for unload alignment report. |
| 4608 | function | `def log_unload_alignment(label, q=None, effector='pour', reference_q=None)` | Handles unload target or dump sequence behavior for log unload alignment. |
| 4631 | function | `def bucket_dump_forward_xy(q=None, reference_q=None)` | Handles bucket dump forward xy behavior. |
| 4656 | function | `def bucket_unload_release_point_world(q=None, reference_q=None)` | Handles unload target or dump sequence behavior for bucket unload release point world. |
| 4671 | function | `def pour_target_for_unload_release_source(release_target, q_estimate=None, q_start=None)` | Handles unload target or dump sequence behavior for pour target for unload release source. |
| 4684 | function | `def unload_drop_drift_model(q=None, reference_q=None, release=None, load=None, wall_z=None)` | Handles unload target or dump sequence behavior for unload drop drift model. |
| 4732 | function | `def unload_release_target_for_landing(landing_target, q_estimate=None, q_start=None, release_z=None, wall_z=None)` | Handles unload target or dump sequence behavior for unload release target for landing. |
| 4746 | function | `def preferred_unload_release_z(ctx=None, wall_z=None)` | Handles unload target or dump sequence behavior for preferred unload release z. |
| 4766 | function | `def predict_unload_drop(q=None, reference_q=None)` | Handles unload target or dump sequence behavior for predict unload drop. |
| 4809 | function | `def unload_drop_report(q=None, reference_q=None)` | Handles unload target or dump sequence behavior for unload drop report. |
| 4908 | function | `def log_unload_drop(label, q=None, reference_q=None)` | Handles unload target or dump sequence behavior for log unload drop. |
| 4944 | function | `def actual_unload_position_report()` | Handles unload target or dump sequence behavior for actual unload position report. |
| 4953 | function | `actual_unload_position_report.point_report` `def point_report(name, point)` | Handles point report behavior. |
| 5027 | function | `def log_actual_unload_position(label)` | Handles unload target or dump sequence behavior for log actual unload position. |
| 5061 | function | `def unload_arrival_report(q_goal=None)` | Handles unload target or dump sequence behavior for unload arrival report. |
| 5206 | function | `def verify_unload_arrival(label, q_goal)` | Handles unload target or dump sequence behavior for verify unload arrival. |
| 5256 | function | `def notify_sand_site_step_done(stage_name)` | Handles sand site or particle sand behavior for notify sand site step done. |
| 5262 | function | `def ensure_sand_site_bucket_colliders(force=False)` | Ensures or creates the required runtime state for sand site bucket colliders for sand site or particle sand. |
| 5266 | function | `def print_ground_contact_diagnostics(label='')` | Prints ground contact diagnostics. |
| 5333 | function | `def print_motion_constraint_diagnostics(label='')` | Prints motion constraint diagnostics for motion execution. |
| 5365 | function | `def q_deg_values(q, wrap_swing_for_display=False)` | Handles q deg values behavior. |
| 5377 | function | `def vec_list(v, n=None)` | Handles vec list behavior. |
| 5386 | function | `def quantized_q_tuple(q, quantum_deg=None, n=4)` | Handles quantized q tuple behavior. |
| 5396 | function | `def quantized_xyz_tuple(v, quantum=0.005, n=3)` | Handles quantized xyz tuple behavior. |
| 5405 | function | `def compact_sand_metrics(metrics)` | Handles sand site or particle sand behavior for compact sand metrics. |
| 5428 | function | `def compact_bucket_load_metrics(metrics)` | Handles compact bucket load metrics behavior. |
| 5452 | function | `def sand_metrics_from_bucket_fast(bucket_metrics, base_metrics=None, scope='bucket_fast_only')` | Build a sand-metrics-shaped row without scanning pile/bin/spill regions. |
| 5496 | function | `def phase_metrics_requires_full(label)` | Handles Omni UI controls behavior for phase metrics requires full. |
| 5504 | function | `def compact_scene_context(ctx=None)` | Handles compact scene context behavior. |
| 5531 | function | `def auto_scene_context_signature(ctx=None)` | Handles auto scene context signature behavior. |
| 5535 | function | `def compact_unload_drop(report=None)` | Handles unload target or dump sequence behavior for compact unload drop. |
| 5576 | function | `def unload_drop_execution_ready(report)` | Handles unload target or dump sequence behavior for unload drop execution ready. |
| 5591 | function | `def unload_drop_loaded_center_ready(report)` | Handles unload target or dump sequence behavior for unload drop loaded center ready. |
| 5606 | function | `def stage_constraint_summary(row)` | Handles stage constraint summary behavior. |
| 5680 | function | `def compact_plan_stage(row)` | Handles compact plan stage behavior. |
| 5728 | function | `def route_diagnostics_from_stages(stages)` | Handles motion route behavior for route diagnostics from stages. |
| 5759 | function | `def unload_ballistics_from_stages(stages)` | Handles unload target or dump sequence behavior for unload ballistics from stages. |
| 5780 | function | `def compact_plan_candidate(row, include_stages=False)` | Handles compact plan candidate behavior. |
| 5810 | function | `def compact_dig_primitive_params(candidate)` | Handles digging behavior for compact dig primitive params. |
| 5877 | function | `def dig_plan_semantic_phase_name(phase)` | Handles dig plan behavior for dig plan semantic phase name. |
| 5887 | function | `def validate_dig_plan_contract(seq=None, points=None, stages=None, trace_points=None)` | Validates dig plan contract. |
| 6074 | function | `def json_sanitize(value)` | Handles JSON payload behavior for json sanitize. |
| 6096 | function | `def ensure_parent_dir(path)` | Ensures or creates the required runtime state for parent dir. |
| 6102 | function | `def write_json_file(path, data)` | Writes json file for JSON payload. |
| 6108 | function | `def append_jsonl(path, data)` | Handles JSON payload behavior for append jsonl. |
| 6114 | function | `def lru_cache_get(cache, key)` | Handles lru cache get behavior. |
| 6125 | function | `def lru_cache_put(cache, key, value, max_size)` | Handles lru cache put behavior. |
| 6146 | function | `def dataset_async_writer_enabled()` | Handles dataset behavior for dataset async writer enabled. |
| 6150 | function | `def dataset_writer_process_job(job)` | Handles dataset behavior for dataset writer process job. |
| 6165 | function | `def dataset_writer_loop(work_queue)` | Handles dataset behavior for dataset writer loop. |
| 6193 | function | `def dataset_writer_ensure()` | Handles dataset behavior for dataset writer ensure. |
| 6225 | function | `def dataset_writer_enqueue(job)` | Handles dataset behavior for dataset writer enqueue. |
| 6243 | function | `def append_jsonl_dataset(path, data)` | Handles dataset behavior for append jsonl dataset. |
| 6250 | function | `def dataset_writer_flush(label='')` | Handles dataset behavior for dataset writer flush. |
| 6267 | function | `def dataset_writer_shutdown(label='shutdown')` | Handles dataset behavior for dataset writer shutdown. |
| 6286 | function | `def ensure_xform_path(stage_obj, path)` | Ensures or creates the required runtime state for xform path for USD transforms. |
| 6303 | function | `def dataset_camera_image_extension()` | Handles dataset behavior for dataset camera image extension. |
| 6310 | function | `def dataset_camera_resolution()` | Handles dataset behavior for dataset camera resolution. |
| 6320 | function | `def dataset_camera_specs()` | Handles dataset behavior for dataset camera specs. |
| 6345 | function | `def is_camera_prim(prim)` | Returns whether camera prim for camera capture. |
| 6354 | function | `def ensure_dataset_camera_prim(stage_obj, spec)` | Ensures or creates the required runtime state for dataset camera prim. |
| 6368 | function | `def dataset_camera_prim_metadata(path)` | Handles dataset behavior for dataset camera prim metadata. |
| 6393 | function | `def dataset_camera_world_pose(path)` | Handles dataset behavior for dataset camera world pose. |
| 6414 | function | `def save_rgb_image(path, rgb, ensure_dir=True)` | Saves rgb image for RGB image payload. |
| 6439 | function | `def dataset_camera_runtime_ready()` | Handles dataset behavior for dataset camera runtime ready. |
| 6463 | function | `def dataset_camera_shutdown(reason='shutdown')` | Handles dataset behavior for dataset camera shutdown. |
| 6485 | function | `def dataset_camera_initialize(force=False)` | Handles dataset behavior for dataset camera initialize. |
| 6556 | function | `def dataset_camera_episode_metadata()` | Handles dataset behavior for dataset camera episode metadata. |
| 6575 | function | `def dataset_camera_episode_cache(reset=False)` | Handles dataset behavior for dataset camera episode cache. |
| 6635 | function | `def dataset_camera_sample_requires_complete_images(sample_index)` | Handles dataset behavior for dataset camera sample requires complete images. |
| 6644 | function | `def dataset_camera_payload_complete(payload, sample_index)` | Handles dataset behavior for dataset camera payload complete. |
| 6672 | function | `def dataset_camera_rgb_ready()` | Handles dataset behavior for dataset camera rgb ready. |
| 6703 | async function | `async def dataset_camera_warmup_for_episode(label='episode')` | Handles dataset behavior for dataset camera warmup for episode. |
| 6762 | function | `def dataset_capture_camera_observations(sample_index)` | Handles dataset behavior for dataset capture camera observations. |
| 6863 | function | `def jsonl_line_count(path)` | Handles JSON payload behavior for jsonl line count. |
| 6871 | function | `def debug_short_string(value, max_len=220)` | Handles debug short string behavior. |
| 6878 | function | `def debug_round_vec(value, n=None, digits=3)` | Handles debug round vec behavior. |
| 6885 | function | `def debug_compact_sand_counts(metrics=None)` | Handles sand site or particle sand behavior for debug compact sand counts. |
| 6903 | function | `def debug_compact_data(data, depth=0, max_items=8)` | Handles debug compact data behavior. |
| 6951 | function | `def debug_timeline_path(create=False)` | Handles debug timeline path behavior. |
| 6966 | function | `def debug_timeline_record(tag, stage='', result='', reason='', data=None, q_cmd=None, q_real=None, include_sand=False)` | Handles debug timeline record behavior. |
| 7011 | function | `def stable_json_hash(data)` | Handles JSON payload behavior for stable json hash. |
| 7019 | function | `def planner_config_snapshot()` | Handles planner state behavior for planner config snapshot. |
| 7066 | function | `def quality_gate_config_snapshot()` | Handles quality gate config snapshot behavior. |
| 7077 | function | `def auto_dataset_config_snapshot()` | Handles dataset behavior for auto dataset config snapshot. |
| 7128 | function | `def camera_config_snapshot()` | Handles camera capture behavior for camera config snapshot. |
| 7158 | function | `def sand_config_snapshot()` | Handles sand site or particle sand behavior for sand config snapshot. |
| 7185 | function | `def full_config_snapshot()` | Handles full config snapshot behavior. |
| 7195 | function | `def current_config_hash()` | Handles current config hash behavior. |
| 7199 | function | `def get_base_yaw_rad()` | Returns or resolves base yaw rad. |
| 7204 | function | `def dataset_sand_status()` | Handles dataset behavior for dataset sand status. |
| 7222 | function | `def dataset_bucket_load_estimate(metrics=None)` | Measured bucket load used by dataset state, as bucket_from_pile particle count. |
| 7235 | function | `def sand_particle_mass()` | Handles sand site or particle sand behavior for sand particle mass. |
| 7248 | function | `def sand_particle_prim()` | Handles sand site or particle sand behavior for sand particle prim. |
| 7266 | function | `def sand_particle_positions()` | Handles sand site or particle sand behavior for sand particle positions. |
| 7299 | function | `def invalidate_sand_runtime_caches(reason='')` | Handles sand site or particle sand behavior for invalidate sand runtime caches. |
| 7310 | function | `def build_sand_spatial_index(points, ctx=None)` | Builds sand spatial index for sand site or particle sand. |
| 7349 | function | `def sand_snapshot_local_indices(snapshot, x, y, radius)` | Handles sand site or particle sand behavior for sand snapshot local indices. |
| 7383 | function | `def sand_snapshot_surface_height(snapshot, x, y, radius=None)` | Handles sand site or particle sand behavior for sand snapshot surface height. |
| 7403 | function | `def sand_snapshot_density_count(snapshot, x, y, radius)` | Handles sand site or particle sand behavior for sand snapshot density count. |
| 7418 | function | `def get_sand_snapshot(force=False, label='', max_age=None)` | Returns or resolves sand snapshot for sand site or particle sand. |
| 7486 | function | `def sand_particle_snapshot()` | Handles sand site or particle sand behavior for sand particle snapshot. |
| 7493 | function | `def sand_reset_displacement_stats(a, b)` | Handles sand site or particle sand behavior for sand reset displacement stats. |
| 7513 | async function | `async def wait_for_sand_particles_stable(label='sand_reset')` | Waits for for sand particles stable for sand site or particle sand. |
| 7568 | async function | `async def wait_for_sand_settled_on_ground(label='sand_settle')` | Waits for for sand settled on ground for sand site or particle sand. |
| 7605 | async function | `async def reset_sand_site_stably(label='')` | Resets sand site stably for sand site or particle sand. |
| 7704 | function | `def request_sand_site_stable_reset(label='ui')` | Requests sand site stable reset for sand site or particle sand. |
| 7716 | async function | `async def delayed_startup_sand_reset()` | Handles sand site or particle sand behavior for delayed startup sand reset. |
| 7742 | function | `def link_world_transform(link_path)` | Handles link world transform behavior. |
| 7754 | function | `def transform_local_points_to_world(link_path, local_points)` | Handles transform local points to world behavior. |
| 7768 | function | `def project_points_to_link_local(points, link_path)` | Handles project points to link local behavior. |
| 7791 | function | `def initial_pile_mask_for_points(points, fallback_pile_mask=None)` | Handles initial pile mask for points behavior. |
| 7815 | function | `def bucket_particle_diagnostic(label, points=None, force_log=True)` | Handles PhysX particle sand behavior for bucket particle diagnostic. |
| 7934 | function | `def mask_points_in_box(points, center, half_xy, z_min, z_max)` | Handles mask points in box behavior. |
| 7948 | function | `def sand_pile_geometry_from_context(ctx=None)` | Handles sand site or particle sand behavior for sand pile geometry from context. |
| 7960 | function | `def sand_pile_xy_mask(points, ctx=None)` | Handles sand site or particle sand behavior for sand pile xy mask. |
| 7970 | function | `def sand_settle_status(points=None, ctx=None)` | Handles sand site or particle sand behavior for sand settle status. |
| 8024 | function | `def filter_settled_sand_particles(points, ctx=None)` | Handles sand site or particle sand behavior for filter settled sand particles. |
| 8034 | function | `def sand_region_masks(points)` | Handles sand site or particle sand behavior for sand region masks. |
| 8075 | function | `def bucket_load_volume_source_path()` | Handles bucket load volume source path behavior. |
| 8081 | function | `def clear_bucket_load_volume_caches()` | Clears bucket load volume caches. |
| 8091 | function | `def point_in_polygon_2d(points_xy, polygon_xy)` | Handles point in polygon 2d behavior. |
| 8113 | function | `def default_bucket_load_volume_mesh_local()` | Handles default bucket load volume mesh local behavior. |
| 8149 | function | `def bucket_load_volume_candidate_mesh_paths()` | Handles bucket load volume candidate mesh paths behavior. |
| 8158 | function | `def mesh_faces_from_usd(mesh)` | Handles mesh faces from usd behavior. |
| 8174 | function | `def mesh_edge_pairs_from_faces(faces)` | Handles mesh edge pairs from faces behavior. |
| 8191 | function | `def bucket_load_source_mesh_local(path)` | Handles bucket load source mesh local behavior. |
| 8203 | function | `bucket_load_source_mesh_local.read_one_mesh` `def read_one_mesh(prim)` | Reads one mesh. |
| 8259 | function | `def authored_bucket_volume_faces(vertices, faces, source_path='', used_paths=None)` | Handles authored bucket volume faces behavior. |
| 8278 | function | `def bucket_load_volume_authored_mesh_local()` | Handles bucket load volume authored mesh local behavior. |
| 8313 | function | `def bucket_load_volume_mesh_local()` | Handles bucket load volume mesh local behavior. |
| 8327 | function | `def triangulate_mesh_faces(faces)` | Handles triangulate mesh faces behavior. |
| 8338 | function | `def bucket_load_volume_topology_cache()` | Handles bucket load volume topology cache behavior. |
| 8397 | function | `def point_in_closed_mesh_topology(points, topology)` | Handles point in closed mesh topology behavior. |
| 8436 | function | `def point_in_closed_mesh(points, vertices, faces)` | Handles point in closed mesh behavior. |
| 8476 | function | `def bucket_load_volume_mask_from_local(local)` | Handles bucket load volume mask from local behavior. |
| 8491 | function | `def bucket_load_volume_local_bounds(expand=0.0)` | Handles bucket load volume local bounds behavior. |
| 8507 | function | `def bucket_load_volume_world_aabb(expand=0.0)` | Handles bucket load volume world aabb behavior. |
| 8520 | function | `def bucket_local_box_world_corners(local_min, local_max)` | Handles bucket local box world corners behavior. |
| 8530 | function | `def bucket_load_volume_world_segments()` | Handles bucket load volume world segments behavior. |
| 8548 | function | `def bucket_load_volume_world_vertices(vertices=None)` | Handles bucket load volume world vertices behavior. |
| 8555 | function | `def bucket_load_volume_world_mesh()` | Handles bucket load volume world mesh behavior. |
| 8565 | function | `def bucket_sand_debug_root_path()` | Handles sand site or particle sand behavior for bucket sand debug root path. |
| 8570 | function | `def draw_bucket_sand_count_debug(force=False)` | Draws bucket sand count debug for sand site or particle sand. |
| 8618 | function | `def maybe_draw_bucket_sand_count_debug(force=False)` | Handles sand site or particle sand behavior for maybe draw bucket sand count debug. |
| 8642 | function | `def dataset_particle_snapshot(label='', build_bucket_index=False)` | Handles dataset behavior for dataset particle snapshot. |
| 8659 | function | `def build_bucket_particle_spatial_index(points)` | Builds bucket particle spatial index for PhysX particle sand. |
| 8698 | function | `def bucket_spatial_aabb_indices(index, aabb_min, aabb_max)` | Handles bucket spatial aabb indices behavior. |
| 8735 | function | `def bucket_load_fast_current(force=False, points=None)` | Handles bucket load fast current behavior. |
| 8923 | function | `def particle_ids_from_mask(mask)` | Handles PhysX particle sand behavior for particle ids from mask. |
| 8929 | function | `def capture_initial_pile_particle_ids()` | Handles PhysX particle sand behavior for capture initial pile particle ids. |
| 8941 | function | `def sand_metrics_current(force=False, snapshot=None)` | Handles sand site or particle sand behavior for sand metrics current. |
| 9003 | function | `sand_metrics_current.effective_region_count` `def effective_region_count(region_name, raw_count, region_count)` | Handles effective region count behavior. |
| 9061 | function | `def dataset_metrics_frame_bundle(label='', q_real=None, need_full=False, force=False)` | Handles dataset behavior for dataset metrics frame bundle. |
| 9111 | function | `def dataset_metrics_frame_full(label='', q_real=None, force=False)` | Handles dataset behavior for dataset metrics frame full. |
| 9124 | function | `def dataset_metrics_frame_fast(label='', q_real=None, force=False)` | Handles dataset behavior for dataset metrics frame fast. |
| 9134 | function | `def is_sand_contact_phase(mode)` | Returns whether sand contact phase for sand site or particle sand. |
| 9139 | function | `def is_sand_cut_geometry_phase(mode)` | Returns whether sand cut geometry phase for sand site or particle sand. |
| 9144 | function | `def phase_collision_context(mode)` | Handles collision state behavior for phase collision context. |
| 9168 | function | `def sand_contact_snapshot(force=False)` | Handles sand site or particle sand behavior for sand contact snapshot. |
| 9196 | function | `def point_distance(a, b)` | Handles point distance behavior. |
| 9205 | function | `def start_sand_contact_stage(stage_name)` | Starts sand contact stage for sand site or particle sand. |
| 9236 | function | `def update_sand_contact_progress(stage_name, q_cmd=None, q_real=None, force=False, log=True)` | Updates sand contact progress for sand site or particle sand. |
| 9363 | function | `def sand_contact_should_suppress_freeze(stage_name, blocked_names, cmd_err_deg, q_cmd, q_real)` | Handles sand site or particle sand behavior for sand contact should suppress freeze. |
| 9382 | function | `def secure_hold_spill_report(stage_name, report, bucket_loaded=None)` | Handles secure hold spill report behavior. |
| 9444 | function | `def sand_contact_stage_can_advance(q_goal, label='', mode='auto', seconds_eff=0.0)` | Handles sand site or particle sand behavior for sand contact stage can advance. |
| 9570 | function | `def sand_contact_stage_should_advance(stage_name, report, q_cmd=None, q_real=None, seconds_eff=0.0)` | Handles sand site or particle sand behavior for sand contact stage should advance. |
| 9690 | function | `def sand_contact_bad_cut_geometry(stage_name, report)` | Handles sand site or particle sand behavior for sand contact bad cut geometry. |
| 9716 | function | `def update_episode_quality_trackers(metrics, phase, q_cmd=None, q_real=None, action=None)` | Updates episode quality trackers for dataset episode. |
| 9749 | function | `def dataset_motion_derivatives(q_cmd, q_real, now)` | Handles dataset behavior for dataset motion derivatives. |
| 9807 | function | `def dataset_joint_error(q_cmd, q_real)` | Handles dataset behavior for dataset joint error. |
| 9819 | function | `def dataset_joint_effort_observation()` | Handles dataset behavior for dataset joint effort observation. |
| 9861 | function | `def dataset_joint_force_torque_observation()` | Handles dataset behavior for dataset joint force torque observation. |
| 9904 | function | `def dataset_phase_index(phase)` | Handles dataset behavior for dataset phase index. |
| 9929 | function | `def dataset_phase_features(phase)` | Handles dataset behavior for dataset phase features. |
| 9942 | function | `def point_aabb_signed_distance(point, mn, mx)` | Handles point aabb signed distance behavior. |
| 9954 | function | `def dataset_rigid_clearance_summary()` | Handles dataset behavior for dataset rigid clearance summary. |
| 9997 | function | `def dataset_local_height_patch(target=None, radius=0.42, grid=3)` | Handles dataset behavior for dataset local height patch. |
| 10049 | function | `def dataset_environment_summary(target=None)` | Handles dataset behavior for dataset environment summary. |
| 10089 | function | `def dataset_contact_flags(phase, sand_metrics, rigid_clearance)` | Handles dataset behavior for dataset contact flags. |
| 10106 | function | `def dataset_cost_summary(q_cmd, q_real, action, sand_metrics, rigid_clearance, dynamics=None)` | Handles dataset behavior for dataset cost summary. |
| 10130 | function | `def dataset_constraint_flags(cost, contact, phase_features)` | Handles dataset behavior for dataset constraint flags. |
| 10146 | function | `def dataset_observation_state(q_real=None, bucket_load_metrics=None)` | Handles dataset behavior for dataset observation state. |
| 10176 | function | `def dataset_record_sample(phase, q_cmd=None, q_real=None, label='', force=False)` | Handles dataset behavior for dataset record sample. |
| 10192 | function | `dataset_record_sample.mark_span` `def mark_span(span_label, started, threshold_ms=5.0, data=None)` | Handles mark span behavior. |
| 10383 | function | `def dataset_record_event(event, detail='', data=None)` | Handles dataset behavior for dataset record event. |
| 10411 | function | `def auto_collect_status_text()` | Handles automatic dataset collection behavior for auto collect status text. |
| 10428 | function | `def ensure_auto_collect_run_dir()` | Ensures or creates the required runtime state for auto collect run dir for automatic dataset collection. |
| 10642 | function | `def auto_collect_write_run_summary()` | Handles automatic dataset collection behavior for auto collect write run summary. |
| 10795 | function | `def lerobot_v3_external_export_python()` | Handles LeRobot export behavior for lerobot v3 external export python. |
| 10804 | function | `def text_tail(value, limit=6000)` | Handles text tail behavior. |
| 10812 | function | `def auto_collect_export_lerobot_v3_subprocess(run_dir, tool_path, export_record)` | Handles automatic dataset collection behavior for auto collect export lerobot v3 subprocess. |
| 10870 | function | `def auto_collect_export_lerobot_v3_sync(run_dir)` | Handles automatic dataset collection behavior for auto collect export lerobot v3 sync. |
| 10885 | function | `auto_collect_export_lerobot_v3_sync.write_failure_marker` `def write_failure_marker(record)` | Writes failure marker. |
| 10960 | async function | `async def auto_collect_export_lerobot_v3_on_finish(run_dir)` | Handles automatic dataset collection behavior for auto collect export lerobot v3 on finish. |
| 11052 | function | `def auto_collect_episode_dir(attempt_index)` | Handles automatic dataset collection behavior for auto collect episode dir. |
| 11057 | function | `def auto_collect_plan_summary(seq)` | Handles automatic dataset collection behavior for auto collect plan summary. |
| 11108 | function | `def auto_collect_episode_metrics()` | Handles automatic dataset collection behavior for auto collect episode metrics. |
| 11146 | function | `def record_phase_metrics(label, q_cmd=None, q_real=None, action=None, need_full=None)` | Handles record phase metrics behavior. |
| 11192 | function | `def stage_contract_summary(stage_name, stage_index=None, q_goal=None, duration=None)` | Handles stage contract summary behavior. |
| 11245 | function | `def stage_timing_key(stage_name, stage_index)` | Handles stage timing key behavior. |
| 11250 | function | `def update_stage_timing_summary(stage_name, result, elapsed_s=None, planned_s=None)` | Updates stage timing summary. |
| 11310 | function | `def stage_timing_attach(row, stage_name, stage_index, result, duration=None)` | Handles stage timing attach behavior. |
| 11366 | function | `def record_stage_audit(stage_name, stage_index, result, reason='', q_goal=None, duration=None, data=None, include_sand=False)` | Handles record stage audit behavior. |
| 11410 | function | `def auto_collect_preflight_report(target_successes=None)` | Handles automatic dataset collection behavior for auto collect preflight report. |
| 11599 | function | `def auto_collect_scene_pre_sample_gate(attempt_index)` | Cheap scene legality gate before any episode recording/camera work. |
| 11677 | function | `def auto_collect_initial_pose_pre_sample_gate(initial_info)` | Reject obviously unsafe initial poses before direct-settle and recording. |
| 11701 | function | `def auto_collect_plan_pre_sample_gate(seq, target=None)` | Keep auto dataset recording for plans that are likely to reach secure/load stages. |
| 11768 | function | `def clamp01(x)` | Handles clamp01 behavior. |
| 11772 | function | `def truncate_text(value, limit=240)` | Handles truncate text behavior. |
| 11780 | function | `def compact_target_score_row(row)` | Handles compact target score row behavior. |
| 11810 | function | `def compact_unload_score_row(row)` | Handles unload target or dump sequence behavior for compact unload score row. |
| 11816 | function | `def compact_auto_plan_attempts(plan_attempts, limit=4)` | Handles compact auto plan attempts behavior. |
| 11844 | function | `def auto_collect_record_planning_diagnostic(attempt_index, target, plan_attempts, reason, initial_info=None)` | Handles automatic dataset collection behavior for auto collect record planning diagnostic. |
| 11911 | function | `def compute_episode_quality_score(execution_success, reason)` | Computes episode quality score for dataset episode. |
| 12007 | function | `def phase_metric_bucket_from_pile(phase_metrics, phase_names)` | Handles phase metric bucket from pile behavior. |
| 12022 | function | `def auto_collect_write_segment_indices(run_dir, meta, index_row, score_report)` | Handles automatic dataset collection behavior for auto collect write segment indices. |
| 12063 | function | `auto_collect_write_segment_indices.append_segment` `def append_segment(filename, segment, count, threshold, extra=None)` | Handles append segment behavior. |
| 12113 | function | `def auto_collect_begin_episode(attempt_index, target, plan_attempts, seq, initial_info=None)` | Handles automatic dataset collection behavior for auto collect begin episode. |
| 12351 | function | `def auto_collect_finish_episode(meta, success, reason)` | Handles automatic dataset collection behavior for auto collect finish episode. |
| 12501 | function | `def auto_collect_finalize_active_episode_if_needed(reason)` | Handles automatic dataset collection behavior for auto collect finalize active episode if needed. |
| 12542 | function | `def auto_collect_sample_target(attempt_index, retry_index=0)` | Handles automatic dataset collection behavior for auto collect sample target. |
| 12571 | function | `def auto_dig_target_z_from_surface(surface_z, depth)` | Handles digging behavior for auto dig target z from surface. |
| 12578 | function | `def auto_dig_local_cut_surface_report(x, y, snapshot=None, fallback_surface_z=None)` | Estimate the sand surface along the bucket's actual cut path. |
| 12638 | function | `def sand_surface_height_for_auto_target(x, y, particles=None, snapshot=None)` | Handles sand site or particle sand behavior for sand surface height for auto target. |
| 12658 | function | `def auto_dig_depth_candidates()` | Handles digging behavior for auto dig depth candidates. |
| 12671 | function | `def auto_dig_target_sand_region_check(x, y, snapshot=None, ctx=None, real_surface_z=None)` | Handles digging behavior for auto dig target sand region check. |
| 12704 | function | `def auto_dig_swept_density_count(target, surface_z, particles=None, snapshot=None)` | Handles digging behavior for auto dig swept density count. |
| 12739 | function | `def auto_dig_approach_quality(target, surface_z, snapshot=None)` | Handles digging behavior for auto dig approach quality. |
| 12768 | function | `def auto_dig_fast_reach_check(target_world, q_seed=None, end_effector='tip')` | Handles digging behavior for auto dig fast reach check. |
| 12810 | function | `def auto_collect_rank_dig_targets(attempt_index)` | Handles automatic dataset collection behavior for auto collect rank dig targets. |
| 13038 | function | `def set_target_models_from_xyz(p)` | Sets or updates target models from xyz. |
| 13052 | function | `def set_unload_models_from_xyz(p)` | Sets or updates unload models from xyz for unload target or dump sequence. |
| 13060 | function | `def update_unload_models_only(p)` | Updates unload models only for unload target or dump sequence. |
| 13108 | function | `def auto_scene_randomization_config()` | Handles auto scene randomization config behavior. |
| 13118 | function | `def auto_scene_randomization_any_enabled()` | Handles auto scene randomization any enabled behavior. |
| 13129 | function | `def auto_scene_xy_radius(xy)` | Handles auto scene xy radius behavior. |
| 13136 | function | `def auto_scene_xy_angle_deg(xy)` | Handles auto scene xy angle deg behavior. |
| 13143 | function | `def auto_scene_sample_polar_xy(rng, radius_range, angle_deg_range)` | Handles auto scene sample polar xy behavior. |
| 13153 | function | `def angle_in_deg_range(angle_deg, range_pair)` | Handles angle in deg range behavior. |
| 13164 | function | `def valid_polygon_xy(poly)` | Handles valid polygon xy behavior. |
| 13176 | function | `def obb_polygon_xy(center_xy, size_xy, yaw_deg=0.0, min_size=0.05)` | Handles obb polygon xy behavior. |
| 13198 | function | `def rotate_xy_deg(vec_xy, yaw_deg)` | Handles rotate xy deg behavior. |
| 13208 | function | `def auto_scene_robot_safety_polygon()` | Handles auto scene robot safety polygon behavior. |
| 13217 | function | `def ellipse_polygon_xy(cx, cy, rx, ry, segments=32)` | Handles ellipse polygon xy behavior. |
| 13226 | function | `def auto_scene_sand_polygon_for_candidate(sand_xy=None, ctx=None)` | Handles sand site or particle sand behavior for auto scene sand polygon for candidate. |
| 13249 | function | `def auto_scene_current_truck_polygon_xy()` | Handles auto scene current truck polygon xy behavior. |
| 13263 | function | `def auto_scene_candidate_truck_polygon_xy(candidate)` | Handles auto scene candidate truck polygon xy behavior. |
| 13279 | function | `def auto_scene_truck_dump_offset_for_yaw(baseline, yaw_deg)` | Handles auto scene truck dump offset for yaw behavior. |
| 13290 | function | `def auto_scene_truck_center_for_unload_xy(unload_xy, yaw_deg, baseline)` | Handles unload target or dump sequence behavior for auto scene truck center for unload xy. |
| 13298 | function | `def auto_scene_robot_xy()` | Handles auto scene robot xy behavior. |
| 13309 | function | `def auto_scene_truck_rear_to_robot_yaw_for_unload(rng, unload_xy, baseline)` | Handles unload target or dump sequence behavior for auto scene truck rear to robot yaw for unload. |
| 13334 | function | `def auto_scene_truck_rear_alignment_error_deg(unload_xy, truck_center_xy)` | Handles auto scene truck rear alignment error deg behavior. |
| 13348 | function | `def auto_scene_sample_truck_yaw_for_unload(rng, unload_xy, unload_angle_deg, baseline)` | Handles unload target or dump sequence behavior for auto scene sample truck yaw for unload. |
| 13352 | function | `def polygons_overlap_with_margin_xy(a, b, margin=0.0)` | Handles polygons overlap with margin xy behavior. |
| 13375 | function | `def polygon_radius_range(poly)` | Handles polygon radius range behavior. |
| 13383 | function | `def auto_scene_geometry_legal(candidate=None, ctx=None, applied=False)` | Handles auto scene geometry legal behavior. |
| 13421 | function | `def auto_scene_random_workspace_bounds()` | Handles auto scene random workspace bounds behavior. |
| 13487 | function | `def auto_scene_sample_box_xy(rng, x_range, y_range)` | Handles auto scene sample box xy behavior. |
| 13494 | function | `def wrap_deg_180(value)` | Handles wrap deg 180 behavior. |
| 13498 | function | `def auto_scene_truck_baseline()` | Handles auto scene truck baseline behavior. |
| 13539 | function | `def auto_scene_candidate_legal(candidate)` | Handles auto scene candidate legal behavior. |
| 13597 | function | `def auto_scene_estimate_particles_for_amount(amount)` | Handles PhysX particle sand behavior for auto scene estimate particles for amount. |
| 13610 | function | `def auto_scene_attempt_record(candidate=None, ok=False, reason='', cfg=None)` | Handles auto scene attempt record behavior. |
| 13670 | function | `def auto_scene_sample_candidate(attempt_index)` | Handles auto scene sample candidate behavior. |
| 13780 | function | `def auto_scene_apply_candidate(candidate)` | Handles auto scene apply candidate behavior. |
| 13892 | async function | `async def auto_collect_apply_scene_randomization(attempt_index)` | Handles automatic dataset collection behavior for auto collect apply scene randomization. |
| 13939 | function | `def auto_collect_prepare_home_needed(policy=None, ready_reset_done=None)` | Handles automatic dataset collection behavior for auto collect prepare home needed. |
| 13987 | async function | `async def auto_collect_prepare_environment()` | Handles automatic dataset collection behavior for auto collect prepare environment. |
| 13994 | function | `auto_collect_prepare_environment.record_gate` `def record_gate(name, ok, reason='ok', detail=None)` | Handles record gate behavior. |
| 14008 | function | `auto_collect_prepare_environment.fail_prepare` `def fail_prepare(reason, gate='', detail=None, q_cmd=None)` | Handles fail prepare behavior. |
| 14299 | async function | `async def auto_collect_find_plan(attempt_index)` | Handles automatic dataset collection behavior for auto collect find plan. |
| 14303 | async function | `async def auto_collect_one_episode()` | Handles automatic dataset collection behavior for auto collect one episode. |
| 14607 | async function | `async def auto_collect_loop(count, max_attempts=None)` | Handles automatic dataset collection behavior for auto collect loop. |
| 14796 | function | `def request_auto_collect(count, max_attempts=None)` | Requests auto collect for automatic dataset collection. |
| 14809 | function | `def stop_auto_collect()` | Stops auto collect for automatic dataset collection. |
| 14825 | function | `def read_jsonl_rows(path)` | Reads jsonl rows for JSON payload. |
| 14842 | function | `def read_json_file_or_none(path)` | Reads json file or none for JSON payload. |
| 14850 | function | `def latest_successful_record_path()` | Handles latest successful record path behavior. |
| 14882 | function | `def q_from_replay_sample(sample)` | Handles episode replay behavior for q from replay sample. |
| 14892 | async function | `async def replay_record(path=None)` | Handles episode replay behavior for replay record. |
| 14965 | function | `def request_replay_latest_record()` | Requests replay latest record for episode replay. |
| 14969 | function | `def q_real_near_command(q_real, q_cmd)` | Handles q real near command behavior. |
| 14980 | function | `def q_delta_abs_deg(q_a, q_b)` | Handles q delta abs deg behavior. |
| 14993 | function | `def dataset_q_delta(q_a, q_b)` | Handles dataset behavior for dataset q delta. |
| 15005 | function | `def plan_joint_motion_metrics(q_to, q_from, duration=0.0)` | Plans joint motion metrics for joint state or command. |
| 15009 | function | `def reload_ik_calculation_module(reason='')` | Handles inverse kinematics behavior for reload ik calculation module. |
| 15029 | function | `def plan_path_penalty_cache_key(q_start, q_goal, mode)` | Plans path penalty cache key. |
| 15038 | function | `def path_penalty_cacheable(detail)` | Handles path penalty cacheable behavior. |
| 15050 | function | `def compute_path_penalty_uncached(q_start, q_goal, mode, deadline=None)` | Computes path penalty uncached. |
| 15073 | function | `def plan_path_penalty(q_start, q_goal, mode, deadline=None)` | Plans path penalty. |
| 15094 | function | `def planning_deadline_exceeded(deadline)` | Handles planning deadline exceeded behavior. |
| 15109 | function | `def child_planning_deadline(parent_deadline, max_seconds, min_seconds=0.05)` | Handles child planning deadline behavior. |
| 15124 | function | `def perf_block_record(label, elapsed_ms, data=None, threshold_ms=None)` | Handles perf block record behavior. |
| 15163 | function | `def clear_planning_runtime_caches(reason='')` | Clears planning runtime caches. |
| 15176 | function | `def stop_manual_motion_after_freeze(q_real, detail='')` | Stops manual motion after freeze for motion execution. |
| 15212 | function | `def stop_auto_motion_after_freeze(q_real, detail='', action_mode='')` | Stops auto motion after freeze for motion execution. |
| 15248 | function | `def freeze_contact_detail(mode='freeze', light=False)` | Handles freeze contact detail behavior. |
| 15301 | function | `def fmt_vec3(v)` | Handles fmt vec3 behavior. |
| 15310 | function | `def swing_drive_detail()` | Handles swing drive detail behavior. |
| 15326 | function | `def link_min_z_detail()` | Handles link min z detail behavior. |
| 15338 | function | `def link_lowest_collision_detail()` | Handles collision state behavior for link lowest collision detail. |
| 15348 | function | `def support_clearance_detail()` | Handles support clearance detail behavior. |
| 15370 | function | `def swing_obstacle_prediction_detail(q_cmd, q_real)` | Handles obstacle or collision checks behavior for swing obstacle prediction detail. |
| 15380 | function | `def swing_freeze_detail(q_cmd, q_real)` | Handles swing freeze detail behavior. |
| 15396 | function | `def log_freeze(reason, mode='', q_cmd=None, q_real=None, extra='', force=False)` | Handles log freeze behavior. |
| 15450 | function | `def check_freeze_state(label='loop')` | Checks freeze state. |
| 15571 | function | `def notify_sand_site_tool_sample(stage_name)` | Handles sand site or particle sand behavior for notify sand site tool sample. |
| 15579 | function | `def setup_paths()` | Handles setup paths behavior. |
| 15605 | function | `def read_imported_limit_deg(joint_name)` | Reads imported limit deg. |
| 15620 | function | `def compute_limits()` | Computes limits. |
| 15644 | function | `def ensure_joint_limits_are_valid()` | Ensures or creates the required runtime state for joint limits are valid for joint state or command. |
| 15695 | function | `def get_joint_drive_api(joint_prim)` | Returns or resolves joint drive api for joint state or command. |
| 15710 | function | `def set_drive_float_attr(drive, get_name, create_name, value)` | Sets or updates drive float attr. |
| 15721 | function | `def read_drive_attr(drive, get_name)` | Reads drive attr. |
| 15731 | function | `def configure_joint_drive_gains()` | Configures joint drive gains for joint state or command. |
| 15757 | function | `def configure_joints()` | Configures joints for joint state or command. |
| 15764 | function | `def make_parking_ground()` | Creates parking ground. |
| 15824 | function | `def support_ground_top_at_xy(x, y)` | Handles support ground top at xy behavior. |
| 15845 | function | `def park_robot_on_support_ground(label='')` | Handles park robot on support ground behavior. |
| 15945 | function | `def settle_robot_on_ground_if_needed(label='')` | Handles settle robot on ground if needed behavior. |
| 16087 | function | `def build_scene()` | Builds scene for Omni UI controls. |
| 16143 | class | `class ArticulationActionController` | Handles articulation action behavior for ArticulationActionController. |
| 16144 | function | `ArticulationActionController.__init__` `def __init__(self)` | Handles init behavior. |
| 16150 | function | `ArticulationActionController.clip_limits` `def clip_limits(self, q)` | Clips or constrains limits. |
| 16160 | function | `ArticulationActionController.clip_action_limits` `def clip_action_limits(self, q)` | Clips or constrains action limits for articulation action. |
| 16167 | function | `ArticulationActionController.ground_ok` `def ground_ok(self, mode='auto')` | Handles ground ok behavior. |
| 16198 | function | `ArticulationActionController.command_ground_ok` `def command_ground_ok(self, q, mode='auto')` | Handles command ground ok behavior. |
| 16227 | function | `ArticulationActionController.send_action` `def send_action(self, q, mode='action')` | Sends action for articulation action. |
| 16297 | function | `ArticulationActionController.hold_current_action` `def hold_current_action(self)` | Handles articulation action behavior for hold current action. |
| 16318 | function | `ArticulationActionController.apply_target` `def apply_target(self, q_target, mode='auto', speed_multiplier_override=None)` | Applies target. |
| 16355 | function | `ArticulationActionController.apply_target_direct` `def apply_target_direct(self, q_target, mode='manual')` | Applies target direct. |
| 16371 | function | `ArticulationActionController.print_state` `def print_state(self)` | Prints state. |
| 16397 | function | `def set_manual_joint_target(q_target, reason='slider')` | Sets or updates manual joint target for joint state or command. |
| 16418 | function | `def manual_recovery_project_target(q_target)` | Handles manual recovery project target behavior. |
| 16460 | function | `def apply_manual_joint_target_step()` | Applies manual joint target step for joint state or command. |
| 16497 | function | `def safe_home_q()` | Handles safe home q behavior. |
| 16505 | function | `def q_from_pose_deg(pose, reference=None)` | Handles q from pose deg behavior. |
| 16514 | function | `def auto_collect_initial_pose_for_attempt(attempt_index)` | Handles automatic dataset collection behavior for auto collect initial pose for attempt. |
| 16538 | async function | `async def auto_collect_move_to_initial_pose(attempt_index)` | Handles automatic dataset collection behavior for auto collect move to initial pose. |
| 16561 | function | `def get_target_pos()` | Returns or resolves target pos. |
| 16565 | function | `def set_target_xyz(x, y, z)` | Sets or updates target xyz. |
| 16573 | function | `def set_manual_unload_point_xyz(x, y, z, source='ui', inner_size=None, z_range=None, selected_path='', range_shape=None)` | Sets or updates manual unload point xyz for unload target or dump sequence. |
| 16613 | function | `def set_manual_unload_from_mesh_path(path, source='selected_mesh', status=True)` | Sets or updates manual unload from mesh path for unload target or dump sequence. |
| 16704 | function | `def set_manual_unload_from_selected_mesh()` | Sets or updates manual unload from selected mesh for unload target or dump sequence. |
| 16708 | function | `def apply_default_unload_source_mesh()` | Applies default unload source mesh for unload target or dump sequence. |
| 16712 | function | `def set_manual_unload_from_selected_point()` | Sets or updates manual unload from selected point for unload target or dump sequence. |
| 16742 | function | `def clear_manual_unload_override()` | Clears manual unload override for unload target or dump sequence. |
| 16760 | function | `def update_target_from_models()` | Updates target from models. |
| 16767 | function | `def get_target_xyz_from_models()` | Returns or resolves target xyz from models. |
| 16775 | function | `def get_unload_xyz_from_models()` | Returns or resolves unload xyz from models for unload target or dump sequence. |
| 16791 | function | `def get_unload_z_range_from_models()` | Returns or resolves unload z range from models for unload target or dump sequence. |
| 16796 | function | `get_unload_z_range_from_models.model_float` `def model_float(model, fallback)` | Handles model float behavior. |
| 16814 | function | `def get_unload_radius_from_model()` | Returns or resolves unload radius from model for unload target or dump sequence. |
| 16827 | function | `def get_unload_mesh_shrink_from_model()` | Returns or resolves unload mesh shrink from model for unload target or dump sequence. |
| 16840 | function | `def sync_target_from_sliders_live(force=False)` | target ball slider author throttle. |
| 16869 | function | `def sync_unload_from_sliders_live(force=False)` | Synchronizes unload from sliders live for unload target or dump sequence. |
| 16960 | function | `def configure_linear_trace_curve(path, color, width, default_points=None)` | Configures linear trace curve for planned or executed trace. |
| 17015 | function | `def ensure_trace_prims()` | Ensures or creates the required runtime state for trace prims for planned or executed trace. |
| 17054 | function | `def set_trace_visibility(paths, visible)` | Sets or updates trace visibility for planned or executed trace. |
| 17068 | function | `def hide_trace_prims()` | Handles planned or executed trace behavior for hide trace prims. |
| 17083 | function | `def show_trace_prims(mode=None)` | Handles planned or executed trace behavior for show trace prims. |
| 17101 | function | `def current_trace_mode()` | Handles planned or executed trace behavior for current trace mode. |
| 17113 | function | `def set_trace_mode(mode, reset_real=False)` | Sets or updates trace mode for planned or executed trace. |
| 17138 | function | `def trace_auto_carry_bucket_world(q0, q1, mode)` | Handles planned or executed trace behavior for trace auto carry bucket world. |
| 17158 | function | `def planned_bucket_segment_points(q_start, q_goal, mode='auto', samples=None)` | Handles planned bucket segment points behavior. |
| 17187 | function | `def cache_dig_plan_trace_points(seq=None, start_q=None)` | Handles dig plan behavior for cache dig plan trace points. |
| 17191 | function | `def cache_active_stage_trace_points(stage_name, q_start, q_goal, stage_index=None, include_remaining=True)` | Handles planned or executed trace behavior for cache active stage trace points. |
| 17202 | function | `def planned_bucket_points_from_dig_plan()` | Handles dig plan behavior for planned bucket points from dig plan. |
| 17206 | function | `def dig_plan_target_matches_current(tol=0.05)` | Handles dig plan behavior for dig plan target matches current. |
| 17219 | function | `def ensure_trace_dig_plan_current()` | Ensures or creates the required runtime state for trace dig plan current. |
| 17223 | function | `def planned_bucket_points_from_active_motion()` | Handles motion execution behavior for planned bucket points from active motion. |
| 17237 | function | `def planned_bucket_trace_points()` | Handles planned or executed trace behavior for planned bucket trace points. |
| 17251 | function | `def trace_points_signature(trace_points)` | Handles planned or executed trace behavior for trace points signature. |
| 17266 | function | `def draw_trace(force=False)` | Draws trace for planned or executed trace. |
| 17362 | async function | `async def move_to(q_goal, seconds=1.0, mode='auto')` | Moves or executes motion for to. |
| 17376 | function | `def hold_real_state_after_verify_failure(q_real, mode)` | Handles hold real state after verify failure behavior. |
| 17390 | function | `def set_execution_failure_reason(reason)` | Sets or updates execution failure reason. |
| 17397 | function | `def execution_failure_status_text(stage_name)` | Handles execution failure status text behavior. |
| 17402 | function | `def current_dig_plan_contract_status()` | Handles dig plan behavior for current dig plan contract status. |
| 17412 | function | `def block_invalid_dig_plan_contract(task_label='dig_plan')` | Handles dig plan behavior for block invalid dig plan contract. |
| 17444 | function | `def sync_motion_start_q(label='')` | Synchronizes motion start q for motion execution. |
| 17457 | function | `def estimate_stage_motion_seconds(q0, q1, requested_seconds=0.0)` | Handles motion execution behavior for estimate stage motion seconds. |
| 17472 | function | `def motion_reach_report(q_goal)` | Handles motion execution behavior for motion reach report. |
| 17503 | function | `def verify_motion_reached(q_goal, label='', mode='auto', record_failure=True)` | Handles motion execution behavior for verify motion reached. |
| 17517 | async function | `async def wait_for_motion_reached(q_goal, label='', mode='auto', seconds_eff=0.0, record_failure=True)` | Waits for for motion reached for motion execution. |
| 17569 | async function | `async def move_to_profile(q_goal, seconds=1.0, label='', task_id=None, mode='auto', q_start_override=None)` | Moves or executes motion for to profile. |
| 18051 | function | `def active_loaded_route_fast_exec(stage_name=None)` | Handles motion route behavior for active loaded route fast exec. |
| 18083 | function | `def clamp(x, lo, hi)` | Handles clamp behavior. |
| 18087 | function | `def q_deg(swing_rad, boom_deg, arm_deg, bucket_deg)` | Handles q deg behavior. |
| 18096 | function | `def get_joint_anchor_world(joint_name)` | USD Physics joint anchor world position. |
| 18140 | function | `def get_swing_center_world()` | Returns or resolves swing center world. |
| 18156 | function | `def get_swing_xy_center()` | Returns or resolves swing xy center. |
| 18161 | function | `def estimate_dynamic_reach_radius()` | bucket tip . |
| 18178 | function | `def transform_local_point_to_world(link_path, local_xyz)` | Handles transform local point to world behavior. |
| 18189 | function | `def bucket_tip_pos()` | Handles bucket tip pos behavior. |
| 18193 | function | `def bucket_load_pos()` | Handles bucket load pos behavior. |
| 18197 | function | `def bucket_pour_pos()` | Handles bucket pour pos behavior. |
| 18201 | function | `def bucket_mid_pos()` | Handles bucket mid pos behavior. |
| 18205 | function | `def is_cutting_phase(mode)` | Returns whether cutting phase. |
| 18210 | function | `def is_curl_phase(mode)` | Returns whether curl phase. |
| 18214 | function | `def strict_path_precheck_phase(mode)` | Handles strict path precheck phase behavior. |
| 18223 | function | `def is_calibrate_phase(mode)` | Returns whether calibrate phase. |
| 18227 | function | `def sand_surface_query_at_xy(x, y)` | Handles sand site or particle sand behavior for sand surface query at xy. |
| 18280 | function | `def sand_surface_z_at_xy(x, y)` | Handles sand site or particle sand behavior for sand surface z at xy. |
| 18285 | function | `def sand_depth_for_point(point)` | Handles sand site or particle sand behavior for sand depth for point. |
| 18300 | function | `def phase_ground_report(mode)` | Handles phase ground report behavior. |
| 18334 | function | `def min_existing(values)` | Handles min existing behavior. |
| 18341 | function | `def predicted_chain_world_z(q, end_effector='tip', reference_q=None)` | Handles predicted chain world z behavior. |
| 18348 | function | `def predicted_chain_world_points(q, end_effector='mid', reference_q=None)` | Handles predicted chain world points behavior. |
| 18400 | function | `def predicted_end_world_point(q, end_effector='mid', reference_q=None)` | Handles predicted end world point behavior. |
| 18407 | function | `def predicted_phase_ground_report(q, mode, reference_q=None)` | Handles predicted phase ground report behavior. |
| 18413 | function | `predicted_phase_ground_report.point_at` `def point_at(points, idx)` | Handles point at behavior. |
| 18418 | function | `predicted_phase_ground_report.z_at` `def z_at(points, idx)` | Handles z at behavior. |
| 18468 | function | `def depth_below_ground(z)` | Handles depth below ground behavior. |
| 18474 | function | `def fmt_optional(x)` | Handles fmt optional behavior. |
| 18478 | function | `def format_ground_report(mode, report, reason)` | Formats ground report. |
| 18519 | function | `def log_phase_ground(prefix, mode)` | Handles log phase ground behavior. |
| 18526 | function | `def log_predicted_phase_ground(prefix, mode, q, reference_q=None)` | Handles log predicted phase ground behavior. |
| 18538 | function | `def phase_ground_ok(mode, report)` | Handles phase ground ok behavior. |
| 18576 | function | `def cut_front_edge_quality(mode, report)` | Handles cut front edge quality behavior. |
| 18625 | function | `def path_phase_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None)` | Handles path phase check behavior. |
| 18656 | function | `def obstacle_path_excluded(path)` | Handles obstacle or collision checks behavior for obstacle path excluded. |
| 18670 | function | `def append_obstacle_bbox(bboxes, seen_paths, path, source='collision_api', collision_required=True)` | Handles obstacle or collision checks behavior for append obstacle bbox. |
| 18721 | function | `def collect_collision_api_obstacles(bboxes, seen_paths)` | Collects collision api obstacles for obstacle or collision checks. |
| 18748 | function | `def rigid_obstacle_bboxes(force=False)` | Handles obstacle or collision checks behavior for rigid obstacle bboxes. |
| 18797 | function | `def clear_rigid_obstacle_cache(reason='')` | Clears rigid obstacle cache for obstacle or collision checks. |
| 18810 | function | `def rigid_obstacle_numpy_cache(obstacles=None)` | Handles obstacle or collision checks behavior for rigid obstacle numpy cache. |
| 18853 | function | `def obstacle_aabb_candidate_indices(pa, pb, obstacle_np, margin_xy=0.0, margin_z=0.0, radius=0.0)` | Handles obstacle or collision checks behavior for obstacle aabb candidate indices. |
| 18879 | function | `def obstacle_aabb_candidate_indices_many(segments, obstacle_np, margin_xy=0.0, margin_z=0.0, radius=0.0)` | Handles obstacle or collision checks behavior for obstacle aabb candidate indices many. |
| 18910 | function | `def store_excavator_runtime_api()` | Handles store excavator runtime api behavior. |
| 18919 | function | `def compact_obstacle_bbox(row)` | Handles obstacle or collision checks behavior for compact obstacle bbox. |
| 18944 | function | `def planning_world_snapshot(force=False, max_obstacles=64)` | Handles planning world snapshot behavior. |
| 18966 | function | `def point_inside_expanded_bbox(point, mn, mx, margin_xy=0.0, margin_z=0.0)` | Handles bounding boxes behavior for point inside expanded bbox. |
| 18979 | function | `def expanded_bbox_arrays(mn, mx, margin_xy=0.0, margin_z=0.0, radius=0.0)` | Handles bounding boxes behavior for expanded bbox arrays. |
| 18992 | function | `def segment_intersects_expanded_bbox(a, b, mn, mx, margin_xy=0.0, margin_z=0.0, radius=0.0)` | Handles bounding boxes behavior for segment intersects expanded bbox. |
| 19018 | function | `def point_in_convex_polygon_xy(point_xy, poly, eps=1e-07)` | Handles point in convex polygon xy behavior. |
| 19034 | function | `def orient2d(a, b, c)` | Handles orient2d behavior. |
| 19041 | function | `def segments_intersect_xy(a0, a1, b0, b1, eps=1e-08)` | Handles segments intersect xy behavior. |
| 19051 | function | `segments_intersect_xy.on_segment` `def on_segment(p, q, r)` | Handles on segment behavior. |
| 19070 | function | `def point_segment_distance_xy(point, a, b)` | Handles point segment distance xy behavior. |
| 19082 | function | `def segment_segment_distance_xy(a0, a1, b0, b1)` | Handles segment segment distance xy behavior. |
| 19093 | function | `def point_in_polygon_with_margin_xy(point_xy, poly, margin_xy=0.0)` | Handles point in polygon with margin xy behavior. |
| 19108 | function | `def segment_intersects_polygon_with_margin_xy(a_xy, b_xy, poly, margin_xy=0.0)` | Handles segment intersects polygon with margin xy behavior. |
| 19128 | function | `def segment_z_overlap_subsegment(a, b, z_min, z_max, margin_z=0.0, radius=0.0)` | Handles segment z overlap subsegment behavior. |
| 19149 | function | `def point_inside_obstacle_proxy(point, obstacle, margin_xy=0.0, margin_z=0.0, radius=0.0)` | Handles obstacle or collision checks behavior for point inside obstacle proxy. |
| 19169 | function | `def segment_intersects_obstacle_proxy(a, b, obstacle, margin_xy=0.0, margin_z=0.0, radius=0.0)` | Handles obstacle or collision checks behavior for segment intersects obstacle proxy. |
| 19203 | function | `def predicted_obstacle_check_points(q, reference_q=None)` | Handles obstacle or collision checks behavior for predicted obstacle check points. |
| 19234 | function | `def predicted_obstacle_check_segments(q, reference_q=None)` | Handles obstacle or collision checks behavior for predicted obstacle check segments. |
| 19270 | function | `def predicted_segment_cache_key(q, reference_q=None)` | Handles predicted segment cache key behavior. |
| 19280 | function | `def predicted_obstacle_check_segments_cached(q, reference_q=None)` | Handles obstacle or collision checks behavior for predicted obstacle check segments cached. |
| 19297 | function | `def format_obstacle_report(mode, report, reason)` | Formats obstacle report for obstacle or collision checks. |
| 19337 | function | `def obstacle_aabb_overlaps_segment(pa, pb, obstacle, margin_xy=0.0, margin_z=0.0, radius=0.0)` | Handles obstacle or collision checks behavior for obstacle aabb overlaps segment. |
| 19359 | function | `def actual_rigid_obstacle_contact_detail()` | Handles obstacle or collision checks behavior for actual rigid obstacle contact detail. |
| 19389 | function | `def is_unload_bin_wall_obstacle(obstacle)` | Returns whether unload bin wall obstacle for unload target or dump sequence. |
| 19394 | function | `def unload_bin_wall_overpass_allowed(mode, obstacle, *points)` | Handles unload target or dump sequence behavior for unload bin wall overpass allowed. |
| 19417 | function | `def path_obstacle_check_cache_key(q_start, q_goal, mode, samples)` | Handles obstacle or collision checks behavior for path obstacle check cache key. |
| 19433 | function | `def path_obstacle_result_cacheable(result)` | Handles obstacle or collision checks behavior for path obstacle result cacheable. |
| 19440 | function | `def path_obstacle_check_debug_profile_data(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None)` | Handles obstacle or collision checks behavior for path obstacle check debug profile data. |
| 19457 | function | `def path_obstacle_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None)` | Handles obstacle or collision checks behavior for path obstacle check. |
| 19488 | function | `path_obstacle_check.finish` `def finish(result)` | Handles finish behavior. |
| 19607 | function | `def path_segment_check_cache_key(q_start, q_goal, mode, samples)` | Handles path segment check cache key behavior. |
| 19623 | function | `def path_segment_result_cacheable(result)` | Handles path segment result cacheable behavior. |
| 19630 | function | `def path_segment_check_debug_profile_data(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None)` | Handles path segment check debug profile data behavior. |
| 19647 | function | `def path_segment_check(q_start, q_goal, mode, samples=PATH_CHECK_SAMPLES, deadline=None)` | Handles path segment check behavior. |
| 19676 | function | `path_segment_check.finish` `def finish(result)` | Handles finish behavior. |
| 19702 | function | `def unload_bin_wall_clearance_required_z(ctx=None)` | Handles unload target or dump sequence behavior for unload bin wall clearance required z. |
| 19712 | function | `def unload_goal_pose_collision_report(q_goal, reference_q=None, mode='unload_to_bin')` | Handles unload target or dump sequence behavior for unload goal pose collision report. |
| 19772 | function | `def unload_collision_report_from_path_obstacle(report, mode='unload_to_bin')` | Handles unload target or dump sequence behavior for unload collision report from path obstacle. |
| 19822 | function | `def unload_goal_reachability_report(q_seed, landing_target=None, label='', deadline=None)` | Handles unload target or dump sequence behavior for unload goal reachability report. |
| 19842 | function | `unload_goal_reachability_report.unique_values` `def unique_values(values, ndigits=6)` | Handles unique values behavior. |
| 19985 | function | `def unload_pose_bucket_clearance_report(q_pose, reference_q=None)` | Handles unload target or dump sequence behavior for unload pose bucket clearance report. |
| 20023 | function | `def validate_unload_goal(q_start, q_pre_dump, q_dump=None, dump_info=None, label='unload_goal', deadline=None)` | Validates unload goal for unload target or dump sequence. |
| 20102 | function | `def path_block_report_text(mode, kind, report, reason)` | Handles path block report text behavior. |
| 20108 | function | `def obstacle_top_z_for_segment(p_start, p_goal)` | Handles obstacle or collision checks behavior for obstacle top z for segment. |
| 20139 | function | `def obstacle_bboxes_for_segment_xy(p_start, p_goal)` | Handles obstacle or collision checks behavior for obstacle bboxes for segment xy. |
| 20173 | function | `def obstacle_corridor_report(p_start, p_goal, obstacle=None, link_name='planned_end')` | Handles obstacle or collision checks behavior for obstacle corridor report. |
| 20196 | function | `def wrap_angle(x)` | Handles wrap angle behavior. |
| 20201 | function | `def normalize_swing_cmd(angle)` | Normalizes swing cmd. |
| 20205 | function | `def swing_delta(target, current)` | Handles swing delta behavior. |
| 20209 | function | `def swing_target_near(target, current)` | Handles swing target near behavior. |
| 20214 | function | `def set_robot_joint_state(q, label='state')` | Sets or updates robot joint state for joint state or command. |
| 20256 | function | `def set_joint_pose_direct(q_goal, label='direct_pose', mode='direct_pose', update_ui=True)` | Sets or updates joint pose direct for joint state or command. |
| 20289 | async function | `async def set_joint_pose_direct_and_settle(q_goal, label='direct_pose', mode='direct_pose', settle_frames=0, task_id=None, record_failure=True)` | Sets or updates joint pose direct and settle for joint state or command. |
| 20324 | function | `def auto_collect_home_reach_detail(q_home)` | Handles automatic dataset collection behavior for auto collect home reach detail. |
| 20368 | function | `def accept_auto_collect_home_recovery(q_real, detail=None)` | Handles automatic dataset collection behavior for accept auto collect home recovery. |
| 20404 | async function | `async def auto_collect_home_direct_with_recovery(q_home, task_id=None)` | Handles automatic dataset collection behavior for auto collect home direct with recovery. |
| 20458 | function | `def maybe_prepare_swing_rebase_for_segment(q_goal, label='segment')` | Handles maybe prepare swing rebase for segment behavior. |
| 20510 | function | `def swing_edge_pose_before_rebase(q_goal, label='segment')` | Handles swing edge pose before rebase behavior. |
| 20545 | function | `def maybe_rebase_swing_for_bounded_joint(q_target)` | Handles joint state or command behavior for maybe rebase swing for bounded joint. |
| 20593 | function | `def clip_command_near(q, reference=None)` | Clips or constrains command near. |
| 20603 | function | `def is_unload_dump_motion(mode='', label='')` | Returns whether unload dump motion for unload target or dump sequence. |
| 20608 | function | `def bucket_dump_branch_target(target_rad, current_rad=None)` | Pick a bucket joint branch that actually opens the bucket for dump. |
| 20633 | function | `def clip_unload_dump_command(q, reference=None)` | Clips or constrains unload dump command for unload target or dump sequence. |
| 20647 | function | `def planner_effective_joint_bounds_rad(name)` | Handles joint state or command behavior for planner effective joint bounds rad. |
| 20663 | function | `def clip_route_command_near(q, reference=None)` | Clips or constrains route command near for motion route. |
| 20674 | function | `def interpolate_q_shortest(q0, q1, s)` | Handles interpolate q shortest behavior. |
| 20684 | function | `def interpolate_q_motion(q0, q1, s, mode='', label='')` | Handles motion execution behavior for interpolate q motion. |
| 20696 | function | `def rotate_xy(v, angle)` | Handles rotate xy behavior. |
| 20705 | function | `def safe_norm(v, default=None)` | Handles safe norm behavior. |
| 20714 | function | `def point_to_2d(point, root, radial_xy)` | Handles point to 2d behavior. |
| 20721 | function | `def ik_end_effector_pos(end_effector='mid')` | Handles inverse kinematics behavior for ik end effector pos. |
| 20731 | function | `def get_ik_world_points(end_effector='mid')` | Returns or resolves ik world points for inverse kinematics. |
| 20741 | function | `def chain_angles_2d(points_2d)` | Handles chain angles 2d behavior. |
| 20749 | function | `def current_planar_chain(end_effector='mid')` | Handles current planar chain behavior. |
| 20784 | function | `def build_ik_offsets(q, chain, signs)` | Builds ik offsets for inverse kinematics. |
| 20794 | function | `def ik_model_part(model=None, end_effector='mid')` | Handles inverse kinematics behavior for ik model part. |
| 20807 | function | `def chain_angles_from_q(q, model=None, end_effector='mid')` | Handles chain angles from q behavior. |
| 20821 | function | `def bucket_joint_for_world_angle(q, world_angle_rad, model=None, end_effector='load')` | Handles joint state or command behavior for bucket joint for world angle. |
| 20845 | function | `def nearest_bucket_level_world_angle(reference_rad)` | Handles nearest bucket level world angle behavior. |
| 20851 | function | `def bucket_carry_world_angle_candidates(reference_rad, tilt_deg=None)` | Candidate bucket world angles for carrying material. |
| 20878 | function | `def bucket_mouth_raise_for_world_angle(q_reference, world_angle_rad, end_effector='load')` | Handles bucket mouth raise for world angle behavior. |
| 20906 | function | `def nearest_bucket_carry_world_angle(reference_rad, q_reference=None, end_effector='load', tilt_deg=None)` | Handles nearest bucket carry world angle behavior. |
| 20963 | function | `def carry_hold_adjusted_q(q_pose, q_reference=None, end_effector='load', max_bucket_adjust_deg=None)` | Handles carry hold adjusted q behavior. |
| 21108 | function | `def carry_report_pour_above_load(carry_report)` | Handles carry report pour above load behavior. |
| 21120 | function | `def carry_report_allows_transitional_load(carry_report)` | Handles carry report allows transitional load behavior. |
| 21131 | function | `def loaded_transitional_hold_allowed(carry_report, loaded_count=0)` | Handles loaded transitional hold allowed behavior. |
| 21150 | function | `def real_loaded_secure_hold_allowed(carry_report, loaded_count=0)` | Handles real loaded secure hold allowed behavior. |
| 21158 | function | `def carry_spill_risk_penalty(carry_report)` | Handles carry spill risk penalty behavior. |
| 21170 | function | `def apply_loaded_bucket_closed_limit(q, label='')` | Applies loaded bucket closed limit. |
| 21198 | function | `def planar_points_from_angles(angles, lengths)` | Handles planar points from angles behavior. |
| 21208 | function | `def planar_points_from_q(q, lengths, model=None, end_effector='mid')` | Handles planar points from q behavior. |
| 21215 | function | `def two_link_ik_2d(target, l1, l2, seed_angles)` | Handles inverse kinematics behavior for two link ik 2d. |
| 21245 | function | `def q_from_chain_angles(swing_goal, angles, model=None, end_effector='mid')` | Handles q from chain angles behavior. |
| 21260 | function | `def predicted_planar_error(q, target_2d, lengths, model=None, end_effector='mid')` | Handles predicted planar error behavior. |
| 21267 | function | `def refine_planar_ik_candidate(q_start, target_2d, lengths, end_effector='mid')` | Handles inverse kinematics behavior for refine planar ik candidate. |
| 21317 | function | `def predicted_min_world_z(q, lengths, boom_root_z, model=None, end_effector='mid', include_end=True)` | Handles predicted min world z behavior. |
| 21325 | function | `def predicted_end_world_z(q, lengths, boom_root_z, model=None, end_effector='mid')` | Handles predicted end world z behavior. |
| 21332 | function | `def default_ik_model()` | Handles inverse kinematics behavior for default ik model. |
| 21358 | function | `def ik_model_is_valid(model=None)` | Handles inverse kinematics behavior for ik model is valid. |
| 21383 | function | `def validate_ik_model_against_current_pose(model, q_reference)` | Validates ik model against current pose for inverse kinematics. |
| 21411 | function | `def solve_priority_ik_to_target(target_world, q_seed=None, preferred_bucket_rad=None, preferred_end_angle_rad=None, bucket_motion_weight=None, bucket_preference_...)` | Solves priority ik to target for inverse kinematics. |
| 21537 | function | `solve_priority_ik_to_target.reject` `def reject(reason)` | Handles reject behavior. |
| 21656 | function | `solve_priority_ik_to_target.info_from_row` `def info_from_row(row)` | Handles info from row behavior. |
| 21696 | function | `def target_radius_from_swing_center(target_xyz)` | Handles target radius from swing center behavior. |
| 21702 | function | `def validate_dig_target(target_xyz, hard_block=False)` | 5.5m . |
| 21716 | function | `def target_to_swing_angle(target_xyz)` | swing joint center . |
| 21730 | function | `def dig_direction_unit(target_xyz)` | swing center. |
| 21746 | function | `def offset_xy(point, direction_xy, amount, z=None)` | Handles offset xy behavior. |
| 21755 | function | `def adaptive_dig_plan_candidates(target_xyz)` | Handles dig plan behavior for adaptive dig plan candidates. |
| 21759 | function | `def solve_dig_pose(label, point, bucket_deg, duration, q_seed, bucket_world_deg=None, ik_effector='tip', accept_err=0.38, bucket_motion_weight=0.45, bucket_prefe...)` | Solves dig pose for digging. |
| 21862 | function | `def solve_dig_pose_candidates(label, point, bucket_deg, duration, q_seed, bucket_world_deg=None, ik_effector='tip', accept_err=0.38, soft_accept_err=None, bucket...)` | Solves dig pose candidates for digging. |
| 21959 | function | `def path_end_effector_for_mode(mode)` | Handles path end effector for mode behavior. |
| 21970 | function | `def solve_clearance_pose(point, q_seed, end_effector, clearance_z, deadline=None)` | Solves clearance pose. |
| 21994 | function | `def route_segments_ok(q_start, route, q_goal, mode, samples=None, deadline=None)` | Handles motion route behavior for route segments ok. |
| 22024 | function | `def clearance_route_cost(q_start, route, q_goal, duration=0.0, clearance_z=0.0, side_offset=0.0)` | Handles motion route behavior for clearance route cost. |
| 22039 | function | `def q_with_joint_degrees(reference_q, joint_degrees, swing_value=None)` | Handles joint state or command behavior for q with joint degrees. |
| 22050 | function | `def q_with_swing_near(reference_q, target_swing)` | Handles q with swing near behavior. |
| 22057 | function | `def q_goal_raised_approach(q_goal, q_reference, lift_deg, arm_delta_deg, bucket_blend=0.55)` | Handles q goal raised approach behavior. |
| 22071 | function | `def swing_corridor_cache_key(q_start, q_goal, mode, samples)` | Handles swing corridor cache key behavior. |
| 22087 | function | `def swing_corridor_summary(q_start, q_goal, mode, samples=25, deadline=None)` | Handles swing corridor summary behavior. |
| 22172 | function | `def find_clearance_route(q_start, q_goal, mode, label, deadline=None, samples=None)` | Finds clearance route for motion route. |
| 22228 | function | `find_clearance_route.budget_expired` `def budget_expired()` | Handles budget expired behavior. |
| 22231 | function | `find_clearance_route.swing_corridor_distance_deg` `def swing_corridor_distance_deg(q_pose)` | Handles swing corridor distance deg behavior. |
| 22251 | function | `find_clearance_route.add_route` `def add_route(route, route_type, clearance_z, detail='', side_offset=0.0)` | Handles motion route behavior for add route. |
| 22291 | function | `find_clearance_route.choose_best_candidate` `def choose_best_candidate()` | Chooses best candidate. |
| 22296 | function | `find_clearance_route.try_joint_rrt_route` `def try_joint_rrt_route()` | Handles joint state or command behavior for try joint rrt route. |
| 22359 | function | `find_clearance_route.add_deterministic_joint_routes` `def add_deterministic_joint_routes()` | Handles joint state or command behavior for add deterministic joint routes. |
| 22487 | function | `find_clearance_route.info_planar_err` `def info_planar_err(info)` | Handles info planar err behavior. |
| 22495 | function | `find_clearance_route.solve_clearance_waypoints` `def solve_clearance_waypoints(points, q_seed, clearance_z)` | Solves clearance waypoints. |
| 22510 | function | `find_clearance_route.obstacle_corner_route_points` `def obstacle_corner_route_points(obstacle, clearance_z)` | Handles obstacle or collision checks behavior for obstacle corner route points. |
| 22710 | function | `def pre_dig_needs_swing_align(q_goal, label, mode)` | Handles digging behavior for pre dig needs swing align. |
| 22744 | async function | `async def prepare_pre_dig_swing_align(q_goal, label='', task_id=None, mode='pre_dig')` | Handles digging behavior for prepare pre dig swing align. |
| 22807 | async function | `async def move_to_profile_with_clearance(q_goal, seconds=1.0, label='', task_id=None, mode='auto', q_start_override=None)` | Moves or executes motion for to profile with clearance. |
| 22910 | function | `def dig_plan_specs_from_candidate(target_xyz, candidate)` | Handles dig plan behavior for dig plan specs from candidate. |
| 22932 | function | `dig_plan_specs_from_candidate.cut_z` `def cut_z(depth, min_clearance=0.035)` | Handles cut z behavior. |
| 23023 | function | `def plan_dig_sequence_candidate(target_xyz, candidate, deadline=None)` | Plans dig sequence candidate for digging. |
| 23039 | function | `plan_dig_sequence_candidate.budget_failure` `def budget_failure(label, partial=None, reasons=None)` | Handles budget failure behavior. |
| 23065 | function | `plan_dig_sequence_candidate.resolve_bucket_world` `def resolve_bucket_world(label, bucket_deg, bucket_world_deg, q_seed, ik_effector)` | Resolves bucket world. |
| 23079 | function | `plan_dig_sequence_candidate.routed_stage_components` `def routed_stage_components(q_seed, q_goal, label, duration, target_point, effector)` | Handles motion route behavior for routed stage components. |
| 23585 | function | `plan_dig_sequence_candidate.add_curl_candidate` `def add_curl_candidate(q_raw, source, reason='', extra_lift_deg=0.0, pose_info=None)` | Handles add curl candidate behavior. |
| 24268 | function | `def clone_candidate_plan_result(seq, points, detail)` | Handles clone candidate plan result behavior. |
| 24285 | function | `def planning_sand_snapshot_cache_key()` | Handles sand site or particle sand behavior for planning sand snapshot cache key. |
| 24315 | function | `def dig_candidate_result_cache_key(target_xyz, candidate)` | Handles digging behavior for dig candidate result cache key. |
| 24339 | function | `def dig_candidate_result_cache_get(target_xyz, candidate)` | Handles digging behavior for dig candidate result cache get. |
| 24355 | function | `def dig_candidate_result_cache_put(target_xyz, candidate, seq, points, detail)` | Handles digging behavior for dig candidate result cache put. |
| 24374 | function | `def evaluate_dig_plan_candidate(seq, points, candidate, stages=None)` | Handles dig plan behavior for evaluate dig plan candidate. |
| 24478 | function | `def build_shared_dig_plan_object(target_xyz, seq, points, chosen_row)` | Builds shared dig plan object. |
| 24565 | function | `def _planned_stage_q(row, key)` | Handles planned stage q behavior. |
| 24580 | function | `def planned_unload_stage_detail(stage_index=None, stage_name=None)` | Handles unload target or dump sequence behavior for planned unload stage detail. |
| 24625 | function | `def dig_plan_staged_prefix_requirements()` | Handles dig plan behavior for dig plan staged prefix requirements. |
| 24634 | function | `def seq_points_from_plan_stages(stages, fallback_point=None)` | Handles seq points from plan stages behavior. |
| 24667 | function | `def staged_prefix_stages_from_failure(failure_row)` | Handles staged prefix stages from failure behavior. |
| 24698 | function | `def install_staged_prefix_plan_from_failure(target_xyz, failure_row)` | Handles install staged prefix plan from failure behavior. |
| 24761 | function | `def make_stage_row_from_q(phase, q_goal, q_from, duration, target_point=None, extra=None, deadline=None)` | Creates stage row from q. |
| 24807 | function | `def bucket_is_dump_branch_for_carry(bucket_deg)` | Handles bucket is dump branch for carry behavior. |
| 24814 | function | `def bucket_joint_in_loaded_carry_state(bucket_deg)` | Handles joint state or command behavior for bucket joint in loaded carry state. |
| 24821 | function | `def set_bucket_loaded_carry_joint(q_pose, reference=None)` | Sets or updates bucket loaded carry joint for joint state or command. |
| 24828 | function | `def force_loaded_carry_bucket_q(q_pose, reference=None, label='')` | Project a pose to a material-carrying bucket orientation. |
| 24855 | function | `def mode_requires_loaded_carry_bucket(mode, label='')` | Handles Omni UI controls behavior for mode requires loaded carry bucket. |
| 24865 | function | `def phase_metric_sand_counts(name)` | Handles sand site or particle sand behavior for phase metric sand counts. |
| 24878 | function | `def secure_material_baseline_sand()` | Handles sand site or particle sand behavior for secure material baseline sand. |
| 24888 | function | `def carry_material_report_for_q(q_pose, end_effector='load')` | Handles carry material report for q behavior. |
| 24938 | function | `def secure_phase_delta_report(current_metrics=None)` | Handles secure phase delta report behavior. |
| 25001 | function | `def secure_post_gate_report(q_start, current_metrics=None)` | Handles secure post gate report behavior. |
| 25042 | function | `def post_lift_material_gate_report(current_metrics=None, q_pose=None)` | Handles post lift material gate report behavior. |
| 25121 | function | `def staged_carry_safe_projection_candidates(q_start)` | Handles staged carry safe projection candidates behavior. |
| 25227 | function | `def staged_lift_candidates(q_start)` | Handles staged lift candidates behavior. |
| 25324 | function | `def staged_high_carry_unload_fallback(q_lift, q_pre_dump, deadline=None)` | Handles unload target or dump sequence behavior for staged high carry unload fallback. |
| 25433 | function | `def staged_dig_secure_candidates(q_start, loaded_count_hint=None)` | Handles digging behavior for staged dig secure candidates. |
| 25469 | function | `staged_dig_secure_candidates.add_secure_candidate` `def add_secure_candidate(q_seed_base, source, boom_lift_deg=0.0, arm_retract_deg=0.0)` | Handles add secure candidate behavior. |
| 25595 | function | `staged_dig_secure_candidates.add_secure_candidate.progressive_secure_specs` `def progressive_secure_specs(q_goal)` | Handles progressive secure specs behavior. |
| 25929 | function | `def append_staged_post_dig_secure_plan(task_label='dig_target_ball')` | Handles digging behavior for append staged post dig secure plan. |
| 26029 | function | `def append_staged_post_secure_load_plan(task_label='dig_target_ball')` | Handles append staged post secure load plan behavior. |
| 26365 | function | `append_staged_post_secure_load_plan.compute_pre_dump_carry_pose` `def compute_pre_dump_carry_pose(q_dump_pose, q_reference, compute_label)` | Computes pre dump carry pose. |
| 26394 | function | `append_staged_post_secure_load_plan.exec_clearance_for` `def exec_clearance_for(q_pose)` | Handles exec clearance for behavior. |
| 26397 | function | `append_staged_post_secure_load_plan.needs_higher_exec_pose` `def needs_higher_exec_pose(clearance)` | Handles needs higher exec pose behavior. |
| 26817 | function | `def should_append_staged_post_secure_load_after_stage(stage_name)` | Handles should append staged post secure load after stage behavior. |
| 26824 | function | `def dig_plan_candidate_early_accept_ok(row, evaluated_count)` | Handles dig plan behavior for dig plan candidate early accept ok. |
| 26840 | function | `def dig_plan_staged_prefix_early_accept_ok(row, evaluated_count)` | Handles dig plan behavior for dig plan staged prefix early accept ok. |
| 26855 | function | `def plan_dig_sequence_from_target(target_xyz, max_seconds=None)` | Plans dig sequence from target for digging. |
| 27063 | function | `def build_dig_plan_from_current_target(force_status=True, max_seconds=None)` | Builds dig plan from current target. |
| 27170 | async function | `async def build_dig_plan_from_current_target_task(force_status=True)` | Builds dig plan from current target task. |
| 27198 | function | `def reset_dig_plan()` | Resets dig plan. |
| 27224 | function | `def get_or_build_dig_plan()` | Returns or resolves or build dig plan. |
| 27231 | function | `def install_loaded_unload_route_test_plan_from_current()` | Handles unload target or dump sequence behavior for install loaded unload route test plan from current. |
| 27355 | async function | `async def execute_loaded_unload_route_test_from_current()` | Executes loaded unload route test from current for unload target or dump sequence. |
| 27391 | function | `def plan_unload_from_current()` | Plans unload from current for unload target or dump sequence. |
| 27427 | function | `def unload_dump_target_deg()` | Handles unload target or dump sequence behavior for unload dump target deg. |
| 27438 | function | `def unload_release_alignment_bucket_deg(dump_deg=None)` | Handles unload target or dump sequence behavior for unload release alignment bucket deg. |
| 27448 | function | `def plan_dump_pose_to_bin(q_seed=None, dump_deg=None, label='unload_dump', log=True, max_correction_iters=None, allow_unaligned=False, deadline=None, goal_obstac...)` | Plans dump pose to bin. |
| 27991 | async function | `async def execute_unload_sequence(stage_name, q_goal, duration, task_id=None)` | Executes unload sequence for unload target or dump sequence. |
| 28166 | function | `def bucket_only_dump_pose(q_reference, dump_deg)` | Handles bucket only dump pose behavior. |
| 28172 | function | `def non_bucket_delta_deg(q_a, q_b)` | Handles non bucket delta deg behavior. |
| 28182 | function | `def current_real_q_near(reference_q=None)` | Handles current real q near behavior. |
| 28187 | function | `def record_unload_trajectory_sample(stage_name, label, q_cmd=None, force=True)` | Capture camera/state rows during direct unload motions that bypass move_to_profile. |
| 28204 | function | `def bucket_only_dump_ready(stage_name, dump_deg, label='before_dump')` | Handles bucket only dump ready behavior. |
| 28227 | async function | `async def wait_for_dump_settle(stage_name, task_id=None)` | Waits for for dump settle. |
| 28247 | async function | `async def execute_unload_bucket_dump_motion(q_dump, stage_name, task_id=None)` | Executes unload bucket dump motion for unload target or dump sequence. |
| 28369 | async function | `async def dump_bucket_at_target(stage_name, task_id=None, planned_q_dump=None, planned_q_release_align=None)` | Handles dump bucket at target behavior. |
| 28617 | function | `def loaded_route_continuous_group(seq, start_index)` | Handles motion route behavior for loaded route continuous group. |
| 28657 | function | `def loaded_route_group_segment_seconds(q0, q1, requested_seconds, stage_name)` | Handles motion route behavior for loaded route group segment seconds. |
| 28674 | function | `def loaded_route_adaptive_scale(q_cmd)` | Handles motion route behavior for loaded route adaptive scale. |
| 28699 | function | `loaded_route_adaptive_scale.ratio` `def ratio(value, soft, hard)` | Handles ratio behavior. |
| 28732 | function | `def loaded_route_corner_scale(profile_t, cumulative)` | Handles motion route behavior for loaded route corner scale. |
| 28751 | function | `def loaded_route_joint_velocity(q0, q1, dt)` | Handles joint state or command behavior for loaded route joint velocity. |
| 28761 | function | `def loaded_route_sample_q_at_path_time(path_time, total_seconds, cumulative)` | Handles motion route behavior for loaded route sample q at path time. |
| 28779 | function | `def loaded_route_accel_limit_scale(q_prev, q_candidate, joint_vel_prev, load_vel_prev, dt)` | Handles motion route behavior for loaded route accel limit scale. |
| 28851 | async function | `async def execute_loaded_route_continuous_group(seq, start_index, task_id=None)` | Executes loaded route continuous group for motion route. |
| 28901 | function | `execute_loaded_route_continuous_group.close_loaded_group_audits` `def close_loaded_group_audits(result, reason_text='')` | Handles close loaded group audits behavior. |
| 29194 | async function | `async def execute_unload_to_bin_from_current()` | Executes unload to bin from current for unload target or dump sequence. |
| 29213 | async function | `async def execute_dig_plan_step(step_index=None)` | Executes dig plan step. |
| 29416 | async function | `async def execute_dig_target_ball(rebuild_plan=True, task_name='dig_target_ball', return_home=True)` | Executes dig target ball for digging. |
| 29676 | async function | `async def calibrate_ik()` | Handles inverse kinematics behavior for calibrate ik. |
| 29684 | async function | `calibrate_ik.fail` `async def fail(reason, data=None)` | Handles fail behavior. |
| 29844 | function | `def follow_step()` | Handles follow step behavior. |
| 29924 | function | `def sync_sliders_from_real_q(force=False)` | Synchronizes sliders from real q. |
| 29952 | function | `def build_ui()` | Builds ui for Omni UI controls. |
| 29958 | function | `build_ui.toggle_follow` `def toggle_follow()` | Toggles follow. |
| 29967 | function | `build_ui.trace_off` `def trace_off()` | Handles planned or executed trace behavior for trace off. |
| 29970 | function | `build_ui.trace_mode_1` `def trace_mode_1()` | Handles planned or executed trace behavior for trace mode 1. |
| 29973 | function | `build_ui.trace_mode_2` `def trace_mode_2()` | Handles planned or executed trace behavior for trace mode 2. |
| 29976 | function | `build_ui.request_calib` `def request_calib()` | Requests calib. |
| 29980 | function | `build_ui.ik_one_step` `def ik_one_step()` | Handles inverse kinematics behavior for ik one step. |
| 29986 | function | `build_ui.home` `def home()` | Handles home behavior. |
| 29995 | function | `build_ui.print_state` `def print_state()` | Prints state. |
| 29999 | function | `build_ui.toggle_render_mode_from_ui` `def toggle_render_mode_from_ui()` | Toggles render mode from ui for Omni UI controls. |
| 30002 | function | `build_ui.toggle_calc_viz_from_ui` `def toggle_calc_viz_from_ui()` | Toggles calc viz from ui for Omni UI controls. |
| 30005 | function | `build_ui.cycle_log_mode_from_ui` `def cycle_log_mode_from_ui()` | Handles Omni UI controls behavior for cycle log mode from ui. |
| 30008 | function | `build_ui.print_log_state_from_ui` `def print_log_state_from_ui()` | Prints log state from ui for Omni UI controls. |
| 30012 | function | `build_ui.auto_collect_count_from_ui` `def auto_collect_count_from_ui()` | Handles automatic dataset collection behavior for auto collect count from ui. |
| 30027 | function | `build_ui.auto_collect_attempts_from_ui` `def auto_collect_attempts_from_ui(count=None)` | Handles automatic dataset collection behavior for auto collect attempts from ui. |
| 30048 | function | `build_ui.start_auto_collect_from_ui` `def start_auto_collect_from_ui()` | Starts auto collect from ui for automatic dataset collection. |
| 30054 | function | `build_ui.auto_scene_bool_from_model` `def auto_scene_bool_from_model(model, default=False)` | Handles auto scene bool from model behavior. |
| 30066 | function | `build_ui.add_auto_scene_checkbox` `def add_auto_scene_checkbox(label, state_key, width=118)` | Handles add auto scene checkbox behavior. |
| 30070 | function | `build_ui.add_auto_scene_checkbox._changed` `def _changed(m, key=state_key, text=label, default=False)` | Handles changed behavior. |
| 30083 | function | `build_ui.open_auto_collect_dir_from_ui` `def open_auto_collect_dir_from_ui()` | Handles automatic dataset collection behavior for open auto collect dir from ui. |
| 30096 | function | `build_ui.stop_all` `def stop_all()` | Stops all. |
| 30110 | function | `build_ui.stop_loop` `def stop_loop()` | Stops loop. |
| 30125 | function | `build_ui.build_dig_plan_button` `def build_dig_plan_button()` | Builds dig plan button. |
| 30128 | function | `build_ui.reset_dig_plan_button` `def reset_dig_plan_button()` | Resets dig plan button. |
| 30131 | function | `build_ui.run_next_dig_step_button` `def run_next_dig_step_button()` | Handles digging behavior for run next dig step button. |
| 30134 | function | `build_ui.run_dig_step_button` `def run_dig_step_button(step_index)` | Handles digging behavior for run dig step button. |
| 30137 | function | `build_ui.run_loaded_unload_route_test_button` `def run_loaded_unload_route_test_button()` | Handles unload target or dump sequence behavior for run loaded unload route test button. |
| 30140 | function | `build_ui.use_selected_unload_mesh_from_ui` `def use_selected_unload_mesh_from_ui()` | Handles unload target or dump sequence behavior for use selected unload mesh from ui. |
| 30143 | function | `build_ui.use_selected_unload_point_from_ui` `def use_selected_unload_point_from_ui()` | Handles unload target or dump sequence behavior for use selected unload point from ui. |
| 30146 | function | `build_ui.clear_unload_override_from_ui` `def clear_unload_override_from_ui()` | Clears unload override from ui for unload target or dump sequence. |
| 30167 | function | `build_ui.on_manual_joint_slider_changed` `def on_manual_joint_slider_changed(model=None)` | Handles joint state or command behavior for on manual joint slider changed. |
| 30251 | function | `build_ui.update_speed_multiplier` `def update_speed_multiplier(model=None)` | Updates speed multiplier. |
| 30391 | async function | `async def main()` | Entry point or long-running loop for this script/module. |

### `scripts/excavator_app/ik_calculation.py`

- Module role: IK and path metric helpers used by the excavator planner.
- Primary imports: `math`, `numpy`

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 5 | function | `def joint_motion_metrics(rt, q_to, q_from, duration=0.0)` | Handles joint state or command behavior for joint motion metrics. |
| 25 | function | `def path_penalty(rt, q_start, q_goal, mode, deadline=None)` | Handles path penalty behavior. |
| 77 | function | `def adaptive_dig_plan_candidates(rt, target_xyz)` | Handles dig plan behavior for adaptive dig plan candidates. |

### `scripts/excavator_app/ik_movement.py`

- Module role: Async motion execution helpers for planned dig and unload stages.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 1 | async function | `async def move_planned_stage(rt, stage_name, q_goal, duration, task_id=None)` | Moves or executes motion for planned stage. |
| 133 | async function | `async def move_unload_stage(rt, stage_name, q_goal, duration, task_id=None)` | Moves or executes motion for unload stage for unload target or dump sequence. |

### `scripts/excavator_app/joint_space_planner.py`

- Module role: Joint-space route planner with sampling, connection, collision checks, shortcutting, and smoothing.
- Primary imports: `math`, `time`, `numpy`

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 7 | function | `def _as_q(q)` | Handles as q behavior. |
| 11 | function | `def _delta(rt, q_to, q_from)` | Handles delta behavior. |
| 20 | function | `def _distance(rt, q_a, q_b, weights)` | Handles distance behavior. |
| 25 | function | `def _nearest(rt, tree, q, weights)` | Handles nearest behavior. |
| 36 | function | `def _path(tree, idx)` | Handles path behavior. |
| 46 | function | `def _clip_near(rt, q, reference)` | Clips or constrains near. |
| 59 | function | `def _carry_world_angle(rt, q_start, q_goal, mode)` | Handles carry world angle behavior. |
| 74 | function | `def _project_carry_bucket(rt, q, reference, carry_world_rad, mode=None)` | Handles project carry bucket behavior. |
| 100 | function | `def _steer(rt, q_from, q_to, max_step, carry_world_rad=None, mode=None)` | Handles steer behavior. |
| 111 | function | `def _edge_ok(rt, q_from, q_to, mode, samples, carry_world_rad=None, deadline=None)` | Handles edge ok behavior. |
| 143 | function | `def _joint_bounds(rt, q_start, q_goal)` | Handles joint state or command behavior for joint bounds. |
| 168 | function | `def _sample(rt, rng, q_start, q_goal, bounds, goal_bias, carry_world_rad=None, mode=None)` | Handles sample behavior. |
| 181 | function | `def _extend(rt, tree, q_target, mode, max_step, samples, weights, carry_world_rad=None, deadline=None)` | Handles extend behavior. |
| 196 | function | `def _connect(rt, tree, q_target, mode, max_step, samples, weights, carry_world_rad=None, deadline=None)` | Handles connect behavior. |
| 223 | function | `def _path_cost(rt, path)` | Handles path cost behavior. |
| 233 | function | `def _local_cost(rt, q_prev, q_mid, q_next, weights, bend_weight)` | Handles local cost behavior. |
| 242 | function | `def _q_deg_list(rt, q)` | Handles q deg list behavior. |
| 249 | function | `def _shortcut(rt, path, mode, samples, deadline, rng, carry_world_rad=None)` | Handles shortcut behavior. |
| 269 | function | `def _elastic_smooth(rt, path, mode, samples, deadline, rng, carry_world_rad=None, weights=None)` | Handles elastic smooth behavior. |
| 336 | function | `def plan_joint_space_route(rt, q_start, q_goal, mode='clearance', label='joint_rrt', deadline=None, samples=None)` | Plans joint space route for joint state or command. |

### `scripts/excavator_app/sand_site_runtime.py`

- Module role: Sand site runtime: PhysX particle sand, sandbox/unload bin geometry, controls, and stable resets.
- Primary imports: `math`, `random`, `time`, `builtins`, `asyncio`, `numpy`, `omni.usd`, `omni.kit.app`, `omni.ui`, `pxr`
- Runtime note: Schedules coroutine(s) with asyncio.ensure_future().
- Runtime note: Builds sand site and UI at import time.

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 271 | function | `def info(*args)` | Handles info behavior. |
| 275 | function | `def get_stage()` | Returns or resolves stage. |
| 279 | function | `def sdf_path(path)` | Handles sdf path behavior. |
| 287 | function | `def get_prim(path)` | Returns or resolves prim. |
| 291 | function | `def get_physx_schema()` | Returns or resolves physx schema for PhysX settings. |
| 306 | function | `def enable_physx_gpu_runtime_settings()` | Handles PhysX settings behavior for enable physx gpu runtime settings. |
| 343 | function | `def root_path()` | Handles root path behavior. |
| 347 | function | `def as_path_string(path_or_prim)` | Handles as path string behavior. |
| 353 | function | `def ui_short_text(text, max_chars=UI_STATUS_MAX_CHARS)` | Handles Omni UI controls behavior for ui short text. |
| 360 | function | `def update_status(text, force=False)` | Updates status. |
| 373 | function | `def clamp_value(value, lo=None, hi=None)` | Handles clamp value behavior. |
| 382 | function | `def lerp_value(a, b, t)` | Handles lerp value behavior. |
| 387 | function | `def sand_mode_label(fidelity=None)` | Handles sand site or particle sand behavior for sand mode label. |
| 396 | function | `def normalize_sand_parameter_mode(mode)` | Normalizes sand parameter mode for sand site or particle sand. |
| 401 | function | `def sand_parameter_mode_label(mode=None)` | Handles sand site or particle sand behavior for sand parameter mode label. |
| 407 | function | `def current_sand_parameter_mode()` | Handles sand site or particle sand behavior for current sand parameter mode. |
| 415 | function | `def clamp_sand_amount(amount=None)` | Handles sand site or particle sand behavior for clamp sand amount. |
| 420 | function | `def sand_height_from_amount(amount=None)` | Handles sand site or particle sand behavior for sand height from amount. |
| 425 | function | `def sand_amount_from_height(height=None)` | Handles sand site or particle sand behavior for sand amount from height. |
| 432 | function | `def sand_amount_density_spacing_scale(amount=None)` | Handles sand site or particle sand behavior for sand amount density spacing scale. |
| 440 | function | `def recompute_derived_scene_params()` | Handles recompute derived scene params behavior. |
| 459 | function | `def derive_stable_particle_params(layer_spacing_z=None)` | Handles PhysX particle sand behavior for derive stable particle params. |
| 507 | function | `def estimate_particle_count_from_config()` | Handles PhysX particle sand behavior for estimate particle count from config. |
| 525 | function | `def particle_performance_target_count()` | Handles PhysX particle sand behavior for particle performance target count. |
| 535 | function | `def apply_particle_performance_budget(announce=False)` | Applies particle performance budget for PhysX particle sand. |
| 606 | function | `def update_particle_runtime_state()` | Updates particle runtime state for PhysX particle sand. |
| 631 | function | `def apply_sand_fidelity_to_particle_globals(fidelity=None, announce=False)` | Applies sand fidelity to particle globals for sand site or particle sand. |
| 687 | function | `def apply_sand_parameter_mode(mode=None, announce=True)` | Applies sand parameter mode for sand site or particle sand. |
| 784 | function | `def set_sand_fidelity(fidelity, rebuild=False)` | Sets or updates sand fidelity for sand site or particle sand. |
| 807 | function | `def apply_parameter_models_to_globals()` | Applies parameter models to globals. |
| 829 | function | `apply_parameter_models_to_globals.model_value` `def model_value(key, current, lo=None, hi=None)` | Handles model value behavior. |
| 949 | function | `def simulation_timeline_is_playing()` | Handles simulation timeline is playing behavior. |
| 961 | function | `def refresh_parameter_models_from_globals()` | Handles refresh parameter models from globals behavior. |
| 1029 | function | `def set_particle_size_and_limit(radius=None, max_count=None, spacing_xy=None, spacing_z=None, rebuild=False)` | Sets or updates particle size and limit for PhysX particle sand. |
| 1070 | function | `def sand_set_xform(prim, translate=None, scale=None, rotate_xyz=None)` | Handles sand site or particle sand behavior for sand set xform. |
| 1081 | function | `def sand_set_color(prim, rgb, opacity=None)` | Handles sand site or particle sand behavior for sand set color. |
| 1088 | function | `def make_imageable_visible(prim)` | Creates imageable visible. |
| 1097 | function | `def world_bbox_min_max_for_prim(prim)` | Handles bounding boxes behavior for world bbox min max for prim. |
| 1120 | function | `def mesh_world_xy_points_under(prim)` | Handles mesh world xy points under behavior. |
| 1143 | function | `def mesh_world_projected_faces_under(prim, max_faces=32)` | Handles mesh world projected faces under behavior. |
| 1195 | function | `def convex_hull_xy(points)` | Handles convex hull xy behavior. |
| 1204 | function | `convex_hull_xy.cross` `def cross(o, a, b)` | Handles cross behavior. |
| 1221 | function | `def polygon_signed_area(poly)` | Handles polygon signed area behavior. |
| 1230 | function | `def polygon_centroid_xy(poly)` | Handles polygon centroid xy behavior. |
| 1247 | function | `def shrink_convex_polygon_xy(poly, shrink_d)` | Handles shrink convex polygon xy behavior. |
| 1257 | function | `shrink_convex_polygon_xy.inside` `def inside(pt, a, b)` | Handles inside behavior. |
| 1262 | function | `shrink_convex_polygon_xy.intersect` `def intersect(s, ept, a, b)` | Handles intersect behavior. |
| 1298 | function | `def point_in_polygon_xy(x, y, poly)` | Handles point in polygon xy behavior. |
| 1315 | function | `def points_in_polygon_xy_mask(x_values, y_values, poly)` | Handles points in polygon xy mask behavior. |
| 1338 | function | `def visual_polygon_xy(poly, max_vertices)` | Handles visual polygon xy behavior. |
| 1365 | function | `def selected_mesh_footprint_polygon_xy(hull_xy, shrink_d)` | Handles selected mesh footprint polygon xy behavior. |
| 1379 | function | `def shrink_face_polygons_xy(polygons, shrink_d)` | Handles shrink face polygons xy behavior. |
| 1392 | function | `def ellipse_polygon_xy(cx, cy, rx, ry, segments=32)` | Handles ellipse polygon xy behavior. |
| 1403 | function | `def current_sand_footprint_polygons_xy()` | Handles sand site or particle sand behavior for current sand footprint polygons xy. |
| 1407 | function | `def current_sand_footprint_polygon_xy()` | Handles sand site or particle sand behavior for current sand footprint polygon xy. |
| 1419 | function | `def current_sand_footprint_bbox_xy(padding=0.0)` | Handles sand site or particle sand behavior for current sand footprint bbox xy. |
| 1428 | function | `def polygon_x_intervals_at_y(poly, y)` | Handles polygon x intervals at y behavior. |
| 1453 | function | `def footprint_xy_samples(spacing_xy)` | Handles footprint xy samples behavior. |
| 1494 | function | `def selected_prim_paths()` | Handles selected prim paths behavior. |
| 1505 | function | `def first_selected_prim_path()` | Handles first selected prim path behavior. |
| 1510 | function | `def refresh_selected_sand_polygon_from_hull()` | Handles sand site or particle sand behavior for refresh selected sand polygon from hull. |
| 1540 | function | `def clear_selected_sand_source_mesh()` | Clears selected sand source mesh for sand site or particle sand. |
| 1567 | function | `def use_mesh_path_as_sand_source(path, source_label='mesh')` | Handles sand site or particle sand behavior for use mesh path as sand source. |
| 1662 | function | `def use_selected_mesh_as_sand_source()` | Handles sand site or particle sand behavior for use selected mesh as sand source. |
| 1666 | function | `def apply_default_sand_source_mesh()` | Applies default sand source mesh for sand site or particle sand. |
| 1670 | function | `def fmt_bbox(mn, mx)` | Handles bounding boxes behavior for fmt bbox. |
| 1676 | function | `def sand_make_cube(path, translate, scale, color, collision=False, opacity=None)` | Handles sand site or particle sand behavior for sand make cube. |
| 1688 | function | `def sand_make_sphere(path, translate, radius, color, collision=False, opacity=None)` | Handles sand site or particle sand behavior for sand make sphere. |
| 1700 | function | `def set_sand_generation_range_box(label='')` | Sets or updates sand generation range box for sand site or particle sand. |
| 1790 | function | `def sand_make_cylinder(path, translate, radius, depth, color, rotate_xyz=(0.0, 90.0, 0.0), collision=False, opacity=None)` | Handles sand site or particle sand behavior for sand make cylinder. |
| 1803 | function | `def sand_make_polygon_prism(path, polygon_xy, z_min, z_max, color, opacity=None)` | Handles sand site or particle sand behavior for sand make polygon prism. |
| 1838 | function | `def sand_x_min()` | Handles sand site or particle sand behavior for sand x min. |
| 1842 | function | `def sand_x_max()` | Handles sand site or particle sand behavior for sand x max. |
| 1846 | function | `def sand_y_min()` | Handles sand site or particle sand behavior for sand y min. |
| 1850 | function | `def sand_y_max()` | Handles sand site or particle sand behavior for sand y max. |
| 1854 | function | `def initial_sand_height_xy(x, y)` | Handles sand site or particle sand behavior for initial sand height xy. |
| 1869 | function | `def in_sand_bounds(x, y)` | Handles sand site or particle sand behavior for in sand bounds. |
| 1873 | function | `def is_inside_diggable_xy(x, y)` | Returns whether inside diggable xy for digging. |
| 1883 | function | `def is_inside_diggable_xy_batch(x_values, y_values)` | Returns whether inside diggable xy batch for digging. |
| 1895 | function | `def initial_sand_height_xy_batch(x_values, y_values)` | Handles sand site or particle sand behavior for initial sand height xy batch. |
| 1925 | function | `def height_from_grid(x, y)` | Handles height from grid behavior. |
| 1944 | function | `def soil_depth_at_point(x, y, z)` | Handles soil depth at point behavior. |
| 1948 | function | `def current_real_particle_positions()` | Handles PhysX particle sand behavior for current real particle positions. |
| 1960 | function | `def current_real_particle_positions_sampled(max_samples=None)` | Handles PhysX particle sand behavior for current real particle positions sampled. |
| 1985 | async function | `async def step_updates(frames=1)` | Handles step updates behavior. |
| 1991 | function | `def sand_reset_health_stats(prev_points=None)` | Handles sand site or particle sand behavior for sand reset health stats. |
| 2051 | async function | `async def wait_for_sand_reset_health(label='sand_reset')` | Waits for for sand reset health for sand site or particle sand. |
| 2087 | async function | `async def reset_sand_surface_stably(label='ui')` | Resets sand surface stably for sand site or particle sand. |
| 2141 | function | `def request_sand_reset(label='ui')` | Requests sand reset for sand site or particle sand. |
| 2153 | function | `def particle_surface_height_at_xy(x, y, radius=None, fallback_reference=True)` | Handles PhysX particle sand behavior for particle surface height at xy. |
| 2166 | function | `def particle_surface_heightmap(res=32, fallback_to_floor=True)` | Handles PhysX particle sand behavior for particle surface heightmap. |
| 2190 | function | `def particle_excavated_volume(res=32)` | Handles PhysX particle sand behavior for particle excavated volume. |
| 2214 | function | `def estimate_particle_vram_gb(count=None)` | Handles PhysX particle sand behavior for estimate particle vram gb. |
| 2230 | function | `def real_sand_stats()` | Handles sand site or particle sand behavior for real sand stats. |
| 2281 | function | `def clamp(x, lo, hi)` | Handles clamp behavior. |
| 2285 | function | `def build_height_arrays()` | Builds height arrays for Omni UI controls. |
| 2296 | function | `def refresh_sand_mesh()` | Handles sand site or particle sand behavior for refresh sand mesh. |
| 2302 | function | `def initialize_sand_reference_grid(root)` | Handles sand site or particle sand behavior for initialize sand reference grid. |
| 2309 | function | `def rebuild_sand_reference_grid(label='')` | Handles sand site or particle sand behavior for rebuild sand reference grid. |
| 2320 | function | `def set_schema_attr(obj, names, value)` | Sets or updates schema attr. |
| 2332 | function | `def set_prim_attr(prim, name, value, type_name=None)` | Sets or updates prim attr. |
| 2353 | function | `def with_root_edit_target(fn)` | Handles with root edit target behavior. |
| 2369 | function | `def set_physx_scene_gpu_attrs(scene_prim)` | Sets or updates physx scene gpu attrs for PhysX settings. |
| 2383 | function | `def apply_api_by_names(prim, names)` | Applies api by names. |
| 2401 | function | `def create_physx_particle_system(root)` | Creates physx particle system for PhysX particle sand. |
| 2445 | function | `def create_sand_particle_material(root)` | Creates sand particle material for sand site or particle sand. |
| 2480 | function | `def apply_current_particle_material_to_stage(root=None)` | Applies current particle material to stage for PhysX particle sand. |
| 2501 | function | `def sand_particle_points_scalar()` | Handles sand site or particle sand behavior for sand particle points scalar. |
| 2555 | function | `def sand_particle_points_vectorized()` | Handles sand site or particle sand behavior for sand particle points vectorized. |
| 2640 | function | `def sand_particle_points()` | Handles sand site or particle sand behavior for sand particle points. |
| 2651 | function | `def print_particle_spacing_diagnostics()` | Prints particle spacing diagnostics for PhysX particle sand. |
| 2675 | function | `def make_real_particle_sand(root)` | Creates real particle sand for sand site or particle sand. |
| 2807 | function | `def clear_path(path)` | Clears path. |
| 2814 | function | `def clear_previous_site(root)` | Clears previous site. |
| 2822 | function | `def ensure_physics_scene()` | Ensures or creates the required runtime state for physics scene. |
| 2877 | function | `def apply_auto_scene_parameters(sand_center_xy=None, sand_amount_multiplier=None, unload_center_xy=None, rebuild=False)` | Applies auto scene parameters. |
| 2957 | function | `def make_sand_retaining_walls(root)` | Creates sand retaining walls for sand site or particle sand. |
| 3017 | function | `def clean_sand_retaining_walls(root=None)` | Handles sand site or particle sand behavior for clean sand retaining walls. |
| 3030 | function | `def unload_bin_dump_point()` | Handles unload target or dump sequence behavior for unload bin dump point. |
| 3037 | function | `def sand_scene_context()` | Handles sand site or particle sand behavior for sand scene context. |
| 3079 | function | `def reset_sand_surface()` | Resets sand surface for sand site or particle sand. |
| 3091 | function | `def clear_real_particle_sand()` | Clears real particle sand for sand site or particle sand. |
| 3103 | function | `def rebuild_real_particle_sand()` | Handles sand site or particle sand behavior for rebuild real particle sand. |
| 3113 | function | `def print_status()` | Prints status. |
| 3209 | function | `def print_physics_scene_diagnostics()` | Prints physics scene diagnostics. |
| 3237 | function | `def print_particle_physics_diagnostics()` | Prints particle physics diagnostics for PhysX particle sand. |
| 3276 | function | `def print_real_sand_particle_diagnostics()` | Prints real sand particle diagnostics for sand site or particle sand. |
| 3308 | function | `def store_runtime_api()` | Handles store runtime api behavior. |
| 3382 | function | `def notify_excavator_obstacle_cache_dirty(reason)` | Handles obstacle or collision checks behavior for notify excavator obstacle cache dirty. |
| 3395 | function | `def build_sand_site()` | Builds sand site for sand site or particle sand. |
| 3426 | function | `def build_ui()` | Builds ui for Omni UI controls. |
| 3429 | function | `build_ui.set_dirty` `def set_dirty(text='Changed, needs Apply')` | Sets or updates dirty. |
| 3437 | function | `build_ui.set_clean` `def set_clean(text='Applied')` | Sets or updates clean. |
| 3445 | function | `build_ui.reset_clicked` `def reset_clicked()` | Resets clicked. |
| 3451 | function | `build_ui.clean_sand_clicked` `def clean_sand_clicked()` | Handles sand site or particle sand behavior for clean sand clicked. |
| 3456 | function | `build_ui.use_point_xyz_sand_clicked` `def use_point_xyz_sand_clicked()` | Handles sand site or particle sand behavior for use point xyz sand clicked. |
| 3462 | function | `build_ui.generate_walls_clicked` `def generate_walls_clicked()` | Handles generate walls clicked behavior. |
| 3470 | function | `build_ui.clean_walls_clicked` `def clean_walls_clicked()` | Handles clean walls clicked behavior. |
| 3476 | function | `build_ui.apply_clicked` `def apply_clicked()` | Applies clicked. |
| 3483 | function | `build_ui.apply_rebuild_clicked` `def apply_rebuild_clicked()` | Applies rebuild clicked for Omni UI controls. |
| 3490 | function | `build_ui.status_clicked` `def status_clicked()` | Handles status clicked behavior. |
| 3496 | function | `build_ui.refresh_sand_param_mode_label` `def refresh_sand_param_mode_label()` | Handles sand site or particle sand behavior for refresh sand param mode label. |
| 3503 | function | `build_ui.toggle_sand_parameter_mode_clicked` `def toggle_sand_parameter_mode_clicked()` | Toggles sand parameter mode clicked for sand site or particle sand. |
| 3510 | function | `build_ui.sync_sand_xyz_from_models_live` `def sync_sand_xyz_from_models_live(key=None)` | Synchronizes sand xyz from models live for sand site or particle sand. |
| 3555 | function | `build_ui.use_selected_sand_mesh_clicked` `def use_selected_sand_mesh_clicked()` | Handles sand site or particle sand behavior for use selected sand mesh clicked. |
| 3564 | function | `build_ui.model_changed` `def model_changed(_model=None, key=None)` | Handles model changed behavior. |
| 3569 | function | `build_ui.add_listener` `def add_listener(model, key, live_fn=None)` | Handles add listener behavior. |
| 3574 | function | `build_ui.add_listener._on_change` `def _on_change(m, k=key, lf=live_fn)` | Handles on change behavior. |
| 3585 | function | `build_ui.add_float_cell` `def add_float_cell(key, label, value, width=82)` | Handles add float cell behavior. |
| 3593 | function | `build_ui.add_param_row` `def add_param_row(items)` | Handles add param row behavior. |
| 3598 | function | `build_ui.section` `def section(title)` | Handles section behavior. |
| 3602 | function | `build_ui.add_sand_amount_row` `def add_sand_amount_row()` | Handles sand site or particle sand behavior for add sand amount row. |
| 3625 | function | `build_ui.add_sand_xyz_sliders` `def add_sand_xyz_sliders()` | Handles sand site or particle sand behavior for add sand xyz sliders. |
| 3649 | function | `build_ui.add_unload_bin_sliders` `def add_unload_bin_sliders()` | Handles unload target or dump sequence behavior for add unload bin sliders. |
| 3675 | function | `build_ui.add_fidelity_row` `def add_fidelity_row()` | Handles add fidelity row behavior. |
| 3696 | function | `build_ui.add_clay_material_controls` `def add_clay_material_controls()` | Handles add clay material controls behavior. |
| 3699 | function | `build_ui.add_clay_material_controls.add_slider` `def add_slider(key, label, value, lo, hi, hint)` | Handles add slider behavior. |

### `scripts/excavator_app/trace_showing.py`

- Module role: Trace cache helpers for planned and active bucket path visualization.
- Primary imports: `numpy`

| Line | Kind | Symbol | Purpose |
| ---: | --- | --- | --- |
| 4 | function | `def _copy_points(points, limit)` | Handles copy points behavior. |
| 8 | function | `def _extend_segment(points, segment)` | Handles extend segment behavior. |
| 16 | function | `def _append_planned_unload_dump(rt, points, breaks, q_cursor, stage_name, stage_index=None, source='planned')` | Handles unload target or dump sequence behavior for append planned unload dump. |
| 33 | function | `def cache_dig_plan_trace_points(rt, seq=None, start_q=None)` | Handles dig plan behavior for cache dig plan trace points. |
| 73 | function | `def cache_active_stage_trace_points(rt, stage_name, q_start, q_goal, stage_index=None, include_remaining=True)` | Cache the execution segment for diagnostics without replacing blue plan trace. |
| 135 | function | `def planned_bucket_points_from_dig_plan(rt)` | Handles dig plan behavior for planned bucket points from dig plan. |

### `scripts/simple_diagnose.py`

- Module role: Minimal import/path diagnostic script for Isaac Sim and USD scene assets.
- Primary imports: `pxr`
- No functions or classes are defined in this file.

