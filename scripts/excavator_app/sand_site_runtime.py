import math
import random
import time
import builtins
import asyncio
import numpy as np

import omni.usd
import omni.kit.app
import omni.ui as ui
try:
    import omni.timeline
    HAS_OMNI_TIMELINE = True
except Exception:
    HAS_OMNI_TIMELINE = False
try:
    import carb
except Exception:
    carb = None

from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf, UsdShade

try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None


# ============================================================
# PhysX particle sand site, with a heightfield reference surface
# ============================================================

SAND_ROOT_NAME = "SandSite"

SANDBOX_INNER_SIZE_X = 2.50
SANDBOX_INNER_SIZE_Y = 2.50
SANDBOX_WALL_THICKNESS = 0.05
SANDBOX_WALL_HEIGHT = 0.60
SANDBOX_FILL_HEIGHT = 3.00
SANDBOX_FLOOR_THICKNESS = 0.16
SANDBOX_WALL_COLOR = (0.34, 0.32, 0.28)
SANDBOX_FLOOR_COLOR = (0.24, 0.23, 0.20)
SAND_AMOUNT_BASE_HEIGHT = 3.00
SAND_AMOUNT_MULTIPLIER = 1.00
SAND_AMOUNT_MIN_MULTIPLIER = 0.25
SAND_AMOUNT_MAX_MULTIPLIER = 10.00
SAND_AMOUNT_HEIGHT_EXPONENT = 0.75

SAND_SIZE_X = SANDBOX_INNER_SIZE_X
SAND_SIZE_Y = SANDBOX_INNER_SIZE_Y
SAND_CENTER_X = 0.0
SAND_CENTER_Y = 6.7
NX = 89
NY = 101

BASE_Z = 0.0
SAND_HEIGHT_MULTIPLIER = 1.0
SAND_RANGE_AREA_FRACTION = 1.0 / 3.0
SAND_RANGE_LINEAR_SCALE = math.sqrt(SAND_RANGE_AREA_FRACTION)
SAND_THICKNESS = 0.36
SAND_FLOOR_Z = BASE_Z
SAND_POINT_Z = SAND_FLOOR_Z
SAND_POINT_RADIUS = min(SANDBOX_INNER_SIZE_X, SANDBOX_INNER_SIZE_Y) * 0.5 - 0.08
MAX_EXCAVATION_DEPTH = SANDBOX_FILL_HEIGHT
GRAVITY_MAGNITUDE = 9.81
GRAVITY_DIRECTION = Gf.Vec3f(0.0, 0.0, -1.0)

WALL_CENTER_X = SAND_CENTER_X
WALL_CENTER_Y = SAND_CENTER_Y
WALL_BASE_Z = BASE_Z
WALL_INNER_SIZE_X = SANDBOX_INNER_SIZE_X
WALL_INNER_SIZE_Y = SANDBOX_INNER_SIZE_Y

PILE_CENTER_X = 0.0
PILE_CENTER_Y = SAND_CENTER_Y
PILE_HEIGHT = SANDBOX_FILL_HEIGHT
PILE_SIGMA_X = 0.56
PILE_SIGMA_Y = 0.56

RIPPLE_AMP = 0.018
NOISE_AMP = 0.010
NOISE_SEED = 27

DIGGABLE_RADIUS_X = SANDBOX_INNER_SIZE_X * 0.5 - 0.08
DIGGABLE_RADIUS_Y = SANDBOX_INNER_SIZE_Y * 0.5 - 0.08
ENABLE_REAL_PARTICLE_SAND = True
AUTO_CREATE_INITIAL_SAND = False
REAL_SAND_PILE_ONLY = True
PARTICLE_DIGGABLE_SPACING = 0.044
PARTICLE_LAYER_SPACING_Z = 0.042
PARTICLE_RADIUS = 0.018
PARTICLE_CONTACT_OFFSET = 0.024
PARTICLE_REST_OFFSET = 0.017
PARTICLE_SOLID_REST_OFFSET = 0.018
PARTICLE_FLUID_REST_OFFSET = 0.0
PARTICLE_MASS = 0.135
PARTICLE_JITTER = 0.004
PARTICLE_MAX_COUNT = 465000
PARTICLE_SOLVER_POSITION_ITERATIONS = 16
PARTICLE_MAX_VELOCITY = 22.0
SAND_FIDELITY = 0.25
SAND_FIDELITY_EFFICIENCY_SPACING = 0.060
SAND_FIDELITY_REALISTIC_SPACING = 0.034
SAND_FIDELITY_EFFICIENCY_RADIUS = 0.022
SAND_FIDELITY_REALISTIC_RADIUS = 0.014
SAND_FIDELITY_EFFICIENCY_MAX_COUNT = 220000
SAND_FIDELITY_REALISTIC_MAX_COUNT = 900000
SAND_FIDELITY_EFFICIENCY_SOLVER_ITERS = 8
SAND_FIDELITY_REALISTIC_SOLVER_ITERS = 20
SAND_FIDELITY_MAX_VELOCITY = 18.0
SAND_FIDELITY_REFERENCE_SPACING = 0.044
SAND_BULK_DENSITY_KG_M3 = 1600.0
SAND_FIDELITY_REFERENCE_MASS = SAND_BULK_DENSITY_KG_M3 * (SAND_FIDELITY_REFERENCE_SPACING ** 3)
SAND_FIDELITY_MIN_MASS = 0.035
SAND_FIDELITY_MAX_MASS = 0.36
PARTICLE_LAYER_SPACING_RATIO = 0.95
PARTICLE_CENTER_SPACING_SAFETY = 2.12
PARTICLE_JITTER_RATIO = 0.08
PARTICLE_JITTER_MAX = 0.004
PARTICLE_MATERIAL_FRICTION = 0.68
PARTICLE_MATERIAL_FRICTION_SCALE = 0.75
PARTICLE_MATERIAL_DAMPING = 0.03
PARTICLE_MATERIAL_VISCOSITY = 0.0
PARTICLE_MATERIAL_COHESION = 0.0
PARTICLE_MATERIAL_ADHESION = 0.0
PARTICLE_MATERIAL_GRAVITY_SCALE = 1.0
PARTICLE_MATERIAL_STATIC_FRICTION = 0.72
PARTICLE_MATERIAL_DYNAMIC_FRICTION = 0.56
PARTICLE_MATERIAL_RESTITUTION = 0.01
STATIC_COLLIDER_CONTACT_OFFSET = 0.030
STATIC_COLLIDER_REST_OFFSET = 0.000
PARTICLE_SYSTEM_PATH_SUFFIX = "ParticleSystem"
PARTICLE_POINTS_PATH_SUFFIX = "RealSandParticles"
PARTICLE_MATERIAL_PATH_SUFFIX = "Materials/SandPBDMaterial"
PHYSX_TIMESTEPS_PER_SECOND = 60
SAND_PERFORMANCE_AUTO_BUDGET = True
SAND_PERFORMANCE_EFFICIENCY_TARGET_PARTICLES = 95000
SAND_PERFORMANCE_REALISTIC_TARGET_PARTICLES = 260000
SAND_PERFORMANCE_MAX_SPACING = 0.115
SAND_PERFORMANCE_MIN_RADIUS_TO_SPACING = 0.30
SAND_PERFORMANCE_BUDGET_MAX_ITERS = 4
SAND_PERFORMANCE_BUDGET_TOLERANCE = 1.08
SAND_PERFORMANCE_MAX_COUNT_HEADROOM = 1.10
SAND_PERFORMANCE_HARD_MAX_PARTICLE_COUNT = 3000000
SAND_RESET_HEALTH_MIN_FRAMES = 45
SAND_RESET_HEALTH_MAX_FRAMES = 260
SAND_RESET_HEALTH_WINDOW_FRAMES = 15
SAND_RESET_MAX_ATTEMPTS = 2
SAND_RESET_Z_ESCAPE_BELOW = 0.75
SAND_RESET_Z_ESCAPE_ABOVE = 3.0
SAND_RESET_XY_ESCAPE_MARGIN = 0.35
SAND_RESET_DISPLACEMENT_P95_BAD = 1.20
SAND_RESET_DISPLACEMENT_MAX_BAD = 6.0
SAND_RESET_HEALTH_SAMPLE_MAX = 12000

UNLOAD_BIN_CENTER = np.array([-10.0, -5.0, 0.0], dtype=np.float32)
UNLOAD_BIN_INNER_SIZE_X = SANDBOX_INNER_SIZE_X
UNLOAD_BIN_INNER_SIZE_Y = SANDBOX_INNER_SIZE_Y
UNLOAD_BIN_WALL_THICKNESS = SANDBOX_WALL_THICKNESS
UNLOAD_BIN_WALL_HEIGHT = SANDBOX_WALL_HEIGHT
UNLOAD_BIN_FLOOR_THICKNESS = SANDBOX_FLOOR_THICKNESS
UNLOAD_BIN_DUMP_HEIGHT = 1.45
UNLOAD_BIN_WALL_COLOR = (0.24, 0.27, 0.30)
UNLOAD_BIN_FLOOR_COLOR = (0.20, 0.22, 0.24)
SAND_SOURCE_SELECTION_EDGE_MARGIN = 0.10
SAND_SOURCE_RANGE_GUIDE_COLOR = (0.05, 0.95, 0.32)
SAND_SOURCE_RANGE_GUIDE_WIDTH = 0.040
SAND_SOURCE_MAX_PROJECTED_FACES = 32
SAND_SOURCE_RANGE_VISUAL_MAX_VERTICES = 128
SAND_POINT_RANGE_VISUAL_SEGMENTS = 32
DEFAULT_SAND_SOURCE_MESH_PATH = "/World/SandSite/SandRetainingWalls"
SAND_SOURCE_SELECTED_PATH = ""
SAND_SOURCE_SELECTED_FACE_COUNT = 0
SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT = 0
SAND_SOURCE_RAW_FACE_POLYGONS_XY = None
SAND_SOURCE_FACE_POLYGONS_XY = None
SAND_SOURCE_SELECTED_HULL_XY = None
SAND_SOURCE_POLYGON_XY = None

if hasattr(builtins, "_SAND_SITE_STATE"):
    try:
        builtins._SAND_SITE_STATE["running"] = False
        old_window = getattr(builtins, "_SAND_SITE_UI_WINDOW", None)
        if old_window is not None:
            try:
                old_window.visible = False
            except Exception:
                pass
    except Exception:
        pass

builtins._SAND_SITE_STATE = {
    "running": True,
    "run_id": time.time(),
    "excavated_volume": 0.0,
    "mesh_dirty": False,
    "real_sand_enabled": False,
    "real_sand_particle_count": 0,
    "real_sand_error": "",
    "particle_mass": PARTICLE_MASS,
    "particle_radius": PARTICLE_RADIUS,
    "particle_max_count": PARTICLE_MAX_COUNT,
    "particle_spacing_xy": PARTICLE_DIGGABLE_SPACING,
    "particle_spacing_z": PARTICLE_LAYER_SPACING_Z,
    "particle_solver_iters": PARTICLE_SOLVER_POSITION_ITERATIONS,
    "particle_max_velocity": PARTICLE_MAX_VELOCITY,
    "sand_fidelity": SAND_FIDELITY,
    "sand_mode_label": "Balanced",
    "estimated_particle_count": 0,
    "needs_reset_after_world_ready": not AUTO_CREATE_INITIAL_SAND,
    "last_status_print_time": 0.0,
    "last_sand_xyz_live_update": 0.0,
    "footprint_sample_cache": {},
    "status": "Sand site script loaded",
}

STATE = builtins._SAND_SITE_STATE

WINDOW = None
STATUS_LABEL = None
DIRTY_LABEL = None
PARAM_MODELS = {}
UI_STATUS_MAX_CHARS = 72
HEIGHTS = None
BASE_HEIGHTS = None
X_VALUES = None
Y_VALUES = None
CELL_AREA = None


def info(*args):
    print("[INFO]", *args)


def get_stage():
    return omni.usd.get_context().get_stage()


def sdf_path(path):
    if isinstance(path, Sdf.Path):
        return path
    if hasattr(path, "GetPath"):
        return path.GetPath()
    return Sdf.Path(str(path))


def get_prim(path):
    return get_stage().GetPrimAtPath(sdf_path(path))


def get_physx_schema():
    global PhysxSchema
    if PhysxSchema is not None:
        return PhysxSchema
    try:
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.physx", True)
        from pxr import PhysxSchema as ImportedPhysxSchema
        PhysxSchema = ImportedPhysxSchema
        info("Enabled omni.physx and imported PhysxSchema")
    except Exception as e:
        info("[WARN] Could not enable/import PhysxSchema:", e)
    return PhysxSchema


def enable_physx_gpu_runtime_settings():
    if carb is None:
        return
    try:
        settings = carb.settings.get_settings()
        for key in [
            "/physics/physx/useGpu",
            "/physics/physx/useGPU",
            "/physics/physx/useGpuDynamics",
            "/physics/physx/enableGPUDynamics",
            "/physics/physx/enableGpuDynamics",
            "/physics/physx/enable_gpu_dynamics",
            "/physics/physx/gpuDynamicsEnabled",
            "/physics/physx/broadphaseType",
            "/physics/physx/enableCCD",
            "/physics/physx/enableCcd",
            "/persistent/physics/physx/useGpu",
            "/persistent/physics/physx/useGPU",
            "/persistent/physics/physx/enableGPUDynamics",
            "/persistent/physics/physx/enableGpuDynamics",
            "/persistent/physics/physx/useGpuDynamics",
            "/persistent/physics/physx/enable_gpu_dynamics",
            "/persistent/physics/physx/gpuDynamicsEnabled",
            "/persistent/physics/physx/broadphaseType",
            "/persistent/physics/physx/enableCCD",
            "/persistent/physics/physx/enableCcd",
        ]:
            if key.endswith("broadphaseType"):
                settings.set_string(key, "GPU")
            elif key.lower().endswith("enableccd"):
                settings.set_bool(key, False)
            else:
                settings.set_bool(key, True)
    except Exception as e:
        info("[WARN] Could not set PhysX GPU runtime settings:", e)


def root_path():
    return f"/World/{SAND_ROOT_NAME}" if get_prim("/World").IsValid() else f"/{SAND_ROOT_NAME}"


def as_path_string(path_or_prim):
    if hasattr(path_or_prim, "GetPath"):
        return str(path_or_prim.GetPath())
    return str(path_or_prim)


def ui_short_text(text, max_chars=UI_STATUS_MAX_CHARS):
    s = str(text).replace("\n", " ")
    if len(s) <= int(max_chars):
        return s
    return s[: max(0, int(max_chars) - 3)] + "..."


def update_status(text, force=False):
    STATE["status"] = str(text)
    if STATUS_LABEL is not None:
        try:
            STATUS_LABEL.text = ui_short_text(text)
        except Exception:
            pass
    now = time.time()
    if force or now - float(STATE.get("last_status_print_time", 0.0)) > 0.45:
        info(text)
        STATE["last_status_print_time"] = now


def clamp_value(value, lo=None, hi=None):
    v = float(value)
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return v


def lerp_value(a, b, t):
    t = clamp_value(t, 0.0, 1.0)
    return float(a) + (float(b) - float(a)) * t


def sand_mode_label(fidelity=None):
    f = SAND_FIDELITY if fidelity is None else float(fidelity)
    if f <= 0.25:
        return "Efficiency"
    if f >= 0.75:
        return "Realistic"
    return "Balanced"


def clamp_sand_amount(amount=None):
    value = SAND_AMOUNT_MULTIPLIER if amount is None else float(amount)
    return clamp_value(value, SAND_AMOUNT_MIN_MULTIPLIER, SAND_AMOUNT_MAX_MULTIPLIER)


def sand_height_from_amount(amount=None):
    amount = clamp_sand_amount(amount)
    return float(SAND_AMOUNT_BASE_HEIGHT) * (float(amount) ** float(SAND_AMOUNT_HEIGHT_EXPONENT))


def sand_amount_from_height(height=None):
    h = max(0.01, float(SANDBOX_FILL_HEIGHT if height is None else height))
    base = max(0.01, float(SAND_AMOUNT_BASE_HEIGHT))
    exponent = max(0.01, float(SAND_AMOUNT_HEIGHT_EXPONENT))
    return clamp_sand_amount((h / base) ** (1.0 / exponent))


def sand_amount_density_spacing_scale(amount=None):
    amount = clamp_sand_amount(amount)
    if amount <= 1.0:
        return 1.0
    exponent = max(0.0, 1.0 - float(SAND_AMOUNT_HEIGHT_EXPONENT))
    return float(amount) ** (-exponent / 3.0)


def recompute_derived_scene_params():
    global SAND_SIZE_X, SAND_SIZE_Y, PILE_HEIGHT, MAX_EXCAVATION_DEPTH
    global DIGGABLE_RADIUS_X, DIGGABLE_RADIUS_Y
    global UNLOAD_BIN_INNER_SIZE_X, UNLOAD_BIN_INNER_SIZE_Y
    global UNLOAD_BIN_WALL_THICKNESS, UNLOAD_BIN_WALL_HEIGHT, UNLOAD_BIN_FLOOR_THICKNESS

    SAND_SIZE_X = float(SANDBOX_INNER_SIZE_X)
    SAND_SIZE_Y = float(SANDBOX_INNER_SIZE_Y)
    PILE_HEIGHT = float(SANDBOX_FILL_HEIGHT)
    MAX_EXCAVATION_DEPTH = float(SANDBOX_FILL_HEIGHT)
    DIGGABLE_RADIUS_X = clamp_value(DIGGABLE_RADIUS_X, 0.05, max(0.05, 0.5 * float(SANDBOX_INNER_SIZE_X) - 0.02))
    DIGGABLE_RADIUS_Y = clamp_value(DIGGABLE_RADIUS_Y, 0.05, max(0.05, 0.5 * float(SANDBOX_INNER_SIZE_Y) - 0.02))
    UNLOAD_BIN_INNER_SIZE_X = clamp_value(UNLOAD_BIN_INNER_SIZE_X, 0.40, 8.0)
    UNLOAD_BIN_INNER_SIZE_Y = clamp_value(UNLOAD_BIN_INNER_SIZE_Y, 0.40, 8.0)
    UNLOAD_BIN_WALL_THICKNESS = clamp_value(UNLOAD_BIN_WALL_THICKNESS, 0.02, 0.40)
    UNLOAD_BIN_WALL_HEIGHT = clamp_value(UNLOAD_BIN_WALL_HEIGHT, 0.10, 2.50)
    UNLOAD_BIN_FLOOR_THICKNESS = clamp_value(UNLOAD_BIN_FLOOR_THICKNESS, 0.04, 0.50)


def derive_stable_particle_params(layer_spacing_z=None):
    global PARTICLE_DIGGABLE_SPACING, PARTICLE_LAYER_SPACING_Z, PARTICLE_RADIUS
    global PARTICLE_CONTACT_OFFSET, PARTICLE_REST_OFFSET, PARTICLE_SOLID_REST_OFFSET, PARTICLE_FLUID_REST_OFFSET
    global PARTICLE_JITTER

    PARTICLE_RADIUS = clamp_value(PARTICLE_RADIUS, 0.006, 0.060)
    min_center_spacing = float(PARTICLE_RADIUS) * PARTICLE_CENTER_SPACING_SAFETY
    PARTICLE_DIGGABLE_SPACING = clamp_value(
        max(float(PARTICLE_DIGGABLE_SPACING), min_center_spacing),
        0.025,
        0.20,
    )

    if layer_spacing_z is None:
        target_layer_spacing = float(PARTICLE_DIGGABLE_SPACING) * PARTICLE_LAYER_SPACING_RATIO
    else:
        target_layer_spacing = float(layer_spacing_z)
    PARTICLE_LAYER_SPACING_Z = clamp_value(
        max(target_layer_spacing, min_center_spacing),
        0.025,
        0.20,
    )

    # These are intentionally derived, not UI-controlled. The ratios preserve
    # the stable backup defaults while keeping particle contacts conservative.
    PARTICLE_REST_OFFSET = clamp_value(
        max(0.92 * float(PARTICLE_RADIUS), 0.38 * float(PARTICLE_DIGGABLE_SPACING)),
        0.0,
        0.12,
    )
    PARTICLE_CONTACT_OFFSET = clamp_value(
        max(1.35 * float(PARTICLE_RADIUS), 1.25 * float(PARTICLE_REST_OFFSET)),
        float(PARTICLE_REST_OFFSET),
        0.12,
    )
    PARTICLE_SOLID_REST_OFFSET = clamp_value(
        max(float(PARTICLE_RADIUS), float(PARTICLE_REST_OFFSET)),
        0.0,
        float(PARTICLE_CONTACT_OFFSET),
    )
    PARTICLE_FLUID_REST_OFFSET = 0.0
    PARTICLE_JITTER = clamp_value(
        min(PARTICLE_JITTER_MAX, PARTICLE_JITTER_RATIO * float(PARTICLE_DIGGABLE_SPACING)),
        0.0,
        0.01,
    )


def estimate_particle_count_from_config():
    try:
        spacing_xy = max(1.0e-6, float(PARTICLE_DIGGABLE_SPACING))
        spacing_z = max(1.0e-6, float(PARTICLE_LAYER_SPACING_Z))
        floor_z = float(SAND_FLOOR_Z)
        count = 0
        for x, y in footprint_xy_samples(spacing_xy):
            surface_z = initial_sand_height_xy(float(x), float(y))
            fill_depth = max(spacing_z, float(surface_z) - floor_z)
            pile_depth = min(float(SAND_THICKNESS) + float(PILE_HEIGHT), fill_depth)
            count += max(1, int(pile_depth / spacing_z))
            if count >= int(PARTICLE_MAX_COUNT):
                return int(PARTICLE_MAX_COUNT)
        return int(min(int(PARTICLE_MAX_COUNT), max(0, count)))
    except Exception:
        return int(PARTICLE_MAX_COUNT)


def particle_performance_target_count():
    f = clamp_value(SAND_FIDELITY, 0.0, 1.0)
    base_target = int(round(lerp_value(
        SAND_PERFORMANCE_EFFICIENCY_TARGET_PARTICLES,
        SAND_PERFORMANCE_REALISTIC_TARGET_PARTICLES,
        f,
    )))
    return int(round(float(base_target) * clamp_sand_amount()))


def apply_particle_performance_budget(announce=False):
    global PARTICLE_DIGGABLE_SPACING, PARTICLE_LAYER_SPACING_Z, PARTICLE_RADIUS
    global PARTICLE_MASS, PARTICLE_MAX_COUNT

    STATE["particle_budget_applied"] = False
    STATE["particle_budget_target"] = int(particle_performance_target_count())
    STATE["particle_budget_estimate_before"] = int(estimate_particle_count_from_config())
    if not SAND_PERFORMANCE_AUTO_BUDGET:
        return

    target = max(1000, int(STATE["particle_budget_target"]))
    estimate = max(1, int(STATE["particle_budget_estimate_before"]))
    if estimate <= target:
        STATE["particle_budget_estimate_after"] = estimate
        return

    start_spacing = float(PARTICLE_DIGGABLE_SPACING)
    start_layer = float(PARTICLE_LAYER_SPACING_Z)
    budget_iters = 0
    after = estimate
    max_iters = max(1, int(SAND_PERFORMANCE_BUDGET_MAX_ITERS))
    tolerance = max(1.0, float(SAND_PERFORMANCE_BUDGET_TOLERANCE))

    for budget_iters in range(1, max_iters + 1):
        if after <= int(round(float(target) * tolerance)):
            break
        spacing_before = float(PARTICLE_DIGGABLE_SPACING)
        layer_before = float(PARTICLE_LAYER_SPACING_Z)
        scale = (float(after) / float(target)) ** (1.0 / 3.0)
        PARTICLE_DIGGABLE_SPACING = clamp_value(
            spacing_before * scale,
            spacing_before,
            SAND_PERFORMANCE_MAX_SPACING,
        )
        PARTICLE_LAYER_SPACING_Z = clamp_value(
            max(layer_before * scale, float(PARTICLE_DIGGABLE_SPACING) * PARTICLE_LAYER_SPACING_RATIO),
            0.025,
            0.20,
        )
        PARTICLE_RADIUS = clamp_value(
            max(float(PARTICLE_RADIUS), float(PARTICLE_DIGGABLE_SPACING) * SAND_PERFORMANCE_MIN_RADIUS_TO_SPACING),
            0.006,
            0.060,
        )
        derive_stable_particle_params(layer_spacing_z=PARTICLE_LAYER_SPACING_Z)
        PARTICLE_MASS = clamp_value(
            float(SAND_BULK_DENSITY_KG_M3) * (float(PARTICLE_DIGGABLE_SPACING) ** 3),
            SAND_FIDELITY_MIN_MASS,
            SAND_FIDELITY_MAX_MASS,
        )
        after = int(estimate_particle_count_from_config())
        if abs(float(PARTICLE_DIGGABLE_SPACING) - spacing_before) < 1.0e-6:
            break

    STATE["particle_budget_applied"] = True
    STATE["particle_budget_estimate_after"] = after
    STATE["particle_budget_iters"] = int(budget_iters)
    if announce:
        info(
            "[SAND PERF BUDGET]",
            f"target={target}",
            f"estimate_before={estimate}",
            f"estimate_after={after}",
            f"iters={budget_iters}",
            f"spacing={start_spacing:.4f}->{PARTICLE_DIGGABLE_SPACING:.4f}",
            f"layer={start_layer:.4f}->{PARTICLE_LAYER_SPACING_Z:.4f}",
            f"radius={PARTICLE_RADIUS:.4f}",
            f"solver_iters={PARTICLE_SOLVER_POSITION_ITERATIONS}",
        )


def update_particle_runtime_state():
    STATE["sand_amount_x"] = float(SAND_AMOUNT_MULTIPLIER)
    STATE["sand_fill_height"] = float(SANDBOX_FILL_HEIGHT)
    STATE["sand_fidelity"] = float(SAND_FIDELITY)
    STATE["sand_mode_label"] = sand_mode_label(SAND_FIDELITY)
    STATE["particle_mass"] = float(PARTICLE_MASS)
    STATE["particle_radius"] = float(PARTICLE_RADIUS)
    STATE["particle_max_count"] = int(PARTICLE_MAX_COUNT)
    STATE["particle_spacing_xy"] = float(PARTICLE_DIGGABLE_SPACING)
    STATE["particle_spacing_z"] = float(PARTICLE_LAYER_SPACING_Z)
    STATE["particle_solver_iters"] = int(PARTICLE_SOLVER_POSITION_ITERATIONS)
    STATE["particle_max_velocity"] = float(PARTICLE_MAX_VELOCITY)
    STATE["particle_contact_offset"] = float(PARTICLE_CONTACT_OFFSET)
    STATE["particle_rest_offset"] = float(PARTICLE_REST_OFFSET)
    STATE["particle_solid_rest_offset"] = float(PARTICLE_SOLID_REST_OFFSET)
    STATE["particle_fluid_rest_offset"] = float(PARTICLE_FLUID_REST_OFFSET)
    STATE["particle_jitter"] = float(PARTICLE_JITTER)
    STATE["estimated_particle_count"] = int(estimate_particle_count_from_config())
    STATE["particle_budget_target"] = int(particle_performance_target_count())
    STATE["particle_budget_applied"] = bool(STATE.get("particle_budget_applied", False))
    STATE["particle_budget_iters"] = int(STATE.get("particle_budget_iters", 0) or 0)


def apply_sand_fidelity_to_particle_globals(fidelity=None, announce=False):
    global SAND_FIDELITY, PARTICLE_DIGGABLE_SPACING, PARTICLE_RADIUS
    global PARTICLE_MAX_COUNT, PARTICLE_SOLVER_POSITION_ITERATIONS, PARTICLE_MAX_VELOCITY, PARTICLE_MASS

    if fidelity is not None:
        SAND_FIDELITY = clamp_value(fidelity, 0.0, 1.0)
    else:
        SAND_FIDELITY = clamp_value(SAND_FIDELITY, 0.0, 1.0)

    f = float(SAND_FIDELITY)
    PARTICLE_DIGGABLE_SPACING = lerp_value(SAND_FIDELITY_EFFICIENCY_SPACING, SAND_FIDELITY_REALISTIC_SPACING, f)
    PARTICLE_RADIUS = lerp_value(SAND_FIDELITY_EFFICIENCY_RADIUS, SAND_FIDELITY_REALISTIC_RADIUS, f)
    amount_spacing_scale = sand_amount_density_spacing_scale()
    PARTICLE_DIGGABLE_SPACING *= amount_spacing_scale
    PARTICLE_RADIUS *= amount_spacing_scale
    base_max_count = int(round(lerp_value(
        SAND_FIDELITY_EFFICIENCY_MAX_COUNT,
        SAND_FIDELITY_REALISTIC_MAX_COUNT,
        f,
    )))
    budget_max_count = int(round(float(particle_performance_target_count()) * SAND_PERFORMANCE_MAX_COUNT_HEADROOM))
    PARTICLE_MAX_COUNT = int(min(
        SAND_PERFORMANCE_HARD_MAX_PARTICLE_COUNT,
        max(base_max_count, budget_max_count),
    ))
    PARTICLE_SOLVER_POSITION_ITERATIONS = int(round(lerp_value(
        SAND_FIDELITY_EFFICIENCY_SOLVER_ITERS,
        SAND_FIDELITY_REALISTIC_SOLVER_ITERS,
        f,
    )))
    PARTICLE_MAX_VELOCITY = float(SAND_FIDELITY_MAX_VELOCITY)
    mass_scale = (float(PARTICLE_DIGGABLE_SPACING) / float(SAND_FIDELITY_REFERENCE_SPACING)) ** 3
    PARTICLE_MASS = clamp_value(
        float(SAND_FIDELITY_REFERENCE_MASS) * mass_scale,
        SAND_FIDELITY_MIN_MASS,
        SAND_FIDELITY_MAX_MASS,
    )
    derive_stable_particle_params()
    apply_particle_performance_budget(announce=announce)
    update_particle_runtime_state()
    if announce:
        info(
            "[SAND FIDELITY]",
            f"value={SAND_FIDELITY:.2f}",
            f"mode={sand_mode_label(SAND_FIDELITY)}",
            f"amount={SAND_AMOUNT_MULTIPLIER:.2f}x",
            f"height={SANDBOX_FILL_HEIGHT:.2f}",
            f"spacing={PARTICLE_DIGGABLE_SPACING:.4f}",
            f"radius={PARTICLE_RADIUS:.4f}",
            f"max_count={PARTICLE_MAX_COUNT}",
            f"solver_iters={PARTICLE_SOLVER_POSITION_ITERATIONS}",
            f"mass={PARTICLE_MASS:.4f}",
            f"estimated={STATE.get('estimated_particle_count')}",
        )


def set_sand_fidelity(fidelity, rebuild=False):
    apply_sand_fidelity_to_particle_globals(fidelity, announce=True)
    refresh_parameter_models_from_globals()
    store_runtime_api()
    update_status(
        f"Sand fidelity {SAND_FIDELITY:.2f} {sand_mode_label(SAND_FIDELITY)}; Apply + Rebuild updates particles",
        force=True,
    )
    if rebuild:
        build_sand_site()
    return {
        "sand_fidelity": float(SAND_FIDELITY),
        "sand_mode_label": sand_mode_label(SAND_FIDELITY),
        "spacing_xy": float(PARTICLE_DIGGABLE_SPACING),
        "spacing_z": float(PARTICLE_LAYER_SPACING_Z),
        "radius": float(PARTICLE_RADIUS),
        "max_count": int(PARTICLE_MAX_COUNT),
        "solver_iters": int(PARTICLE_SOLVER_POSITION_ITERATIONS),
        "mass": float(PARTICLE_MASS),
        "estimated_particle_count": int(STATE.get("estimated_particle_count", 0)),
    }


def apply_parameter_models_to_globals():
    global SANDBOX_INNER_SIZE_X, SANDBOX_INNER_SIZE_Y, SANDBOX_WALL_THICKNESS, SANDBOX_WALL_HEIGHT
    global SANDBOX_FILL_HEIGHT, SANDBOX_FLOOR_THICKNESS, SAND_CENTER_X, SAND_CENTER_Y
    global SAND_AMOUNT_MULTIPLIER
    global SAND_POINT_Z, SAND_FLOOR_Z, SAND_POINT_RADIUS
    global SAND_SOURCE_SELECTED_PATH, SAND_SOURCE_SELECTED_FACE_COUNT, SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT
    global SAND_SOURCE_RAW_FACE_POLYGONS_XY, SAND_SOURCE_FACE_POLYGONS_XY
    global SAND_SOURCE_SELECTED_HULL_XY, SAND_SOURCE_POLYGON_XY
    global WALL_CENTER_X, WALL_CENTER_Y, WALL_BASE_Z, WALL_INNER_SIZE_X, WALL_INNER_SIZE_Y
    global PILE_CENTER_X, PILE_CENTER_Y, PILE_SIGMA_X, PILE_SIGMA_Y, SAND_THICKNESS
    global DIGGABLE_RADIUS_X, DIGGABLE_RADIUS_Y
    global SAND_SOURCE_SELECTION_EDGE_MARGIN
    global PARTICLE_DIGGABLE_SPACING, PARTICLE_LAYER_SPACING_Z, PARTICLE_RADIUS, PARTICLE_CONTACT_OFFSET
    global PARTICLE_REST_OFFSET, PARTICLE_SOLID_REST_OFFSET, PARTICLE_FLUID_REST_OFFSET, PARTICLE_MASS
    global PARTICLE_JITTER, PARTICLE_SOLVER_POSITION_ITERATIONS, PARTICLE_MAX_VELOCITY, PARTICLE_MAX_COUNT
    global SAND_FIDELITY
    global UNLOAD_BIN_CENTER, UNLOAD_BIN_INNER_SIZE_X, UNLOAD_BIN_INNER_SIZE_Y, UNLOAD_BIN_WALL_HEIGHT
    global UNLOAD_BIN_WALL_THICKNESS, UNLOAD_BIN_FLOOR_THICKNESS, UNLOAD_BIN_DUMP_HEIGHT

    def model_value(key, current, lo=None, hi=None):
        model = PARAM_MODELS.get(key)
        if model is None:
            return current
        try:
            raw = model.get_value_as_float()
        except Exception:
            try:
                raw = model.as_float
            except Exception:
                raw = current
        return clamp_value(raw, lo, hi)

    next_sand_x = model_value("sand_center_x", SAND_CENTER_X, -20.0, 20.0)
    next_sand_y = model_value("sand_center_y", SAND_CENTER_Y, -20.0, 20.0)
    next_sand_z = model_value("sand_point_z", SAND_POINT_Z, -2.0, 8.0)
    next_sand_r = model_value("sand_point_r", SAND_POINT_RADIUS, 0.05, 6.0)
    xy_changed_by_user = (
        abs(float(next_sand_x) - float(SAND_CENTER_X)) > 1.0e-4
        or abs(float(next_sand_y) - float(SAND_CENTER_Y)) > 1.0e-4
    )
    r_changed_by_user = abs(float(next_sand_r) - float(SAND_POINT_RADIUS)) > 1.0e-4
    if (xy_changed_by_user or r_changed_by_user) and SAND_SOURCE_SELECTED_HULL_XY is not None:
        SAND_SOURCE_SELECTED_PATH = ""
        SAND_SOURCE_SELECTED_FACE_COUNT = 0
        SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT = 0
        SAND_SOURCE_RAW_FACE_POLYGONS_XY = None
        SAND_SOURCE_FACE_POLYGONS_XY = None
        SAND_SOURCE_SELECTED_HULL_XY = None
        SAND_SOURCE_POLYGON_XY = None
        info("[SAND SOURCE SELECT] source=point_xyz reason=sand_point_xy_or_radius_changed")

    SAND_CENTER_X = next_sand_x
    SAND_CENTER_Y = next_sand_y
    SAND_POINT_Z = next_sand_z
    SAND_POINT_RADIUS = next_sand_r
    SAND_FLOOR_Z = SAND_POINT_Z
    PILE_CENTER_X = SAND_CENTER_X
    PILE_CENTER_Y = SAND_CENTER_Y
    SAND_SOURCE_SELECTION_EDGE_MARGIN = model_value("sand_source_shrink_d", SAND_SOURCE_SELECTION_EDGE_MARGIN, 0.0, 2.0)

    WALL_CENTER_X = model_value("wall_center_x", WALL_CENTER_X, -20.0, 20.0)
    WALL_CENTER_Y = model_value("wall_center_y", WALL_CENTER_Y, -20.0, 20.0)
    WALL_BASE_Z = model_value("wall_center_z", WALL_BASE_Z, -2.0, 8.0)
    WALL_INNER_SIZE_X = model_value("wall_size_x", WALL_INNER_SIZE_X, 0.50, 12.0)
    WALL_INNER_SIZE_Y = model_value("wall_size_y", WALL_INNER_SIZE_Y, 0.50, 12.0)

    SANDBOX_INNER_SIZE_X = model_value("sandbox_x", SANDBOX_INNER_SIZE_X, 0.50, 8.0)
    SANDBOX_INNER_SIZE_Y = model_value("sandbox_y", SANDBOX_INNER_SIZE_Y, 0.50, 8.0)
    if SAND_SOURCE_SELECTED_HULL_XY is None:
        SANDBOX_INNER_SIZE_X = max(0.10, 2.0 * float(SAND_POINT_RADIUS))
        SANDBOX_INNER_SIZE_Y = max(0.10, 2.0 * float(SAND_POINT_RADIUS))
    SANDBOX_WALL_THICKNESS = model_value("wall_thickness", SANDBOX_WALL_THICKNESS, 0.02, 0.40)
    SANDBOX_WALL_HEIGHT = model_value("wall_height", SANDBOX_WALL_HEIGHT, 0.10, 2.50)
    SAND_AMOUNT_MULTIPLIER = model_value(
        "sand_amount_x",
        SAND_AMOUNT_MULTIPLIER,
        SAND_AMOUNT_MIN_MULTIPLIER,
        SAND_AMOUNT_MAX_MULTIPLIER,
    )
    SANDBOX_FILL_HEIGHT = sand_height_from_amount(SAND_AMOUNT_MULTIPLIER)
    SANDBOX_FLOOR_THICKNESS = model_value("floor_thickness", SANDBOX_FLOOR_THICKNESS, 0.04, 0.50)
    SAND_THICKNESS = model_value("sand_thickness", SAND_THICKNESS, 0.02, SANDBOX_FILL_HEIGHT)

    DIGGABLE_RADIUS_X = model_value("diggable_x", DIGGABLE_RADIUS_X, 0.05, 0.5 * SANDBOX_INNER_SIZE_X - 0.02)
    DIGGABLE_RADIUS_Y = model_value("diggable_y", DIGGABLE_RADIUS_Y, 0.05, 0.5 * SANDBOX_INNER_SIZE_Y - 0.02)
    if SAND_SOURCE_SELECTED_HULL_XY is None:
        DIGGABLE_RADIUS_X = float(SAND_POINT_RADIUS)
        DIGGABLE_RADIUS_Y = float(SAND_POINT_RADIUS)
    PILE_SIGMA_X = model_value("pile_sigma_x", PILE_SIGMA_X, 0.05, max(0.05, DIGGABLE_RADIUS_X))
    PILE_SIGMA_Y = model_value("pile_sigma_y", PILE_SIGMA_Y, 0.05, max(0.05, DIGGABLE_RADIUS_Y))
    refresh_selected_sand_polygon_from_hull()

    recompute_derived_scene_params()

    SAND_FIDELITY = model_value("sand_fidelity", SAND_FIDELITY, 0.0, 1.0)
    apply_sand_fidelity_to_particle_globals(SAND_FIDELITY, announce=True)

    unload_x = model_value("unload_x", float(UNLOAD_BIN_CENTER[0]), -20.0, 20.0)
    unload_y = model_value("unload_y", float(UNLOAD_BIN_CENTER[1]), -20.0, 20.0)
    UNLOAD_BIN_CENTER = np.array([unload_x, unload_y, 0.0], dtype=np.float32)
    UNLOAD_BIN_INNER_SIZE_X = model_value("unload_size_x", UNLOAD_BIN_INNER_SIZE_X, 0.40, 8.0)
    UNLOAD_BIN_INNER_SIZE_Y = model_value("unload_size_y", UNLOAD_BIN_INNER_SIZE_Y, 0.40, 8.0)
    UNLOAD_BIN_WALL_HEIGHT = model_value("unload_wall_height", UNLOAD_BIN_WALL_HEIGHT, 0.10, 2.50)
    UNLOAD_BIN_WALL_THICKNESS = model_value("unload_wall_thickness", UNLOAD_BIN_WALL_THICKNESS, 0.02, 0.40)
    UNLOAD_BIN_FLOOR_THICKNESS = model_value("unload_floor_thickness", UNLOAD_BIN_FLOOR_THICKNESS, 0.04, 0.50)
    UNLOAD_BIN_DUMP_HEIGHT = model_value("unload_dump_height", UNLOAD_BIN_DUMP_HEIGHT, UNLOAD_BIN_WALL_HEIGHT + 0.10, 4.0)

    recompute_derived_scene_params()
    update_particle_runtime_state()
    rebuild_sand_reference_grid("apply_parameters")
    set_sand_generation_range_box("apply_parameters")


def simulation_timeline_is_playing():
    if not HAS_OMNI_TIMELINE:
        return True
    try:
        timeline = omni.timeline.get_timeline_interface()
        if timeline is None:
            return True
        return bool(timeline.is_playing())
    except Exception:
        return True


def refresh_parameter_models_from_globals():
    values = {
        "sand_center_x": SAND_CENTER_X,
        "sand_center_y": SAND_CENTER_Y,
        "sand_point_z": SAND_POINT_Z,
        "sand_point_r": SAND_POINT_RADIUS,
        "sand_amount_x": SAND_AMOUNT_MULTIPLIER,
        "sand_source_shrink_d": SAND_SOURCE_SELECTION_EDGE_MARGIN,
        "wall_center_x": WALL_CENTER_X,
        "wall_center_y": WALL_CENTER_Y,
        "wall_center_z": WALL_BASE_Z,
        "wall_size_x": WALL_INNER_SIZE_X,
        "wall_size_y": WALL_INNER_SIZE_Y,
        "sandbox_x": SANDBOX_INNER_SIZE_X,
        "sandbox_y": SANDBOX_INNER_SIZE_Y,
        "wall_thickness": SANDBOX_WALL_THICKNESS,
        "wall_height": SANDBOX_WALL_HEIGHT,
        "fill_height": SANDBOX_FILL_HEIGHT,
        "floor_thickness": SANDBOX_FLOOR_THICKNESS,
        "sand_thickness": SAND_THICKNESS,
        "diggable_x": DIGGABLE_RADIUS_X,
        "diggable_y": DIGGABLE_RADIUS_Y,
        "pile_sigma_x": PILE_SIGMA_X,
        "pile_sigma_y": PILE_SIGMA_Y,
        "sand_fidelity": SAND_FIDELITY,
        "particle_spacing_xy": PARTICLE_DIGGABLE_SPACING,
        "particle_spacing_z": PARTICLE_LAYER_SPACING_Z,
        "particle_radius": PARTICLE_RADIUS,
        "particle_contact_offset": PARTICLE_CONTACT_OFFSET,
        "particle_rest_offset": PARTICLE_REST_OFFSET,
        "particle_solid_rest_offset": PARTICLE_SOLID_REST_OFFSET,
        "particle_fluid_rest_offset": PARTICLE_FLUID_REST_OFFSET,
        "particle_mass": PARTICLE_MASS,
        "particle_jitter": PARTICLE_JITTER,
        "particle_solver_iters": float(PARTICLE_SOLVER_POSITION_ITERATIONS),
        "particle_max_velocity": PARTICLE_MAX_VELOCITY,
        "particle_max_count": float(PARTICLE_MAX_COUNT),
        "unload_x": float(UNLOAD_BIN_CENTER[0]),
        "unload_y": float(UNLOAD_BIN_CENTER[1]),
        "unload_size_x": UNLOAD_BIN_INNER_SIZE_X,
        "unload_size_y": UNLOAD_BIN_INNER_SIZE_Y,
        "unload_wall_height": UNLOAD_BIN_WALL_HEIGHT,
        "unload_wall_thickness": UNLOAD_BIN_WALL_THICKNESS,
        "unload_floor_thickness": UNLOAD_BIN_FLOOR_THICKNESS,
        "unload_dump_height": UNLOAD_BIN_DUMP_HEIGHT,
    }
    STATE["suppress_parameter_callbacks"] = True
    try:
        for key, value in values.items():
            model = PARAM_MODELS.get(key)
            if model is not None:
                try:
                    model.set_value(float(value))
                except Exception:
                    pass
    finally:
        STATE["suppress_parameter_callbacks"] = False


def set_particle_size_and_limit(radius=None, max_count=None, spacing_xy=None, spacing_z=None, rebuild=False):
    global PARTICLE_RADIUS, PARTICLE_CONTACT_OFFSET, PARTICLE_REST_OFFSET
    global PARTICLE_SOLID_REST_OFFSET, PARTICLE_FLUID_REST_OFFSET
    global PARTICLE_MAX_COUNT, PARTICLE_DIGGABLE_SPACING, PARTICLE_LAYER_SPACING_Z
    global SAND_FIDELITY

    if radius is not None:
        PARTICLE_RADIUS = clamp_value(radius, 0.006, 0.060)
        PARTICLE_CONTACT_OFFSET = max(float(PARTICLE_CONTACT_OFFSET), float(PARTICLE_RADIUS))
        PARTICLE_REST_OFFSET = min(float(PARTICLE_REST_OFFSET), float(PARTICLE_CONTACT_OFFSET))
        PARTICLE_SOLID_REST_OFFSET = min(float(PARTICLE_SOLID_REST_OFFSET), float(PARTICLE_CONTACT_OFFSET))
        PARTICLE_FLUID_REST_OFFSET = min(float(PARTICLE_FLUID_REST_OFFSET), float(PARTICLE_CONTACT_OFFSET))
    if max_count is not None:
        PARTICLE_MAX_COUNT = int(round(clamp_value(max_count, 1000, SAND_PERFORMANCE_HARD_MAX_PARTICLE_COUNT)))
    if spacing_xy is not None:
        PARTICLE_DIGGABLE_SPACING = clamp_value(spacing_xy, 0.025, 0.20)
    if spacing_z is not None:
        PARTICLE_LAYER_SPACING_Z = clamp_value(spacing_z, 0.025, 0.20)
    derive_stable_particle_params(layer_spacing_z=PARTICLE_LAYER_SPACING_Z if spacing_z is not None else None)
    apply_particle_performance_budget(announce=True)
    denom = max(1.0e-6, SAND_FIDELITY_EFFICIENCY_SPACING - SAND_FIDELITY_REALISTIC_SPACING)
    SAND_FIDELITY = clamp_value((SAND_FIDELITY_EFFICIENCY_SPACING - float(PARTICLE_DIGGABLE_SPACING)) / denom, 0.0, 1.0)
    update_particle_runtime_state()
    refresh_parameter_models_from_globals()
    store_runtime_api()
    update_status(
        f"Particle size/limit set: fidelity={SAND_FIDELITY:.2f}, radius={PARTICLE_RADIUS:.4f}, max_count={PARTICLE_MAX_COUNT}",
        force=True,
    )
    if rebuild:
        build_sand_site()
    return {
        "radius": PARTICLE_RADIUS,
        "max_count": PARTICLE_MAX_COUNT,
        "spacing_xy": PARTICLE_DIGGABLE_SPACING,
        "spacing_z": PARTICLE_LAYER_SPACING_Z,
        "sand_fidelity": SAND_FIDELITY,
        "rebuild": bool(rebuild),
    }


def sand_set_xform(prim, translate=None, scale=None, rotate_xyz=None):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translate is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if rotate_xyz is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))


def sand_set_color(prim, rgb, opacity=None):
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])
    if opacity is not None:
        gprim.CreateDisplayOpacityAttr([float(opacity)])


def make_imageable_visible(prim):
    try:
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()
        imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
    except Exception:
        pass


def world_bbox_min_max_for_prim(prim):
    if prim is None or not prim.IsValid():
        return None, None
    try:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
            useExtentsHint=False,
        )
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = box.GetMin()
        mx = box.GetMax()
        vals = [float(mn[0]), float(mn[1]), float(mn[2]), float(mx[0]), float(mx[1]), float(mx[2])]
        if not all(math.isfinite(v) for v in vals):
            return None, None
        return (
            np.array([vals[0], vals[1], vals[2]], dtype=np.float32),
            np.array([vals[3], vals[4], vals[5]], dtype=np.float32),
        )
    except Exception:
        return None, None


def mesh_world_xy_points_under(prim):
    if prim is None or not prim.IsValid():
        return np.empty((0, 2), dtype=np.float32)
    pts = []
    for p in Usd.PrimRange(prim):
        try:
            if not p.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(p)
            local_points = mesh.GetPointsAttr().Get()
            if not local_points:
                continue
            mat = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for lp in local_points:
                wp = mat.Transform(Gf.Vec3d(float(lp[0]), float(lp[1]), float(lp[2])))
                pts.append((float(wp[0]), float(wp[1])))
        except Exception:
            continue
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def mesh_world_projected_faces_under(prim, max_faces=32):
    if prim is None or not prim.IsValid():
        return np.empty((0, 2), dtype=np.float32), [], 0
    candidates = []
    total_projected_faces = 0
    for p in Usd.PrimRange(prim):
        try:
            if not p.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(p)
            local_points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if local_points is None or counts is None or indices is None:
                continue
            if len(local_points) == 0 or len(counts) == 0 or len(indices) == 0:
                continue
            mat = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            cursor = 0
            for face_idx, count in enumerate(counts):
                count = int(count)
                face_indices = indices[cursor:cursor + count]
                cursor += count
                if count < 3:
                    continue
                poly = []
                for raw_idx in face_indices:
                    idx = int(raw_idx)
                    if idx < 0 or idx >= len(local_points):
                        continue
                    lp = local_points[idx]
                    wp = mat.Transform(Gf.Vec3d(float(lp[0]), float(lp[1]), float(lp[2])))
                    poly.append((float(wp[0]), float(wp[1])))
                if len(poly) < 3:
                    continue
                area = abs(polygon_signed_area(poly))
                if area <= 1.0e-6:
                    continue
                total_projected_faces += 1
                candidates.append((area, str(p.GetPath()), int(face_idx), np.array(poly, dtype=np.float32)))
        except Exception:
            continue

    if not candidates:
        return np.empty((0, 2), dtype=np.float32), [], total_projected_faces
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max(1, int(max_faces))]
    polys = [item[3] for item in selected]
    xy_points = np.vstack(polys).astype(np.float32)
    return xy_points, polys, total_projected_faces


def convex_hull_xy(points):
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return None
    pts = np.unique(np.round(pts, 6), axis=0)
    if pts.shape[0] < 3:
        return None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1.0e-9:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1.0e-9:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)
    return hull if hull.shape[0] >= 3 else None


def polygon_signed_area(poly):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid_xy(poly):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    area = polygon_signed_area(p)
    if abs(area) < 1.0e-8:
        return np.mean(p, axis=0).astype(np.float32)
    x = p[:, 0]
    y = p[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    cx = np.sum((x + x2) * cross) / (6.0 * area)
    cy = np.sum((y + y2) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=np.float32)


def shrink_convex_polygon_xy(poly, shrink_d):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return None
    if polygon_signed_area(p) < 0.0:
        p = p[::-1].copy()
    d = max(0.0, float(shrink_d))
    if d <= 1.0e-6:
        return p.astype(np.float32)

    def inside(pt, a, b):
        e = b - a
        length = max(1.0e-9, float(np.linalg.norm(e)))
        return float(e[0] * (pt[1] - a[1]) - e[1] * (pt[0] - a[0])) >= d * length - 1.0e-8

    def intersect(s, ept, a, b):
        edge = b - a
        seg = ept - s
        rhs = d * max(1.0e-9, float(np.linalg.norm(edge))) - (edge[0] * (s[1] - a[1]) - edge[1] * (s[0] - a[0]))
        denom = edge[0] * seg[1] - edge[1] * seg[0]
        if abs(float(denom)) < 1.0e-12:
            return ept
        t = float(rhs / denom)
        return s + np.clip(t, 0.0, 1.0) * seg

    out = p.copy()
    for i in range(p.shape[0]):
        a = p[i]
        b = p[(i + 1) % p.shape[0]]
        if out.shape[0] == 0:
            break
        new_pts = []
        s = out[-1]
        s_inside = inside(s, a, b)
        for ept in out:
            e_inside = inside(ept, a, b)
            if e_inside:
                if not s_inside:
                    new_pts.append(intersect(s, ept, a, b))
                new_pts.append(ept)
            elif s_inside:
                new_pts.append(intersect(s, ept, a, b))
            s = ept
            s_inside = e_inside
        out = np.array(new_pts, dtype=np.float64) if new_pts else np.empty((0, 2), dtype=np.float64)
    if out.shape[0] < 3 or abs(polygon_signed_area(out)) < 1.0e-6:
        c = polygon_centroid_xy(p).astype(np.float64)
        return (c + 0.80 * (p - c)).astype(np.float32)
    return out.astype(np.float32)


def point_in_polygon_xy(x, y, poly):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return False
    inside = False
    px = float(x)
    py = float(y)
    j = p.shape[0] - 1
    for i in range(p.shape[0]):
        xi, yi = p[i]
        xj, yj = p[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / max(1.0e-12, yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def visual_polygon_xy(poly, max_vertices):
    p = np.array(poly, dtype=np.float32).reshape(-1, 2)
    n = int(p.shape[0])
    limit = max(3, int(max_vertices))
    if n <= limit:
        return p

    closed = np.vstack([p, p[0]])
    edge_vec = closed[1:] - closed[:-1]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    perimeter = float(np.sum(edge_len))
    if perimeter <= 1.0e-6:
        idx = np.linspace(0, n - 1, limit, dtype=np.int32)
        return p[idx]

    cumulative = np.concatenate([[0.0], np.cumsum(edge_len)])
    samples = []
    for target_dist in np.linspace(0.0, perimeter, limit, endpoint=False):
        edge_idx = int(np.searchsorted(cumulative, target_dist, side="right") - 1)
        edge_idx = max(0, min(edge_idx, n - 1))
        local_len = max(1.0e-9, float(edge_len[edge_idx]))
        t = float((target_dist - cumulative[edge_idx]) / local_len)
        pt = closed[edge_idx] + t * edge_vec[edge_idx]
        samples.append((float(pt[0]), float(pt[1])))
    return np.array(samples, dtype=np.float32)


def selected_mesh_footprint_polygon_xy(hull_xy, shrink_d):
    hull = np.array(hull_xy, dtype=np.float32).reshape(-1, 2)
    if hull.shape[0] < 3:
        return None, None

    # Fallback for non-face mesh data; selected mesh generation normally uses face polygons directly.
    poly = shrink_convex_polygon_xy(hull, shrink_d)
    if poly is None:
        return hull, None
    if poly.shape[0] < 3:
        return hull, None
    return hull, poly


def shrink_face_polygons_xy(polygons, shrink_d):
    out = []
    for poly in polygons or []:
        p = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if p.shape[0] < 3:
            continue
        shrunk = shrink_convex_polygon_xy(p, shrink_d)
        if shrunk is None or np.array(shrunk).reshape(-1, 2).shape[0] < 3:
            continue
        out.append(np.array(shrunk, dtype=np.float32).reshape(-1, 2))
    return out


def ellipse_polygon_xy(cx, cy, rx, ry, segments=32):
    n = max(8, int(segments))
    rx = max(0.01, float(rx))
    ry = max(0.01, float(ry))
    pts = []
    for i in range(n):
        a = 2.0 * math.pi * float(i) / float(n)
        pts.append((float(cx) + rx * math.cos(a), float(cy) + ry * math.sin(a)))
    return np.array(pts, dtype=np.float32)


def current_sand_footprint_polygons_xy():
    return [current_sand_footprint_polygon_xy()]


def current_sand_footprint_polygon_xy():
    if SAND_SOURCE_POLYGON_XY is not None:
        return np.array(SAND_SOURCE_POLYGON_XY, dtype=np.float32).reshape(-1, 2)
    return ellipse_polygon_xy(
        float(PILE_CENTER_X),
        float(PILE_CENTER_Y),
        float(DIGGABLE_RADIUS_X),
        float(DIGGABLE_RADIUS_Y),
        SAND_POINT_RANGE_VISUAL_SEGMENTS,
    )


def current_sand_footprint_bbox_xy(padding=0.0):
    polys = current_sand_footprint_polygons_xy()
    pts = np.vstack([np.array(p, dtype=np.float32).reshape(-1, 2) for p in polys])
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    pad = max(0.0, float(padding))
    return float(mn[0] - pad), float(mx[0] + pad), float(mn[1] - pad), float(mx[1] + pad)


def polygon_x_intervals_at_y(poly, y):
    p = np.array(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return []
    yy = float(y)
    xs = []
    n = int(p.shape[0])
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        if abs(float(y2 - y1)) < 1.0e-12:
            continue
        if (float(y1) <= yy < float(y2)) or (float(y2) <= yy < float(y1)):
            t = (yy - float(y1)) / float(y2 - y1)
            xs.append(float(x1) + t * float(x2 - x1))
    xs.sort()
    intervals = []
    for i in range(0, len(xs) - 1, 2):
        lo = float(xs[i])
        hi = float(xs[i + 1])
        if hi > lo:
            intervals.append((lo, hi))
    return intervals


def footprint_xy_samples(spacing_xy):
    poly = current_sand_footprint_polygon_xy()
    x_min, x_max, y_min, y_max = current_sand_footprint_bbox_xy()
    spacing = max(1.0e-6, float(spacing_xy))
    try:
        poly_key = tuple((round(float(x), 5), round(float(y), 5)) for x, y in np.array(poly, dtype=np.float32).reshape(-1, 2))
        cache_key = (round(spacing, 5), poly_key, round(float(x_min), 5), round(float(x_max), 5), round(float(y_min), 5), round(float(y_max), 5))
        cache = STATE.get("footprint_sample_cache")
        if not isinstance(cache, dict):
            cache = {}
            STATE["footprint_sample_cache"] = cache
        cached = cache.get(cache_key)
        if cached is not None:
            for x, y in cached:
                yield float(x), float(y)
            return
    except Exception:
        cache_key = None
        cache = None

    samples = []
    y = float(y_min) + 0.5 * spacing
    row = 0
    while y <= float(y_max) + 1.0e-9:
        intervals = polygon_x_intervals_at_y(poly, y)
        row_offset = 0.5 * spacing if (row % 2) else 0.0
        for lo, hi in intervals:
            start = float(x_min) + row_offset + math.ceil((float(lo) - float(x_min) - row_offset) / spacing) * spacing
            x = max(float(lo), start)
            while x <= float(hi) + 1.0e-9:
                if is_inside_diggable_xy(float(x), float(y)):
                    samples.append((float(x), float(y)))
                    yield float(x), float(y)
                x += spacing
        row += 1
        y += spacing
    if cache_key is not None and isinstance(cache, dict):
        cache.clear()
        cache[cache_key] = samples


def selected_prim_paths():
    try:
        selection = omni.usd.get_context().get_selection()
        if selection is None:
            return []
        return [str(p) for p in (selection.get_selected_prim_paths() or []) if str(p)]
    except Exception as e:
        info("[WARN] selection read failed:", type(e).__name__, e)
        return []


def first_selected_prim_path():
    paths = selected_prim_paths()
    return paths[0] if paths else ""


def refresh_selected_sand_polygon_from_hull():
    global SAND_SOURCE_RAW_FACE_POLYGONS_XY, SAND_SOURCE_FACE_POLYGONS_XY
    global SAND_SOURCE_SELECTED_HULL_XY, SAND_SOURCE_POLYGON_XY, SAND_CENTER_X, SAND_CENTER_Y, PILE_CENTER_X, PILE_CENTER_Y
    global SANDBOX_INNER_SIZE_X, SANDBOX_INNER_SIZE_Y
    global DIGGABLE_RADIUS_X, DIGGABLE_RADIUS_Y, PILE_SIGMA_X, PILE_SIGMA_Y
    SAND_SOURCE_FACE_POLYGONS_XY = None
    if SAND_SOURCE_SELECTED_HULL_XY is None:
        return False
    hull, poly = selected_mesh_footprint_polygon_xy(SAND_SOURCE_SELECTED_HULL_XY, SAND_SOURCE_SELECTION_EDGE_MARGIN)
    if poly is None:
        return False
    if hull is not None:
        SAND_SOURCE_SELECTED_HULL_XY = np.array(hull, dtype=np.float32)
    SAND_SOURCE_POLYGON_XY = np.array(poly, dtype=np.float32)
    center_xy = polygon_centroid_xy(poly)
    poly_min = np.min(poly, axis=0)
    poly_max = np.max(poly, axis=0)
    SANDBOX_INNER_SIZE_X = max(0.50, float(poly_max[0] - poly_min[0]))
    SANDBOX_INNER_SIZE_Y = max(0.50, float(poly_max[1] - poly_min[1]))
    SAND_CENTER_X = float(center_xy[0])
    SAND_CENTER_Y = float(center_xy[1])
    PILE_CENTER_X = float(center_xy[0])
    PILE_CENTER_Y = float(center_xy[1])
    DIGGABLE_RADIUS_X = max(0.05, 0.5 * SANDBOX_INNER_SIZE_X - 0.08)
    DIGGABLE_RADIUS_Y = max(0.05, 0.5 * SANDBOX_INNER_SIZE_Y - 0.08)
    PILE_SIGMA_X = max(0.05, 0.70 * DIGGABLE_RADIUS_X)
    PILE_SIGMA_Y = max(0.05, 0.70 * DIGGABLE_RADIUS_Y)
    return True


def clear_selected_sand_source_mesh():
    global SAND_SOURCE_SELECTED_PATH, SAND_SOURCE_SELECTED_FACE_COUNT, SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT
    global SAND_SOURCE_RAW_FACE_POLYGONS_XY, SAND_SOURCE_FACE_POLYGONS_XY
    global SAND_SOURCE_SELECTED_HULL_XY, SAND_SOURCE_POLYGON_XY
    global PILE_CENTER_X, PILE_CENTER_Y
    SAND_SOURCE_SELECTED_PATH = ""
    SAND_SOURCE_SELECTED_FACE_COUNT = 0
    SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT = 0
    SAND_SOURCE_RAW_FACE_POLYGONS_XY = None
    SAND_SOURCE_FACE_POLYGONS_XY = None
    SAND_SOURCE_SELECTED_HULL_XY = None
    SAND_SOURCE_POLYGON_XY = None
    PILE_CENTER_X = SAND_CENTER_X
    PILE_CENTER_Y = SAND_CENTER_Y
    rebuild_sand_reference_grid("sand_point_xyz")
    set_sand_generation_range_box("sand_point_xyz")
    store_runtime_api()
    update_status("Sand source set to point XYZ mode; Reset Sand regenerates particles")
    info(
        "[SAND SOURCE SELECT]",
        "source=point_xyz",
        "center=", np.round([SAND_CENTER_X, SAND_CENTER_Y, SAND_FLOOR_Z], 3),
        "inner_size=", (round(float(SANDBOX_INNER_SIZE_X), 3), round(float(SANDBOX_INNER_SIZE_Y), 3)),
    )
    return True


def use_mesh_path_as_sand_source(path, source_label="mesh"):
    global SAND_SOURCE_SELECTED_PATH, SAND_SOURCE_SELECTED_FACE_COUNT, SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT
    global SAND_SOURCE_RAW_FACE_POLYGONS_XY, SAND_SOURCE_FACE_POLYGONS_XY
    global SAND_SOURCE_SELECTED_HULL_XY, SAND_SOURCE_POLYGON_XY
    global SAND_CENTER_X, SAND_CENTER_Y, PILE_CENTER_X, PILE_CENTER_Y
    global SANDBOX_INNER_SIZE_X, SANDBOX_INNER_SIZE_Y
    global DIGGABLE_RADIUS_X, DIGGABLE_RADIUS_Y, PILE_SIGMA_X, PILE_SIGMA_Y

    if not path:
        update_status("Select a sand source mesh/group first")
        return False
    prim = get_stage().GetPrimAtPath(Sdf.Path(path))
    if not prim or not prim.IsValid():
        update_status(f"Sand source path is invalid: {path}")
        info("[WARN] [SAND SOURCE SELECT] invalid path:", path)
        return False
    mn, mx = world_bbox_min_max_for_prim(prim)
    if mn is None or mx is None:
        update_status(f"Selected sand source has no valid bbox: {path}")
        return False

    xy_points, selected_faces, total_projected_faces = mesh_world_projected_faces_under(
        prim,
        SAND_SOURCE_MAX_PROJECTED_FACES,
    )
    used_bbox_fallback = False
    if xy_points.shape[0] == 0:
        xy_points = mesh_world_xy_points_under(prim)
    raw_hull = convex_hull_xy(xy_points)
    if raw_hull is None:
        used_bbox_fallback = True
        raw_hull = np.array(
            [
                [float(mn[0]), float(mn[1])],
                [float(mx[0]), float(mn[1])],
                [float(mx[0]), float(mx[1])],
                [float(mn[0]), float(mx[1])],
            ],
            dtype=np.float32,
        )
    shrink = float(SAND_SOURCE_SELECTION_EDGE_MARGIN)
    hull, poly = selected_mesh_footprint_polygon_xy(raw_hull, shrink)
    if poly is None:
        update_status(f"Selected sand source footprint failed: {path}")
        return False

    center_xy = polygon_centroid_xy(poly)
    poly_min = np.min(poly, axis=0)
    poly_max = np.max(poly, axis=0)
    inner_x = max(0.50, float(poly_max[0] - poly_min[0]))
    inner_y = max(0.50, float(poly_max[1] - poly_min[1]))

    SAND_SOURCE_SELECTED_PATH = path
    SAND_SOURCE_SELECTED_FACE_COUNT = int(len(selected_faces))
    SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT = int(total_projected_faces)
    SAND_SOURCE_RAW_FACE_POLYGONS_XY = None
    SAND_SOURCE_FACE_POLYGONS_XY = None
    SAND_SOURCE_SELECTED_HULL_XY = np.array(hull, dtype=np.float32)
    SAND_SOURCE_POLYGON_XY = np.array(poly, dtype=np.float32)

    SAND_CENTER_X = float(center_xy[0])
    SAND_CENTER_Y = float(center_xy[1])
    PILE_CENTER_X = float(center_xy[0])
    PILE_CENTER_Y = float(center_xy[1])
    SANDBOX_INNER_SIZE_X = inner_x
    SANDBOX_INNER_SIZE_Y = inner_y
    DIGGABLE_RADIUS_X = max(0.05, 0.5 * inner_x - 0.08)
    DIGGABLE_RADIUS_Y = max(0.05, 0.5 * inner_y - 0.08)
    PILE_SIGMA_X = max(0.05, 0.70 * DIGGABLE_RADIUS_X)
    PILE_SIGMA_Y = max(0.05, 0.70 * DIGGABLE_RADIUS_Y)

    recompute_derived_scene_params()
    update_particle_runtime_state()
    rebuild_sand_reference_grid("selected_sand_mesh")
    store_runtime_api()
    set_sand_generation_range_box("selected_sand_mesh")
    update_status(f"Sand source mesh/group selected: {path}; Reset Sand to regenerate particles")
    info(
        "[SAND SOURCE SELECT]",
        f"source={source_label}",
        "path=", path,
        "center=", np.round([SAND_CENTER_X, SAND_CENTER_Y, SAND_FLOOR_Z], 3),
        "inner_size=", (round(float(SANDBOX_INNER_SIZE_X), 3), round(float(SANDBOX_INNER_SIZE_Y), 3)),
        "diggable_radius=", (round(float(DIGGABLE_RADIUS_X), 3), round(float(DIGGABLE_RADIUS_Y), 3)),
        "mesh_shrink_d=", round(float(shrink), 3),
        "projected_faces_used=", int(SAND_SOURCE_SELECTED_FACE_COUNT),
        "projected_faces_total=", int(SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT),
        "max_projected_faces=", int(SAND_SOURCE_MAX_PROJECTED_FACES),
        "bbox_fallback=", bool(used_bbox_fallback),
        "hull_vertices=", int(len(hull)),
        "active_vertices=", int(len(poly)),
    )
    return True


def use_selected_mesh_as_sand_source():
    return use_mesh_path_as_sand_source(first_selected_prim_path(), source_label="selected_mesh")


def apply_default_sand_source_mesh():
    return use_mesh_path_as_sand_source(DEFAULT_SAND_SOURCE_MESH_PATH, source_label="preset_mesh")


def fmt_bbox(mn, mx):
    if mn is None or mx is None:
        return "None"
    return f"min={np.round(mn, 3)} max={np.round(mx, 3)}"


def sand_make_cube(path, translate, scale, color, collision=False, opacity=None):
    stage = get_stage()
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    sand_set_xform(prim, translate=translate, scale=scale)
    sand_set_color(prim, color, opacity)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def sand_make_sphere(path, translate, radius, color, collision=False, opacity=None):
    stage = get_stage()
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    prim = sphere.GetPrim()
    sand_set_xform(prim, translate=translate)
    sand_set_color(prim, color, opacity)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def set_sand_generation_range_box(label=""):
    stage = get_stage()
    cleanup_key = "sand_range_guide_cleanup_done"
    if not bool(STATE.get(cleanup_key, False)):
        for old_path in [
            f"{root_path()}/SandGenerationRangeColumn",
            f"{root_path()}/SandGenerationRangeBox",
        ]:
            try:
                stage.RemovePrim(Sdf.Path(old_path))
            except Exception:
                pass
        STATE[cleanup_key] = True
    path = f"{root_path()}/SandGenerationRangeGuide"
    height = max(0.20, float(SANDBOX_FILL_HEIGHT))
    poly = current_sand_footprint_polygon_xy()
    visual_poly = visual_polygon_xy(poly, SAND_SOURCE_RANGE_VISUAL_MAX_VERTICES)
    n = int(len(visual_poly))
    z0 = float(SAND_FLOOR_Z)
    z1 = float(SAND_FLOOR_Z) + height
    points = []
    counts = []
    if n >= 3:
        bottom = [Gf.Vec3f(float(x), float(y), z0) for x, y in visual_poly] + [
            Gf.Vec3f(float(visual_poly[0][0]), float(visual_poly[0][1]), z0)
        ]
        top = [Gf.Vec3f(float(x), float(y), z1) for x, y in visual_poly] + [
            Gf.Vec3f(float(visual_poly[0][0]), float(visual_poly[0][1]), z1)
        ]
        points.extend(bottom)
        counts.append(len(bottom))
        points.extend(top)
        counts.append(len(top))
        step = max(1, n // 8)
        for i in range(0, n, step):
            x, y = visual_poly[i]
            points.extend([Gf.Vec3f(float(x), float(y), z0), Gf.Vec3f(float(x), float(y), z1)])
            counts.append(2)
    prim = get_prim(path)
    created = False
    if not prim.IsValid() or not prim.IsA(UsdGeom.BasisCurves):
        try:
            stage.RemovePrim(Sdf.Path(path))
        except Exception:
            pass
        curve = UsdGeom.BasisCurves.Define(stage, path)
        created = True
    else:
        curve = UsdGeom.BasisCurves(prim)
    if len(points) < 2:
        points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.001, 0.0, 0.0)]
        counts = [2]
    try:
        curve.GetPointsAttr().Set(points) if curve.GetPointsAttr().IsValid() else curve.CreatePointsAttr(points)
        curve.GetCurveVertexCountsAttr().Set(counts) if curve.GetCurveVertexCountsAttr().IsValid() else curve.CreateCurveVertexCountsAttr(counts)
    except Exception:
        curve.CreatePointsAttr(points)
        curve.CreateCurveVertexCountsAttr(counts)
    if created or label != "sand_xyz_live":
        try:
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            try:
                curve.GetBasisAttr().Clear()
            except Exception:
                pass
            curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            curve.CreateWidthsAttr([float(SAND_SOURCE_RANGE_GUIDE_WIDTH)])
        except Exception:
            pass
    prim = curve.GetPrim()
    if created or label != "sand_xyz_live":
        sand_set_color(prim, SAND_SOURCE_RANGE_GUIDE_COLOR, None)
    poly_min = np.min(np.array(poly, dtype=np.float32).reshape(-1, 2), axis=0)
    poly_max = np.max(np.array(poly, dtype=np.float32).reshape(-1, 2), axis=0)
    center = polygon_centroid_xy(poly)
    if label and label != "sand_xyz_live":
        info(
            "[SAND SOURCE RANGE]",
            f"label={label}",
            "path=", path,
            "center=", np.round([float(center[0]), float(center[1]), float(SAND_FLOOR_Z) + 0.5 * height], 3),
            "bbox_size=", (round(float(poly_max[0] - poly_min[0]), 3), round(float(poly_max[1] - poly_min[1]), 3)),
            "vertices=", int(len(poly)),
            "visual_vertices=", int(len(visual_poly)),
            "height=", round(height, 3),
            "mesh_shrink_d=", round(float(SAND_SOURCE_SELECTION_EDGE_MARGIN), 3),
        )
    return path


def sand_make_cylinder(path, translate, radius, depth, color, rotate_xyz=(0.0, 90.0, 0.0), collision=False, opacity=None):
    stage = get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(depth))
    prim = cyl.GetPrim()
    sand_set_xform(prim, translate=translate, rotate_xyz=rotate_xyz)
    sand_set_color(prim, color, opacity)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def sand_make_polygon_prism(path, polygon_xy, z_min, z_max, color, opacity=None):
    poly = np.array(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return None
    mesh = UsdGeom.Mesh.Define(get_stage(), path)
    pts = []
    for xy in poly:
        pts.append(Gf.Vec3f(float(xy[0]), float(xy[1]), float(z_min)))
    for xy in poly:
        pts.append(Gf.Vec3f(float(xy[0]), float(xy[1]), float(z_max)))
    n = int(poly.shape[0])
    counts = [n, n]
    indices = list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        indices.extend([i, j, n + j, n + i])
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    try:
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(False)
    except Exception:
        pass
    prim = mesh.GetPrim()
    sand_set_color(prim, color, opacity)
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    except Exception:
        pass
    return prim


def sand_x_min():
    return current_sand_footprint_bbox_xy()[0]


def sand_x_max():
    return current_sand_footprint_bbox_xy()[1]


def sand_y_min():
    return current_sand_footprint_bbox_xy()[2]


def sand_y_max():
    return current_sand_footprint_bbox_xy()[3]


def initial_sand_height_xy(x, y):
    if not is_inside_diggable_xy(x, y):
        return SAND_FLOOR_Z
    dx = (float(x) - PILE_CENTER_X) / PILE_SIGMA_X
    dy = (float(y) - PILE_CENTER_Y) / PILE_SIGMA_Y
    mound = PILE_HEIGHT * (0.82 + 0.18 * math.exp(-0.5 * (dx * dx + dy * dy)))
    ripple = RIPPLE_AMP * (
        math.sin(3.7 * float(x) + 0.55 * math.sin(0.8 * float(y)))
        + 0.45 * math.sin(2.2 * float(y) + 0.25 * float(x))
    )
    n = math.sin(12.9898 * float(x) + 78.233 * float(y) + NOISE_SEED) * 43758.5453
    n = n - math.floor(n)
    return SAND_FLOOR_Z + mound + ripple + NOISE_AMP * (2.0 * n - 1.0)


def in_sand_bounds(x, y):
    return sand_x_min() <= float(x) <= sand_x_max() and sand_y_min() <= float(y) <= sand_y_max()


def is_inside_diggable_xy(x, y):
    if SAND_SOURCE_POLYGON_XY is not None:
        return point_in_polygon_xy(x, y, SAND_SOURCE_POLYGON_XY)
    rx = max(1.0e-6, float(DIGGABLE_RADIUS_X))
    ry = max(1.0e-6, float(DIGGABLE_RADIUS_Y))
    dx = (float(x) - float(PILE_CENTER_X)) / rx
    dy = (float(y) - float(PILE_CENTER_Y)) / ry
    return (dx * dx + dy * dy) <= 1.0


def height_from_grid(x, y):
    if HEIGHTS is None or X_VALUES is None or Y_VALUES is None:
        return initial_sand_height_xy(x, y)
    if not in_sand_bounds(x, y):
        return BASE_Z

    fx = (float(x) - sand_x_min()) / (sand_x_max() - sand_x_min()) * (NX - 1)
    fy = (float(y) - sand_y_min()) / (sand_y_max() - sand_y_min()) * (NY - 1)
    i0 = int(max(0, min(NX - 2, math.floor(fx))))
    j0 = int(max(0, min(NY - 2, math.floor(fy))))
    tx = fx - i0
    ty = fy - j0
    z00 = float(HEIGHTS[j0, i0])
    z10 = float(HEIGHTS[j0, i0 + 1])
    z01 = float(HEIGHTS[j0 + 1, i0])
    z11 = float(HEIGHTS[j0 + 1, i0 + 1])
    return (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11


def soil_depth_at_point(x, y, z):
    return height_from_grid(x, y) - float(z)


def current_real_particle_positions():
    root = root_path()
    prim = get_prim(f"{root}/{PARTICLE_POINTS_PATH_SUFFIX}")
    if not prim.IsValid():
        return np.empty((0, 3), dtype=np.float32)
    points_attr = prim.GetAttribute("points")
    points = points_attr.Get() if points_attr.IsValid() else None
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(points, dtype=np.float32).reshape(-1, 3)


def current_real_particle_positions_sampled(max_samples=None):
    root = root_path()
    prim = get_prim(f"{root}/{PARTICLE_POINTS_PATH_SUFFIX}")
    if not prim.IsValid():
        return np.empty((0, 3), dtype=np.float32), 0
    points_attr = prim.GetAttribute("points")
    points = points_attr.Get() if points_attr.IsValid() else None
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32), 0

    total = int(len(points))
    limit = total if max_samples is None else max(1, int(max_samples))
    if total <= limit:
        return np.array(points, dtype=np.float32).reshape(-1, 3), total

    indices = np.linspace(0, total - 1, limit, dtype=np.int64)
    sampled = np.empty((len(indices), 3), dtype=np.float32)
    for out_i, src_i in enumerate(indices):
        p = points[int(src_i)]
        sampled[out_i, 0] = float(p[0])
        sampled[out_i, 1] = float(p[1])
        sampled[out_i, 2] = float(p[2])
    return sampled, total


async def step_updates(frames=1):
    app = omni.kit.app.get_app()
    for _ in range(max(1, int(frames))):
        await app.next_update_async()


def sand_reset_health_stats(prev_points=None):
    points, total_count = current_real_particle_positions_sampled(SAND_RESET_HEALTH_SAMPLE_MAX)
    if points.shape[0] == 0:
        return {"ok": False, "reason": "no_particles", "n": 0}

    n = int(total_count)
    sample_n = int(points.shape[0])
    x_min = float(SAND_CENTER_X - float(DIGGABLE_RADIUS_X) - SAND_RESET_XY_ESCAPE_MARGIN)
    x_max = float(SAND_CENTER_X + float(DIGGABLE_RADIUS_X) + SAND_RESET_XY_ESCAPE_MARGIN)
    y_min = float(SAND_CENTER_Y - float(DIGGABLE_RADIUS_Y) - SAND_RESET_XY_ESCAPE_MARGIN)
    y_max = float(SAND_CENTER_Y + float(DIGGABLE_RADIUS_Y) + SAND_RESET_XY_ESCAPE_MARGIN)
    z_escape_below = max(5.0, 2.0 * float(SANDBOX_FILL_HEIGHT))
    z_escape_above = max(5.0, float(SAND_RESET_Z_ESCAPE_ABOVE))
    z_min_allowed = min(float(BASE_Z) - 3.0, float(SAND_FLOOR_Z) - z_escape_below)
    z_max_allowed = float(SAND_FLOOR_Z) + float(SANDBOX_FILL_HEIGHT) + z_escape_above

    outside = (
        (points[:, 0] < x_min)
        | (points[:, 0] > x_max)
        | (points[:, 1] < y_min)
        | (points[:, 1] > y_max)
    )
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    outside_count_sample = int(np.count_nonzero(outside))
    outside_count = int(round(outside_count_sample * (float(n) / float(max(1, sample_n)))))

    displacement_p95 = 0.0
    displacement_max = 0.0
    if prev_points is not None and isinstance(prev_points, np.ndarray) and prev_points.shape == points.shape:
        d = np.linalg.norm(points - prev_points, axis=1)
        if d.size > 0:
            displacement_p95 = float(np.percentile(d, 95))
            displacement_max = float(np.max(d))

    bad_reasons = []
    if z_min < z_min_allowed:
        bad_reasons.append(f"z_min={z_min:.3f}<limit={z_min_allowed:.3f}")
    if z_max > z_max_allowed:
        bad_reasons.append(f"z_max={z_max:.3f}>limit={z_max_allowed:.3f}")
    # Particles outside the sandbox XY footprint are diagnostic only. In real
    # sand interaction we allow material to spill over walls, so this must not
    # trigger an automatic reset/retry.
    # Displacement during the first seconds is diagnostic only. Tall piles are
    # expected to collapse and flow; retry only for true z explosion/escape.

    return {
        "ok": not bad_reasons,
        "reason": "ok" if not bad_reasons else ";".join(bad_reasons[:3]),
        "n": n,
        "sample_n": sample_n,
        "z_min": z_min,
        "z_max": z_max,
        "outside_xy": outside_count,
        "outside_xy_sample": outside_count_sample,
        "disp_p95": displacement_p95,
        "disp_max": displacement_max,
    }


async def wait_for_sand_reset_health(label="sand_reset"):
    prev, _ = current_real_particle_positions_sampled(SAND_RESET_HEALTH_SAMPLE_MAX)
    stats = None
    elapsed = 0
    max_frames = max(int(SAND_RESET_HEALTH_MIN_FRAMES), int(SAND_RESET_HEALTH_MAX_FRAMES))
    window = max(1, int(SAND_RESET_HEALTH_WINDOW_FRAMES))
    while elapsed < max_frames:
        if not bool(STATE.get("running", False)) or not simulation_timeline_is_playing():
            return False, {
                "ok": False,
                "reason": "timeline_stopped_or_runtime_stopped",
                "n": int(STATE.get("real_sand_particle_count", 0) or 0),
            }
        await step_updates(window)
        elapsed += window
        stats = sand_reset_health_stats(prev)
        prev, _ = current_real_particle_positions_sampled(SAND_RESET_HEALTH_SAMPLE_MAX)
        info(
            "[SAND RESET HEALTH]",
            f"label={label}",
            f"frames={elapsed}",
            f"ok={stats.get('ok')}",
            f"reason={stats.get('reason')}",
            f"n={stats.get('n')}",
            f"sample_n={stats.get('sample_n', 0)}",
            f"z=({stats.get('z_min', 0.0):.3f},{stats.get('z_max', 0.0):.3f})",
            f"outside_xy={stats.get('outside_xy', 0)}",
            f"disp_p95={stats.get('disp_p95', 0.0):.3f}",
        )
        if not bool(stats.get("ok", False)):
            return False, stats
        if elapsed >= int(SAND_RESET_HEALTH_MIN_FRAMES):
            return True, stats
    return bool(stats and stats.get("ok", False)), stats


async def reset_sand_surface_stably(label="ui"):
    if bool(STATE.get("sand_reset_active", False)):
        update_status("Sand reset already running", force=True)
        return False
    if not bool(STATE.get("running", False)) or not simulation_timeline_is_playing():
        info("[SAND RESET] skipped: timeline stopped or runtime stopped", f"label={label}")
        return False
    STATE["sand_reset_active"] = True
    try:
        attempts = max(1, int(SAND_RESET_MAX_ATTEMPTS))
        last_stats = None
        for attempt in range(1, attempts + 1):
            if not bool(STATE.get("running", False)) or not simulation_timeline_is_playing():
                last_stats = {"ok": False, "reason": "timeline_stopped_or_runtime_stopped"}
                break
            update_status(f"Reset real particle sand attempt {attempt}/{attempts}", force=True)
            info("[SAND RESET NATIVE]", f"label={label}", f"attempt={attempt}/{attempts}")
            if BASE_HEIGHTS is not None and HEIGHTS is not None:
                HEIGHTS[:, :] = BASE_HEIGHTS
                STATE["excavated_volume"] = 0.0
                refresh_sand_mesh()
            clear_real_particle_sand()
            await step_updates(3)
            rebuild_real_particle_sand()
            store_runtime_api()
            STATE["needs_reset_after_world_ready"] = not bool(STATE.get("real_sand_enabled", False))
            await step_updates(2)
            ok, last_stats = await wait_for_sand_reset_health(f"{label}_attempt{attempt}")
            if ok:
                STATE["last_reset_healthy"] = True
                STATE["last_reset_time"] = time.time()
                STATE["last_reset_label"] = str(label)
                store_runtime_api()
                update_status(f"Sand reset healthy: {label}", force=True)
                info("[SAND RESET DONE]", f"label={label}", "healthy=True", f"stats={last_stats}")
                return True
            if attempt < attempts:
                info("[SAND RESET RETRY]", f"label={label}", f"attempt={attempt}", f"reason={last_stats}")
                await step_updates(20)
        update_status(f"Sand reset unhealthy after retry: {label}", force=True)
        STATE["last_reset_healthy"] = False
        STATE["last_reset_time"] = time.time()
        STATE["last_reset_label"] = str(label)
        store_runtime_api()
        info("[SAND RESET DONE]", f"label={label}", "healthy=False", f"stats={last_stats}")
        return False
    except Exception as e:
        update_status(f"Sand reset failed: {type(e).__name__}", force=True)
        info("[WARN] [SAND RESET] failed:", type(e).__name__, e)
        return False
    finally:
        STATE["sand_reset_active"] = False


def request_sand_reset(label="ui"):
    try:
        if not bool(STATE.get("running", False)) or not simulation_timeline_is_playing():
            info("[SAND RESET] request skipped: timeline stopped or runtime stopped", f"label={label}")
            return False
        asyncio.ensure_future(reset_sand_surface_stably(label))
        return True
    except Exception as e:
        info("[WARN] [SAND RESET] request failed:", type(e).__name__, e)
        return False


def particle_surface_height_at_xy(x, y, radius=None, fallback_reference=True):
    points = current_real_particle_positions()
    if points.shape[0] == 0:
        return height_from_grid(x, y) if fallback_reference else float("-inf")
    r = float(radius) if radius is not None else max(float(PARTICLE_DIGGABLE_SPACING) * 1.5, float(PARTICLE_RADIUS) * 3.0)
    dx = points[:, 0] - float(x)
    dy = points[:, 1] - float(y)
    mask = (dx * dx + dy * dy) <= r * r
    if not np.any(mask):
        return height_from_grid(x, y) if fallback_reference else float("-inf")
    return float(np.max(points[mask, 2]))


def particle_surface_heightmap(res=32, fallback_to_floor=True):
    res = max(4, min(256, int(res)))
    points = current_real_particle_positions()
    hm = np.full((res, res), np.nan, dtype=np.float32)
    x_min, x_max, y_min, y_max = current_sand_footprint_bbox_xy()
    if points.shape[0] > 0 and x_max > x_min and y_max > y_min:
        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
        )
        p = points[mask]
        if p.shape[0] > 0:
            ix = np.clip(((p[:, 0] - x_min) / max(1.0e-6, x_max - x_min) * res).astype(np.int32), 0, res - 1)
            iy = np.clip(((p[:, 1] - y_min) / max(1.0e-6, y_max - y_min) * res).astype(np.int32), 0, res - 1)
            for gx, gy, gz in zip(ix, iy, p[:, 2]):
                if np.isnan(hm[gy, gx]) or float(gz) > float(hm[gy, gx]):
                    hm[gy, gx] = float(gz)
    if fallback_to_floor:
        hm = np.where(np.isnan(hm), float(SAND_FLOOR_Z), hm).astype(np.float32)
    return hm


def particle_excavated_volume(res=32):
    res = max(4, min(256, int(res)))
    hm = particle_surface_heightmap(res=res, fallback_to_floor=True)
    x_min, x_max, y_min, y_max = current_sand_footprint_bbox_xy()
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    xs = np.linspace(x_min, x_max, res, dtype=np.float32)
    ys = np.linspace(y_min, y_max, res, dtype=np.float32)
    drop_sum = 0.0
    valid_cells = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if not is_inside_diggable_xy(float(x), float(y)):
                continue
            reference_z = float(initial_sand_height_xy(float(x), float(y)))
            live_z = float(hm[j, i])
            drop_sum += max(0.0, reference_z - live_z)
            valid_cells += 1
    if valid_cells <= 0:
        return 0.0
    cell_area = float((x_max - x_min) / max(1, res - 1)) * float((y_max - y_min) / max(1, res - 1))
    return float(drop_sum * cell_area)


def estimate_particle_vram_gb(count=None):
    n = int(STATE.get("estimated_particle_count", 0) if count is None else count)
    solver_state = n * 250.0
    user_buffers = n * (12.0 + 12.0 + 4.0)
    render_copy = n * 12.0
    baseline = 2.5e9
    return {
        "particles": n,
        "solver_state_gb": round(solver_state / 1.0e9, 3),
        "user_buffers_gb": round(user_buffers / 1.0e9, 3),
        "render_copy_gb": round(render_copy / 1.0e9, 3),
        "baseline_gb": round(baseline / 1.0e9, 3),
        "total_gb": round((solver_state + user_buffers + render_copy + baseline) / 1.0e9, 3),
    }


def real_sand_stats():
    points = current_real_particle_positions()
    stats = {
        "enabled": bool(STATE.get("real_sand_enabled", False)),
        "count": int(points.shape[0]),
        "estimated_count": int(STATE.get("estimated_particle_count", 0)),
        "spacing": float(PARTICLE_DIGGABLE_SPACING),
        "radius": float(PARTICLE_RADIUS),
        "mass": float(PARTICLE_MASS),
        "solver_iters": int(PARTICLE_SOLVER_POSITION_ITERATIONS),
        "fidelity": float(SAND_FIDELITY),
        "mode": sand_mode_label(SAND_FIDELITY),
        "performance_budget_target": int(STATE.get("particle_budget_target", 0)),
        "performance_budget_applied": bool(STATE.get("particle_budget_applied", False)),
        "physics_timesteps_per_second": int(PHYSX_TIMESTEPS_PER_SECOND),
        "vram_gb": estimate_particle_vram_gb(points.shape[0]),
    }
    if points.shape[0] == 0:
        stats.update({"z_min": None, "z_max": None, "in_diggable": 0, "above_ground": 0, "excavated_volume_live": 0.0})
        return stats
    in_diggable = (
        (np.abs(points[:, 0] - float(PILE_CENTER_X)) <= float(DIGGABLE_RADIUS_X))
        & (np.abs(points[:, 1] - float(PILE_CENTER_Y)) <= float(DIGGABLE_RADIUS_Y))
    )
    center_surface = float(particle_surface_height_at_xy(PILE_CENTER_X, PILE_CENTER_Y, fallback_reference=False))
    if not math.isfinite(center_surface):
        center_surface = None
    stats.update(
        {
            "z_min": float(np.min(points[:, 2])),
            "z_max": float(np.max(points[:, 2])),
            "in_diggable": int(np.count_nonzero(in_diggable)),
            "above_ground": int(np.count_nonzero(points[:, 2] > float(BASE_Z) + 0.18)),
            "surface_center_z": center_surface,
            "excavated_volume_live": float(particle_excavated_volume(res=32)),
        }
    )
    return stats


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def build_height_arrays():
    global HEIGHTS, BASE_HEIGHTS, X_VALUES, Y_VALUES, CELL_AREA
    x_min, x_max, y_min, y_max = current_sand_footprint_bbox_xy()
    X_VALUES = np.linspace(x_min, x_max, NX, dtype=np.float32)
    Y_VALUES = np.linspace(y_min, y_max, NY, dtype=np.float32)
    HEIGHTS = np.zeros((NY, NX), dtype=np.float32)
    for j, y in enumerate(Y_VALUES):
        for i, x in enumerate(X_VALUES):
            HEIGHTS[j, i] = initial_sand_height_xy(float(x), float(y))
    BASE_HEIGHTS = HEIGHTS.copy()
    CELL_AREA = float(((x_max - x_min) / max(1, NX - 1)) * ((y_max - y_min) / max(1, NY - 1)))


def refresh_sand_mesh():
    # No visual/collision height mesh is authored. Height arrays remain only as
    # planner reference data; real sand is represented by PhysX particles.
    STATE["mesh_dirty"] = False


def initialize_sand_reference_grid(root):
    build_height_arrays()
    info("Reference height grid kept for planner only; no visual/collision height mesh authored")
    info("Sand reference grid:", NX, "x", NY, "vertices=", NX * NY)
    return None


def rebuild_sand_reference_grid(label=""):
    build_height_arrays()
    refresh_sand_mesh()
    info(
        "[SAND REFERENCE GRID]",
        f"label={label}",
        "center=", np.round([PILE_CENTER_X, PILE_CENTER_Y, SAND_FLOOR_Z], 3),
        "bbox=", tuple(round(float(v), 3) for v in current_sand_footprint_bbox_xy()),
    )


def set_schema_attr(obj, names, value):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn().Set(value)
                return True
            except Exception:
                pass
    return False


def set_prim_attr(prim, name, value, type_name=None):
    try:
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            if type_name is None:
                if isinstance(value, bool):
                    type_name = Sdf.ValueTypeNames.Bool
                elif isinstance(value, int):
                    type_name = Sdf.ValueTypeNames.Int
                elif isinstance(value, float):
                    type_name = Sdf.ValueTypeNames.Float
                else:
                    type_name = Sdf.ValueTypeNames.Token
            attr = prim.CreateAttribute(name, type_name)
        attr.Set(value)
        return True
    except Exception as e:
        info("[WARN] set attr failed:", prim.GetPath(), name, e)
        return False


def with_root_edit_target(fn):
    stage = get_stage()
    old_target = stage.GetEditTarget()
    try:
        stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))
    except Exception as e:
        info("[WARN] Could not switch to root edit target:", e)
    try:
        return fn()
    finally:
        try:
            stage.SetEditTarget(old_target)
        except Exception:
            pass


def set_physx_scene_gpu_attrs(scene_prim):
    set_prim_attr(scene_prim, "physxScene:enableGPUDynamics", True, Sdf.ValueTypeNames.Bool)
    set_prim_attr(scene_prim, "physxScene:enableGpuDynamics", True, Sdf.ValueTypeNames.Bool)
    set_prim_attr(scene_prim, "physxScene:gpuDynamicsEnabled", True, Sdf.ValueTypeNames.Bool)
    set_prim_attr(scene_prim, "physxScene:broadphaseType", "GPU", Sdf.ValueTypeNames.Token)
    set_prim_attr(scene_prim, "physxScene:gpuBroadphase", True, Sdf.ValueTypeNames.Bool)
    set_prim_attr(scene_prim, "physxScene:solverType", "TGS", Sdf.ValueTypeNames.Token)
    set_prim_attr(scene_prim, "physxScene:timeStepsPerSecond", int(PHYSX_TIMESTEPS_PER_SECOND), Sdf.ValueTypeNames.Int)
    # CCD is not supported with GPU dynamics. Keeping it authored false avoids
    # a PhysX-side disable/rebuild path during particle resets.
    set_prim_attr(scene_prim, "physxScene:enableCCD", False, Sdf.ValueTypeNames.Bool)
    set_prim_attr(scene_prim, "physics:enableCCD", False, Sdf.ValueTypeNames.Bool)


def apply_api_by_names(prim, names):
    schema = get_physx_schema()
    if schema is None:
        return None
    for name in names:
        api_cls = getattr(schema, name, None)
        if api_cls is None:
            continue
        try:
            return api_cls.Apply(prim)
        except Exception:
            try:
                return api_cls(prim)
            except Exception:
                pass
    return None


def create_physx_particle_system(root):
    schema = get_physx_schema()
    if schema is None:
        raise RuntimeError("pxr.PhysxSchema is not available; enable Isaac Sim PhysX extensions")

    stage = get_stage()
    system_path = f"{root}/{PARTICLE_SYSTEM_PATH_SUFFIX}"
    system_cls = getattr(schema, "PhysxParticleSystem", None)
    if system_cls is None:
        raise RuntimeError("PhysxSchema.PhysxParticleSystem is not available in this Isaac Sim build")

    scene_prim = ensure_physics_scene()
    system = system_cls.Define(stage, system_path)
    rel_fn = getattr(system, "CreateSimulationOwnerRel", None)
    if callable(rel_fn):
        try:
            rel_fn().SetTargets([scene_prim.GetPath()])
        except Exception:
            pass
    for rel_name in ["simulationOwner", "physics:simulationOwner", "physxParticle:simulationOwner"]:
        try:
            rel = system.GetPrim().CreateRelationship(rel_name)
            rel.SetTargets([scene_prim.GetPath()])
        except Exception:
            pass
    set_schema_attr(system, ["CreateContactOffsetAttr"], float(PARTICLE_CONTACT_OFFSET))
    set_schema_attr(system, ["CreateRestOffsetAttr"], float(PARTICLE_REST_OFFSET))
    set_schema_attr(system, ["CreateParticleContactOffsetAttr"], float(PARTICLE_CONTACT_OFFSET))
    set_schema_attr(system, ["CreateSolidRestOffsetAttr"], float(PARTICLE_SOLID_REST_OFFSET))
    set_schema_attr(system, ["CreateFluidRestOffsetAttr"], float(PARTICLE_FLUID_REST_OFFSET))
    set_schema_attr(system, ["CreateSolverPositionIterationCountAttr"], int(PARTICLE_SOLVER_POSITION_ITERATIONS))
    set_schema_attr(system, ["CreateMaxVelocityAttr"], float(PARTICLE_MAX_VELOCITY))
    system_prim = system.GetPrim()
    set_prim_attr(system_prim, "physxParticle:contactOffset", float(PARTICLE_CONTACT_OFFSET), Sdf.ValueTypeNames.Float)
    set_prim_attr(system_prim, "physxParticle:restOffset", float(PARTICLE_REST_OFFSET), Sdf.ValueTypeNames.Float)
    set_prim_attr(system_prim, "physxParticle:particleContactOffset", float(PARTICLE_CONTACT_OFFSET), Sdf.ValueTypeNames.Float)
    set_prim_attr(system_prim, "physxParticle:solidRestOffset", float(PARTICLE_SOLID_REST_OFFSET), Sdf.ValueTypeNames.Float)
    set_prim_attr(system_prim, "physxParticle:fluidRestOffset", float(PARTICLE_FLUID_REST_OFFSET), Sdf.ValueTypeNames.Float)
    set_prim_attr(system_prim, "physxParticle:solverPositionIterationCount", int(PARTICLE_SOLVER_POSITION_ITERATIONS), Sdf.ValueTypeNames.Int)
    set_prim_attr(system_prim, "physxParticle:maxVelocity", float(PARTICLE_MAX_VELOCITY), Sdf.ValueTypeNames.Float)

    return system


def create_sand_particle_material(root):
    stage = get_stage()
    mat_path = f"{root}/{PARTICLE_MATERIAL_PATH_SUFFIX}"
    mat = UsdShade.Material.Define(stage, mat_path)
    prim = mat.GetPrim()

    pbd = apply_api_by_names(prim, ["PhysxPBDMaterialAPI"])
    if pbd is not None:
        set_schema_attr(pbd, ["CreateFrictionAttr"], float(PARTICLE_MATERIAL_FRICTION))
        set_schema_attr(pbd, ["CreateParticleFrictionScaleAttr"], float(PARTICLE_MATERIAL_FRICTION_SCALE))
        set_schema_attr(pbd, ["CreateDampingAttr"], float(PARTICLE_MATERIAL_DAMPING))
        set_schema_attr(pbd, ["CreateViscosityAttr"], float(PARTICLE_MATERIAL_VISCOSITY))
        set_schema_attr(pbd, ["CreateCohesionAttr"], float(PARTICLE_MATERIAL_COHESION))
        set_schema_attr(pbd, ["CreateAdhesionAttr"], float(PARTICLE_MATERIAL_ADHESION))
        set_schema_attr(pbd, ["CreateGravityScaleAttr"], float(PARTICLE_MATERIAL_GRAVITY_SCALE))
    set_prim_attr(prim, "physxPBDMaterial:friction", float(PARTICLE_MATERIAL_FRICTION), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxPBDMaterial:particleFrictionScale", float(PARTICLE_MATERIAL_FRICTION_SCALE), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxPBDMaterial:damping", float(PARTICLE_MATERIAL_DAMPING), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxPBDMaterial:viscosity", float(PARTICLE_MATERIAL_VISCOSITY), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxPBDMaterial:cohesion", float(PARTICLE_MATERIAL_COHESION), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxPBDMaterial:adhesion", float(PARTICLE_MATERIAL_ADHESION), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxPBDMaterial:gravityScale", float(PARTICLE_MATERIAL_GRAVITY_SCALE), Sdf.ValueTypeNames.Float)

    phys_mat = apply_api_by_names(prim, ["PhysxMaterialAPI", "PhysxPhysicsMaterialAPI"])
    if phys_mat is not None:
        set_schema_attr(phys_mat, ["CreateStaticFrictionAttr"], float(PARTICLE_MATERIAL_STATIC_FRICTION))
        set_schema_attr(phys_mat, ["CreateDynamicFrictionAttr"], float(PARTICLE_MATERIAL_DYNAMIC_FRICTION))
        set_schema_attr(phys_mat, ["CreateRestitutionAttr"], float(PARTICLE_MATERIAL_RESTITUTION))
    set_prim_attr(prim, "physxMaterial:staticFriction", float(PARTICLE_MATERIAL_STATIC_FRICTION), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxMaterial:dynamicFriction", float(PARTICLE_MATERIAL_DYNAMIC_FRICTION), Sdf.ValueTypeNames.Float)
    set_prim_attr(prim, "physxMaterial:restitution", float(PARTICLE_MATERIAL_RESTITUTION), Sdf.ValueTypeNames.Float)

    return mat


def sand_particle_points():
    rng = random.Random(4107)
    points = []
    floor_z = float(SAND_FLOOR_Z)
    generated_outside = 0
    candidate_count = 0
    z_min = None
    z_max = None

    for x, y in footprint_xy_samples(PARTICLE_DIGGABLE_SPACING):
        surface_z = initial_sand_height_xy(float(x), float(y))
        fill_depth = max(PARTICLE_LAYER_SPACING_Z, surface_z - floor_z)
        pile_depth = min(SAND_THICKNESS + PILE_HEIGHT, fill_depth)
        layers = max(1, int(pile_depth / PARTICLE_LAYER_SPACING_Z))
        for layer in range(layers):
            if len(points) >= PARTICLE_MAX_COUNT:
                meta = {
                    "candidate_count": candidate_count,
                    "generated_outside": generated_outside,
                    "z_min": z_min,
                    "z_max": z_max,
                    "limited": True,
                }
                velocities = [Gf.Vec3f(0.0, 0.0, 0.0)] * len(points)
                widths = [float(PARTICLE_RADIUS * 2.0)] * len(points)
                return points, velocities, widths, meta
            z = floor_z + PARTICLE_RADIUS * 1.2 + layer * PARTICLE_LAYER_SPACING_Z
            if z > surface_z + PARTICLE_RADIUS * 0.8:
                continue
            candidate_count += 1
            px = float(x) + rng.uniform(-PARTICLE_JITTER, PARTICLE_JITTER)
            py = float(y) + rng.uniform(-PARTICLE_JITTER, PARTICLE_JITTER)
            if not is_inside_diggable_xy(px, py):
                generated_outside += 1
                continue
            pz = float(z) + rng.uniform(-PARTICLE_JITTER * 0.4, PARTICLE_JITTER * 0.4)
            z_min = pz if z_min is None else min(z_min, pz)
            z_max = pz if z_max is None else max(z_max, pz)
            points.append(Gf.Vec3f(px, py, pz))

    meta = {
        "candidate_count": candidate_count,
        "generated_outside": generated_outside,
        "z_min": z_min,
        "z_max": z_max,
        "limited": False,
    }
    velocities = [Gf.Vec3f(0.0, 0.0, 0.0)] * len(points)
    widths = [float(PARTICLE_RADIUS * 2.0)] * len(points)
    return points, velocities, widths, meta


def print_particle_spacing_diagnostics():
    min_center_spacing = 2.0 * float(PARTICLE_RADIUS)
    stable = (
        float(PARTICLE_DIGGABLE_SPACING) >= min_center_spacing
        and float(PARTICLE_LAYER_SPACING_Z) >= min_center_spacing
    )
    prefix = "[PARTICLE SPACING OK]" if stable else "[WARN] [PARTICLE SPACING OVERLAP]"
    info(
        prefix,
        f"diameter={min_center_spacing:.3f}",
        f"xy_spacing={PARTICLE_DIGGABLE_SPACING:.3f}",
        f"z_spacing={PARTICLE_LAYER_SPACING_Z:.3f}",
        f"contact_offset={PARTICLE_CONTACT_OFFSET:.3f}",
        f"rest_offset={PARTICLE_REST_OFFSET:.3f}",
        f"jitter={PARTICLE_JITTER:.4f}",
        f"mass={PARTICLE_MASS:.3f}",
        f"solver_iters={PARTICLE_SOLVER_POSITION_ITERATIONS}",
        f"max_velocity={PARTICLE_MAX_VELOCITY:.1f}",
        f"max_count={PARTICLE_MAX_COUNT}",
        f"budget_target={STATE.get('particle_budget_target')}",
        f"budget_applied={STATE.get('particle_budget_applied')}",
    )


def make_real_particle_sand(root):
    STATE["real_sand_enabled"] = False
    STATE["real_sand_particle_count"] = 0
    STATE["real_sand_error"] = ""

    if not ENABLE_REAL_PARTICLE_SAND:
        STATE["real_sand_error"] = "disabled by ENABLE_REAL_PARTICLE_SAND"
        info("[WARN] Real particle sand disabled by config")
        return None

    try:
        timing_start = time.perf_counter()
        print_particle_spacing_diagnostics()
        system = create_physx_particle_system(root)
        material = create_sand_particle_material(root)
        timing_setup = time.perf_counter()
        points, velocities, widths, gen_meta = sand_particle_points()
        timing_points = time.perf_counter()
        raw_count = int(gen_meta.get("candidate_count", len(points)))
        generated_outside = int(gen_meta.get("generated_outside", 0))
        limited_by_max_count = bool(gen_meta.get("limited", False)) or len(points) >= int(PARTICLE_MAX_COUNT)
        bbox = current_sand_footprint_bbox_xy()
        footprint_vertices = int(len(current_sand_footprint_polygon_xy()))
        info(
            "[SAND GEN FOOTPRINT]",
            "source=", "mesh" if SAND_SOURCE_POLYGON_XY is not None else "point_xyz",
            "sampler=polygon_scanline",
            "particles_raw=", int(raw_count),
            "particles_kept=", int(len(points)),
            "generated_outside_footprint=", int(generated_outside),
            "footprint_vertices=", footprint_vertices,
            "projected_faces=", int(SAND_SOURCE_SELECTED_FACE_COUNT),
            "bbox=", tuple(round(float(v), 3) for v in bbox),
        )

        particles = UsdGeom.Points.Define(get_stage(), f"{root}/{PARTICLE_POINTS_PATH_SUFFIX}")
        particles.CreatePointsAttr(points)
        particles.CreateVelocitiesAttr(velocities)
        particles.CreateWidthsAttr(widths)
        prim = particles.GetPrim()
        sand_set_color(prim, (0.72, 0.56, 0.33), 1.0)
        timing_author = time.perf_counter()

        particle_set_api = apply_api_by_names(prim, ["PhysxParticleSetAPI"])
        particle_api = apply_api_by_names(prim, ["PhysxParticleAPI"])
        if particle_set_api is None and particle_api is None:
            raise RuntimeError("cannot apply PhysxParticleAPI/PhysxParticleSetAPI to sand points")

        for api in [particle_set_api, particle_api]:
            if api is None:
                continue
            set_schema_attr(api, ["CreateParticleEnabledAttr"], True)
            set_schema_attr(api, ["CreateSelfCollisionAttr"], True)
            set_schema_attr(api, ["CreateFluidAttr"], False)
            set_schema_attr(api, ["CreateParticleGroupAttr"], 0)
            set_schema_attr(api, ["CreateParticleMassAttr"], PARTICLE_MASS)

        set_prim_attr(prim, "physxParticle:particleEnabled", True, Sdf.ValueTypeNames.Bool)
        set_prim_attr(prim, "physxParticle:selfCollision", True, Sdf.ValueTypeNames.Bool)
        set_prim_attr(prim, "physxParticle:fluid", False, Sdf.ValueTypeNames.Bool)
        set_prim_attr(prim, "physxParticle:particleGroup", 0, Sdf.ValueTypeNames.Int)
        set_prim_attr(prim, "physxParticle:particleMass", float(PARTICLE_MASS), Sdf.ValueTypeNames.Float)

        rel_api = particle_api if particle_api is not None else particle_set_api
        rel_fn = getattr(rel_api, "CreateParticleSystemRel", None)
        if callable(rel_fn):
            rel_fn().SetTargets([system.GetPath()])
        rel = prim.CreateRelationship("physxParticle:particleSystem")
        rel.SetTargets([system.GetPath()])

        try:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(material)
        except Exception as e:
            info("[WARN] real sand material bind failed:", e)
        timing_physx = time.perf_counter()

        STATE["real_sand_enabled"] = True
        STATE["real_sand_particle_count"] = len(points)
        STATE["real_sand_error"] = "" if not limited_by_max_count else "limited_by_particle_max_count"
        update_particle_runtime_state()
        z_min = gen_meta.get("z_min")
        z_max = gen_meta.get("z_max")
        info("Created real PhysX particle sand:", prim.GetPath())
        info(
            "[SAND GEN TIMING]",
            f"setup_ms={(timing_setup - timing_start) * 1000.0:.1f}",
            f"points_ms={(timing_points - timing_setup) * 1000.0:.1f}",
            f"usd_author_ms={(timing_author - timing_points) * 1000.0:.1f}",
            f"physx_bind_ms={(timing_physx - timing_author) * 1000.0:.1f}",
            f"total_ms={(timing_physx - timing_start) * 1000.0:.1f}",
            f"particles={len(points)}",
        )
        info(
            "Real sand particle count:",
            len(points),
            "radius=", PARTICLE_RADIUS,
            "max_count=", PARTICLE_MAX_COUNT,
            "limited=", limited_by_max_count,
            "contact_offset=", PARTICLE_CONTACT_OFFSET,
            "rest_offset=", PARTICLE_REST_OFFSET,
            "mass=", PARTICLE_MASS,
            "solver_iters=", PARTICLE_SOLVER_POSITION_ITERATIONS,
            "max_velocity=", PARTICLE_MAX_VELOCITY,
            "friction=", PARTICLE_MATERIAL_FRICTION,
            "friction_scale=", PARTICLE_MATERIAL_FRICTION_SCALE,
            "damping=", PARTICLE_MATERIAL_DAMPING,
            "height_multiplier=", SAND_HEIGHT_MULTIPLIER,
            "range_area_fraction=", SAND_RANGE_AREA_FRACTION,
            "range_linear_scale=", SAND_RANGE_LINEAR_SCALE,
            "floor_z=", SAND_FLOOR_Z,
            f"z_range=({float(z_min):.3f},{float(z_max):.3f})" if z_min is not None and z_max is not None else "z_range=(none)",
        )
        try:
            info("Real sand applied schemas:", list(prim.GetAppliedSchemas()))
        except Exception as e:
            info("[WARN] Could not read real sand applied schemas:", e)
        return prim
    except Exception as e:
        STATE["real_sand_enabled"] = False
        STATE["real_sand_error"] = f"{type(e).__name__}: {repr(e)}"
        info("[WARN] real particle sand unavailable:", STATE["real_sand_error"])
        info("[WARN] Keeping heightfield reference surface so IK and target placement still work")
        return None


def clear_path(path):
    stage = get_stage()
    path_str = as_path_string(path)
    if get_prim(path_str).IsValid():
        stage.RemovePrim(Sdf.Path(path_str))


def clear_previous_site(root):
    stage = get_stage()
    root_str = as_path_string(root)
    if get_prim(root_str).IsValid():
        stage.RemovePrim(Sdf.Path(root_str))
        info("Removed sand site:", root_str)


def ensure_physics_scene():
    enable_physx_gpu_runtime_settings()
    stage = get_stage()
    try:
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    except Exception as e:
        info("[WARN] Could not set stage metersPerUnit=1.0:", e)
    scene_path = "/physicsScene"
    prim = get_prim(scene_path)
    if not prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, sdf_path(scene_path))
        prim = scene.GetPrim()
        info("Created physics scene:", scene_path)
    physics_scenes = []
    scene_paths = set()
    for p in stage.Traverse():
        if p.GetTypeName() == "PhysicsScene" or str(p.GetPath()) == scene_path:
            p_path = str(p.GetPath())
            if p.IsValid() and p_path not in scene_paths:
                physics_scenes.append(p)
                scene_paths.add(p_path)
    if str(prim.GetPath()) not in scene_paths:
        physics_scenes.append(prim)
    for scene_prim in physics_scenes:
        scene = UsdPhysics.Scene(scene_prim)
        try:
            scene.CreateGravityDirectionAttr().Set(GRAVITY_DIRECTION)
            scene.CreateGravityMagnitudeAttr().Set(float(GRAVITY_MAGNITUDE))
        except Exception as e:
            info("[WARN] Could not set scene gravity:", scene_prim.GetPath(), e)
        scene_api = apply_api_by_names(scene_prim, ["PhysxSceneAPI"])
        if scene_api is not None:
            set_schema_attr(scene_api, ["CreateEnableGPUDynamicsAttr"], True)
            set_schema_attr(scene_api, ["CreateGpuDynamicsEnabledAttr"], True)
            set_schema_attr(scene_api, ["CreateBroadphaseTypeAttr"], "GPU")
            set_schema_attr(scene_api, ["CreateSolverTypeAttr"], "TGS")
            set_schema_attr(scene_api, ["CreateTimeStepsPerSecondAttr"], int(PHYSX_TIMESTEPS_PER_SECOND))
            set_schema_attr(scene_api, ["CreateEnableCCDAttr", "CreateEnableCcdAttr"], False)
        set_physx_scene_gpu_attrs(scene_prim)
        gpu_attr = scene_prim.GetAttribute("physxScene:enableGPUDynamics")
        broadphase_attr = scene_prim.GetAttribute("physxScene:broadphaseType")
        gravity_attr = scene_prim.GetAttribute("physics:gravityMagnitude")
        gpu_value = gpu_attr.Get() if gpu_attr.IsValid() else None
        broadphase_value = broadphase_attr.Get() if broadphase_attr.IsValid() else None
        gravity_value = gravity_attr.Get() if gravity_attr.IsValid() else None
        info(
            "Configured PhysX scene for GPU particles:",
            scene_prim.GetPath(),
            f"gpu={gpu_value}",
            f"broadphase={broadphase_value}",
            f"gravity={gravity_value}",
        )
    return prim


def make_unload_bin(root):
    bin_root = f"{root}/UnloadBin"
    UsdGeom.Xform.Define(get_stage(), bin_root)

    cx = float(UNLOAD_BIN_CENTER[0])
    cy = float(UNLOAD_BIN_CENTER[1])
    sx = float(UNLOAD_BIN_INNER_SIZE_X)
    sy = float(UNLOAD_BIN_INNER_SIZE_Y)
    t = float(UNLOAD_BIN_WALL_THICKNESS)
    h = float(UNLOAD_BIN_WALL_HEIGHT)
    f = float(UNLOAD_BIN_FLOOR_THICKNESS)
    z = BASE_Z + 0.5 * h

    sand_make_cube(
        f"{bin_root}/Floor",
        (cx, cy, BASE_Z - 0.5 * f),
        (sx + 2.0 * t, sy + 2.0 * t, f),
        UNLOAD_BIN_FLOOR_COLOR,
        collision=True,
    )

    sand_make_cube(
        f"{bin_root}/WallLeft",
        (cx - 0.5 * sx - 0.5 * t, cy, z),
        (t, sy + 2.0 * t, h),
        UNLOAD_BIN_WALL_COLOR,
        collision=True,
    )
    sand_make_cube(
        f"{bin_root}/WallRight",
        (cx + 0.5 * sx + 0.5 * t, cy, z),
        (t, sy + 2.0 * t, h),
        UNLOAD_BIN_WALL_COLOR,
        collision=True,
    )
    sand_make_cube(
        f"{bin_root}/WallFront",
        (cx, cy - 0.5 * sy - 0.5 * t, z),
        (sx + 2.0 * t, t, h),
        UNLOAD_BIN_WALL_COLOR,
        collision=True,
    )
    sand_make_cube(
        f"{bin_root}/WallBack",
        (cx, cy + 0.5 * sy + 0.5 * t, z),
        (sx + 2.0 * t, t, h),
        UNLOAD_BIN_WALL_COLOR,
        collision=True,
    )

    info(
        "Created unload bin:",
        bin_root,
        "center=",
        np.round(UNLOAD_BIN_CENTER, 3),
        "inner_size=",
        (sx, sy),
        "wall_height=",
        h,
        "floor_thickness=",
        f,
    )
    info("Unload bin dump point:", np.round(unload_bin_dump_point(), 3))
    return bin_root


def make_sand_retaining_walls(root):
    wall_root = f"{root}/SandRetainingWalls"
    try:
        get_stage().RemovePrim(Sdf.Path(wall_root))
    except Exception:
        pass
    UsdGeom.Xform.Define(get_stage(), wall_root)

    cx = float(WALL_CENTER_X)
    cy = float(WALL_CENTER_Y)
    sx = float(WALL_INNER_SIZE_X)
    sy = float(WALL_INNER_SIZE_Y)
    t = float(SANDBOX_WALL_THICKNESS)
    h = float(SANDBOX_WALL_HEIGHT)
    z = float(WALL_BASE_Z) + 0.5 * h

    sand_make_cube(
        f"{wall_root}/WallLeft",
        (cx - 0.5 * sx - 0.5 * t, cy, z),
        (t, sy + 2.0 * t, h),
        SANDBOX_WALL_COLOR,
        collision=True,
    )
    sand_make_cube(
        f"{wall_root}/WallRight",
        (cx + 0.5 * sx + 0.5 * t, cy, z),
        (t, sy + 2.0 * t, h),
        SANDBOX_WALL_COLOR,
        collision=True,
    )
    sand_make_cube(
        f"{wall_root}/WallFront",
        (cx, cy - 0.5 * sy - 0.5 * t, z),
        (sx + 2.0 * t, t, h),
        SANDBOX_WALL_COLOR,
        collision=True,
    )
    sand_make_cube(
        f"{wall_root}/WallBack",
        (cx, cy + 0.5 * sy + 0.5 * t, z),
        (sx + 2.0 * t, t, h),
        SANDBOX_WALL_COLOR,
        collision=True,
    )

    info(
        "Created sand retaining walls:",
        wall_root,
        "inner_size=",
        (WALL_INNER_SIZE_X, WALL_INNER_SIZE_Y),
        "center=",
        (WALL_CENTER_X, WALL_CENTER_Y, WALL_BASE_Z),
        "wall_height=",
        SANDBOX_WALL_HEIGHT,
        "wall_thickness=",
        SANDBOX_WALL_THICKNESS,
    )
    return wall_root


def clean_sand_retaining_walls(root=None):
    root = root_path() if root is None else str(root)
    wall_root = f"{root}/SandRetainingWalls"
    try:
        get_stage().RemovePrim(Sdf.Path(wall_root))
        info("Cleaned sand retaining walls:", wall_root)
        update_status("Sand walls cleaned")
        return True
    except Exception as e:
        info("[WARN] clean sand retaining walls failed:", type(e).__name__, e)
        return False


def unload_bin_dump_point():
    return np.array(
        [float(UNLOAD_BIN_CENTER[0]), float(UNLOAD_BIN_CENTER[1]), BASE_Z + UNLOAD_BIN_DUMP_HEIGHT],
        dtype=np.float32,
    )


def sand_scene_context():
    live_stats = real_sand_stats()
    try:
        footprint = current_sand_footprint_polygon_xy()
        footprint_center = polygon_centroid_xy(footprint)
        footprint_min = np.min(footprint, axis=0)
        footprint_max = np.max(footprint, axis=0)
        footprint_radius = np.maximum((footprint_max - footprint_min) * 0.5, np.array([0.05, 0.05], dtype=np.float32))
    except Exception:
        footprint_center = np.array([float(PILE_CENTER_X), float(PILE_CENTER_Y)], dtype=np.float32)
        footprint_radius = np.array([float(DIGGABLE_RADIUS_X), float(DIGGABLE_RADIUS_Y)], dtype=np.float32)
    return {
        "sand_center": np.array([float(SAND_CENTER_X), float(SAND_CENTER_Y), float(SAND_FLOOR_Z)], dtype=np.float32),
        "sand_size": np.array([float(SAND_SIZE_X), float(SAND_SIZE_Y)], dtype=np.float32),
        "pile_center": np.array([float(footprint_center[0]), float(footprint_center[1]), float(SAND_FLOOR_Z)], dtype=np.float32),
        "diggable_radius": np.array([float(footprint_radius[0]), float(footprint_radius[1])], dtype=np.float32),
        "sandbox_inner_size": np.array([float(SANDBOX_INNER_SIZE_X), float(SANDBOX_INNER_SIZE_Y)], dtype=np.float32),
        "sandbox_wall_height": float(SANDBOX_WALL_HEIGHT),
        "wall_center": np.array([float(WALL_CENTER_X), float(WALL_CENTER_Y), float(WALL_BASE_Z)], dtype=np.float32),
        "wall_inner_size": np.array([float(WALL_INNER_SIZE_X), float(WALL_INNER_SIZE_Y)], dtype=np.float32),
        "sand_floor_z": float(SAND_FLOOR_Z),
        "sand_base_z": float(BASE_Z),
        "sand_fill_height": float(SANDBOX_FILL_HEIGHT),
        "live_surface_center_z": live_stats.get("surface_center_z"),
        "live_excavated_volume": live_stats.get("excavated_volume_live"),
        "live_particle_count": live_stats.get("count"),
        "estimated_vram_gb": live_stats.get("vram_gb", {}).get("total_gb"),
        "unload_bin_center": np.array(UNLOAD_BIN_CENTER, dtype=np.float32).copy(),
        "unload_bin_inner_size": np.array([float(UNLOAD_BIN_INNER_SIZE_X), float(UNLOAD_BIN_INNER_SIZE_Y)], dtype=np.float32),
        "unload_bin_wall_height": float(UNLOAD_BIN_WALL_HEIGHT),
        "unload_bin_wall_thickness": float(UNLOAD_BIN_WALL_THICKNESS),
        "unload_bin_dump_height": float(UNLOAD_BIN_DUMP_HEIGHT),
        "unload_bin_dump_point": unload_bin_dump_point(),
        "unload_bin_z_range": np.array(
            [float(BASE_Z) - 0.05, float(BASE_Z) + max(float(UNLOAD_BIN_WALL_HEIGHT) + 0.45, float(UNLOAD_BIN_DUMP_HEIGHT) * 0.65)],
            dtype=np.float32,
        ),
    }


def reset_sand_surface():
    if BASE_HEIGHTS is None or HEIGHTS is None:
        return
    HEIGHTS[:, :] = BASE_HEIGHTS
    STATE["excavated_volume"] = 0.0
    refresh_sand_mesh()
    rebuild_real_particle_sand()
    store_runtime_api()
    STATE["needs_reset_after_world_ready"] = not bool(STATE.get("real_sand_enabled", False))
    update_status("Reset real particle sand")


def clear_real_particle_sand():
    root = root_path()
    clear_path(f"{root}/{PARTICLE_POINTS_PATH_SUFFIX}")
    clear_path(f"{root}/{PARTICLE_SYSTEM_PATH_SUFFIX}")
    clear_path(f"{root}/{PARTICLE_MATERIAL_PATH_SUFFIX}")
    STATE["real_sand_enabled"] = False
    STATE["real_sand_particle_count"] = 0
    STATE["real_sand_error"] = ""
    store_runtime_api()
    update_status("Cleared real particle sand")


def rebuild_real_particle_sand():
    root = root_path()
    ensure_physics_scene()
    clear_path(f"{root}/{PARTICLE_POINTS_PATH_SUFFIX}")
    clear_path(f"{root}/{PARTICLE_SYSTEM_PATH_SUFFIX}")
    clear_path(f"{root}/{PARTICLE_MATERIAL_PATH_SUFFIX}")
    if ENABLE_REAL_PARTICLE_SAND:
        make_real_particle_sand(root)


def print_status():
    info("========== SAND SITE STATUS ==========")
    info("root:", root_path())
    info("sand_bounds:", (sand_x_min(), sand_x_max(), sand_y_min(), sand_y_max()))
    info(
        "sand_point_xyz:",
        (SAND_CENTER_X, SAND_CENTER_Y, SAND_POINT_Z),
        "amount_x:",
        SAND_AMOUNT_MULTIPLIER,
        "actual_fill_height:",
        SANDBOX_FILL_HEIGHT,
        "sand_floor_z:",
        SAND_FLOOR_Z,
        "sand_surface_base_z:",
        BASE_Z,
        "sand_thickness:",
        SAND_THICKNESS,
    )
    info(
        "sandbox:",
        "inner_size=", (SANDBOX_INNER_SIZE_X, SANDBOX_INNER_SIZE_Y),
        "fill_height=", SANDBOX_FILL_HEIGHT,
        "wall_height=", SANDBOX_WALL_HEIGHT,
        "wall_thickness=", SANDBOX_WALL_THICKNESS,
    )
    info(
        "sand_shape:",
        "height_multiplier=", SAND_HEIGHT_MULTIPLIER,
        "pile_height=", PILE_HEIGHT,
        "range_area_fraction=", SAND_RANGE_AREA_FRACTION,
        "range_linear_scale=", SAND_RANGE_LINEAR_SCALE,
        "max_excavation_depth=", MAX_EXCAVATION_DEPTH,
    )
    info("real_sand_scope:", "pile_only" if REAL_SAND_PILE_ONLY else "full_bed")
    info(
        "sand_source:",
        "mesh" if SAND_SOURCE_POLYGON_XY is not None else "point_xyz",
        "selected_path:", SAND_SOURCE_SELECTED_PATH,
        "projected_faces=", SAND_SOURCE_SELECTED_FACE_COUNT,
        "projected_faces_total=", SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT,
        "vertices:", 0 if SAND_SOURCE_POLYGON_XY is None else len(SAND_SOURCE_POLYGON_XY),
    )
    info("diggable_center:", (PILE_CENTER_X, PILE_CENTER_Y), "diggable_radius:", (DIGGABLE_RADIUS_X, DIGGABLE_RADIUS_Y))
    info("wall_point_xyz:", (WALL_CENTER_X, WALL_CENTER_Y, WALL_BASE_Z), "wall_inner_size:", (WALL_INNER_SIZE_X, WALL_INNER_SIZE_Y))
    info("real_sand_enabled:", STATE.get("real_sand_enabled"))
    info("real_sand_particle_count:", STATE.get("real_sand_particle_count"))
    info(
        "sand_fidelity:",
        f"value={SAND_FIDELITY:.2f}",
        f"mode={sand_mode_label(SAND_FIDELITY)}",
        f"estimated_particles={STATE.get('estimated_particle_count')}",
        f"budget_target={STATE.get('particle_budget_target')}",
        f"budget_applied={STATE.get('particle_budget_applied')}",
        f"physics_tps={PHYSX_TIMESTEPS_PER_SECOND}",
    )
    info("needs_reset_after_world_ready:", STATE.get("needs_reset_after_world_ready"))
    info(
        "real_sand_particle_params:",
        f"radius={PARTICLE_RADIUS}",
        f"contact_offset={PARTICLE_CONTACT_OFFSET}",
        f"rest_offset={PARTICLE_REST_OFFSET}",
        f"solid_rest_offset={PARTICLE_SOLID_REST_OFFSET}",
        f"mass={PARTICLE_MASS}",
        f"diggable_spacing={PARTICLE_DIGGABLE_SPACING}",
        f"layer_spacing={PARTICLE_LAYER_SPACING_Z}",
        f"solver_iters={PARTICLE_SOLVER_POSITION_ITERATIONS}",
        f"max_velocity={PARTICLE_MAX_VELOCITY}",
        f"friction={PARTICLE_MATERIAL_FRICTION}",
        f"friction_scale={PARTICLE_MATERIAL_FRICTION_SCALE}",
        f"damping={PARTICLE_MATERIAL_DAMPING}",
        f"max_count={PARTICLE_MAX_COUNT}",
    )
    info("real_sand_error:", STATE.get("real_sand_error"))
    info("excavated_volume:", STATE.get("excavated_volume"))
    live_stats = real_sand_stats()
    info(
        "live_sand_stats:",
        f"count={live_stats.get('count')}",
        f"surface_center_z={live_stats.get('surface_center_z')}",
        f"excavated_volume_live={live_stats.get('excavated_volume_live')}",
        f"budget_target={live_stats.get('performance_budget_target')}",
        f"budget_applied={live_stats.get('performance_budget_applied')}",
        f"physics_tps={live_stats.get('physics_timesteps_per_second')}",
        f"vram_total_gb={live_stats.get('vram_gb', {}).get('total_gb')}",
    )
    info("unload_bin_center:", np.round(UNLOAD_BIN_CENTER, 3), "feet_z:", BASE_Z)
    info("unload_bin_dump_point:", np.round(unload_bin_dump_point(), 3))
    print_physics_scene_diagnostics()
    print_particle_physics_diagnostics()
    print_real_sand_particle_diagnostics()


def print_physics_scene_diagnostics():
    stage = get_stage()
    try:
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    except Exception:
        meters_per_unit = None
    info("[PHYSICS SCENE DIAG]", "metersPerUnit=", meters_per_unit)
    count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene" and str(prim.GetPath()) != "/physicsScene":
            continue
        count += 1
        gravity_mag_attr = prim.GetAttribute("physics:gravityMagnitude")
        gravity_dir_attr = prim.GetAttribute("physics:gravityDirection")
        gpu_attr = prim.GetAttribute("physxScene:enableGPUDynamics")
        broadphase_attr = prim.GetAttribute("physxScene:broadphaseType")
        info(
            "[PHYSICS SCENE DIAG]",
            prim.GetPath(),
            "gravityMagnitude=", gravity_mag_attr.Get() if gravity_mag_attr.IsValid() else None,
            "gravityDirection=", gravity_dir_attr.Get() if gravity_dir_attr.IsValid() else None,
            "gpu=", gpu_attr.Get() if gpu_attr.IsValid() else None,
            "broadphase=", broadphase_attr.Get() if broadphase_attr.IsValid() else None,
        )
    if count == 0:
        info("[PHYSICS SCENE DIAG] no PhysicsScene found")


def print_particle_physics_diagnostics():
    root = root_path()
    system = get_prim(f"{root}/{PARTICLE_SYSTEM_PATH_SUFFIX}")
    material = get_prim(f"{root}/{PARTICLE_MATERIAL_PATH_SUFFIX}")
    if system.IsValid():
        attrs = [
            "physxParticle:contactOffset",
            "physxParticle:restOffset",
            "physxParticle:particleContactOffset",
            "physxParticle:solidRestOffset",
            "physxParticle:maxVelocity",
            "physxParticle:solverPositionIterationCount",
        ]
        values = []
        for name in attrs:
            attr = system.GetAttribute(name)
            values.append(f"{name.split(':')[-1]}={attr.Get() if attr.IsValid() else None}")
        info("[PARTICLE SYSTEM DIAG]", system.GetPath(), " ".join(values))
    else:
        info("[PARTICLE SYSTEM DIAG] particle system missing")
    if material.IsValid():
        attrs = [
            "physxPBDMaterial:gravityScale",
            "physxPBDMaterial:damping",
            "physxPBDMaterial:cohesion",
            "physxPBDMaterial:adhesion",
            "physxPBDMaterial:friction",
            "physxPBDMaterial:particleFrictionScale",
        ]
        values = []
        for name in attrs:
            attr = material.GetAttribute(name)
            values.append(f"{name.split(':')[-1]}={attr.Get() if attr.IsValid() else None}")
        info("[PARTICLE MATERIAL DIAG]", material.GetPath(), " ".join(values))
    else:
        info("[PARTICLE MATERIAL DIAG] particle material missing")


def print_real_sand_particle_diagnostics():
    root = root_path()
    prim = get_prim(f"{root}/{PARTICLE_POINTS_PATH_SUFFIX}")
    if not prim.IsValid():
        info("[REAL SAND DIAG] particle prim missing")
        return
    points_attr = prim.GetAttribute("points")
    points = points_attr.Get() if points_attr.IsValid() else None
    if points is None or len(points) == 0:
        info("[REAL SAND DIAG] no particle points")
        return
    above_ground = 0
    in_diggable = 0
    z_min = 1e9
    z_max = -1e9
    for p in points:
        pp = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
        z_min = min(z_min, float(pp[2]))
        z_max = max(z_max, float(pp[2]))
        if is_inside_diggable_xy(float(pp[0]), float(pp[1])):
            in_diggable += 1
        if pp[2] > BASE_Z + 0.18:
            above_ground += 1
    info(
        "[REAL SAND DIAG]",
        "particles=", len(points),
        f"z_range=({z_min:.3f},{z_max:.3f})",
        "in_diggable=", in_diggable,
        "above_ground=", above_ground,
    )


def store_runtime_api():
    context = sand_scene_context()
    builtins._SAND_SITE = {
        "root_path": root_path(),
        "height_fn": height_from_grid,
        "initial_height_fn": initial_sand_height_xy,
        "particle_positions_fn": current_real_particle_positions,
        "particle_surface_height_fn": particle_surface_height_at_xy,
        "particle_heightmap_fn": particle_surface_heightmap,
        "particle_excavated_volume_fn": particle_excavated_volume,
        "real_sand_stats_fn": real_sand_stats,
        "estimate_particle_vram_gb": estimate_particle_vram_gb,
        "is_inside_diggable_xy": is_inside_diggable_xy,
        "soil_depth_at_point": soil_depth_at_point,
        "reset": reset_sand_surface,
        "request_reset": request_sand_reset,
        "reset_stably": reset_sand_surface_stably,
        "reset_health_stats": sand_reset_health_stats,
        "last_reset_healthy": bool(STATE.get("last_reset_healthy", False)),
        "last_reset_time": float(STATE.get("last_reset_time", 0.0) or 0.0),
        "last_reset_label": str(STATE.get("last_reset_label", "")),
        "clear_real_particle_sand": clear_real_particle_sand,
        "rebuild_real_particle_sand": rebuild_real_particle_sand,
        "set_particle_size_and_limit": set_particle_size_and_limit,
        "set_sand_fidelity": set_sand_fidelity,
        "sand_fidelity": float(SAND_FIDELITY),
        "sand_mode_label": sand_mode_label(SAND_FIDELITY),
        "derived_particle_config": {
            "spacing_xy": float(PARTICLE_DIGGABLE_SPACING),
            "spacing_z": float(PARTICLE_LAYER_SPACING_Z),
            "radius": float(PARTICLE_RADIUS),
            "max_count": int(PARTICLE_MAX_COUNT),
            "solver_iters": int(PARTICLE_SOLVER_POSITION_ITERATIONS),
            "max_velocity": float(PARTICLE_MAX_VELOCITY),
            "mass": float(PARTICLE_MASS),
            "contact_offset": float(PARTICLE_CONTACT_OFFSET),
            "rest_offset": float(PARTICLE_REST_OFFSET),
            "solid_rest_offset": float(PARTICLE_SOLID_REST_OFFSET),
            "fluid_rest_offset": float(PARTICLE_FLUID_REST_OFFSET),
            "jitter": float(PARTICLE_JITTER),
            "estimated_particle_count": int(STATE.get("estimated_particle_count", 0)),
            "performance_budget_target": int(STATE.get("particle_budget_target", 0)),
            "performance_budget_applied": bool(STATE.get("particle_budget_applied", False)),
            "physics_timesteps_per_second": int(PHYSX_TIMESTEPS_PER_SECOND),
        },
        "unload_bin_dump_point": unload_bin_dump_point,
        "dump_point": unload_bin_dump_point(),
        "scene_context": context,
        "sand_pile_center": context["pile_center"],
        "sand_pile_radius": context["diggable_radius"],
        "unload_bin_center": context["unload_bin_center"],
        "unload_bin_inner_size": context["unload_bin_inner_size"],
        "unload_bin_z_range": context["unload_bin_z_range"],
        "get_status": lambda: dict(STATE),
        "suppress_control_ground": True,
    }
    info("Stored runtime API in builtins._SAND_SITE")


def notify_excavator_obstacle_cache_dirty(reason):
    api = getattr(builtins, "_EXCAVATOR_RUNTIME", None)
    if not isinstance(api, dict):
        return
    fn = api.get("clear_rigid_obstacle_cache")
    if not callable(fn):
        return
    try:
        fn(str(reason))
    except Exception as exc:
        info("[WARN] obstacle cache invalidate failed:", type(exc).__name__, exc)


def build_sand_site():
    root = root_path()
    clear_previous_site(root)
    ensure_physics_scene()
    UsdGeom.Xform.Define(get_stage(), root)
    make_sand_retaining_walls(root)
    apply_default_sand_source_mesh()
    initialize_sand_reference_grid(root)
    set_sand_generation_range_box("build_sand_site")
    if AUTO_CREATE_INITIAL_SAND:
        make_real_particle_sand(root)
        STATE["needs_reset_after_world_ready"] = False
    else:
        STATE["real_sand_enabled"] = False
        STATE["real_sand_particle_count"] = 0
        STATE["real_sand_error"] = "waiting for stable reset after main.py world is ready"
        STATE["needs_reset_after_world_ready"] = True
        info("[SAND SITE] initial real particle sand creation delayed until stable reset")
    make_unload_bin(root)
    store_runtime_api()
    notify_excavator_obstacle_cache_dirty("sand_site_rebuilt")
    if STATE.get("real_sand_enabled", False):
        update_status(f"Sand site ready: real PhysX particle sand ({STATE['real_sand_particle_count']} particles); unload uses unload bin")
    else:
        update_status("Sand site ready: reset sand after main.py world is ready")
        info("Real sand deferred:", STATE.get("real_sand_error"))
    info("Sand surface collision: OFF")
    info("Real particle sand:", STATE.get("real_sand_enabled"), "count=", STATE.get("real_sand_particle_count"))
    info("Hard/base floor generation: OFF")
    info("Run main.py after this script, then use target ball and dig/unload steps")


def build_ui():
    global WINDOW, STATUS_LABEL, DIRTY_LABEL, PARAM_MODELS

    def set_dirty(text="Changed, needs Apply"):
        STATE["ui_dirty"] = True
        if DIRTY_LABEL is not None:
            try:
                DIRTY_LABEL.text = ui_short_text(text, 96)
            except Exception:
                pass

    def set_clean(text="Applied"):
        STATE["ui_dirty"] = False
        if DIRTY_LABEL is not None:
            try:
                DIRTY_LABEL.text = ui_short_text(text, 96)
            except Exception:
                pass

    def reset_clicked():
        apply_parameter_models_to_globals()
        request_sand_reset("sand_site_window")
        refresh_parameter_models_from_globals()
        set_clean("Reset requested")

    def clean_sand_clicked():
        clear_real_particle_sand()
        set_sand_generation_range_box("clean_sand")
        set_clean("Sand cleaned")

    def use_point_xyz_sand_clicked():
        apply_parameter_models_to_globals()
        clear_selected_sand_source_mesh()
        refresh_parameter_models_from_globals()
        set_clean("Sand source uses point XYZ; click Reset Sand")

    def generate_walls_clicked():
        apply_parameter_models_to_globals()
        make_sand_retaining_walls(root_path())
        store_runtime_api()
        notify_excavator_obstacle_cache_dirty("sand_walls_generated")
        set_clean("Walls generated")
        update_status("Sand walls generated from wall point XYZ", force=True)

    def clean_walls_clicked():
        clean_sand_retaining_walls(root_path())
        store_runtime_api()
        notify_excavator_obstacle_cache_dirty("sand_walls_cleaned")
        set_clean("Walls cleaned")

    def apply_clicked():
        apply_parameter_models_to_globals()
        store_runtime_api()
        refresh_parameter_models_from_globals()
        set_clean("Applied; rebuild for geometry changes")
        update_status("Sand parameters applied", force=True)

    def apply_rebuild_clicked():
        apply_parameter_models_to_globals()
        build_sand_site()
        refresh_parameter_models_from_globals()
        set_clean("Applied and rebuilt; reset sand if needed")
        update_status("Sand parameters applied and site rebuilt", force=True)

    def status_clicked():
        print_status()
        set_clean(str(STATE.get("status", "Status printed")))

    PARAM_MODELS = {}

    def sync_sand_xyz_from_models_live(key=None):
        global SAND_CENTER_X, SAND_CENTER_Y, PILE_CENTER_X, PILE_CENTER_Y
        global SAND_POINT_Z, SAND_FLOOR_Z, SAND_POINT_RADIUS
        global SANDBOX_INNER_SIZE_X, SANDBOX_INNER_SIZE_Y, DIGGABLE_RADIUS_X, DIGGABLE_RADIUS_Y, PILE_SIGMA_X, PILE_SIGMA_Y
        global SAND_SOURCE_SELECTED_PATH, SAND_SOURCE_SELECTED_FACE_COUNT, SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT
        global SAND_SOURCE_RAW_FACE_POLYGONS_XY, SAND_SOURCE_FACE_POLYGONS_XY
        global SAND_SOURCE_SELECTED_HULL_XY, SAND_SOURCE_POLYGON_XY
        if bool(STATE.get("suppress_parameter_callbacks", False)):
            return
        try:
            x = float(PARAM_MODELS["sand_center_x"].get_value_as_float())
            y = float(PARAM_MODELS["sand_center_y"].get_value_as_float())
            z = float(PARAM_MODELS["sand_point_z"].get_value_as_float())
            r = max(0.05, float(PARAM_MODELS["sand_point_r"].get_value_as_float()))
        except Exception:
            return
        if key in ("sand_center_x", "sand_center_y", "sand_point_r") and SAND_SOURCE_SELECTED_HULL_XY is not None:
            SAND_SOURCE_SELECTED_PATH = ""
            SAND_SOURCE_SELECTED_FACE_COUNT = 0
            SAND_SOURCE_TOTAL_PROJECTED_FACE_COUNT = 0
            SAND_SOURCE_RAW_FACE_POLYGONS_XY = None
            SAND_SOURCE_FACE_POLYGONS_XY = None
            SAND_SOURCE_SELECTED_HULL_XY = None
            SAND_SOURCE_POLYGON_XY = None
            info("[SAND SOURCE SELECT] source=point_xyz reason=sand_xyz_slider")
        SAND_CENTER_X = x
        SAND_CENTER_Y = y
        PILE_CENTER_X = x
        PILE_CENTER_Y = y
        SAND_POINT_Z = z
        SAND_FLOOR_Z = z
        SAND_POINT_RADIUS = r
        if SAND_SOURCE_SELECTED_HULL_XY is None:
            SANDBOX_INNER_SIZE_X = 2.0 * r
            SANDBOX_INNER_SIZE_Y = 2.0 * r
            DIGGABLE_RADIUS_X = r
            DIGGABLE_RADIUS_Y = r
            PILE_SIGMA_X = max(0.05, 0.70 * r)
            PILE_SIGMA_Y = max(0.05, 0.70 * r)
        now = time.time()
        if now - float(STATE.get("last_sand_xyz_live_update", 0.0) or 0.0) < 0.05:
            return
        STATE["last_sand_xyz_live_update"] = now
        set_sand_generation_range_box("sand_xyz_live")

    def use_selected_sand_mesh_clicked():
        apply_parameter_models_to_globals()
        if use_selected_mesh_as_sand_source():
            refresh_parameter_models_from_globals()
            set_sand_generation_range_box("selected_sand_mesh_ui")
            store_runtime_api()
            refresh_parameter_models_from_globals()
            set_clean("Sand source set; click Reset Sand")

    def model_changed(_model=None, key=None):
        if bool(STATE.get("suppress_parameter_callbacks", False)):
            return
        set_dirty("Changed, needs Apply")

    def add_listener(model, key, live_fn=None):
        for fn_name in ["add_value_changed_fn", "add_value_changed_fn"]:
            try:
                fn = getattr(model, fn_name, None)
                if callable(fn):
                    def _on_change(m, k=key, lf=live_fn):
                        if bool(STATE.get("suppress_parameter_callbacks", False)):
                            return
                        if callable(lf):
                            lf(k)
                        model_changed(m, k)
                    fn(_on_change)
                    return
            except Exception:
                pass

    def add_float_cell(key, label, value, width=82):
        if key not in PARAM_MODELS:
            PARAM_MODELS[key] = ui.SimpleFloatModel(float(value))
            add_listener(PARAM_MODELS[key], key)
        with ui.VStack(width=178, height=42, spacing=1):
            ui.Label(label)
            ui.FloatField(model=PARAM_MODELS[key], width=width)

    def add_param_row(items):
        with ui.HStack(spacing=8, height=44):
            for key, label, value in items:
                add_float_cell(key, label, value)

    def section(title):
        ui.Separator()
        ui.Label(title)

    def add_sand_amount_row():
        key = "sand_amount_x"
        if key not in PARAM_MODELS:
            PARAM_MODELS[key] = ui.SimpleFloatModel(float(SAND_AMOUNT_MULTIPLIER))
            add_listener(PARAM_MODELS[key], key)
        with ui.VStack(height=48, spacing=3):
            with ui.HStack(spacing=8, height=24):
                ui.Label("Sand amount x", width=96)
                ui.FloatSlider(
                    model=PARAM_MODELS[key],
                    min=SAND_AMOUNT_MIN_MULTIPLIER,
                    max=SAND_AMOUNT_MAX_MULTIPLIER,
                    width=382,
                )
                ui.FloatField(model=PARAM_MODELS[key], width=66)
            ui.Label(
                ui_short_text(
                    "1x equals old 3m. Up to 10x uses compressed height, denser spacing, and a larger particle budget.",
                    112,
                ),
                width=610,
            )

    def add_sand_xyz_sliders():
        specs = [
            ("sand_center_x", "Sand X", SAND_CENTER_X, -20.0, 20.0),
            ("sand_center_y", "Sand Y", SAND_CENTER_Y, -20.0, 20.0),
            ("sand_point_z", "Sand Z", SAND_POINT_Z, -2.0, 8.0),
            ("sand_point_r", "Sand R", SAND_POINT_RADIUS, 0.05, 6.0),
        ]
        with ui.VStack(spacing=3):
            for key, label, value, lo, hi in specs:
                if key not in PARAM_MODELS:
                    PARAM_MODELS[key] = ui.SimpleFloatModel(float(value))
                    add_listener(PARAM_MODELS[key], key, live_fn=sync_sand_xyz_from_models_live)
                with ui.HStack(spacing=8, height=24):
                    ui.Label(label, width=72)
                    ui.FloatSlider(model=PARAM_MODELS[key], min=float(lo), max=float(hi), width=400)
                    ui.FloatField(model=PARAM_MODELS[key], width=76)
            ui.Label(
                ui_short_text(
                    "Drag XYZ/R to move the guide live. R is for point mode; Reset Sand rebuilds particles there.",
                    112,
                ),
                width=610,
            )

    def add_unload_bin_sliders():
        specs = [
            ("unload_x", "Bin X", float(UNLOAD_BIN_CENTER[0]), -20.0, 20.0),
            ("unload_y", "Bin Y", float(UNLOAD_BIN_CENTER[1]), -20.0, 20.0),
            ("unload_size_x", "Bin size X", UNLOAD_BIN_INNER_SIZE_X, 0.40, 8.0),
            ("unload_size_y", "Bin size Y", UNLOAD_BIN_INNER_SIZE_Y, 0.40, 8.0),
            ("unload_wall_height", "Wall H", UNLOAD_BIN_WALL_HEIGHT, 0.10, 2.50),
            ("unload_dump_height", "Dump H", UNLOAD_BIN_DUMP_HEIGHT, 0.20, 4.0),
        ]
        with ui.VStack(spacing=3):
            for key, label, value, lo, hi in specs:
                if key not in PARAM_MODELS:
                    PARAM_MODELS[key] = ui.SimpleFloatModel(float(value))
                    add_listener(PARAM_MODELS[key], key)
                with ui.HStack(spacing=8, height=24):
                    ui.Label(label, width=82)
                    ui.FloatSlider(model=PARAM_MODELS[key], min=float(lo), max=float(hi), width=390)
                    ui.FloatField(model=PARAM_MODELS[key], width=76)
            ui.Label(
                ui_short_text(
                    "Apply + Rebuild moves the physical UnloadBin walls/floor. Excavator unload planning uses this bin context.",
                    112,
                ),
                width=610,
            )

    def add_fidelity_row():
        key = "sand_fidelity"
        if key not in PARAM_MODELS:
            PARAM_MODELS[key] = ui.SimpleFloatModel(float(SAND_FIDELITY))
            add_listener(PARAM_MODELS[key], key)
        with ui.VStack(height=50, spacing=4):
            with ui.HStack(spacing=8, height=24):
                ui.Label("Efficiency", width=86)
                ui.FloatSlider(model=PARAM_MODELS[key], min=0.0, max=1.0, width=330)
                ui.Label("Realistic", width=78)
                ui.FloatField(model=PARAM_MODELS[key], width=64)
            ui.Label(
                ui_short_text(
                    f"Derived: spacing={PARTICLE_DIGGABLE_SPACING:.3f}, radius={PARTICLE_RADIUS:.3f}, "
                    f"solver={PARTICLE_SOLVER_POSITION_ITERATIONS}, est={STATE.get('estimated_particle_count', 0)}, "
                    f"target={STATE.get('particle_budget_target', 0)}, budget={STATE.get('particle_budget_applied', False)}",
                    110,
                ),
                width=610,
            )

    WINDOW = ui.Window("Sand Site Control", width=650, height=650)
    with WINDOW.frame:
        with ui.VStack(spacing=5):
            with ui.HStack(spacing=8, height=22):
                ui.Label("Sand Site Control", width=145, height=20)
                DIRTY_LABEL = ui.Label("Applied", width=128, height=20)
                STATUS_LABEL = ui.Label(ui_short_text(STATE.get("status", "Ready"), 70), width=350, height=20)
            with ui.HStack(spacing=6, height=26):
                ui.Button("Apply", width=82, clicked_fn=apply_clicked)
                ui.Button("Apply + Rebuild", width=132, clicked_fn=apply_rebuild_clicked)
                ui.Button("Reset Sand", width=112, clicked_fn=reset_clicked)
                ui.Button("Clean Sand", width=104, clicked_fn=clean_sand_clicked)
                ui.Button("Status", width=82, clicked_fn=status_clicked)
            with ui.ScrollingFrame(height=ui.Fraction(1)):
                with ui.VStack(spacing=5):
                    section("Sand Amount")
                    with ui.HStack(spacing=6, height=26):
                        ui.Button("Use Selected Sand Mesh", width=178, clicked_fn=use_selected_sand_mesh_clicked)
                        ui.Button("Use Point XYZ Sand", width=150, clicked_fn=use_point_xyz_sand_clicked)
                        ui.Label("Mesh affects sand footprint only, not walls", width=240)
                    add_sand_amount_row()
                    add_sand_xyz_sliders()
                    add_param_row([
                        ("sand_source_shrink_d", "mesh shrink d", SAND_SOURCE_SELECTION_EDGE_MARGIN),
                        ("sandbox_x", "source size x", SANDBOX_INNER_SIZE_X),
                        ("sandbox_y", "source size y", SANDBOX_INNER_SIZE_Y),
                    ])
                    add_param_row([
                        ("sand_thickness", "base layer", SAND_THICKNESS),
                        ("diggable_x", "active radius x", DIGGABLE_RADIUS_X),
                        ("diggable_y", "active radius y", DIGGABLE_RADIUS_Y),
                    ])

                    section("Walls")
                    with ui.HStack(spacing=6, height=26):
                        ui.Button("Generate Walls", width=128, clicked_fn=generate_walls_clicked)
                        ui.Button("Clean Walls", width=104, clicked_fn=clean_walls_clicked)
                        ui.Label("Walls use wall point XYZ and do not follow selected sand mesh", width=360)
                    add_param_row([
                        ("wall_center_x", "wall point x", WALL_CENTER_X),
                        ("wall_center_y", "wall point y", WALL_CENTER_Y),
                        ("wall_center_z", "wall point z", WALL_BASE_Z),
                    ])
                    add_param_row([
                        ("wall_size_x", "wall size x", WALL_INNER_SIZE_X),
                        ("wall_size_y", "wall size y", WALL_INNER_SIZE_Y),
                        ("wall_height", "wall height", SANDBOX_WALL_HEIGHT),
                    ])
                    add_param_row([
                        ("wall_thickness", "wall thick", SANDBOX_WALL_THICKNESS),
                        ("floor_thickness", "floor thick", SANDBOX_FLOOR_THICKNESS),
                    ])

                    section("Sand Fidelity")
                    add_fidelity_row()

                    section("Unload Bin")
                    add_unload_bin_sliders()

    builtins._SAND_SITE_UI_WINDOW = WINDOW
    WINDOW.visible = True
    try:
        WINDOW.focus()
    except Exception:
        pass


apply_sand_fidelity_to_particle_globals(SAND_FIDELITY, announce=False)
build_sand_site()
build_ui()
