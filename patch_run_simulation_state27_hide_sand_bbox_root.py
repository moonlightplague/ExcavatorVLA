#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

DEFAULT_TARGET = Path(
    "/root/isaacsim/ExcavatorVLA/run_simulation.py"
)

HIDE_FUNC = '    def hide_sand_source_guides():\n        """Hide sand BBox/debug visuals while preserving particle physics."""\n        hidden = []\n        preserved = []\n        tokens = (\n            "bbox", "bounding_box", "boundingbox", "bounding",\n            "bounds", "aabb", "range", "guide", "selection", "debug",\n        )\n\n        def contains_particle_geometry(root_prim):\n            try:\n                for descendant in Usd.PrimRange(root_prim):\n                    if descendant == root_prim:\n                        continue\n                    if (\n                        str(descendant.GetTypeName()) == "PhysxParticleSystem"\n                        or descendant.IsA(UsdGeom.Points)\n                    ):\n                        return True\n            except Exception:\n                pass\n            return False\n\n        for prim in stage.Traverse():\n            path = str(prim.GetPath())\n            lower_path = path.lower()\n            type_name = str(prim.GetTypeName())\n\n            if "sand" not in lower_path:\n                continue\n            if not any(token in lower_path for token in tokens):\n                continue\n            if type_name == "PhysxParticleSystem" or prim.IsA(UsdGeom.Points):\n                preserved.append(path)\n                continue\n            if type_name == "Xform" and contains_particle_geometry(prim):\n                preserved.append(path)\n                continue\n\n            try:\n                imageable = UsdGeom.Imageable(prim)\n                if imageable:\n                    imageable.MakeInvisible()\n                    hidden.append(path)\n            except Exception as exc:\n                print(\n                    "[SAND] Failed to hide visual helper:",\n                    path,\n                    repr(exc),\n                    flush=True,\n                )\n\n        print("[SAND] Hidden guide/BBox visuals:", hidden, flush=True)\n        if preserved:\n            print(\n                "[SAND] Preserved particle-related prims:",\n                preserved,\n                flush=True,\n            )\n        return hidden\n\n    # Import bridge server components.'
HELPER_CODE = '\n    # -----------------------------------------------------------------\n    # State27 deployment features. phase_index was removed from the model\n    # input and is supervised separately.\n    # -----------------------------------------------------------------\n    STATE27_NAMES = [\n        "base_x", "base_y", "base_yaw",\n        "swing", "boom", "arm", "bucket",\n        "bucket_load_estimate",\n        "bucket_tip_x", "bucket_tip_y", "bucket_tip_z",\n        "bucket_load_x", "bucket_load_y", "bucket_load_z",\n        "swing_velocity", "boom_velocity", "arm_velocity", "bucket_velocity",\n        "dig_target_local_x", "dig_target_local_y", "dig_target_local_z",\n        "unload_landing_local_x", "unload_landing_local_y",\n        "unload_landing_local_z",\n        "truck_heading_relative_sin", "truck_heading_relative_cos",\n        "bucket_load_rate",\n    ]\n\n    def quaternion_wxyz_to_yaw(orientation):\n        quat = np.asarray(orientation, dtype=np.float64).reshape(-1)\n        if quat.size < 4 or not np.all(np.isfinite(quat[:4])):\n            return 0.0\n        w, x, y, z = [float(value) for value in quat[:4]]\n        return math.atan2(\n            2.0 * (w * z + x * y),\n            1.0 - 2.0 * (y * y + z * z),\n        )\n\n    def wrap_angle_radians(angle):\n        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))\n\n    def world_point_to_robot_local_feature(point_world, base_world, base_yaw):\n        point = np.asarray(point_world, dtype=np.float32).reshape(-1)\n        base = np.asarray(base_world, dtype=np.float32).reshape(-1)\n        if point.size < 3 or base.size < 2:\n            raise RuntimeError(\n                f"Invalid target/base point: point={point}, base={base}"\n            )\n        dx = float(point[0] - base[0])\n        dy = float(point[1] - base[1])\n        c = math.cos(float(base_yaw))\n        s = math.sin(float(base_yaw))\n        return np.asarray(\n            [c * dx + s * dy, -s * dx + c * dy, float(point[2])],\n            dtype=np.float32,\n        )\n\n    def prim_world_yaw(prim):\n        if prim is None or not prim.IsValid():\n            return 0.0\n        try:\n            matrix = UsdGeom.Xformable(\n                prim\n            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())\n            quat = matrix.ExtractRotationQuat()\n            imag = quat.GetImaginary()\n            w = float(quat.GetReal())\n            x, y, z = [float(imag[i]) for i in range(3)]\n            return math.atan2(\n                2.0 * (w * z + x * y),\n                1.0 - 2.0 * (y * y + z * z),\n            )\n        except Exception:\n            return 0.0\n\n    def prim_world_aligned_bbox(prim):\n        if prim is None or not prim.IsValid():\n            return None\n        try:\n            cache = UsdGeom.BBoxCache(\n                Usd.TimeCode.Default(),\n                [\n                    UsdGeom.Tokens.default_,\n                    UsdGeom.Tokens.render,\n                    UsdGeom.Tokens.proxy,\n                ],\n                useExtentsHint=True,\n            )\n            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()\n            minimum = aligned.GetMin()\n            maximum = aligned.GetMax()\n            result = np.asarray(\n                [\n                    [float(minimum[0]), float(minimum[1]), float(minimum[2])],\n                    [float(maximum[0]), float(maximum[1]), float(maximum[2])],\n                ],\n                dtype=np.float32,\n            )\n            return result if np.all(np.isfinite(result)) else None\n        except Exception:\n            return None\n\n    def resolve_truck_prim():\n        for candidate_path in (\n            "/World/truck", "/World/Truck",\n            "/World/DumpTruck", "/World/dump_truck",\n        ):\n            candidate = stage.GetPrimAtPath(candidate_path)\n            if candidate.IsValid():\n                return candidate\n\n        candidates = []\n        for prim in stage.Traverse():\n            path = str(prim.GetPath()).lower()\n            if "truck" in path and prim.GetTypeName() == "Xform":\n                candidates.append(prim)\n        candidates.sort(key=lambda prim: len(str(prim.GetPath())))\n        return candidates[0] if candidates else None\n\n    def resolve_unload_landing_world(truck_prim):\n        preferred = []\n        if truck_prim is not None and truck_prim.IsValid():\n            for prim in Usd.PrimRange(truck_prim):\n                lower_path = str(prim.GetPath()).lower()\n                if not any(\n                    token in lower_path\n                    for token in ("bed", "bin", "cargo", "container", "dump")\n                ):\n                    continue\n                bbox = prim_world_aligned_bbox(prim)\n                if bbox is None:\n                    continue\n                extent = bbox[1] - bbox[0]\n                area = max(0.0, float(extent[0] * extent[1]))\n                if area > 0.0:\n                    preferred.append((area, len(lower_path), prim, bbox))\n\n        if preferred:\n            preferred.sort(key=lambda item: (-item[0], item[1]))\n            _, _, selected_prim, bbox = preferred[0]\n            extent = bbox[1] - bbox[0]\n            point = 0.5 * (bbox[0] + bbox[1])\n            point[2] = bbox[0, 2] + 0.15 * max(0.0, float(extent[2]))\n            return point.astype(np.float32), str(selected_prim.GetPath())\n\n        bbox = prim_world_aligned_bbox(truck_prim)\n        if bbox is not None:\n            extent = bbox[1] - bbox[0]\n            point = 0.5 * (bbox[0] + bbox[1])\n            point[2] = bbox[0, 2] + 0.55 * max(0.0, float(extent[2]))\n            return point.astype(np.float32), "truck_bbox_fallback"\n\n        return (\n            np.asarray([4.14439, 6.72012, 4.222683], dtype=np.float32),\n            "configured_fallback",\n        )\n\n    def resolve_dig_target_world(sand_module, sand_api):\n        center_x = float(\n            getattr(sand_module, "SAND_CENTER_X", SAND_INITIAL_CENTER[0])\n        )\n        center_y = float(\n            getattr(sand_module, "SAND_CENTER_Y", SAND_INITIAL_CENTER[1])\n        )\n        target_z = 0.35\n        source = "configured_sand_center"\n\n        positions_fn = (\n            sand_api.get("particle_positions_fn")\n            if isinstance(sand_api, dict)\n            else None\n        )\n        if callable(positions_fn):\n            try:\n                points = np.asarray(\n                    positions_fn(),\n                    dtype=np.float32,\n                ).reshape(-1, 3)\n                if len(points) > 0:\n                    radius2 = (\n                        np.square(points[:, 0] - center_x)\n                        + np.square(points[:, 1] - center_y)\n                    )\n                    local_points = points[radius2 <= 1.0]\n                    if len(local_points) < 32:\n                        local_points = points\n                    target_z = float(np.percentile(local_points[:, 2], 65.0))\n                    source = "sand_particle_percentile65"\n            except Exception as exc:\n                print(\n                    "[STATE27] dig target particle lookup failed:",\n                    repr(exc),\n                    flush=True,\n                )\n\n        return (\n            np.asarray([center_x, center_y, target_z], dtype=np.float32),\n            source,\n        )\n\n    truck_prim_for_state = resolve_truck_prim()\n    unload_landing_world, unload_landing_source = (\n        resolve_unload_landing_world(truck_prim_for_state)\n    )\n    truck_yaw_world = prim_world_yaw(truck_prim_for_state)\n    dig_target_world, dig_target_source = resolve_dig_target_world(\n        sand_module,\n        sand_api,\n    )\n\n    print(\n        "[STATE27] fixed environment features:",\n        f"dig_target_world={dig_target_world.tolist()}",\n        f"dig_source={dig_target_source}",\n        f"unload_landing_world={unload_landing_world.tolist()}",\n        f"unload_source={unload_landing_source}",\n        f"truck_prim={str(truck_prim_for_state.GetPath()) if truck_prim_for_state is not None and truck_prim_for_state.IsValid() else None}",\n        f"truck_yaw={truck_yaw_world:.6f}",\n        flush=True,\n    )\n\n'
STATE_BLOCK = '            # Compute the exact 27D observation.state used by the final\n            # checkpoint. phase_index is not an input feature.\n            try:\n                base_pos, base_ori = robot.get_world_pose()\n                base_pos = np.asarray(base_pos, dtype=np.float32).reshape(-1)\n                base_x = float(base_pos[0])\n                base_y = float(base_pos[1])\n                base_yaw = quaternion_wxyz_to_yaw(base_ori)\n            except Exception as exc:\n                print(\n                    "[WARN] Failed to read robot base pose:",\n                    repr(exc),\n                    flush=True,\n                )\n                base_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)\n                base_x = base_y = base_yaw = 0.0\n\n            joint_positions = np.asarray(q, dtype=np.float32).reshape(-1)\n            joint_velocities_state = np.asarray(\n                qd, dtype=np.float32\n            ).reshape(-1)\n            if joint_positions.size < 4:\n                raise RuntimeError(\n                    f"State27 requires four joint positions, got "\n                    f"{joint_positions.shape}"\n                )\n            if joint_velocities_state.size < 4:\n                raise RuntimeError(\n                    f"State27 requires four joint velocities, got "\n                    f"{joint_velocities_state.shape}"\n                )\n            joint_positions = joint_positions[:4]\n            joint_velocities_state = joint_velocities_state[:4]\n\n            bucket_load_metrics = estimate_bucket_load_particles()\n            bucket_load_count = float(bucket_load_metrics["count"])\n            if previous_bucket_load_count is None:\n                bucket_load_rate = 0.0\n            else:\n                bucket_load_rate = (\n                    bucket_load_count - float(previous_bucket_load_count)\n                ) / max(float(bridge_step_seconds), 1.0e-8)\n            previous_bucket_load_count = bucket_load_count\n\n            tip_xyz = [0.0, 0.0, 0.0]\n            load_xyz = [0.0, 0.0, 0.0]\n            try:\n                bucket_prim = stage.GetPrimAtPath(\n                    "/World/URDF_real3/bucket_link"\n                )\n                if bucket_prim.IsValid():\n                    world_xf = UsdGeom.Xformable(\n                        bucket_prim\n                    ).ComputeLocalToWorldTransform(\n                        Usd.TimeCode.Default()\n                    )\n                    tip_world = world_xf.Transform(\n                        Gf.Vec3d(0.75, 0.0, -0.18)\n                    )\n                    load_world = world_xf.Transform(\n                        Gf.Vec3d(0.35, 0.0, 0.08)\n                    )\n                    tip_xyz = [\n                        float(tip_world[0]),\n                        float(tip_world[1]),\n                        float(tip_world[2]),\n                    ]\n                    load_xyz = [\n                        float(load_world[0]),\n                        float(load_world[1]),\n                        float(load_world[2]),\n                    ]\n            except Exception as exc:\n                print(\n                    "[WARN] Failed to compute bucket points:",\n                    repr(exc),\n                    flush=True,\n                )\n\n            dig_target_local = world_point_to_robot_local_feature(\n                dig_target_world, base_pos, base_yaw\n            )\n            unload_landing_local = world_point_to_robot_local_feature(\n                unload_landing_world, base_pos, base_yaw\n            )\n            relative_heading = wrap_angle_radians(\n                float(truck_yaw_world) - float(base_yaw)\n            )\n\n            observation_state = [\n                base_x,\n                base_y,\n                base_yaw,\n                float(joint_positions[0]),\n                float(joint_positions[1]),\n                float(joint_positions[2]),\n                float(joint_positions[3]),\n                bucket_load_count,\n                float(tip_xyz[0]),\n                float(tip_xyz[1]),\n                float(tip_xyz[2]),\n                float(load_xyz[0]),\n                float(load_xyz[1]),\n                float(load_xyz[2]),\n                float(joint_velocities_state[0]),\n                float(joint_velocities_state[1]),\n                float(joint_velocities_state[2]),\n                float(joint_velocities_state[3]),\n                float(dig_target_local[0]),\n                float(dig_target_local[1]),\n                float(dig_target_local[2]),\n                float(unload_landing_local[0]),\n                float(unload_landing_local[1]),\n                float(unload_landing_local[2]),\n                math.sin(relative_heading),\n                math.cos(relative_heading),\n                float(bucket_load_rate),\n            ]\n            if len(observation_state) != 27:\n                raise RuntimeError(\n                    f"State27 construction produced "\n                    f"{len(observation_state)} values"\n                )\n            if not np.all(\n                np.isfinite(np.asarray(observation_state, dtype=np.float32))\n            ):\n                raise RuntimeError(\n                    f"State27 contains NaN/Inf: {observation_state}"\n                )\n\n            task_text ='


def exact(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one match, found {count}"
        )
    return text.replace(old, new, 1)


def regex(text, pattern, new, label):
    result, count = re.subn(
        pattern, new, text, count=1, flags=re.DOTALL
    )
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one match, found {count}"
        )
    return result


def patch(target):
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)

    text = target.read_text(encoding="utf-8")
    if "excavator_state27_no_phase_v1" in text:
        raise RuntimeError("State27 patch is already present")

    backup = target.with_suffix(target.suffix + ".before_state27_bbox")
    if not backup.exists():
        shutil.copy2(target, backup)
        print("[OK] backup:", backup)

    text = regex(
        text,
        r"    def hide_sand_source_guides\(\):.*?    # Import bridge server components\.",
        HIDE_FUNC,
        "replace sand BBox hiding",
    )

    text = exact(
        text,
        "    # The standalone launcher intentionally does not import the full excavator\n",
        "    # Hide sand BBox/range/debug visuals after particle settling.\n"
        "    hidden_sand_visuals = hide_sand_source_guides()\n\n"
        "    # The standalone launcher intentionally does not import the full excavator\n",
        "call sand BBox hiding",
    )

    text = exact(
        text,
        "    # TCP Bridge Server functions.\n",
        HELPER_CODE + "    # TCP Bridge Server functions.\n",
        "insert State27 helpers",
    )

    text = exact(
        text,
        "    active_contract = None\n"
        "    active_connection_id = None\n"
        "    tick_scheduler = None\n",
        "    active_contract = None\n"
        "    active_connection_id = None\n"
        "    tick_scheduler = None\n"
        "    previous_bucket_load_count = None\n",
        "add bucket-load-rate state",
    )

    text = exact(
        text,
        '                    tick_scheduler = PhysicsTickScheduler(active_contract["training_fps"])\n'
        "                    reply = {\n",
        '                    tick_scheduler = PhysicsTickScheduler(active_contract["training_fps"])\n'
        "                    previous_bucket_load_count = None\n"
        "                    reply = {\n",
        "reset bucket load rate on handshake",
    )

    text = regex(
        text,
        r"            # Compute full 14D observation_state matching dataset schema:.*?            task_text =",
        STATE_BLOCK,
        "replace 14D state with State27",
    )

    text = exact(
        text,
        '                "observation_state": observation_state,\n'
        '                "observation_effort": _read_bridge_measured_effort(\n',
        '                "observation_state": observation_state,\n'
        '                "observation_state_schema": "excavator_state27_no_phase_v1",\n'
        '                "observation_state_names": STATE27_NAMES,\n'
        '                "observation_state_dim": 27,\n'
        '                "observation_effort": _read_bridge_measured_effort(\n',
        "add reply schema metadata",
    )

    text = exact(
        text,
        '        "camera_mode": "per_camera_persistent_viewport",\n'
        "    }\n",
        '        "camera_mode": "per_camera_persistent_viewport",\n'
        '        "observation_state_schema": "excavator_state27_no_phase_v1",\n'
        '        "observation_state_names": list(STATE27_NAMES),\n'
        '        "observation_state_dim": 27,\n'
        '        "sand_hidden_visuals": list(hidden_sand_visuals),\n'
        "    }\n",
        "add runtime metadata",
    )

    target.write_text(text, encoding="utf-8")
    print("[OK] patched:", target)
    print("[INFO] observation.state: 27D")
    print("[INFO] observation.effort: separate 4D")
    print("[INFO] sand BBox/debug visuals: hidden")


def restore(target):
    target = target.expanduser().resolve()
    backup = target.with_suffix(target.suffix + ".before_state27_bbox")
    if not backup.is_file():
        raise FileNotFoundError(backup)
    shutil.copy2(backup, target)
    print("[OK] restored:", target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target", nargs="?", type=Path, default=DEFAULT_TARGET
    )
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    restore(args.target) if args.restore else patch(args.target)


if __name__ == "__main__":
    main()
