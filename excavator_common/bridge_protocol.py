import base64
import json
import struct
import zlib


def recv_exact(sock, n):
    chunks = []
    remaining = int(n)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_json(sock):
    header = recv_exact(sock, 4)
    n = struct.unpack("!I", header)[0]
    data = recv_exact(sock, n)
    return json.loads(data.decode("utf-8"))


def write_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


async def async_read_json(reader):
    header = await reader.readexactly(4)
    n = struct.unpack("!I", header)[0]
    data = await reader.readexactly(n)
    return json.loads(data.decode("utf-8"))


async def async_write_json(writer, obj):
    data = json.dumps(obj).encode("utf-8")
    writer.write(struct.pack("!I", len(data)))
    writer.write(data)
    await writer.drain()


def validate_bridge_command(cmd):
    has_pos = "joint_positions" in cmd and cmd["joint_positions"] is not None
    has_vel = "joint_velocities" in cmd and cmd["joint_velocities"] is not None
    has_eff = "joint_efforts" in cmd and cmd["joint_efforts"] is not None
    if sum([has_pos, has_vel, has_eff]) > 1:
        raise ValueError("Send only one of joint_positions, joint_velocities, joint_efforts per command.")
    ticks = int(cmd.get("ticks", 4))
    cmd["ticks"] = max(1, min(ticks, 120))
    return cmd


def make_articulation_action(cmd, action_cls, np_module):
    validate_bridge_command(cmd)
    if "joint_positions" in cmd and cmd["joint_positions"] is not None:
        return action_cls(joint_positions=np_module.asarray(cmd["joint_positions"], dtype=np_module.float32))
    if "joint_velocities" in cmd and cmd["joint_velocities"] is not None:
        return action_cls(joint_velocities=np_module.asarray(cmd["joint_velocities"], dtype=np_module.float32))
    if "joint_efforts" in cmd and cmd["joint_efforts"] is not None:
        return action_cls(joint_efforts=np_module.asarray(cmd["joint_efforts"], dtype=np_module.float32))
    return None


def encode_rgb_payload(rgb, np_module=None, camera_path=None):
    if np_module is None:
        import numpy as np_module

    rgb = np_module.asarray(rgb)
    if rgb.dtype != np_module.uint8:
        rgb = np_module.clip(rgb, 0, 255).astype(np_module.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    payload = {
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "rgb_zlib_b64": base64.b64encode(zlib.compress(rgb.tobytes(), level=1)).decode("ascii"),
    }
    if camera_path is not None:
        payload["camera_path"] = str(camera_path)
    return payload


def decode_rgb_payload(payload, np_module=None):
    if np_module is None:
        import numpy as np_module

    raw = base64.b64decode(payload["rgb_zlib_b64"])
    data = zlib.decompress(raw)
    rgb = np_module.frombuffer(data, dtype=np_module.dtype(payload.get("rgb_dtype", "uint8"))).reshape(payload["rgb_shape"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    if rgb.dtype != np_module.uint8:
        rgb = np_module.clip(rgb, 0, 255).astype(np_module.uint8)
    return rgb


def decode_camera_images(reply, np_module=None):
    cameras = reply.get("cameras")
    if not isinstance(cameras, dict):
        return {"front": decode_rgb_payload(reply, np_module=np_module)}
    return {
        str(camera_name): decode_rgb_payload(camera_payload, np_module=np_module)
        for camera_name, camera_payload in cameras.items()
    }

