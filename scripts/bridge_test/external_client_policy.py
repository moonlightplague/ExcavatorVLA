# 03_external_client_policy.py
#
# Run this outside Isaac Sim, for example in WSL:
#   python 03_external_client_policy.py --host 127.0.0.1 --port 5555
#
# If WSL cannot connect to Windows Isaac Sim, try:
#   cat /etc/resolv.conf | grep nameserver
# Then use that IP:
#   python 03_external_client_policy.py --host <windows_host_ip> --port 5555

import argparse
import os
import socket
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from excavator_common.bridge_protocol import (
    decode_rgb_payload as decode_rgb,
    read_json as recv_json,
    write_json as send_json,
)


JOINT_ORDER_HINT = ["swing", "boom", "arm", "bucket"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--save-debug-image", action="store_true")
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        target = None

        for t in range(args.steps):
            if target is None:
                cmd = {"ticks": args.ticks}
            else:
                cmd = {
                    "ticks": args.ticks,
                    "joint_positions": target.tolist(),
                }

            send_json(sock, cmd)
            reply = recv_json(sock)

            q = np.asarray(reply["joint_positions"], dtype=np.float32)
            qd = np.asarray(reply["joint_velocities"], dtype=np.float32)
            rgb = decode_rgb(reply)

            print(
                f"t={t:04d} "
                f"q={np.round(q, 3)} "
                f"qd={np.round(qd, 3)} "
                f"rgb={rgb.shape}"
            )

            if t == 0 and args.save_debug_image:
                try:
                    from PIL import Image
                    Image.fromarray(rgb).save("debug_rgb.png")
                    print("Saved debug_rgb.png")
                except Exception as exc:
                    print("Could not save debug_rgb.png:", exc)

            # Replace this block with your VLA / policy.
            # Joint order for the cleaned URDF is expected to be:
            #   [swing, boom, arm, bucket]
            target = q.copy()

            if len(target) >= 4:
                # A tiny safe demo motion.
                target[0] = 0.2 * np.sin(t * 0.03)       # swing
                target[1] = 0.3 + 0.2 * np.sin(t * 0.02) # boom
                target[2] = -0.2                         # arm
                target[3] = 0.3                          # bucket

            time.sleep(0.01)


if __name__ == "__main__":
    main()
