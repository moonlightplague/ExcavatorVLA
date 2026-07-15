#!/usr/bin/env python3

import argparse
import json
import socket
import struct
import sys
import time


def recv_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        chunk = sock.recv(
            size - len(data)
        )

        if not chunk:
            raise ConnectionError(
                "Simulator closed the connection."
            )

        data.extend(chunk)

    return bytes(data)


def send_json(sock, obj):
    payload = json.dumps(
        obj
    ).encode("utf-8")

    header = struct.pack(
        "!I",
        len(payload),
    )

    sock.sendall(
        header + payload
    )


def recv_json(sock):
    header = recv_exact(
        sock,
        4,
    )

    payload_size = struct.unpack(
        "!I",
        header,
    )[0]

    payload = recv_exact(
        sock,
        payload_size,
    )

    return json.loads(
        payload.decode("utf-8")
    )


def get_tip_z(reply):
    state = reply.get(
        "observation_state"
    )

    if (
        isinstance(state, list)
        and len(state) > 10
    ):
        return float(
            state[10]
        )

    return None


def send_command(
    sock,
    velocity,
    ticks,
):
    command = {
        "joint_velocities": velocity,
        "ticks": ticks,
    }

    send_json(
        sock,
        command,
    )

    return recv_json(
        sock
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply constant boom and arm "
            "velocities without loading a policy."
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5555,
    )

    parser.add_argument(
        "--boom-vel",
        type=float,
        default=-0.05,
        help=(
            "Boom velocity. Current observed "
            "downward direction is negative."
        ),
    )

    parser.add_argument(
        "--arm-vel",
        type=float,
        default=0.05,
        help=(
            "Arm velocity. Current test "
            "assumes positive extends/lowers."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--ticks",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    velocity = [
        0.0,
        float(args.boom_vel),
        float(args.arm_vel),
        0.0,
    ]

    print(
        "[TEST] Constant command:"
    )

    print(
        "  swing = 0.0"
    )

    print(
        f"  boom  = {velocity[1]:+.4f}"
    )

    print(
        f"  arm   = {velocity[2]:+.4f}"
    )

    print(
        "  bucket = 0.0"
    )

    print(
        f"  steps = {args.steps}"
    )

    print(
        f"  ticks = {args.ticks}"
    )

    print()

    sock = socket.create_connection(
        (
            args.host,
            args.port,
        ),
        timeout=30.0,
    )

    sock.settimeout(
        120.0
    )

    first_tip_z = None
    minimum_tip_z = None
    maximum_tip_z = None

    try:
        # Read the initial state with zero velocity.
        initial_reply = send_command(
            sock,
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            1,
        )

        first_tip_z = get_tip_z(
            initial_reply
        )

        initial_q = initial_reply.get(
            "joint_positions",
            [],
        )

        print(
            "[INITIAL]"
        )

        print(
            f"  tip_z = {first_tip_z}"
        )

        print(
            f"  q = {initial_q}"
        )

        print()

        for step in range(
            args.steps
        ):
            reply = send_command(
                sock,
                velocity,
                args.ticks,
            )

            tip_z = get_tip_z(
                reply
            )

            q = reply.get(
                "joint_positions",
                [],
            )

            qd = reply.get(
                "joint_velocities",
                [],
            )

            if tip_z is not None:

                if minimum_tip_z is None:
                    minimum_tip_z = tip_z
                    maximum_tip_z = tip_z

                minimum_tip_z = min(
                    minimum_tip_z,
                    tip_z,
                )

                maximum_tip_z = max(
                    maximum_tip_z,
                    tip_z,
                )

            if (
                step % args.print_every
                == 0
            ):
                if (
                    tip_z is not None
                    and first_tip_z is not None
                ):
                    delta_z = (
                        tip_z
                        - first_tip_z
                    )

                    delta_text = (
                        f"{delta_z:+.5f}"
                    )

                else:
                    delta_text = "N/A"

                print(
                    f"[STEP {step:04d}] "
                    f"tip_z={tip_z} "
                    f"delta_z={delta_text} "
                    f"q={q} "
                    f"qd={qd}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print(
            "\n[STOP] Ctrl+C received."
        )

    finally:
        # Explicitly stop every joint.
        try:
            stop_reply = send_command(
                sock,
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                1,
            )

            print()
            print(
                "[STOP] Zero velocity sent."
            )

            print(
                "[STOP] Final q:",
                stop_reply.get(
                    "joint_positions",
                    [],
                ),
            )

            print(
                "[STOP] Final tip_z:",
                get_tip_z(
                    stop_reply
                ),
            )

        except Exception as exc:
            print(
                "[WARN] Could not send "
                f"stop command: {exc}"
            )

        sock.close()

    print()

    if (
        first_tip_z is not None
        and minimum_tip_z is not None
    ):
        print(
            "[SUMMARY]"
        )

        print(
            f"  initial tip_z = "
            f"{first_tip_z:.5f}"
        )

        print(
            f"  minimum tip_z = "
            f"{minimum_tip_z:.5f}"
        )

        print(
            f"  maximum tip_z = "
            f"{maximum_tip_z:.5f}"
        )

        print(
            f"  deepest delta = "
            f"{minimum_tip_z - first_tip_z:+.5f}"
        )


if __name__ == "__main__":
    main()
