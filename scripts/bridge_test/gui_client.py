# 04_gui_client.py
#
# Simple pygame GUI to teleoperate the excavator over the same socket protocol
# as 03_external_client_policy.py. Isaac Sim must already be running its server.
#
#   conda activate pygame
#   python 04_gui_client.py --host 127.0.0.1 --port 5555

import argparse
import os
import socket
import sys

import numpy as np
import pygame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from excavator_common.bridge_protocol import (
    decode_camera_images,
    read_json as recv_json,
    write_json as send_json,
)

JOINT_NAMES = ["swing", "boom", "arm", "bucket"]
JOINT_LIMITS = {
    "swing": (-3.14, 3.14),
    "boom": (-1.57, 1.57),
    "arm": (-1.57, 1.57),
    "bucket": (-1.57, 1.57),
}
DEFAULT_JOINTS = [0.0, 0.3, -0.2, 0.3]
STEP = 0.05

CAMERA_NAMES = ["left", "front", "right"]
CAMERA_ALIASES = {
    "left": ("left", "0"),
    "front": ("front", "1"),
    "right": ("right", "2"),
}
CAMERA_W, CAMERA_H = 320, 240
RGB_W, RGB_H = CAMERA_W * len(CAMERA_NAMES), CAMERA_H
PANEL_H = 180
WIN_W = RGB_W
WIN_H = RGB_H + PANEL_H

BTN_W, BTN_H = 48, 32
ROW_H = 36
MARGIN = 12

BG = (24, 24, 28)
FG = (230, 230, 230)
BTN = (60, 60, 70)
BTN_HOVER = (90, 90, 105)
ACCENT = (80, 160, 255)
ERR = (220, 80, 80)


def clamp_joint(name, value):
    lo, hi = JOINT_LIMITS[name]
    return float(np.clip(value, lo, hi))


class Button:
    def __init__(self, rect, label, callback):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.hover = False

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surf, font):
        color = BTN_HOVER if self.hover else BTN
        pygame.draw.rect(surf, color, self.rect, border_radius=4)
        text = font.render(self.label, True, FG)
        surf.blit(text, text.get_rect(center=self.rect.center))


class GuiClient:
    def __init__(self, host, port, ticks):
        self.host = host
        self.port = port
        self.ticks = ticks
        self.sock = None
        self.status = "Connecting..."
        self.joints = DEFAULT_JOINTS.copy()
        self.actual = self.joints.copy()
        self.velocities = [0.0] * 4
        self.camera_surfaces = {}
        self.buttons = []
        self._build_buttons()

    def _build_buttons(self):
        self.buttons.clear()
        y0 = RGB_H + MARGIN
        for i, name in enumerate(JOINT_NAMES):
            y = y0 + i * ROW_H
            minus_x = 120
            plus_x = WIN_W - MARGIN - BTN_W
            self.buttons.append(
                Button((minus_x, y, BTN_W, BTN_H), "-", lambda idx=i: self.nudge(idx, -STEP))
            )
            self.buttons.append(
                Button((plus_x, y, BTN_W, BTN_H), "+", lambda idx=i: self.nudge(idx, +STEP))
            )

    def nudge(self, idx, delta):
        name = JOINT_NAMES[idx]
        self.joints[idx] = clamp_joint(name, self.joints[idx] + delta)

    def connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=5)
            self.sock.settimeout(2.0)
            self.status = f"Connected {self.host}:{self.port}"
            return True
        except OSError as exc:
            self.status = f"Connect failed: {exc}"
            self.sock = None
            return False

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def step_sim(self):
        if self.sock is None:
            return
        try:
            send_json(
                self.sock,
                {"ticks": self.ticks, "joint_positions": self.joints},
            )
            reply = recv_json(self.sock)
            self.actual = list(map(float, reply["joint_positions"]))
            self.velocities = list(map(float, reply["joint_velocities"]))
            camera_images = decode_camera_images(reply)
            self.camera_surfaces = {}
            for camera_name, rgb in camera_images.items():
                rgb = np.ascontiguousarray(rgb)
                surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                self.camera_surfaces[camera_name] = pygame.transform.smoothscale(
                    surface,
                    (CAMERA_W, CAMERA_H),
                )
            camera_list = ",".join(sorted(self.camera_surfaces.keys()))
            self.status = f"Connected {self.host}:{self.port} cameras={camera_list}"
        except OSError as exc:
            self.status = f"Lost connection: {exc}"
            self.close()

    def handle_key(self, key):
        key_map = {
            pygame.K_a: (0, -STEP),
            pygame.K_q: (0, +STEP),
            pygame.K_s: (1, -STEP),
            pygame.K_w: (1, +STEP),
            pygame.K_d: (2, -STEP),
            pygame.K_e: (2, +STEP),
            pygame.K_f: (3, -STEP),
            pygame.K_r: (3, +STEP),
        }
        if key in key_map:
            idx, delta = key_map[key]
            self.nudge(idx, delta)

    def draw(self, surf, font, small_font):
        surf.fill(BG)

        for i, camera_name in enumerate(CAMERA_NAMES):
            x = i * CAMERA_W
            rect = pygame.Rect(x, 0, CAMERA_W, CAMERA_H)
            pygame.draw.rect(surf, (12, 12, 16), rect)

            camera_surface = None
            for key in CAMERA_ALIASES.get(camera_name, (camera_name,)):
                camera_surface = self.camera_surfaces.get(key)
                if camera_surface is not None:
                    break
            if camera_surface is not None:
                surf.blit(camera_surface, rect.topleft)
            else:
                placeholder = font.render(f"Waiting for {camera_name}...", True, FG)
                surf.blit(placeholder, placeholder.get_rect(center=rect.center))

            pygame.draw.rect(surf, (50, 50, 58), rect, width=1)
            label_bg = pygame.Rect(x + 8, 8, 88, 24)
            pygame.draw.rect(surf, (0, 0, 0), label_bg, border_radius=3)
            label = small_font.render(camera_name, True, FG)
            surf.blit(label, (x + 14, 12))

        y0 = RGB_H + MARGIN
        for i, name in enumerate(JOINT_NAMES):
            y = y0 + i * ROW_H
            label = font.render(f"{name:>6}", True, ACCENT)
            surf.blit(label, (MARGIN, y + 4))
            val = small_font.render(
                f"cmd {self.joints[i]:+.2f}   act {self.actual[i]:+.2f}",
                True,
                FG,
            )
            surf.blit(val, (180, y + 8))

        for btn in self.buttons:
            btn.draw(surf, font)

        status_color = ERR if "failed" in self.status.lower() or "lost" in self.status.lower() else FG
        status = small_font.render(self.status, True, status_color)
        surf.blit(status, (MARGIN, WIN_H - 24))
        hint = small_font.render("Keys: Q/A swing  W/S boom  E/D arm  R/F bucket", True, (140, 140, 150))
        surf.blit(hint, (MARGIN, WIN_H - 44))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Excavator GUI Client")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    small_font = pygame.font.SysFont("consolas", 14)

    client = GuiClient(args.host, args.port, args.ticks)
    client.connect()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    client.connect()
                else:
                    client.handle_key(event.key)
            else:
                for btn in client.buttons:
                    btn.handle(event)

        client.step_sim()
        client.draw(screen, font, small_font)
        pygame.display.flip()
        clock.tick(args.fps)

    client.close()
    pygame.quit()


if __name__ == "__main__":
    main()
