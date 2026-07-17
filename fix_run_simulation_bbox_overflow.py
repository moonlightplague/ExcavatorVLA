#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_TARGET = Path("/root/isaacsim/ExcavatorVLA/run_simulation.py")

NEW_BBOX_FUNCTION = '    def prim_world_aligned_bbox(prim):\n        """\n        Return a finite, non-empty world-aligned BBox as float64.\n\n        USD BBoxCache may return an empty range whose endpoints are close to\n        +/-FLT_MAX. Those values are technically finite, but subtracting them\n        overflows float32. Reject such ranges before any extent calculation.\n        """\n        if prim is None or not prim.IsValid():\n            return None\n\n        try:\n            cache = UsdGeom.BBoxCache(\n                Usd.TimeCode.Default(),\n                [\n                    UsdGeom.Tokens.default_,\n                    UsdGeom.Tokens.render,\n                    UsdGeom.Tokens.proxy,\n                ],\n                useExtentsHint=True,\n            )\n            aligned_range = (\n                cache.ComputeWorldBound(prim)\n                .ComputeAlignedRange()\n            )\n\n            try:\n                if aligned_range.IsEmpty():\n                    return None\n            except Exception:\n                pass\n\n            minimum = aligned_range.GetMin()\n            maximum = aligned_range.GetMax()\n\n            bbox = np.asarray(\n                [\n                    [\n                        float(minimum[0]),\n                        float(minimum[1]),\n                        float(minimum[2]),\n                    ],\n                    [\n                        float(maximum[0]),\n                        float(maximum[1]),\n                        float(maximum[2]),\n                    ],\n                ],\n                dtype=np.float64,\n            )\n\n            if bbox.shape != (2, 3):\n                return None\n            if not np.all(np.isfinite(bbox)):\n                return None\n\n            # Reject FLT_MAX-style sentinel values and unreasonable bounds\n            # before any subtraction.\n            if np.max(np.abs(bbox)) > 1.0e6:\n                return None\n\n            extent = bbox[1] - bbox[0]\n            if not np.all(np.isfinite(extent)):\n                return None\n            if np.any(extent < 0.0):\n                return None\n            if float(np.max(extent)) <= 1.0e-6:\n                return None\n\n            return bbox\n\n        except Exception as exc:\n            print(\n                "[STATE27] Ignoring invalid BBox:",\n                str(prim.GetPath()) if prim is not None else None,\n                repr(exc),\n                flush=True,\n            )\n            return None\n'


def patch_file(target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)

    text = target.read_text(encoding="utf-8")
    pattern = re.compile(
        r"    def prim_world_aligned_bbox\(prim\):\n"
        r".*?"
        r"(?=\n    def resolve_truck_prim\(\):)",
        flags=re.DOTALL,
    )

    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one prim_world_aligned_bbox function, "
            f"found {len(matches)}"
        )

    backup = target.with_suffix(
        target.suffix + ".before_bbox_overflow_fix"
    )
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[OK] backup: {backup}")

    updated = pattern.sub(
        NEW_BBOX_FUNCTION.rstrip(),
        text,
        count=1,
    )
    target.write_text(updated, encoding="utf-8")

    print(f"[OK] patched: {target}")
    print("[INFO] Empty/FLT_MAX USD bounds will now be ignored")
    print("[INFO] BBox calculations remain float64 until validated")


def restore_file(target: Path) -> None:
    target = target.expanduser().resolve()
    backup = target.with_suffix(
        target.suffix + ".before_bbox_overflow_fix"
    )
    if not backup.is_file():
        raise FileNotFoundError(backup)

    shutil.copy2(backup, target)
    print(f"[OK] restored: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=DEFAULT_TARGET,
    )
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        restore_file(args.target)
    else:
        patch_file(args.target)


if __name__ == "__main__":
    main()
