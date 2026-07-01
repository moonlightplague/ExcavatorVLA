#!/usr/bin/env python3
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from excavator_common.paths import default_scene_path, resolve_existing_path

parser = argparse.ArgumentParser(description="Print a compact USD scene prim listing.")
parser.add_argument("--scene", default=default_scene_path(PROJECT_ROOT), help="USD scene path.")
args = parser.parse_args()

from pxr import Usd, UsdGeom

scene_path = resolve_existing_path(args.scene, root=PROJECT_ROOT)
stage = Usd.Stage.Open(scene_path)
print('Stage opened:', stage.GetRootLayer().identifier)
print('=' * 60)

for prim in stage.Traverse():
    path = prim.GetPath().pathString
    type_name = prim.GetTypeName()
    
    # Check if mesh has geometry
    mesh = UsdGeom.Mesh(prim)
    if mesh:
        points = mesh.GetPointsAttr().Get()
        if points:
            print(f"[MESH] {path} ({type_name}) - {len(points)} points")
        else:
            print(f"[MESH-EMPTY] {path} ({type_name})")
    elif UsdGeom.Camera(prim):
        print(f"[CAMERA] {path} ({type_name})")
    elif UsdGeom.Xform(prim):
        print(f"[XFORM] {path} ({type_name})")
    else:
        print(f"[{type_name}] {path}")
