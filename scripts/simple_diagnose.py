#!/usr/bin/env python3
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open('/isaac-sim/ExcavatorVLA/assets/usd/excavator_scene.usd')
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