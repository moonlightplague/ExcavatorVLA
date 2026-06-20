#!/usr/bin/env python3
"""
Scene Diagnostic Script - Check USD scene content without SimulationApp
"""

import os
import sys

SCENE_USD_PATH = "/isaac-sim/ExcavatorVLA/assets/usd/excavator_scene.usd"

def diagnose_scene():
    # Use pure USD API without SimulationApp
    from pxr import Usd, UsdGeom, UsdPhysics
    
    # Open USD stage directly
    stage = Usd.Stage.Open(SCENE_USD_PATH)
    
    if not stage:
        print(f"[ERROR] Cannot open USD file: {SCENE_USD_PATH}")
        return
    
    print("=" * 60)
    print("Scene Diagnostic Report")
    print("=" * 60)
    print(f"USD File: {SCENE_USD_PATH}")
    print(f"Root Layer: {stage.GetRootLayer().identifier}")
    print("=" * 60)
    
    # Count prims by type
    prim_counts = {}
    mesh_prims = []
    camera_prims = []
    physics_prims = []
    
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        prim_path = prim.GetPath().pathString
        
        if prim_type not in prim_counts:
            prim_counts[prim_type] = 0
        prim_counts[prim_type] += 1
        
        # Check for meshes
        if UsdGeom.Mesh(prim):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                mesh_prims.append((prim_path, len(points)))
        
        # Check for cameras
        if UsdGeom.Camera(prim):
            camera_prims.append(prim_path)
        
        # Check for physics
        if UsdPhysics.RigidBodyAPI(prim) or UsdPhysics.Joint(prim):
            physics_prims.append(prim_path)
    
    # Print summary
    print("\nPrim Type Counts:")
    for prim_type, count in sorted(prim_counts.items()):
        print(f"  {prim_type}: {count}")
    
    print("\nMesh Prims (with points):")
    if mesh_prims:
        for path, num_points in mesh_prims[:20]:  # Show first 20
            print(f"  {path}: {num_points} points")
        if len(mesh_prims) > 20:
            print(f"  ... and {len(mesh_prims) - 20} more meshes")
    else:
        print("  [WARN] No mesh prims with geometry found!")
    
    print("\nCamera Prims:")
    if camera_prims:
        for path in camera_prims:
            print(f"  {path}")
    else:
        print("  [WARN] No camera prims found!")
    
    print("\nPhysics Prims:")
    if physics_prims:
        for path in physics_prims[:10]:  # Show first 10
            print(f"  {path}")
        if len(physics_prims) > 10:
            print(f"  ... and {len(physics_prims) - 10} more")
    else:
        print("  [WARN] No physics prims found!")
    
    # Check root prim
    print("\nRoot Prim Structure:")
    root_prim = stage.GetPseudoRoot()
    for child in root_prim.GetChildren():
        print(f"  {child.GetPath().pathString} ({child.GetTypeName()})")
        for subchild in child.GetChildren():
            print(f"    {subchild.GetPath().pathString} ({subchild.GetTypeName()})")
    
    # Check for references
    print("\nReferences:")
    for prim in stage.Traverse():
        refs = prim.GetReferences()
        if refs and refs.GetAuthoredReferences():
            print(f"  {prim.GetPath().pathString}:")
            for ref in refs.GetAuthoredReferences():
                print(f"    -> {ref.assetPath}")
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_scene()