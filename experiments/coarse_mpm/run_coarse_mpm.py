"""Large-looking material points, still Newton MPM, in the original Isaac scene.

Run with the existing Isaac Python wrapper. --dry-run uses CPU only.
This entry does not start collection, export, policy evaluation or a TCP bridge.
No substep, constitutive-material or excavator-controller setting is changed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spacing', type=float, default=.15)
    parser.add_argument('--grid', type=float, default=.20)
    parser.add_argument('--diameter', type=float, default=.18, help='Display diameter in meters')
    parser.add_argument('--max-points', type=int, default=6000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--scene')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.spacing, args.grid, args.diameter)):
        parser.error('Spacing, grid and diameter must be finite and positive')
    if args.spacing > args.grid or not 1 <= args.max_points <= 40000 or args.gpu < 0:
        parser.error('Require spacing <= grid, 1..40000 points and a nonnegative GPU index')
    settings = {
        'EXCAVATOR_SOIL_BACKEND': 'newton_mpm',
        'EXCAVATOR_MPM_PROFILE': 'coarse',
        'EXCAVATOR_MPM_PARTICLE_SPACING': str(args.spacing),
        'EXCAVATOR_MPM_VOXEL_SIZE': str(args.grid),
        'EXCAVATOR_MPM_RENDER_DIAMETER': str(args.diameter),
        'EXCAVATOR_MPM_MAX_PARTICLES': str(args.max_points),
        'EXCAVATOR_MPM_DEVICE': f'cuda:{args.gpu}',
        'EXCAVATOR_DATASET_CAMERA_ENABLED': '0',
    }
    print(json.dumps({'settings': settings, 'substeps': 'unchanged',
                      'rendering': 'one sphere per MPM material point',
                      'physics': 'MPM, not rigid spheres',
                      'note': 'Spacing and grid may grow together to fit the full source into the point budget.'},
                     indent=2), flush=True)
    if args.dry_run:
        return
    # Check before importing Isaac, allocating GPU memory or creating a window.
    result = subprocess.run(['nvidia-smi', '-i', str(args.gpu),
                             '--query-gpu=utilization.gpu,memory.free', '--format=csv,noheader,nounits'],
                            check=True, text=True, capture_output=True, timeout=15)
    utilization, free = [int(value.strip()) for value in result.stdout.strip().split(',')]
    if utilization > 20 or free < 8000:
        raise RuntimeError(f'GPU {args.gpu} busy ({utilization}%, {free} MiB free); simulation NOT started.')
    repo = Path(__file__).resolve().parents[2]
    os.environ.update(settings)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo/'scripts'))
    from excavator_app.newton_mpm_soil import soil_resolution_settings
    assert soil_resolution_settings()['render_material_centers']
    entry = repo/'run_vla_train_scene.py'
    sys.argv = [str(entry), '--with-ui', '--no-bridge', '--disable-export',
                '--active-gpu', str(args.gpu), '--physics-gpu', str(args.gpu),
                '--width', '1280', '--height', '720']
    if args.scene:
        sys.argv.extend(['--scene', args.scene])
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
