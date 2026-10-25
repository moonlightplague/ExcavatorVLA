"""Restore only the backed-up soil integration sources, refusing unknown edits.

Server usage (stop collection/evaluation first):
  python restore_soil_integration.py --backup /absolute/path/to/soil_mpm_... 
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', required=True, type=Path)
    args = parser.parse_args()
    backup = args.backup.resolve()
    baseline = json.loads((backup/'manifest.json').read_text())
    deployed = json.loads((backup/'deployed_manifest.json').read_text())
    base = Path(baseline['base']).resolve()
    originals = {
        'ExcavatorVLA-isaac/scripts/excavator_app/sand_site_runtime.py',
        'ExcavatorVLA-smolvla/scripts/excavator_app/sand_site_runtime.py',
        'ExcavatorVLA-smolvla/run_simulation.py',
    }
    added = {repo+'/scripts/excavator_app/newton_mpm_soil.py'
             for repo in ('ExcavatorVLA-isaac', 'ExcavatorVLA-smolvla')}
    contents = {}
    with tarfile.open(backup/'code_before.tar.gz', 'r:gz') as archive:
        for name in originals:
            data = archive.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != baseline['files'][name]:
                raise RuntimeError('Backup checksum failed: '+name)
            contents[name] = data
    for name in originals | added:
        target = (base/name).resolve()
        target.relative_to(base)
        if target.is_file():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest not in {deployed.get(name), baseline['files'].get(name)}:
                raise RuntimeError('Refusing to overwrite edits made after integration: '+name)
    for name, data in contents.items():
        target = base/name
        temporary = target.with_name(target.name+'.restore_tmp')
        mode = target.stat().st_mode & 0o777
        temporary.write_bytes(data)
        temporary.chmod(mode)
        os.replace(temporary, target)
    for name in added:
        target = base/name
        if target.exists():
            target.unlink()
    print('Restored the three original source files; removed the two adapter copies.')
    print('Outer wrappers, assets, installed environments and collected data were not changed.')


if __name__ == '__main__':
    main()
