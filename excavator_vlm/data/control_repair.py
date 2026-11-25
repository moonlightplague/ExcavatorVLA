"""Verify the immutable Newton ten-cycle control-reconstruction gate."""
import json
from pathlib import Path

from .action_limits import action_contract, BUCKET20_PROFILE
from .contract import file_hash


def require_control_repair_gate(root, fingerprint):
    root = Path(root).resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    if info.get('action_limit_profile') != BUCKET20_PROFILE:
        return None
    path = root / 'control_replay_gate.json'
    if not path.is_file():
        raise ValueError('Newton control repair requires three passed full ten-cycle expert replays')
    gate = json.loads(path.read_text())
    if (gate.get('schema_version') != 'newton_control_repair_gate_v1'
            or gate.get('status') != 'passed' or gate.get('soil_backend') != 'newton_mpm'
            or gate.get('dataset_fingerprint') != fingerprint
            or gate.get('source_fingerprint') != info.get('relabel', {}).get('source_fingerprint')
            or gate.get('action_contract') != action_contract(info)):
        raise ValueError('Control replay gate differs from the candidate data/profile')
    entries = gate.get('episodes', [])
    if len(entries) < 3 or len({e['episode_index'] for e in entries}) != len(entries):
        raise ValueError('Three distinct complete control-repair replays are required')
    for entry in entries:
        if (entry.get('cycles_tested') != 10 or not entry.get('full_reference_executed')
                or not entry.get('numerical_target_parity_passed')
                or not entry.get('measured_joint_parity_passed')
                or not entry.get('all_cycle_deliveries_passed')
                or entry.get('clipped_action_dimensions') != 0):
            raise ValueError('Incomplete or failed ten-cycle control-repair evidence')
        evidence = entry.get('evidence', [])
        if not {'status', 'trace', 'reference_controls', 'compatibility'} <= {e.get('kind') for e in evidence}:
            raise ValueError('Replay gate is missing its independent numeric/physical evidence')
        for item in evidence:
            source = (root / item['path']).resolve()
            if not source.is_relative_to(root) or not source.is_file() or file_hash(source) != item['sha256']:
                raise ValueError('Control replay evidence path/hash differs')
    return gate
