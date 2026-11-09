"""Versioned command-target slew limits; these are not measured joint speeds."""
import json
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_PROFILE = 'excavator_velocity_v1'
BUCKET20_PROFILE = 'excavator_bucket_slew20_v1'
PROFILES = {
    DEFAULT_PROFILE: [6., 2.5, 2.5, 6.],
    BUCKET20_PROFILE: [6., 2.5, 2.5, 20.],
}


def action_contract(info):
    profile = info.get('action_limit_profile', DEFAULT_PROFILE)
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('Unknown action-limit profile')
    limits = np.asarray(info.get('action_limits_rad_s'), dtype=float)
    if limits.shape != (4,) or not np.array_equal(limits, PROFILES[profile]):
        raise ValueError('Unexpected deployment action limits for the declared profile')
    return {'profile': profile, 'units': 'rad/s', 'kind': 'joint_command_target_slew_rate',
            'limits_rad_s': list(PROFILES[profile])}


def default_contract():
    return action_contract({'action_limits_rad_s': PROFILES[DEFAULT_PROFILE]})


def checkpoint_contract(manifest):
    value = manifest.get('action_contract', default_contract())
    if not isinstance(value, dict):
        raise ValueError('Invalid checkpoint action contract')
    canonical = action_contract({'action_limit_profile': value.get('profile'),
                                 'action_limits_rad_s': value.get('limits_rad_s')})
    if value != canonical:
        raise ValueError('Checkpoint action contract differs from the versioned profile')
    return canonical


def check_checkpoint_contract(manifest, info):
    if checkpoint_contract(manifest) != action_contract(info):
        raise ValueError('Checkpoint and evaluation dataset use different action-limit profiles')


def collection_payload_threshold(root, criterion):
    """Read the archived collection threshold; do not substitute a new benchmark goal."""
    if not isinstance(criterion, dict) or criterion.get('version') != 'collection_min_dump_particles_v1':
        raise ValueError('Replay gate requires an evidence-backed collection payload criterion')
    threshold = criterion.get('min_payload_particles_per_cycle')
    if type(threshold) is not int or threshold < 1:
        raise ValueError('Collection payload threshold must be a positive integer')
    evidence = criterion.get('source_evidence', [])
    if not evidence or len({entry['path'] for entry in evidence}) != len(evidence):
        raise ValueError('Collection criterion needs distinct immutable evidence files')
    for entry in evidence:
        path = (Path(root) / entry['path']).resolve()
        if (not path.is_relative_to(Path(root).resolve()) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']):
            raise ValueError('Collection criterion evidence is missing or changed')
        payload = json.loads(path.read_text())
        if (entry.get('field') != 'quality_min_dump_particles'
                or type(payload.get(entry['field'])) is not int
                or payload[entry['field']] != threshold
                or payload.get('quality_gate_version') != criterion.get('quality_gate_version')):
            raise ValueError('Collection threshold/version differs from the recorded criterion')
    return threshold


def require_replay_gate(root, fingerprint):
    """A relaxed numerical bound alone must never authorize formal training."""
    path = Path(root) / 'expert_replay_gate.json'
    if not path.is_file():
        raise ValueError('Experimental action limits require a passed full-episode expert replay gate')
    gate = json.loads(path.read_text())
    if (gate.get('schema_version') != 'excavator_repeat_expert_gate_v2'
            or gate.get('status') != 'passed' or gate.get('dataset_fingerprint') != fingerprint
            or gate.get('soil_backend') != 'physx'
            or gate.get('action_contract') != action_contract(json.loads((Path(root) / 'meta/info.json').read_text()))):
        raise ValueError('Expert replay gate is missing, failed, or belongs to different data')
    threshold = collection_payload_threshold(root, gate.get('collection_criterion'))
    results = gate.get('episodes', [])
    if (len(results) < 3 or len({r.get('episode_index') for r in results}) != len(results)
            or any(r.get('full_reference_executed') is not True or r.get('success') is not True
                   or r.get('cycles_passed') != 5
                   or r.get('min_net_deposited_particles_per_cycle') != threshold for r in results)):
        raise ValueError('At least three distinct complete five-cycle expert replays must pass')
    for result in results:
        evidence = result.get('evidence', [])
        if len(evidence) != 2:
            raise ValueError('Expert replay gate requires immutable status and full-trace evidence')
        for entry in evidence:
            evidence_path = (Path(root) / entry['path']).resolve()
            if not evidence_path.is_relative_to(Path(root).resolve()) or not evidence_path.is_file():
                raise ValueError('Replay evidence must be inside the candidate dataset')
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Replay evidence checksum differs from the passed gate')
    return gate
