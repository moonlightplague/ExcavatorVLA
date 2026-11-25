"""Verify archived collection evidence without modifying a frozen publication."""
import json
from pathlib import Path

from .contract import audit_backend_provenance, file_hash


def audit_external_backend_provenance(dataset, fingerprint, evidence_root, episode_ids, expected):
    dataset, root = Path(dataset).resolve(), Path(evidence_root).resolve()
    manifest = json.loads((root / 'backend_provenance.json').read_text())
    if (manifest.get('dataset_fingerprint') != fingerprint
            or Path(manifest.get('dataset_root', '')).resolve() != dataset):
        raise ValueError('External collection evidence belongs to a different publication/fingerprint')
    report = audit_backend_provenance(root, episode_ids, expected)
    for entry in manifest['episodes']:
        reference = entry['publication_receipt']
        path = (root / reference['path']).resolve()
        if not path.is_relative_to(root) or file_hash(path) != reference['sha256']:
            raise ValueError('Archived publication receipt path/hash differs')
        receipt = json.loads(path.read_text())
        meta = json.loads((root / entry['evidence']['path']).read_text())
        if (receipt.get('validated') is not True
                or receipt.get('export_episode_index') != entry['episode_index']
                or Path(receipt.get('export_dir', '')).resolve() != dataset
                or receipt.get('episode_id') != entry['parent_id']
                or meta.get('episode_id') != entry['parent_id']):
            raise ValueError('Collection parent/receipt does not map to the frozen publication episode')
    return {**report, 'external_evidence_root': str(root),
            'bound_dataset_fingerprint': fingerprint,
            'publication_receipts_verified': len(manifest['episodes'])}
