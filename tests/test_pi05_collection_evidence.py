"""External provenance must bind to the unchanged source and frozen split."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SUBMODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMODULE))
from excavator_vlm.data.collection_evidence import audit_external_backend_provenance
from excavator_vlm.data.contract import file_hash
from excavator_vlm.policies.pi05.train import prepare


class CollectionEvidenceTests(unittest.TestCase):
    def test_bucket20_uses_the_correct_backend_replay_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            split=dict(dataset_fingerprint='candidate',train=list(range(7)),validation=[7],test=[8,9])
            split_file=root/'split.json';split_file.write_text(json.dumps(split))
            settings=dict(dataset=dict(root=str(root),episodes=None),policy=dict(chunk_size=10),
                          training={},benchmark=dict(split_file=str(split_file),split_seed=1000))
            report=dict(fingerprint='candidate',action_contract=dict(profile='excavator_bucket_slew20_v1'))
            frames=pd.DataFrame({'episode_index':list(range(10))})
            for kind, newton in [('newton_repeat_control_reconstruction_v1',True),('legacy_physx',False)]:
                with self.subTest(kind=kind), \
                        patch('excavator_vlm.policies.pi05.train.audit_dataset',return_value=(report,frames)), \
                        patch('excavator_vlm.data.contract.load_info',return_value={'relabel':{'kind':kind}}), \
                        patch('excavator_vlm.data.control_repair.require_control_repair_gate') as repaired, \
                        patch('excavator_vlm.data.action_limits.require_replay_gate') as legacy:
                    _,actual,_,_=prepare(settings)
                    self.assertEqual(actual,split)
                    self.assertEqual(repaired.call_count,int(newton))
                    self.assertEqual(legacy.call_count,int(not newton))
                    if newton:
                        repaired.side_effect=ValueError('Control replay evidence hash differs')
                        with self.assertRaisesRegex(ValueError,'hash'):prepare(settings)

    def evidence(self, root, count=10):
        dataset, proof = root/'dataset', root/'proof'
        dataset.mkdir(); proof.mkdir()
        entries = []
        for eid in range(count):
            meta = proof/f'meta_{eid}.json'
            meta.write_text(json.dumps({'episode_id':f'parent{eid}', 'repeat_baseline':{'soil_backend':'newton_mpm'}}))
            receipt = proof/f'export_{eid}.json'
            receipt.write_text(json.dumps(dict(validated=True,export_episode_index=eid,
                export_dir=str(dataset),episode_id=f'parent{eid}')))
            entries.append(dict(episode_index=eid,parent_id=f'parent{eid}',soil_backend='newton_mpm',
                evidence=dict(path=meta.name,sha256=file_hash(meta),field='repeat_baseline.soil_backend'),
                publication_receipt=dict(path=receipt.name,sha256=file_hash(receipt))))
        manifest=dict(schema_version='excavator_backend_provenance_v1',soil_backend='newton_mpm',
            dataset_root=str(dataset),dataset_fingerprint='a'*64,episodes=entries)
        (proof/'backend_provenance.json').write_text(json.dumps(manifest))
        return dataset,proof,manifest

    def test_preparation_uses_the_literal_frozen_70_10_20_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,proof,_=self.evidence(root)
            split=dict(dataset_fingerprint='a'*64,train=list(range(7)),validation=[7],test=[8,9])
            split_file=root/'split.json';split_file.write_text(json.dumps(split))
            settings=dict(dataset=dict(root=str(dataset),episodes=None),policy=dict(chunk_size=10),
                training=dict(data_soil_backend='newton_mpm',backend_provenance_root=str(proof)),
                benchmark=dict(split_file=str(split_file),split_seed=1000))
            frames=pd.DataFrame({'episode_index':list(range(10))})
            report=dict(fingerprint='a'*64)
            before=split_file.read_bytes()
            with patch('excavator_vlm.policies.pi05.train.audit_dataset',return_value=(report,frames)), \
                    patch('excavator_vlm.policies.pi05.train.make_split',side_effect=AssertionError('Must not resplit')):
                audited,actual,_,train=prepare(settings)
            self.assertEqual(actual,split);self.assertEqual(train,split['train'])
            self.assertEqual(audited['fingerprint'],'a'*64)
            self.assertEqual(audited['collection_backend']['publication_receipts_verified'],10)
            self.assertEqual(split_file.read_bytes(),before)

    def test_different_dataset_or_fingerprint_cannot_satisfy_the_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,proof,_=self.evidence(root)
            for actual,fingerprint in [(dataset,'b'*64),(root/'different','a'*64)]:
                with self.subTest(actual=actual),self.assertRaises(ValueError):
                    audit_external_backend_provenance(actual,fingerprint,proof,range(10),'newton_mpm')

    def test_wrong_parent_mapping_and_missing_episode_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset,proof,manifest=self.evidence(Path(directory))
            receipt=proof/manifest['episodes'][0]['publication_receipt']['path']
            data=json.loads(receipt.read_text());data['export_episode_index']=9
            receipt.write_text(json.dumps(data))
            manifest['episodes'][0]['publication_receipt']['sha256']=file_hash(receipt)
            (proof/'backend_provenance.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit_external_backend_provenance(dataset,'a'*64,proof,range(10),'newton_mpm')
            with self.assertRaises(ValueError):
                audit_external_backend_provenance(dataset,'a'*64,proof,range(11),'newton_mpm')

    def test_changed_backend_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset,proof,manifest=self.evidence(Path(directory))
            meta=proof/manifest['episodes'][0]['evidence']['path']
            meta.write_text(json.dumps({'episode_id':'parent0','repeat_baseline':{'soil_backend':'physx'}}))
            with self.assertRaises(ValueError):
                audit_external_backend_provenance(dataset,'a'*64,proof,range(10),'newton_mpm')


if __name__ == '__main__':
    unittest.main()
