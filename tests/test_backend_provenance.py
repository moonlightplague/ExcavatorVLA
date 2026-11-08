"""Collection backend evidence must be complete, explicit and tamper evident."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import importlib.util
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from excavator_vlm.data.contract import audit_backend_provenance


class BackendProvenanceTests(unittest.TestCase):
    def fixture(self, root, backend='physx'):
        evidence=root/'provenance/episode_0000/meta.json'
        evidence.parent.mkdir(parents=True)
        evidence.write_text(json.dumps({'repeat_baseline':{'soil_backend':backend}}))
        manifest={'schema_version':'excavator_backend_provenance_v1','soil_backend':'physx',
                  'episodes':[{'episode_index':0,'soil_backend':'physx',
                  'evidence':{'path':'provenance/episode_0000/meta.json',
                              'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),
                              'field':'repeat_baseline.soil_backend'}}]}
        (root/'backend_provenance.json').write_text(json.dumps(manifest))
        return evidence,manifest

    def test_verified_backend_and_requested_backend_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            self.assertEqual(audit_backend_provenance(root,[0],'physx')['episodes_verified'],1)
            with self.assertRaises(ValueError):audit_backend_provenance(root,[0],'newton_mpm')

    def test_mislabeled_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,'newton_mpm')
            with self.assertRaises(ValueError):audit_backend_provenance(root,[0])


    def test_modified_source_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evidence,_=self.fixture(root)
            evidence.write_text('{}')
            with self.assertRaises(ValueError):audit_backend_provenance(root,[0])

    def test_missing_duplicate_or_unknown_episode_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,manifest=self.fixture(root)
            with self.assertRaises(ValueError):audit_backend_provenance(root,[0,1])
            manifest['episodes']*=2
            (root/'backend_provenance.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):audit_backend_provenance(root,[0])


class ControllerCompatibilityTests(unittest.TestCase):
    def parity_function(self):
        path=Path(__file__).resolve().parents[1]/'scripts/build_pi05_singlecycle_pilot.py'
        spec=importlib.util.spec_from_file_location('singlecycle_builder_contract_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.controller_parity

    def test_held_velocity_at_a_joint_limit_is_valid(self):
        states=np.zeros((16,28),dtype=np.float32)
        actions=np.zeros((16,4),dtype=np.float32);actions[:,3]=-6
        states[0,3]=-1.5
        for i in range(15):
            value=states[i,3]
            for _ in range(6):value=np.clip(value-.1,-2.0943951607,1.5707963705)
            states[i+1,3]=value
        error,_=self.parity_function()(states,actions)
        self.assertLess(error,1e-5)

    def test_unencoded_target_reset_is_rejected(self):
        states=np.zeros((16,28),dtype=np.float32);actions=np.zeros((16,4),dtype=np.float32)
        states[6:,8]=.3
        error,_=self.parity_function()(states,actions)
        self.assertGreater(error,.29)

if __name__=='__main__':unittest.main()
