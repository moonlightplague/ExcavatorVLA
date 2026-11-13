from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from baseline.coarse_acceptance import episode_mass_quality, validate_mass_report, VERSION
from baseline.repeat_runtime import RepeatConfig
from baseline.tuning import install_quality_policy


class CoarseAcceptanceTests(unittest.TestCase):
    def config(self):
        return RepeatConfig(scoops=10, acceptance_profile='coarse_mass', min_delivered_kg=50.)

    def measurements(self, mass=5., per_dig_kg=100., final_kg=800.):
        results = [dict(success=True, execution_success=True, freeze_count=0,
                        strict_success=False, legacy_episode_success=False) for _ in range(10)]
        metrics = dict(available=True, particle_mass_kg=mass, payload_particles=int(final_kg / mass),
                       scoop_cycles=[dict(success=True, payload_particles=int(per_dig_kg / mass)) for _ in range(10)])
        return results, metrics

    def test_equal_mass_has_equal_acceptance_at_different_material_resolutions(self):
        reports = []
        for mass in (.5, 5., 20.):
            results, metrics = self.measurements(mass)
            reports.append(episode_mass_quality(results, self.config(), metrics))
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[1], reports[2])
        self.assertTrue(reports[0]['success'])
        self.assertAlmostEqual(reports[0]['deposit_retention_ratio'], .8)

    def test_large_early_loads_cannot_hide_one_small_final_delivery(self):
        results, metrics = self.measurements()
        metrics['scoop_cycles'][-1]['payload_particles'] = 9
        report = episode_mass_quality(results, self.config(), metrics)
        self.assertFalse(report['success'])
        self.assertIn('10:45.000<50.000', report['reason'])

    def test_net_load_retention_and_physical_completion_are_independent_gates(self):
        results, metrics = self.measurements()
        metrics['payload_particles'] = 159
        self.assertIn('insufficient_deposit_retention', episode_mass_quality(results, self.config(), metrics)['reason'])
        metrics['payload_particles'] = 78
        self.assertIn('insufficient_final_payload_mass', episode_mass_quality(results, self.config(), metrics)['reason'])
        for field, value in [('execution_success', False), ('freeze_count', 1), ('success', False)]:
            results, metrics = self.measurements()
            results[-1][field] = value
            self.assertIn('incomplete_physical_cycles', episode_mass_quality(results, self.config(), metrics)['reason'])
        results, metrics = self.measurements()
        metrics['available'] = False
        self.assertFalse(episode_mass_quality(results, self.config(), metrics)['success'])

    def test_validator_checks_actual_model_mass_and_recomputes_the_outcome(self):
        results, metrics = self.measurements()
        model = dict(profile='coarse', active=True, error='', metric_point_mass_kg=5.)
        meta = dict(scoop_results=results, repeat_baseline=dict(config=asdict(self.config()),
            soil=dict(backend='newton_mpm', mpm=model.copy()), mpm_final=model.copy()))
        meta['repeat_episode_quality'] = episode_mass_quality(results, self.config(), metrics)
        validate_mass_report(meta, metrics)
        meta['repeat_baseline']['mpm_final']['metric_point_mass_kg'] = .5
        with self.assertRaisesRegex(ValueError, 'material-point mass'):
            validate_mass_report(meta, metrics)
        meta['repeat_baseline']['mpm_final'] = model.copy()
        meta['repeat_episode_quality']['final_payload_mass_kg'] = 999.
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            validate_mass_report(meta, metrics)

    def runtime(self, delivered=10, execution=True, freeze=0):
        report = dict(success=False, reason='low_bucket_particles:1;low_lift_particles:1;score_below_minimum:55',
            scoop_index=0, score=55., spill_ratio=.1, execution_success=execution, freeze_count=freeze,
            max_bucket_from_pile_particles=1, lift_bucket_from_pile_particles=1)
        rt = SimpleNamespace(STATE={'dataset_scoop_results': []}, info_print=Mock(), dataset_record_event=Mock(),
            sand_particle_mass=lambda: 5., repeat_baseline=SimpleNamespace(cycles=[dict(scoop_index=0,
                metrics=dict(scooped_particles=1, payload_particles=delivered, spillage_particles=0))], meta={}))
        def original(*_args):
            rt.STATE['dataset_scoop_results'].append(report)
            return report
        rt.multi_scoop_cycle_report = original
        rt.compute_episode_quality_score = Mock(return_value=dict(success=False,
            failure_reason='quality_rejected/low_bucket_particles:1', execution_success=True, score=55.))
        return rt

    def test_small_point_count_does_not_reject_a_real_fifty_kg_deposit(self):
        rt = self.runtime()
        install_quality_policy(rt, self.config())
        report = rt.multi_scoop_cycle_report(True)
        self.assertTrue(report['success'])
        self.assertFalse(report['strict_success'])
        self.assertFalse(report['legacy_episode_success'])
        self.assertEqual(report['source_delivered_mass_kg'], 50.)
        self.assertEqual(rt.STATE['dataset_scoops_completed'], 1)

    def test_low_mass_execution_failure_and_freeze_still_reject(self):
        for delivered, execution, freeze in ((9, True, 0), (10, False, 0), (10, True, 1)):
            rt = self.runtime(delivered, execution, freeze)
            install_quality_policy(rt, self.config())
            self.assertFalse(rt.multi_scoop_cycle_report(execution)['success'])

    def test_global_point_gate_can_only_be_replaced_after_verified_mass_acceptance(self):
        rt = self.runtime()
        install_quality_policy(rt, self.config())
        self.assertFalse(rt.compute_episode_quality_score(True, '')['success'])
        rt.repeat_baseline.meta['repeat_episode_quality'] = dict(policy=VERSION, success=True)
        self.assertTrue(rt.compute_episode_quality_score(True, '')['success'])
        self.assertFalse(rt.compute_episode_quality_score(False, 'execution_failed/collision')['success'])
        rt.compute_episode_quality_score.__wrapped__.return_value = dict(success=False,
            failure_reason='execution_failed/freeze_detected:1', execution_success=False)
        self.assertFalse(rt.compute_episode_quality_score(True, '')['success'])


if __name__ == '__main__':
    unittest.main()
