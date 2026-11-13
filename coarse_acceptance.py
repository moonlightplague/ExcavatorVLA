"""Resolution-independent outcome gates for coarse material-point collection."""
from __future__ import annotations

from functools import wraps
import math

from baseline.tuning import delivery_mass_quality


VERSION = "repeat_coarse_mass_v1"
COUNT_REASONS = ("low_bucket_particles:", "low_lift_particles:", "low_bin_gain:",
                 "low_source_capture:", "low_source_delivery:")


def episode_mass_quality(results, config, metrics):
    """Grade useful deposits and retained truck mass, never a raw-point score."""
    cycles = metrics.get("scoop_cycles", [])
    mass = metrics.get("particle_mass_kg")
    delivery = delivery_mass_quality(cycles, mass, config.min_delivered_kg, config.scoops)
    failures = [delivery['reason']] if not delivery['success'] else []
    if not metrics.get("available"):
        failures.append("source_metrics_unavailable")
    if (len(results) != config.scoops or len(cycles) != config.scoops
            or not all(row.get("success") and row.get("execution_success")
                       and row.get("freeze_count", 0) == 0 for row in results)
            or not all(row.get("success") for row in cycles)):
        failures.append("incomplete_physical_cycles")
    payload = metrics.get("payload_particles")
    valid = (mass is not None and math.isfinite(float(mass)) and float(mass) > 0
             and payload is not None and math.isfinite(float(payload)) and payload >= 0)
    final_kg = float(payload) * float(mass) if valid else None
    loads = delivery['delivered_mass_kg']
    total = sum(loads) if all(v is not None and math.isfinite(v) for v in loads) else None
    retention = final_kg / total if final_kg is not None and total is not None and total > 0 else None
    required = config.min_retained_delivery_ratio * config.scoops * config.min_delivered_kg
    if final_kg is None or final_kg < required:
        failures.append("insufficient_final_payload_mass")
    if retention is None or retention < config.min_retained_delivery_ratio:
        failures.append("insufficient_deposit_retention")
    return dict(policy=VERSION, success=not failures, reason=";".join(failures),
                delivered_mass_kg=loads, minimum_per_dig_kg=config.min_delivered_kg,
                final_payload_mass_kg=final_kg, minimum_final_payload_kg=required,
                cumulative_delivered_mass_kg=total, deposit_retention_ratio=retention,
                minimum_deposit_retention_ratio=config.min_retained_delivery_ratio,
                legacy_strict_scoops=sum(bool(r.get('strict_success')) for r in results),
                legacy_episode_scoops=sum(bool(r.get('legacy_episode_success')) for r in results))


def validate_mass_report(meta, metrics):
    from types import SimpleNamespace
    config = SimpleNamespace(**meta['repeat_baseline']['config'])
    baseline = meta['repeat_baseline']
    if baseline.get('soil', {}).get('backend') != 'newton_mpm':
        raise ValueError('Coarse mass policy requires Newton MPM')
    for model in (baseline.get('soil', {}).get('mpm', {}), baseline.get('mpm_final', {})):
        mass = model.get('metric_point_mass_kg')
        if (model.get('profile') != 'coarse' or not model.get('active') or model.get('error')
                or mass is None or metrics.get('particle_mass_kg') is None
                or not math.isfinite(float(mass)) or mass <= 0
                or not math.isclose(mass, metrics['particle_mass_kg'], rel_tol=1.e-9)):
            raise ValueError('Coarse model or material-point mass does not match the acceptance report')
    report = episode_mass_quality(meta.get('scoop_results', []), config, metrics)
    if not report['success'] or meta.get('repeat_episode_quality') != report:
        raise ValueError('Coarse mass acceptance failed or disagrees with measured source delivery')
    return report


def install_mass_policy(rt, config):
    """Wrap the already installed legacy episode policy, retaining its verdicts."""
    original = rt.multi_scoop_cycle_report
    snapshot = getattr(rt, 'quality_gate_config_snapshot', None)
    if snapshot is not None:
        def quality_snapshot():
            return {**snapshot(), 'repeat_acceptance_profile': 'coarse_mass',
                    'repeat_mass_policy_version': VERSION,
                    'repeat_min_delivered_kg': config.min_delivered_kg,
                    'repeat_min_retained_delivery_ratio': config.min_retained_delivery_ratio}
        rt.quality_gate_config_snapshot = quality_snapshot

    @wraps(original)
    def cycle_report(execution_success, reason=''):
        event, log = rt.dataset_record_event, rt.info_print
        def legacy_event(name, *args, **kwargs):
            return event('repeat_legacy_episode_end' if name == 'scoop_end' else name, *args, **kwargs)
        def legacy_log(tag, *args, **kwargs):
            return log('[REPEAT LEGACY QUALITY]' if tag == '[REPEAT QUALITY]' else tag, *args, **kwargs)
        rt.dataset_record_event, rt.info_print = legacy_event, legacy_log
        try:
            report = original(execution_success, reason)
        finally:
            rt.dataset_record_event, rt.info_print = event, log
        report.update(legacy_episode_success=bool(report['success']), legacy_episode_reason=report.get('reason', ''))
        failures = [item for item in report.get('reason', '').split(';') if item and not item.startswith(COUNT_REASONS)]
        particle_mass = rt.sand_particle_mass()
        count = report.get('source_delivered_particles')
        kg = (float(count) * float(particle_mass) if count is not None and particle_mass is not None
              and math.isfinite(float(particle_mass)) and particle_mass > 0 else None)
        if kg is None or not math.isfinite(kg):
            failures.append('delivery_mass_unavailable')
        elif kg < config.min_delivered_kg:
            failures.append(f'low_delivered_mass_kg:{kg:.3f}<{config.min_delivered_kg:.3f}')
        if not execution_success and not failures:
            failures.append(str(reason or 'execution_failed'))
        if report.get('freeze_count', 0) > 0 and not any('freeze' in item for item in failures):
            failures.append('freeze_detected')
        report.update(success=not failures, reason=';'.join(failures), acceptance_profile='coarse_mass',
                      source_delivered_mass_kg=kg, minimum_delivered_mass_kg=config.min_delivered_kg,
                      mass_policy_version=VERSION)
        rt.STATE['dataset_scoops_completed'] = sum(bool(row.get('success')) for row in rt.STATE['dataset_scoop_results'])
        event('scoop_end', data={k:report[k] for k in ('scoop_index','success','reason','source_delivered_mass_kg',
              'strict_success','legacy_episode_success','acceptance_profile')})
        log('[REPEAT QUALITY]', f"scoop={report['scoop_index'] + 1} accepted={report['success']}",
            f"profile=coarse_mass delivered_kg={kg} min_kg={config.min_delivered_kg}", force_log=True)
        return report

    rt.multi_scoop_cycle_report = cycle_report
    original_score = getattr(rt, 'compute_episode_quality_score', None)
    if original_score is not None:
        @wraps(original_score)
        def score(execution_success, reason):
            result = dict(original_score(execution_success, reason))
            result['legacy_global_success'] = bool(result.get('success'))
            result['legacy_global_failure_reason'] = result.get('failure_reason', '')
            failures = [item.strip() for item in result.get('failure_reason', '').split(';') if item.strip()]
            hard = [item for item in failures if not item.startswith('quality_rejected/low_bucket_particles:')]
            meta = getattr(getattr(rt, 'repeat_baseline', None), 'meta', None) or {}
            meta['repeat_legacy_global_quality'] = {
                'success': result['legacy_global_success'], 'score': result.get('score'),
                'failure_reason': result['legacy_global_failure_reason'],
                'thresholds': result.get('thresholds', {})}
            verdict = meta.get('repeat_episode_quality', {})
            verified = verdict.get('policy') == VERSION and verdict.get('success')
            if execution_success and verified and not hard:
                result.update(success=True, failure_reason='')
            elif not execution_success:
                result.update(success=False, failure_reason=str(reason or 'execution_failed'))
            elif not verified:
                result.update(success=False, failure_reason='quality_rejected/coarse_mass_verdict_unavailable')
            return result
        rt.compute_episode_quality_score = score
