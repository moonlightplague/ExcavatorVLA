"""Particle accounting and timing for online evaluation, without Isaac/ML imports."""
from collections import Counter
import math
from numbers import Integral
import statistics


METRIC_FIELDS = (
    'payload_particles', 'scooped_particles', 'retained_particles',
    'spillage_particles', 'spillage_fraction', 'cycle_time_seconds',
    'cycle_time_wall_seconds', 'simulated_rollout_time_seconds',
    'policy_inference_calls', 'policy_inference_time_seconds',
    'policy_latency_ms_mean', 'policy_latency_ms_p50', 'policy_latency_ms_p95',
    'policy_latency_ms_max', 'bridge_latency_ms_mean', 'bridge_latency_ms_p95',
    'step_latency_ms_mean', 'step_latency_ms_p95', 'fps',
)
METRIC_DEFINITIONS = {
    'payload_particles': 'Initial-pile particles currently in the truck, excluding IDs already there at the first observation.',
    'scooped_particles': 'Unique initial-pile particle IDs observed inside the bucket, excluding initially deposited IDs.',
    'retained_particles': 'Scooped particles currently inside the bucket.',
    'spillage_particles': 'Scooped particles currently outside both bucket and truck; includes material returned to the pile or still in flight.',
    'spillage_fraction': 'Spillage / scooped particles; null when no particles have been scooped.',
    'cycle_time_seconds': 'Simulated action time until success and bucket open; null for incomplete cycles. Excludes the initial zero-action observation.',
    'cycle_time_wall_seconds': 'Wall time from policy rollout start until success and bucket open; null for incomplete cycles. Excludes startup/reset.',
    'simulated_rollout_time_seconds': 'Sum of simulator action durations, including failed/incomplete rollouts; excludes initial zero-action observation.',
    'policy_latency_ms': 'Per replan: image/state preparation, preprocessing, model prediction, postprocessing and CPU action transfer. Cached actions have no inference sample.',
    'bridge_latency_ms': 'Per action: command/reply round trip, including simulation, rendering, encoding and transport.',
    'step_latency_ms': 'Per action: input validation, optional inference, action preparation, bridge round trip and returned-state validation; excludes trace/video writing.',
    'fps': 'Executed policy actions / rollout wall seconds, including trace/video writing; excludes startup/reset, initial observation and final video hold.',
    'aggregation': 'Numeric summary statistics describe completed episodes. Latency means/percentiles are computed within each episode; summary distributions are over episode values. Unavailable measurements are null.',
    'failure_mode': 'none, no_payload, lost_payload, insufficient_payload, unstable_deposit, bucket_not_open, infrastructure_error, validation_error; unclassified for historical results without enough measurements.',
    'particle_sampling': 'Particle IDs are sampled once per returned observation. Transfers occurring entirely between observations are not observed. Counts use the strict initial-pile mask, independently of the observation load fallback.',
}


def numeric_summary(values):
    finite = sorted(float(value) for value in values
                    if value is not None and math.isfinite(float(value)))
    if not finite:
        return dict(count=0, min=None, mean=None, median=None, p95=None, max=None)
    return dict(count=len(finite), min=finite[0], mean=statistics.fmean(finite),
                median=statistics.median(finite),
                p95=finite[max(0, math.ceil(.95 * len(finite)) - 1)], max=finite[-1])


class ParticleTransferTracker:
    """Account for sampled particle IDs; lifetime is one negotiated episode."""
    def __init__(self):
        self.particle_count = None
        self.initial_truck = set()
        self.scooped = set()

    def observe(self, particle_count, bucket_indices, truck_indices):
        if not isinstance(particle_count, Integral) or isinstance(particle_count, bool) or particle_count <= 0:
            raise ValueError('Particle count must be a positive integer')
        if self.particle_count is not None and particle_count != self.particle_count:
            raise ValueError('Particle count changed during evaluation')
        bucket, truck = set(bucket_indices), set(truck_indices)
        for index in bucket | truck:
            if (not isinstance(index, Integral) or isinstance(index, bool)
                    or index < 0 or index >= particle_count):
                raise ValueError(f'Invalid particle index: {index}')
        if bucket & truck:
            raise ValueError('Truck particles must exclude bucket overlap')
        if self.particle_count is None:
            self.particle_count = int(particle_count)
            self.initial_truck = truck.copy()
        self.scooped.update(bucket - self.initial_truck)
        spilled = self.scooped - bucket - truck
        return {
            'available': True, 'method': 'sampled_initial_pile_particle_ids_v1',
            'payload_particles': len(truck - self.initial_truck),
            'scooped_particles': len(self.scooped),
            'retained_particles': len(self.scooped & bucket),
            'spillage_particles': len(spilled),
            'spillage_fraction': len(spilled) / len(self.scooped) if self.scooped else None,
        }


def failure_mode(status):
    if status.get('status') == 'validation_error':
        return 'validation_error'
    if status.get('status') != 'completed':
        return 'infrastructure_error'
    if status.get('task_completed_success_and_bucket_open'):
        return 'none'
    if status.get('success'):
        return 'bucket_not_open'
    payload = status.get('payload_particles')
    if payload is None:
        return 'unclassified'
    minimum = status.get('min_particles')
    if minimum is None:
        return 'unclassified'
    if payload >= minimum:
        return 'unstable_deposit'
    if payload > 0:
        return 'insufficient_payload'
    if status.get('retained_particles', 0) > 0:
        return 'no_payload'
    if status.get('spillage_particles', 0) > 0:
        return 'lost_payload'
    return 'no_payload'


def failure_mode_counts(rows):
    return dict(sorted(Counter(row.get('failure_mode') or failure_mode(row) for row in rows).items()))


class EpisodeMetrics:
    def __init__(self):
        self.transfer = {}
        self.simulated_seconds = 0.0
        self.action_steps = 0
        self.policy_ms = []
        self.bridge_ms = []
        self.step_ms = []

    def observe(self, reply, *, action_step=False):
        transfer = reply.get('particle_transfer_metrics') or {}
        if not transfer.get('available') or transfer.get('method') != 'sampled_initial_pile_particle_ids_v1':
            raise ValueError(f'Invalid particle transfer metric: {transfer}')
        for field in ('payload_particles', 'scooped_particles', 'retained_particles', 'spillage_particles'):
            value = transfer.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'Invalid {field}: {value}')
        scooped = transfer['scooped_particles']
        fraction = transfer.get('spillage_fraction')
        expected = transfer['spillage_particles'] / scooped if scooped else None
        if (transfer['retained_particles'] + transfer['spillage_particles'] > scooped
                or fraction != expected):
            raise ValueError(f'Inconsistent particle transfer metric: {transfer}')
        duration = reply['bridge_timing']['simulated_seconds'] if action_step else 0.0
        if not math.isfinite(duration) or (duration <= 0 and action_step):
            raise ValueError(f'Invalid simulated action duration: {duration}')
        self.transfer = {field: transfer[field] for field in (
            'payload_particles', 'scooped_particles', 'retained_particles',
            'spillage_particles', 'spillage_fraction')}
        if action_step:
            self.simulated_seconds += duration
            self.action_steps += 1

    def finalize(self, status, rollout_wall_seconds):
        result = dict.fromkeys(METRIC_FIELDS)
        result.update(self.transfer)
        result.update(policy_inference_calls=len(self.policy_ms),
                      policy_inference_time_seconds=sum(self.policy_ms) / 1000,
                      simulated_rollout_time_seconds=self.simulated_seconds if self.transfer else None)
        for prefix, values in (('policy', self.policy_ms), ('bridge', self.bridge_ms), ('step', self.step_ms)):
            summary = numeric_summary(values)
            result[f'{prefix}_latency_ms_mean'] = summary['mean']
            result[f'{prefix}_latency_ms_p95'] = summary['p95']
            if prefix == 'policy':
                result['policy_latency_ms_p50'] = summary['median']
                result['policy_latency_ms_max'] = summary['max']
        if rollout_wall_seconds is not None and rollout_wall_seconds > 0 and self.action_steps:
            result['fps'] = self.action_steps / rollout_wall_seconds
        if status.get('status') == 'completed' and status.get('task_completed_success_and_bucket_open'):
            result['cycle_time_seconds'] = self.simulated_seconds
            result['cycle_time_wall_seconds'] = status['task_completion_time_from_policy_start_seconds']
        status.update(result, metrics_schema_version='excavator_evaluation_metrics_v1')
        status['failure_mode'] = failure_mode(status)
