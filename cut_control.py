"""Complete the checked cut when a small native-grain progress cue fires early."""
from functools import wraps
import math


def install_cut_load_goal(rt, config):
    original = getattr(rt, 'sand_contact_stage_should_advance', None)
    if original is None or config.acceptance_profile != 'coarse_mass':
        return
    goal = 2. * config.min_delivered_kg
    rt.STATE['repeat_cut_advance_policy'] = dict(
        policy='checked_cut_continue_until_load_goal_v1',
        early_advance_bucket_goal_kg=goal,
        minimum_delivery_kg=config.min_delivered_kg,
        scope=['insert_cut', 'pull_mid_cut'],
        full_planned_endpoint_may_complete_below_goal=True,
        physical_limits_changed=False)
    last_logged = None

    @wraps(original)
    def advance(stage_name, report, *args, **kwargs):
        nonlocal last_logged
        decision, reason = original(stage_name, report, *args, **kwargs)
        if (not decision or not rt.STATE.get('repeat_planning_resolution')
                or stage_name not in ('insert_cut', 'pull_mid_cut')):
            return decision, reason
        mass = rt.sand_particle_mass()
        count = report.get('bucket') if isinstance(report, dict) else None
        if (mass is None or not math.isfinite(float(mass)) or mass <= 0
                or count is None or not math.isfinite(float(count)) or count < 0):
            return False, 'repeat_cut_load_measurement_unavailable'
        kg = float(count) * float(mass)
        if kg < goal:
            key = (stage_name, rt.STATE.get('sand_contact_stage_started'))
            logger = getattr(rt, 'info_print', None)
            if key != last_logged and logger is not None:
                logger('[REPEAT CUT GOAL]', f'stage={stage_name} bucket_kg={kg:.3f} goal_kg={goal:.3f}',
                       'continuing_original_checked_trajectory', force_log=True)
                last_logged = key
            return False, f'repeat_continue_checked_cut:bucket_kg={kg:.3f}<goal_kg={goal:.3f}'
        return decision, reason

    rt.sand_contact_stage_should_advance = advance
