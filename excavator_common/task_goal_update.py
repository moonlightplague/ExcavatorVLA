"""Opt-in task-goal revision at an existing observation; never advance physics."""
import copy
import math

from .vla_observation_contract import point_delta_in_heading_frame


def update_task_goals(context, observation, command, upper_heading_rad):
    if context.get('allow_task_goal_updates') is not True:
        raise ValueError('Task-goal updates were not enabled in the handshake')
    if observation is None or not observation.get('observation_32d_ready'):
        raise ValueError('A valid current-v4 observation is required before a goal update')
    required = {'type', 'revision', 'dig_target_xyz', 'unload_landing_xyz'}
    if set(command) - required - {'_bridge_connection_id'} or not required.issubset(command):
        raise ValueError('Goal updates accept only revision and two task coordinates')
    revision = command['revision']
    if type(revision) is not int or revision != context.get('goal_revision', 0) + 1:
        raise ValueError('Task-goal revision must increase by exactly one')
    updated = dict(context)
    for name in ('dig_target_xyz', 'unload_landing_xyz'):
        value = command[name]
        if (not isinstance(value, list) or len(value) != 3 or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
            raise ValueError(f'Expected three finite coordinates: {name}')
        updated[name] = list(value)
    updated['goal_revision'] = revision
    reply = copy.deepcopy(observation)
    state = list(reply['observation_state_28d'])
    if len(state) != 28:
        raise ValueError('Task-goal updates require the current-v4 28D schema')
    landmarks = reply['bucket_landmarks_world']
    state[18:21] = point_delta_in_heading_frame(updated['dig_target_xyz'], landmarks['tip'], upper_heading_rad)
    state[21:24] = point_delta_in_heading_frame(updated['unload_landing_xyz'], landmarks['load'], upper_heading_rad)
    reply.update(type='task_goals_ack', ok=True, goal_revision=revision,
                 observation_state_28d=state, observation_state=list(state),
                 observation_context=updated, task_text=updated['task_text'])
    reply['bridge_timing'] = {**reply['bridge_timing'], 'physics_ticks': 0,
        'simulated_seconds': 0.0, 'physics_time_advanced_seconds': 0.0}
    return updated, reply
