"""Resolve the soil settings that must travel with an evaluation identity."""
import os


def resolve_simulation_settings(soil_backend=None, sand_walls_during_settle=None, environ=None):
    environ = os.environ if environ is None else environ
    backend = (soil_backend if soil_backend is not None else
               environ.get('EXCAVATOR_SOIL_BACKEND', 'newton_mpm')).strip().lower()
    if backend not in ('physx', 'newton_mpm'):
        raise ValueError(f'Unknown soil backend: {backend!r}')
    walls = sand_walls_during_settle
    if walls is None:
        value = environ.get('EXCAVATOR_SAND_WALLS_DURING_SETTLE', '1').strip().lower()
        if value not in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
            raise ValueError(f'Invalid EXCAVATOR_SAND_WALLS_DURING_SETTLE: {value!r}')
        walls = value in ('1', 'true', 'yes', 'on')
    if type(walls) is not bool:
        raise ValueError('Sand wall setting must be a boolean')
    # Newton disables the generation walls' collisions, regardless of their
    # requested cleanup time. Its free pile must not claim PhysX initialization.
    method = ('newton_mpm_free_pile_settle_v1' if backend == 'newton_mpm' else
              'temporary_rectangular_walls_during_settle_v1' if walls else
              'free_pile_settle_without_retaining_walls_v1')
    return {
        'soil_backend': backend,
        'sand_walls_during_settle': walls,
        'sand_initialization_method': method,
        'mpm_environment': {key: value for key, value in sorted(environ.items())
                            if key.startswith('EXCAVATOR_MPM_')} if backend == 'newton_mpm' else {},
    }


def apply_simulation_settings(settings, environ=None):
    """Pin worker/simulator environments to the parent's recorded settings."""
    environ = os.environ if environ is None else environ
    environ['EXCAVATOR_SOIL_BACKEND'] = settings['soil_backend']
    environ['EXCAVATOR_SAND_WALLS_DURING_SETTLE'] = (
        '1' if settings['sand_walls_during_settle'] else '0')
    for key in list(environ):
        if key.startswith('EXCAVATOR_MPM_'):
            del environ[key]
    environ.update(settings['mpm_environment'])
