"""Newton provenance follows material IDs, including soil avalanched during settle."""
import numbers

import numpy as np

SOURCE_IDENTITY = 'newton_source_material_ids_excluding_initial_bucket_and_truck'


def initial_source_mask(points, bucket_ids, truck_ids):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Newton source identity requires finite material-point positions')
    mask = np.ones(len(points), dtype=bool)
    for index in set(bucket_ids) | set(truck_ids):
        if isinstance(index, bool) or not isinstance(index, numbers.Integral) or not 0 <= index < len(points):
            raise ValueError('Invalid initial bucket/truck material ID')
        mask[int(index)] = False
    return mask
