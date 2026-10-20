"""Small I/O and data-loader helpers shared by training/evaluation entrypoints."""
import json
from pathlib import Path


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def loader_for(dataset, batch_size, workers, shuffle=False):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=False, multiprocessing_context='spawn' if workers else None,
                      persistent_workers=workers > 0, timeout=60 if workers else 0)
