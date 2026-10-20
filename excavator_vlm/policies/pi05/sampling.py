"""Deterministic consumed-batch addressing, unaffected by worker prefetch."""
import math

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler):
    def __init__(self, size, batch_size, seed, rank=0, world_size=1, start_batch=0):
        if size < 1 or batch_size < 1 or not 0 <= rank < world_size or start_batch < 0:
            raise ValueError('Invalid sampler dimensions or progress')
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.rank, self.world_size, self.start_batch = rank, world_size, start_batch
        self.rank_size = math.ceil(size / world_size)
        self.batches_per_epoch = math.ceil(self.rank_size / batch_size)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        cursor = self.start_batch
        while True:
            epoch, offset = divmod(cursor, self.batches_per_epoch)
            generator = torch.Generator().manual_seed(self.seed + epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
            needed = self.rank_size * self.world_size
            indices = (indices * math.ceil(needed / self.size))[:needed]
            indices = indices[self.rank::self.world_size]
            for index in range(offset, self.batches_per_epoch):
                start = index * self.batch_size
                yield indices[start:start + self.batch_size]
                cursor += 1
