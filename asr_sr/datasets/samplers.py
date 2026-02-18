import math
import random
from torch.utils.data import Sampler


class NoisyBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, sort=True, sample_weights=None):
        """
        Bucket batching sampler with noisy sorting.

        Args:
            lengths: list of sample durations (float, in seconds)
            batch_size: batch size
            shuffle: shuffle batch order between epochs
            sort: enable bucket batching by length
                True  + shuffle=True  — noisy bucket sort + shuffle batches
                True  + shuffle=False — deterministic length sorting
                False + shuffle=True  — full random shuffle
                False + shuffle=False — sequential order
            sample_weights: list of sample weights (float, 0..1)
                Weight = probability of including sample in an epoch
                1.0 = always included, 0.33 = ~1/3 epochs
                None = all samples included (weight 1.0)
        """
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sort = sort
        self.noise = 0.05 if (shuffle and sort) else 0.0
        self.sample_weights = sample_weights

    def __iter__(self):
        n = len(self.lengths)

        # Subsampling by weights: each sample included with probability = weight
        if self.sample_weights is None:
            active_indices = list(range(n))
        else:
            active_indices = []
            for i in range(n):
                w = self.sample_weights[i]
                if w >= 1.0 or random.random() < w:
                    active_indices.append(i)


        if self.sort:
            # Noisy bucket sort: length * (1 + small random noise)
            if self.noise > 0.000001:
                noisy_lengths = {
                    i: self.lengths[i] * (1.0 + random.uniform(-self.noise, self.noise))
                    for i in active_indices
                }
            else:
                noisy_lengths = {i: self.lengths[i] for i in active_indices}

            ordered = sorted(active_indices, key=lambda i: noisy_lengths[i])
        else:
            ordered = active_indices
            if self.shuffle:
                random.shuffle(ordered)

        batches = [
            ordered[i:i + self.batch_size]
            for i in range(0, len(ordered), self.batch_size)
        ]

        if self.shuffle and self.sort:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.sample_weights is not None:
            expected_n = sum(min(1.0, w) for w in self.sample_weights)
            return math.ceil(expected_n / self.batch_size)
        return math.ceil(len(self.lengths) / self.batch_size)

