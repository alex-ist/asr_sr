import math
import random
from torch.utils.data import Sampler


class NoisyBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, sample_weights=None):
        """
        семплер создаёт батчи из похожих по длине элементов, с небольшим шумом, чтобы порядок менялся от эпохи к эпохе

        lengths: список длин (float) для каждого элемента датасета (duration_sec)
        batch_size: размер батча
        shuffle: перемешивать ли порядок батчей между эпохами
        sample_weights: список весов (float) для каждого сэмпла, 0..1.
                        Вес = вероятность включения сэмпла в эпоху.
                        1.0 = всегда включён, 0.33 = ~1/3 эпох.
                        None = все включены (вес 1.0).
        """
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise = 0.05 if shuffle else 0.0   # шум только если есть перемешивание
        self.sample_weights = sample_weights

    def __iter__(self):
        n = len(self.lengths)

        # === Субсемплирование по весам ===
        # Для каждого сэмпла бросаем монетку: включать ли его в эту эпоху.
        # Вес 1.0 — всегда включён, вес 0.33 — включён примерно в 1 из 3 эпох.
        if self.sample_weights is None:
            active_indices = list(range(n))
        else:
            active_indices = []
            for i in range(n):
                w = self.sample_weights[i]
                if w >= 1.0 or random.random() < w:
                    active_indices.append(i)


        # "noisy sort": длина * (1 + небольшой случайный шум) -> батчи похожих длин, но состав меняется от эпохи к эпохе
        if self.noise > 0.000001:
            noisy_lengths = {
                i: self.lengths[i] * (1.0 + random.uniform(-self.noise, self.noise))
                for i in active_indices
            }
        else:
            noisy_lengths = {i: self.lengths[i] for i in active_indices}

        # индексы датасета, отсортированные по зашумлённым длинам
        sorted_indices = sorted(active_indices, key=lambda i: noisy_lengths[i])

        # режем на батчи
        batches = [
            sorted_indices[i:i + self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]

        # порядок батчей перемешать (внутри батча порядок по длине сохраняется)
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.sample_weights is not None:
            expected_n = sum(min(1.0, w) for w in self.sample_weights)
            return math.ceil(expected_n / self.batch_size)
        return math.ceil(len(self.lengths) / self.batch_size)

