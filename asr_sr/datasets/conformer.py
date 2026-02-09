# asr_sr/datasets/conformer.py
import math
import random
from typing import List

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, Sampler

from ..text import normalize_sr_text
from .readers import HFDatasetReader, SegmentsTSVReader, YodasCTCReader


class ConformerDataset(Dataset):
    """Conformer CTC dataset — чтение аудио/текста через Reader."""

    def __init__(self, reader, processor, dataset_name):
        self.reader = reader
        self.processor = processor
        self.dataset_name = dataset_name
        self.target_sr = 16000

    @property
    def lengths(self) -> List[float]:
        return self.reader.lengths

    def total_duration(self) -> float:
        return self.reader.total_duration()

    def __len__(self):
        return len(self.reader)

    def get_audio_text(self, idx: int):
        return self.reader.get_audio_text(idx)

    def __getitem__(self, idx: int):
        audio, text, uid = self.get_audio_text(idx)
            
        text = normalize_sr_text(text)

        features = self.processor.audio_to_features(audio)
        input_length = features.shape[0]

        idxs = self.processor.text_to_indices(text)
        label_indices = torch.tensor(idxs, dtype=torch.long)

        return {
            "input_features": features,
            "input_length": input_length,
            "labels": label_indices,
            "dataset_name": self.dataset_name,
            "text": text,
            "path": uid,
        }


# ===== Конкретные датасеты (аналог whisper.py) =====

class BookConformerDataset(ConformerDataset):
    def __init__(self, dataset_dir, processor, dataset_name, pseudo_conf = None):
        reader = SegmentsTSVReader(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            pseudo_conf=pseudo_conf,
        )
        super().__init__(
            reader=reader, 
            processor=processor, 
            dataset_name=dataset_name
        )


class CommonVoiceConformerDataset(ConformerDataset):
    def __init__(self, dataset_dir, processor, split_name):
        dataset_name = f"common_voice/{split_name}"
        reader = SegmentsTSVReader(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            tsv_name=f"{split_name}.tsv",
        )
        super().__init__(reader=reader, processor=processor, dataset_name=dataset_name)


class YodasConformerDataset(ConformerDataset):
    def __init__(self, dataset_dir, processor):
        reader = YodasCTCReader(dataset_dir)
        super().__init__(reader=reader, processor=processor, dataset_name="yodas_sr")


class HFConformerDataset(ConformerDataset):
    def __init__(self, dataset_dir, processor, split, dataset_name=None):
        reader = HFDatasetReader(
            dataset_dir=dataset_dir,
            split=split,
            dataset_name=dataset_name,
        )
        super().__init__(
            reader=reader,
            processor=processor,
            dataset_name=dataset_name or reader.dataset_name,
        )


# ===== Collate =====

def collate_fn(batch):
    """Батчинг последовательностей разной длины."""
    features = [item["features"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_lengths = torch.tensor([item["input_length"] for item in batch], dtype=torch.long)
    label_lengths = torch.tensor([item["label_length"] for item in batch], dtype=torch.long)

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    transcripts = [item["transcript"] for item in batch]

    return {
        "features": features_padded,      # (B, T_max, n_mels)
        "labels": labels_padded,           # (B, L_max)
        "input_lengths": input_lengths,    # (B,)
        "label_lengths": label_lengths,    # (B,)
        "transcripts": transcripts,
    }


# ===== Subset / ConcatDataset =====
class ConformerSubset(Subset):
    @property
    def reader(self):
        return self.dataset.reader

    @property
    def lengths(self) -> List[float]:
        base = self.dataset.lengths
        return [base[i] for i in self.indices]

    def total_duration(self) -> float:
        return sum(self.lengths)


class ConformerConcatDataset(ConcatDataset):
    @property
    def lengths(self) -> List[float]:
        out: List[float] = []
        for ds in self.datasets:
            if hasattr(ds, "lengths"):
                out.extend(ds.lengths)
            else:
                raise TypeError(f"Dataset {type(ds)} has no lengths")
        return out

    def total_duration(self) -> float:
        total = 0.0
        for ds in self.datasets:
            if hasattr(ds, "total_duration"):
                total += ds.total_duration()
            else:
                raise TypeError(f"Dataset {type(ds)} has no total_duration()")
        return total


# ===== Lengths / Sampler =====
class NoisyBucketBatchSampler(Sampler):
    """Семплер: батчи из похожих по длине элементов с небольшим шумом."""

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise = 0.05 if shuffle else 0.0

    def __iter__(self):
        n = len(self.lengths)

        if self.noise > 1e-6:
            noisy_lengths = [
                self.lengths[i] * (1.0 + random.uniform(-self.noise, self.noise))
                for i in range(n)
            ]
        else:
            noisy_lengths = self.lengths

        sorted_indices = sorted(range(n), key=lambda i: noisy_lengths[i])

        batches = [
            sorted_indices[i : i + self.batch_size]
            for i in range(0, n, self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)
