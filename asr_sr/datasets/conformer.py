# asr_sr/datasets/conformer.py
from typing import List

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, Sampler

from ..text import normalize_sr_text
from .readers import HFDatasetReader, SegmentsTSVReader, YodasCTCReader


class ConformerDataset(Dataset):
    """Conformer CTC dataset with audio/text reading via Reader."""

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


class ConformerSubset(Subset):
    def __init__(self, dataset, indices, dataset_name=None):
        super().__init__(dataset, indices)
        self.dataset_name = dataset_name or getattr(dataset, "dataset_name", type(dataset).__name__)

    @property
    def reader(self):
        return self.dataset.reader

    @property
    def lengths(self) -> List[float]:
        base = self.dataset.lengths
        return [base[i] for i in self.indices]

    def total_duration(self) -> float:
        return sum(self.lengths)

    def get_audio_text(self, idx: int):
        return self.dataset.get_audio_text(self.indices[idx])


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

    def get_audio_text(self, idx: int):
        from bisect import bisect_right
        ds_idx = bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx - (self.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0)
        return self.datasets[ds_idx].get_audio_text(sample_idx)
