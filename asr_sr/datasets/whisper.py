# asr_sr/datasets/whisper.py
import torch
from torch.utils.data import Dataset

from ..text import normalize_sr_text
from .readers import HFDatasetReader, SegmentsTSVReader, YodasCTCReader


class WhisperDataset(Dataset):
    def __init__(self, reader, processor, dataset_name):
        self.reader = reader
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor
        self.target_sr = 16000
        self.dataset_name = dataset_name
        self.max_target_length = 448

    def __len__(self):
        return len(self.reader)

    def total_duration(self):
        return self.reader.total_duration()

    def get_audio_text(self, idx: int):
        return self.reader.get_audio_text(idx)

    def load_audio_from_path(self, path: str):
        """Load audio from path and return as numpy array (float32, mono, 16kHz)."""
        return self.reader.load_audio_from_path(path)

    def create_subset_by_paths(self, paths, dataset_name=None):
        """Create a subset containing only samples with paths in the given list.

        Args:
            paths: list of paths (strings) to include in the subset
            dataset_name: optional name for the subset

        Returns:
            WhisperSubset with only the matching samples
        """
        paths_set = set(paths)
        indices = []

        for idx in range(len(self)):
            _, _, uid = self.get_audio_text(idx)
            if uid in paths_set:
                indices.append(idx)

        return WhisperSubset(self, indices, dataset_name=dataset_name)

    def __getitem__(self, idx: int):
        audio, text, uid = self.get_audio_text(idx)

        text = normalize_sr_text(text)
        text = " " + text.strip()  # Whisper generates with leading space

        f = self.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            padding="max_length",
            return_tensors="pt"
        ).input_features[0]  # np.ndarray (n_mels, 3000)

        # input_length is the number of audio frames (features are always 3000 frames = 30s)
        input_length = len(audio)//160

        labels = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        if labels.size(0) > self.max_target_length:
            print(f"Warning: target length {labels.size(0)} exceeds max_target_length at {uid}")


        return {
            "input_features": f,
            "input_length": input_length,
            "labels": labels,
            "dataset_name": self.dataset_name,
            "text": text,
            "path": uid,
        }


#fixme: dataset_name после processor'а - когда будет сдан проект
class BookWhisperDataset(WhisperDataset):
    def __init__(self, dataset_dir, dataset_name, processor, pseudo_conf = None):
        reader = SegmentsTSVReader(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            pseudo_conf=pseudo_conf,
        )

        super().__init__(
            reader=reader,
            processor=processor,
            dataset_name=dataset_name,
        )


class CommonVoiceWhisperDataset(WhisperDataset):
    def __init__(self, dataset_dir, split_name, processor):
        dataset_name = f"common_voice/{split_name}"
        reader = SegmentsTSVReader(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            tsv_name=f"{split_name}.tsv",
        )

        super().__init__(
            reader=reader,
            processor=processor,
            dataset_name=dataset_name,
        )

class YodasWhisperDataset(WhisperDataset):
    def __init__(self, dataset_dir, processor):
        reader = YodasCTCReader(dataset_dir)
        super().__init__(
            reader=reader,
            processor=processor,
            dataset_name="yodas_sr",
        )


# ===== HF =====
class HFWhisperDataset(WhisperDataset):
    def __init__(self, dataset_dir, split, processor, dataset_name=None, exclude_paths=None):

        reader = HFDatasetReader(
            dataset_dir=dataset_dir,
            split=split,
            dataset_name=dataset_name,
            exclude_paths=exclude_paths,
        )

        super().__init__(
            reader=reader,
            processor=processor,
            dataset_name=dataset_name or reader.dataset_name,
        )
        
from torch.utils.data import Subset, ConcatDataset

class WhisperSubset(Subset):
    def __init__(self, dataset, indices, dataset_name=None):
        super().__init__(dataset, indices)
        self.dataset_name = dataset_name or getattr(dataset, "dataset_name", type(dataset).__name__)
    
    @property
    def reader(self):
        return self.dataset.reader

    @property
    def lengths(self):
        base = self.dataset.reader.lengths
        return [base[i] for i in self.indices]

    def total_duration(self) -> float:
        return sum(self.lengths)

    def get_audio_text(self, idx: int):
        return self.dataset.get_audio_text(self.indices[idx])
    
class WhisperConcatDataset(ConcatDataset):
    @property
    def reader(self):
        # Return the reader of the first sub-dataset (for compatibility)
        for ds in self.datasets:
            if hasattr(ds, "reader"):
                return ds.reader
        raise AttributeError("No sub-dataset has a reader attribute")

    @property
    def lengths(self):
        out = []
        for ds in self.datasets:
            if hasattr(ds, "lengths"):
                out.extend(ds.lengths)
            elif hasattr(ds, "reader"):
                out.extend(ds.reader.lengths)
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