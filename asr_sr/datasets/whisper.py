# asr_sr/datasets/whisper.py
import numpy as np
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

    def __getitem__(self, idx: int):
        audio, text, uid = self.reader.get_audio_text(idx)

        text = normalize_sr_text(text)
        text = " " + text.strip()  # Whisper генерирует с ведущим пробелом

        f = self.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            padding="max_length",
        ).input_features[0].astype(np.float32)

        input_length = int(len(audio) * 100 / self.target_sr)
        labels = self.tokenizer(text).input_ids

        if len(labels) > self.max_target_length:
            print(f"Warning: target length {len(labels)} exceeds max_target_length at {uid}")

        return {
            "labels": labels,
            "input_features": f,
            "input_length": input_length,
            "dataset_name": self.dataset_name,
            "text": text,
            "path": uid,
        }


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
    def __init__(self, dataset_dir, split, processor, dataset_name=None):

        reader = HFDatasetReader(
            dataset_dir=dataset_dir,
            split=split,
            dataset_name=dataset_name,
        )

        super().__init__(
            reader=reader,
            processor=processor,
            dataset_name=dataset_name,
        )
        