# asr_sr/datasets/collators.py
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor  # оставляю как у тебя, не типизирую WhisperProcessor

    def __call__(self, inputs) -> BatchFeature:
        input_features = [x["input_features"] for x in inputs]
        input_features_batch = torch.from_numpy(np.stack(input_features, axis=0))

        input_lengths = torch.LongTensor([x["input_length"] for x in inputs])

        labels = [x["labels"] for x in inputs]
        lengths = [len(x) for x in labels]
        max_length = max(lengths)
        labels_padded = [x + [-100] * (max_length - len(x)) for x in labels]
        labels_padded = torch.LongTensor(labels_padded)

        audio_paths = [x["path"] for x in inputs]
        texts = [x["text"] for x in inputs]
        dataset_names = [x["dataset_name"] for x in inputs]

        return BatchFeature(
            {
                "input_features": input_features_batch,
                "input_lengths": input_lengths,
                "labels": labels_padded,
                "dataset_name": dataset_names,
                "path": audio_paths,
                "text": texts,
            }
        )
