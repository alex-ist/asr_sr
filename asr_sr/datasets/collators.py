# asr_sr/datasets/collators.py
import torch
from transformers.feature_extraction_utils import BatchFeature


class WhisperDataCollator:
    def __init__(self, processor=None):
        pass

    def __call__(self, inputs) -> BatchFeature:
        input_features = [x["input_features"] for x in inputs]
        
        input_features_batch = torch.stack(input_features)
        input_lengths = torch.LongTensor([x["input_length"] for x in inputs])

        labels = [x["labels"] for x in inputs]
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        audio_paths = [x["path"] for x in inputs]
        texts = [x["text"] for x in inputs]
        dataset_names = [x["dataset_name"] for x in inputs]

        return BatchFeature({
                "input_features": input_features_batch,
                "input_lengths": input_lengths,
                "labels": labels_padded,
                "dataset_name": dataset_names,
                "path": audio_paths,
                "text": texts,
            })


class ConformerDataCollator:
    def __call__(self, inputs):
        input_features = [x["input_features"] for x in inputs]
        input_lengths = torch.LongTensor([x['input_length'] for x in inputs])
        features_padded = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        
        labels = [x["labels"] for x in inputs]
        lengths = [x.size(0) for x in labels]
        label_lengths = torch.LongTensor(lengths)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

        text = [x['text'] for x in inputs]
        audio_paths = [x["path"] for x in inputs]
        dataset_names = [x["dataset_name"] for x in inputs]        

        return BatchFeature({
            'input_features': features_padded,      # (B, T_max, n_mels)
            'input_lengths': input_lengths,   # (B,)
            'labels': labels_padded,          # (B, L_max)
            'label_lengths': label_lengths,   # (B,)
            "dataset_name": dataset_names,
            "path": audio_paths,
            'text': text,
        })