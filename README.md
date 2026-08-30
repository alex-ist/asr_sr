# asr_sr

A collection of reusable components for automatic Serbian speech recognition
(ASR) experiments in Google Colab.

The project emerged from training and comparing two approaches:

- a custom **Conformer-CTC** model;
- fine-tuning **Whisper**.

Shared components were extracted from experimental notebooks to make them
smaller, avoid code duplication, and simplify further experimentation. This
repository is a library for notebooks rather than a ready-to-use application or
speech recognition service.

## Features

- custom Conformer encoder with a CTC classifier;
- relative positional encoding and relative position attention;
- log-mel feature extraction for Conformer;
- feature and token preparation for Whisper;
- Serbian text normalization, including Cyrillic-to-Latin transliteration;
- conversion of integers to Serbian words;
- support for the Serbian digraphs `dž`, `lj`, and `nj` in the CTC vocabulary;
- readers for TSV datasets, YODAS, and Hugging Face Datasets;
- dataset concatenation and subsets;
- padded batch collation;
- bucket batching by audio duration;
- filtering of excessively short and long recordings;
- heuristics for detecting repetition loops in Whisper output.

## Project structure

```text
asr_sr/
├── conformer/
│   └── conformer.py     # Conformer encoder and CTC model
├── datasets/
│   ├── readers.py       # Local and Hugging Face dataset readers
│   ├── conformer.py     # Datasets for Conformer-CTC
│   ├── whisper.py       # Datasets for Whisper
│   ├── collators.py     # Batch collation
│   └── samplers.py      # Bucket batch sampler
├── processor.py         # Log-mel features and Serbian CTC vocabulary
├── text.py              # Serbian text normalization
└── whisper_utils.py     # Additional Whisper utilities
```

## Conformer-CTC

The model is implemented directly in this project and is not a wrapper around a
third-party Conformer implementation.

Default architecture:

1. two 2D convolutional layers that reduce the temporal resolution by a factor
   of four;
2. linear projection of the input features;
3. 17 Conformer blocks with a model dimension of 512;
4. 8 relative position attention heads;
5. a convolution module with a kernel size of 31;
6. a linear CTC projection into the character vocabulary.

Each Conformer block has the following structure:

```text
1/2 Feed Forward → Relative Self-Attention → Convolution
                 → 1/2 Feed Forward → LayerNorm
```

Residual connections are used throughout the main modules. The relative
position and masking implementations were refined and debugged separately
during experimentation.

Minimal model initialization:

```python
from asr_sr.conformer import ConformerCTC
from asr_sr.processor import SerbianCTCProcessor

processor = SerbianCTCProcessor()

model = ConformerCTC(
    num_classes=processor.vocab_size,
    input_dim=80,
    encoder_dim=512,
    num_layers=17,
    num_attention_heads=8,
    conv_kernel_size=31,
    dropout_p=0.1,
)
```

Forward pass:

```python
import torch

# input_features: [batch, time, 80]
# input_lengths:  [batch]
logits, output_lengths = model(input_features, input_lengths)

# torch.nn.CTCLoss expects [time, batch, classes].
log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
```

Loss function example:

```python
criterion = torch.nn.CTCLoss(
    blank=processor.blank_id,
    reduction="mean",
    zero_infinity=True,
)

loss = criterion(
    log_probs,
    labels,
    output_lengths,
    label_lengths,
)
```

## Serbian text normalization

Before training, text is converted to a consistent representation:

```python
from asr_sr.text import normalize_sr_text

text = normalize_sr_text("Данас је 25. август!")
print(text)
# danas je dvadeset pet avgust
```

The normalizer:

- converts text to lowercase;
- transliterates Serbian Cyrillic to Latin;
- replaces `w`, `q`, `y`, and `x` with supported equivalents;
- converts numbers from 0 to 999,999,999 into words;
- removes punctuation and unsupported characters;
- normalizes whitespace.

The CTC processor uses dedicated tokens for `dž`, `lj`, and `nj`, so they are
encoded as individual letters of the Serbian alphabet.

## Data formats

All readers return a tuple:

```text
(audio, text, uid)
```

Here, `audio` is mono `float32` sampled at 16 kHz, `text` is the transcript, and
`uid` is the path or recording identifier.

Recordings are automatically filtered by duration when a reader is created. By
default, segments between 0.2 and 30 seconds are accepted.

### TSV dataset

`BookConformerDataset`, `BookWhisperDataset`, and the Common Voice datasets
expect the following structure:

```text
dataset/
├── segments.tsv
└── clips/
    ├── sample-0001.wav
    └── sample-0002.mp3
```

Minimal `segments.tsv`:

```tsv
path<TAB>sentence<TAB>duration
sample-0001.wav<TAB>dobar dan<TAB>2.4
sample-0002.mp3<TAB>kako ste<TAB>1.8
```

Here, `<TAB>` represents an actual tab character.

The `duration` column is optional. If it is absent, file durations are computed
during initialization. The `pseudo_text` and `pseudo_conf` columns are also
supported for pseudo-labeling experiments.

### YODAS

Expected structure:

```text
dataset/
├── text.ctc
└── audio/
    └── <utterance_id>.flac
```

### Hugging Face Datasets

Both datasets saved with `save_to_disk` and datasets loaded by name from the
Hugging Face Hub are supported. A dataset must contain an `audio` column and one
of the following text columns:

- `transcript`;
- `sentence`;
- `text`;
- `transcription`.

## Conformer DataLoader example

```python
from torch.utils.data import DataLoader

from asr_sr.datasets import (
    BookConformerDataset,
    ConformerDataCollator,
    NoisyBucketBatchSampler,
)
from asr_sr.processor import SerbianCTCProcessor

processor = SerbianCTCProcessor()
dataset = BookConformerDataset(
    dataset_dir="/content/data/my_dataset",
    processor=processor,
    dataset_name="my_dataset",
)

batch_sampler = NoisyBucketBatchSampler(
    lengths=dataset.lengths,
    batch_size=16,
    shuffle=True,
)

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=ConformerDataCollator(),
    num_workers=2,
)

batch = next(iter(loader))
logits, output_lengths = model(
    batch["input_features"],
    batch["input_lengths"],
)
```

## Whisper DataLoader example

The Whisper dataset accepts a Transformers `WhisperProcessor` and uses its
tokenizer and feature extractor:

```python
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from asr_sr.datasets import BookWhisperDataset, WhisperDataCollator

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Serbian",
    task="transcribe",
)

dataset = BookWhisperDataset(
    dataset_dir="/content/data/my_dataset",
    dataset_name="my_dataset",
    processor=processor,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=WhisperDataCollator(),
    num_workers=2,
)
```

Whisper training remains in the experimental notebook. This repository handles
data reading, normalization, and preparation.

## Dependencies

The code uses the following libraries:

- Python 3.10+;
- PyTorch and TorchAudio;
- Transformers;
- Hugging Face Datasets;
- NumPy and pandas;
- librosa and SoundFile;
- Mutagen;
- srtools.

Example dependency installation in Colab:

```bash
pip install torch torchaudio transformers datasets numpy pandas \
    librosa soundfile mutagen srtools
```

Dependency versions are not currently pinned and should match the environment
used by the corresponding experimental notebook.

## Current limitations

- The repository does not contain a complete training loop; training and result
  analysis are performed in separate Colab notebooks.
- There is no built-in beam-search decoder or language model.

These limitations reflect the project's current purpose: maintaining stable,
shared code used by several research notebooks.
