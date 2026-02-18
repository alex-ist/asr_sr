# asr_sr/datasets/readers.py
import os
import io
import csv

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from datasets import load_dataset, load_from_disk, Audio
from mutagen.mp3 import MP3


class BaseReader:
    MIN_AUDIO_SEC = 0.2
    MAX_AUDIO_SEC = 30.0

    def __len__(self):
        raise NotImplementedError

    def get_audio_text(self, idx: int):
        """Return: (audio: np.ndarray float32 mono 16k, text: str, uid: str)"""
        raise NotImplementedError

    def total_duration(self) -> float:
        return sum(self.lengths)

    def duration(self, idx: int) -> float:
        raise NotImplementedError

    def load_audio_from_path(self, path: str):
        """Load audio from path and return as numpy array (float32, mono, 16kHz)."""
        raise NotImplementedError


class TSVFileReader(BaseReader):
    def __init__(self, df, dataset_dir, dataset_name):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.target_sr = 16000

    def __len__(self):
        return len(self.df)

    def get_wav_path(self, row):
        raise NotImplementedError

    def duration_from_path(self, path: str) -> float:
        ext = path.lower().rsplit(".", 1)[-1]
        if ext == "mp3":
            m = MP3(path)
            if m.info and m.info.length:
                return float(m.info.length)
        return float(librosa.get_duration(path=path))

    def duration(self, idx: int) -> float:
        row = self.df.iloc[idx]
        wav_path = self.get_wav_path(row)
        return self.duration_from_path(wav_path)

    def filter_by_duration(self):
        """Filter dataframe by duration, populate self.lengths."""
        if "duration" not in self.df.columns:
            durations = []
            for i in range(len(self.df)):
                try:
                    durations.append(self.duration(i))
                except Exception:
                    durations.append(0.0)
            self.df["duration"] = durations

        original_len = len(self.df)
        too_short = int((self.df["duration"] < self.MIN_AUDIO_SEC).sum())
        too_long = int((self.df["duration"] > self.MAX_AUDIO_SEC).sum())

        self.df = self.df[
            (self.df["duration"] >= self.MIN_AUDIO_SEC)
            & (self.df["duration"] <= self.MAX_AUDIO_SEC)
        ].reset_index(drop=True)

        self.lengths = self.df["duration"].astype(float).tolist()

        print(f"Reader '{self.dataset_name}':")
        print(f"  Original: {original_len} samples")
        if too_short:
            print(f"  Removed too short (<{self.MIN_AUDIO_SEC}s): {too_short}")
        if too_long:
            print(f"  Removed too long (>{self.MAX_AUDIO_SEC}s): {too_long}")
        if too_long or too_short:
            print(f"  Kept: {len(self.df)} samples")

    def get_audio_text(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = self.get_wav_path(row)
        audio, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
        audio = audio.astype(np.float32, copy=False)
        text = row["sentence"] if "sentence" in row else ""

        return audio, text, wav_path

    def load_audio_from_path(self, path: str):
        """Load audio from path and return as numpy array (float32, mono, 16kHz)."""
        audio, sr = librosa.load(path, sr=self.target_sr, mono=True)
        audio = audio.astype(np.float32, copy=False)
        return audio

class SegmentsTSVReader(TSVFileReader):
    def __init__(self, dataset_dir, dataset_name, tsv_name="segments.tsv", pseudo_conf=None):
        tsv_path = os.path.join(dataset_dir, tsv_name)
        self.pseudo_conf = pseudo_conf

        df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE)
        if pseudo_conf is not None:
            df = df[df["pseudo_conf"] >= pseudo_conf].copy()
            if "sentence" in df.columns:
                df = df.drop(columns=["sentence"])
            df = df.rename(columns={"pseudo_text": "sentence"})

        if "sentence" in df.columns:
            df = df.dropna(subset=["path", "sentence"]).copy()
        else:
            df = df.dropna(subset=["path"]).copy()

        super().__init__(
            df=df,
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
        )
        self.filter_by_duration()

    def get_wav_path(self, row):
        return os.path.join(self.dataset_dir, "clips", row["path"])


class YodasCTCReader(TSVFileReader):
    def __init__(self, dataset_dir):
        items = []
        with open(os.path.join(dataset_dir, "text.ctc"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(maxsplit=1)
                wav_path = os.path.join(dataset_dir, "audio", f"{utt_id}.flac")
                if os.path.exists(wav_path):
                    items.append((utt_id, text))

        df = pd.DataFrame(items, columns=["utt_id", "sentence"])

        super().__init__(
            df=df,
            dataset_dir=dataset_dir,
            dataset_name="yodas_sr",
        )
        self.filter_by_duration()

    def get_wav_path(self, row):
        return os.path.join(self.dataset_dir, "audio", f"{row['utt_id']}.flac")


class HFDatasetReader(BaseReader):
    def __init__(self, dataset_dir, split, dataset_name=None, exclude_paths=None):
        self.path = f"{dataset_dir}/{split}"
        self.split = split
        self.exclude_paths = set(exclude_paths) if exclude_paths else set()

        if dataset_name is None:
            self.ds = load_from_disk(self.path).with_format(None)
            self.dataset_name = os.path.basename(dataset_dir.rstrip("/"))
        else:
            self.ds = load_dataset(
                dataset_name,
                split=split,
                cache_dir=dataset_dir
            )
            self.dataset_name = dataset_name

        # Disable built-in audio decoder
        self.ds = self.ds.cast_column("audio", Audio(decode=False))
        self.table = self.ds.data
        self.target_sr = 16000

        candidates = ["transcript", "sentence", "text", "transcription"]
        self.text_col = next((c for c in candidates if c in self.ds.column_names), None)
        if self.text_col is None:
            raise ValueError(f"No text column found. Available columns: {self.ds.column_names}")

        self._filter_by_duration()

    def __len__(self):
        return len(self.ds)

    def _hf_audio_to_np(self, audio_cell):
        # HF Audio can be {'bytes': ...} or {'array': ..., 'sampling_rate': ...}
        if isinstance(audio_cell, dict) and "array" in audio_cell and "sampling_rate" in audio_cell:
            audio = np.asarray(audio_cell["array"], dtype=np.float32)
            sr = int(audio_cell["sampling_rate"])
        else:
            wav_bytes = audio_cell["bytes"]
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

        if audio.size == 0:
            raise ValueError("empty audio (0 samples)")

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return audio

    def _hf_duration(self, audio_cell) -> float:
        if isinstance(audio_cell, dict) and "bytes" in audio_cell and audio_cell["bytes"] is not None:
            info = sf.info(io.BytesIO(audio_cell["bytes"]))
            if info.frames == 0:
                return 0.0
            return float(info.frames) / float(info.samplerate)

        if isinstance(audio_cell, dict) and "array" in audio_cell and "sampling_rate" in audio_cell:
            sr = int(audio_cell["sampling_rate"])
            arr = np.asarray(audio_cell["array"])
            if arr.ndim == 2:
                arr = arr.mean(axis=-1)
            return float(arr.shape[0]) / float(sr)

        return 0.0

    def duration(self, idx: int) -> float:
        audio_cell = self.ds[idx]["audio"]
        return self._hf_duration(audio_cell)

    def _filter_by_duration(self):
        """Filter HF dataset by duration, populate self.lengths."""
        keep = []
        keep_durations = []
        self.audio_paths = {}
        too_short = 0
        too_long = 0
        unreadable = 0
        excluded_by_path = 0

        print(f"Reader '{self.dataset_name}/{self.split}':")

        # Before select(), table matches ds and can be read directly
        audio_col = self.table.column("audio")

        for i in range(len(self.ds)):
            try:
                audio_cell = audio_col[i].as_py()

                audio_path = audio_cell.get("path", "")
                if audio_path in self.exclude_paths:
                    excluded_by_path += 1
                    continue

                dur = self._hf_duration(audio_cell)
                if dur < self.MIN_AUDIO_SEC:
                    too_short += 1
                elif dur > self.MAX_AUDIO_SEC:
                    too_long += 1
                else:
                    self.audio_paths[audio_path] = len(keep)
                    keep.append(i)
                    keep_durations.append(float(dur))
            except Exception:
                unreadable += 1

        original_len = len(self.ds)
        self.ds = self.ds.select(keep)
        # After select(), ds.data may be the original table with index mapping
        self.lengths = keep_durations

        print(f"  Original: {original_len} samples")
        if excluded_by_path:
            print(f"  Excluded by path: {excluded_by_path}")
        if too_short:
            print(f"  Removed too short (<{self.MIN_AUDIO_SEC}s): {too_short}")
        if too_long:
            print(f"  Removed too long (>{self.MAX_AUDIO_SEC}s): {too_long}")
        if unreadable:
            print(f"  Unreadable: {unreadable}")
        if too_long or too_short or unreadable or excluded_by_path:
            print(f"  Kept: {len(self.ds)} samples")

    def get_audio_text(self, idx: int):
        row = self.ds[idx]
        audio_cell = row["audio"]
        text = row[self.text_col]
        audio = self._hf_audio_to_np(audio_cell)
        audio_path = audio_cell.get("path")
        return audio, text, audio_path

    def load_audio_from_path(self, path: str):
        """Load audio from path and return as numpy array (float32, mono, 16kHz)."""
        filtered_idx = self.audio_paths.get(path, None)
        if filtered_idx is None:
            raise ValueError(f"Audio path '{path}' not found in dataset.")
        audio_cell = self.ds[filtered_idx]["audio"]
        return self._hf_audio_to_np(audio_cell)
