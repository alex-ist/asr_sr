# asr_sr/datasets/readers.py
import os
import io
import csv

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from datasets import load_dataset, load_from_disk
from mutagen.mp3 import MP3


class BaseReader:
    def __len__(self):
        raise NotImplementedError

    def get_audio_text(self, idx: int):
        """Return: (audio: np.ndarray float32 mono 16k, text: str, uid: str)"""
        raise NotImplementedError

    def total_duration(self):
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

    def total_duration(self):
        total = 0.0
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            wav_path = self.get_wav_path(row)
            total += self.duration_from_path(wav_path)
        return total

    def get_audio_text(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = self.get_wav_path(row)
        audio, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
        audio = audio.astype(np.float32, copy=False)
        text = row["sentence"] if "sentence" in row else ""
            
        return audio, text, wav_path

class SegmentsTSVReader(TSVFileReader):
    def __init__(self, dataset_dir, dataset_name, tsv_name="segments.tsv", pseudo_conf=None):
        tsv_path = os.path.join(dataset_dir, tsv_name)

        df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE)
        if pseudo_conf is not None:
            df = df[df["pseudo_conf"] >= pseudo_conf].copy()
            if "sentence" in df.columns:
                df = df.drop(columns=["sentence"])
            df = df.rename(columns={"pseudo_text": "sentence"})

        # Проверяем наличие колонки sentence
        if "sentence" in df.columns:
            df = df.dropna(subset=["path", "sentence"]).copy()
        else:
            df = df.dropna(subset=["path"]).copy()

        super().__init__(
            df=df,
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
        )

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

    def get_wav_path(self, row):
        return os.path.join(self.dataset_dir, "audio", f"{row['utt_id']}.flac")




class HFDatasetReader(BaseReader):
    def __init__(self, dataset_dir, split, dataset_name=None):
        self.path = f"{dataset_dir}/{split}"
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

        self.table = self.ds.data
        self.target_sr = 16000

        candidates = ["transcript", "sentence", "text", "transcription"]
        self.text_col = next((c for c in candidates if c in self.ds.column_names), None)
        if self.text_col is None:
            raise ValueError(f"нет текстовой колонки. Есть колонки: {self.ds.column_names}")

    def __len__(self):
        return len(self.ds)

    def _hf_audio_to_np(self, audio_cell):
        # HF Audio может быть {'bytes': ...} или {'array': ..., 'sampling_rate': ...}
        if isinstance(audio_cell, dict) and "array" in audio_cell and "sampling_rate" in audio_cell:
            audio = np.asarray(audio_cell["array"], dtype=np.float32)
            sr = int(audio_cell["sampling_rate"])
        else:
            wav_bytes = audio_cell["bytes"]
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return audio

    def _hf_duration(self, audio_cell) -> float:
        # bytes
        if isinstance(audio_cell, dict) and "bytes" in audio_cell and audio_cell["bytes"] is not None:
            info = sf.info(io.BytesIO(audio_cell["bytes"]))
            return float(info.frames) / float(info.samplerate)

        # array + sampling_rate
        if isinstance(audio_cell, dict) and "array" in audio_cell and "sampling_rate" in audio_cell:
            sr = int(audio_cell["sampling_rate"])
            arr = np.asarray(audio_cell["array"])
            if arr.ndim == 2:
                arr = arr.mean(axis=-1)
            return float(arr.shape[0]) / float(sr)

        return 0.0

    def total_duration(self):
        total = 0.0
        audio_col = self.table.column("audio")
        for i in range(audio_col.length()):
            audio_cell = audio_col[i].as_py()
            total += self._hf_duration(audio_cell)
        return total

    def get_audio_text(self, idx: int):
        row = self.table.slice(idx, 1)
        audio_cell = row.column("audio")[0].as_py()
        
        
        text = row.column(self.text_col)[0].as_py()
        audio = self._hf_audio_to_np(audio_cell)
        return audio, text, f"{idx}"
