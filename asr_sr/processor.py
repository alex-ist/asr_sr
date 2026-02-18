import numpy as np
import torch
import torchaudio


class SerbianCTCProcessor:
    def __init__(self, sample_rate=16000, n_mels=80, win_length=400, hop_length=160, n_fft=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft
        )

        self.build_vocabulary()

    def build_vocabulary(self):
        self.blank_id = 0
        self.char_to_idx = {
            '_': 0,
            ' ': 1,
            'a': 2,
            'b': 3,
            'c': 4,
            'č': 5,
            'ć': 6,
            'd': 7,
            'dž': 8,
            'đ': 9,
            'e': 10,
            'f': 11,
            'g': 12,
            'h': 13,
            'i': 14,
            'j': 15,
            'k': 16,
            'l': 17,
            'lj': 18,
            'm': 19,          
            'n': 20,
            'nj': 21,
            'o': 22,
            'p': 23,
            'r': 24,
            's': 25,
            'š': 26,
            't': 27,
            'u': 28,
            'v': 29,
            'z': 30,
            'ž': 31,
        }
        self.digraph_to_idx = {
                "dž": self.char_to_idx.get("dž"),
                "lj": self.char_to_idx.get("lj"),
                "nj": self.char_to_idx.get("nj"),
            }
        
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        return self.char_to_idx

    def audio_to_features(self, waveform):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))

        # (n_mels, time) -> (time, n_mels)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)

        return mel_spec
    
    def text_to_indices(self, text):
        idxs = []
        n = len(text)
        i = 0
        while i < n:
            if i + 1 < n:
                di = self.digraph_to_idx.get(text[i : i + 2])
                if di is not None:
                    idxs.append(di)
                    i += 2
                    continue

            ci = self.char_to_idx.get(text[i])
            if ci is not None:
                idxs.append(ci)
            i += 1
        return idxs

    def indices_to_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        res = ''
        for idx in indices:
            c = self.idx_to_char.get(int(idx), '')
            if c == '_':
                continue
            res += c
        return res

