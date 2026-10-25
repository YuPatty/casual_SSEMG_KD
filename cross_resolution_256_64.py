"""Utilities for 512/128 Teacher -> 256/64 Student response distillation.

The teacher and student operate on different STFT grids.  Teacher responses
are therefore brought back to the waveform domain and re-analysed on the
student grid before computing response-level KD losses.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectrogram_dataset import collect_waveform_pairs
from spectrogram_utils import mag_pha_stft


class WaveformPairDataset(Dataset):
    """Loads paired waveforms; STFT is intentionally deferred to the pipeline."""

    def __init__(self, data_root: str, split: str):
        self.pairs = collect_waveform_pairs(f"{data_root}/{split}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.pairs[idx]
        clean = torch.as_tensor(np.load(clean_path), dtype=torch.float32).contiguous()
        noisy = torch.as_tensor(np.load(noisy_path), dtype=torch.float32).contiguous()
        if clean.ndim != 1 or noisy.ndim != 1:
            raise ValueError(f"Expected 1-D waveforms, got clean={clean.shape}, noisy={noisy.shape}")
        return clean, noisy


def waveform_batch_to_spec(clean_wav: torch.Tensor, noisy_wav: torch.Tensor,
                           cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a waveform batch to [B, 2, F, T] using the supplied model config."""
    h = cfg["model"]
    n_fft = int(h["n_fft"])
    hop = int(h["hop_size"])
    win = int(h["win_size"])
    compress = float(h.get("compress_factor", 1.0))

    def to_spec(wav: torch.Tensor) -> torch.Tensor:
        mag, pha, _ = mag_pha_stft(wav.float(), n_fft, hop, win, compress)
        return torch.stack((mag, pha), dim=1).contiguous()

    return to_spec(clean_wav), to_spec(noisy_wav)


