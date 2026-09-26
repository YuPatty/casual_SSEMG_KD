# spectrogram_dataset.py
"""
在 Dataset 讀取時即時（on-the-fly）把 sEMG 波形 (.npy) 轉成頻譜圖，
取代原本需要先跑 convert.py 產生 spectrogram_cf05*/ *.pt 快取檔的做法。

STFT 參數與 convert.py 完全對齊：
    n_fft=512, hop_size=128, win_size=512, compress_factor=0.5

用法（放進 repo 根目錄，跟 spectrogram_utils.py 同一層）：
    from spectrogram_dataset import OnTheFlySpectrogramDataset

    train_ds = OnTheFlySpectrogramDataset(data_root="semg_data/processed", split="train")
    valid_ds = OnTheFlySpectrogramDataset(data_root="semg_data/processed", split="valid")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                               drop_last=True, num_workers=num_workers,
                               pin_memory=use_pin, persistent_workers=(num_workers > 0))

    for clean_b, noisy_b in train_loader:
        ...

注意：data_root 要指向 prepare_data.py 產生的原始 waveform 目錄
（也就是 config 裡的 sEMG_dataset_dir，例如 semg_data/processed），
而不是 make_dataset_spectrogram.py 輸出的 dataset/ 目錄。
"""
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from spectrogram_utils import mag_pha_stft

# 必須與 convert.py 完全一致，否則模型輸入分布會跟訓練好的權重對不上
N_FFT = 512
HOP_SIZE = 128
WIN_SIZE = 512
COMPRESS_FACTOR = 0.5


def _wav_to_spec_2FT(wav: np.ndarray) -> torch.Tensor:
    """
    單一波形 -> [2, F, T] float32 tensor
    channel 0 = 壓縮幅值 (magnitude ** compress_factor)
    channel 1 = 相位 (phase)

    邏輯對齊 convert.py 的 _wav2spec()，只是不落地存檔、
    也不轉 half precision（訓練時用 float32 精度更好、更省一次轉換）。
    """
    x = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)  # [1, T_samples]
    with torch.no_grad():
        mag, pha, _ = mag_pha_stft(x, N_FFT, HOP_SIZE, WIN_SIZE, COMPRESS_FACTOR)
        # mag, pha: [1, F, T] -> squeeze(0) -> [F, T]
        spec = torch.stack([mag.squeeze(0), pha.squeeze(0)], dim=0)  # [2, F, T]
    return spec.contiguous().float()


def collect_waveform_pairs(split_dir, noisy_dirname="noisy", clean_dirname="clean", ext=".npy"):
    """
    收集 (noisy_wav_path, clean_wav_path) 路徑配對清單。
    邏輯對齊 make_dataset_spectrogram.py 的 collect_spectrogram_pairs，
    只是比對對象從 .pt 頻譜圖換成 .npy 波形檔。

    noisy 目錄預期為巢狀： split/noisy_dir/{SNR}/{ECG_ID}/*.npy
    clean 目錄可為同名層級，或平坦目錄用檔名比對。
    """
    noisy_root = os.path.join(split_dir, noisy_dirname)
    clean_root = os.path.join(split_dir, clean_dirname)

    if not os.path.isdir(noisy_root):
        raise FileNotFoundError(f"Noisy root not found: {noisy_root}")
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f"Clean root not found: {clean_root}")

    pattern = os.path.join(noisy_root, "**", f"*{ext}")
    noisy_files = sorted(glob(pattern, recursive=True))

    pairs = []
    missing_clean = 0
    for noisy_path in noisy_files:
        rel = os.path.relpath(noisy_path, noisy_root)
        clean_path = os.path.join(clean_root, rel)

        if not os.path.exists(clean_path):
            fname = os.path.basename(noisy_path)
            alt_clean = os.path.join(clean_root, fname)
            if os.path.exists(alt_clean):
                clean_path = alt_clean

        if os.path.exists(clean_path):
            pairs.append((noisy_path, clean_path))
        else:
            missing_clean += 1

    if missing_clean > 0:
        print(f"[Warning] {missing_clean} clean waveforms not found under '{clean_root}' "
              f"(by relative path or filename).")

    if len(pairs) == 0:
        raise RuntimeError(f"No (noisy, clean) waveform pairs found under {split_dir}")

    return pairs


class OnTheFlySpectrogramDataset(Dataset):
    """
    直接讀取 {data_root}/{split}/{noisy,clean}/*.npy 波形，
    在 __getitem__ 裡即時做 STFT，回傳 (clean_spec, noisy_spec)，
    形狀皆為 [2, F, T]，與原本 make_dataset_spectrogram.py 產生的
    {split}_spectrogram.pt 內容相容（可直接餵給 pipeline_spectrogram.py
    的訓練迴圈，不需要改動 model 端）。

    注意：STFT 在 CPU 上計算（DataLoader 用多個 num_workers 時比較安全，
    因為 CUDA context 不適合在 fork 出來的 worker process 裡使用）。
    """

    def __init__(self, data_root, split, noisy_dirname="noisy", clean_dirname="clean"):
        split_dir = os.path.join(data_root, split)
        self.pairs = collect_waveform_pairs(split_dir, noisy_dirname, clean_dirname, ext=".npy")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy_wav = np.load(noisy_path)
        clean_wav = np.load(clean_path)

        noisy_spec = _wav_to_spec_2FT(noisy_wav)
        clean_spec = _wav_to_spec_2FT(clean_wav)

        # 與 pipeline_spectrogram.py 原本的 (clean_b, noisy_b) 順序對齊
        return clean_spec, noisy_spec
