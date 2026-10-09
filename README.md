# SSEMG-Net

Official implementation of **SSEMG-Net: A Spectrogram-Based Mamba Network for Surface Electromyography Denoising**, together with a cross-architecture knowledge distillation (KD) pipeline that compresses SSEMG-Net into a lightweight, CPU-deployable student model.

SSEMG-Net (Teacher) removes ECG artifacts from surface electromyography signals using:

- compressed magnitude and wrapped-phase spectrograms
- time-frequency bidirectional Mamba blocks
- a physiologically guided sub-band magnitude mask
- explicit phase estimation
- time-domain, complex-domain, consistency, and multi-resolution STFT losses

The Student model (`StudentSSEMGNet`) replaces the TF-Bi-Mamba backbone with depthwise separable, dilated convolutions, removing any dependency on `mamba_ssm` / CUDA / Triton so it can run on CPU-only devices, and is trained via knowledge distillation from the Teacher.

> **⚡ Data pipeline update: `.pt` spectrogram cache is now optional.**
> This fork adds `spectrogram_dataset.py`, which lets every training/inference
> script read `prepare_data.py`'s raw waveform `.npy` output directly and
> compute STFT spectrograms on-the-fly inside the `Dataset`. **Steps 2 and 3
> below (`convert.py` and `make_dataset_spectrogram.py`) are no longer
> required** — go straight from step 1 (`prepare_data.py`) to step 4/6/7,
> pointing `--data_root` / `--dataset` at `semg_data/processed` (the
> waveform directory) instead of `dataset/` (the old `.pt` bundle directory).
> The two scripts are kept in the repo for backward compatibility / anyone
> who prefers precomputed `.pt` files (e.g. to trade disk space for less
> per-epoch CPU work), but are not part of the default workflow anymore.
> See `spectrogram_dataset.py`'s docstring and `verify_dataset_coverage.py`
> for details and a sanity-check tool.
>
> **⚠️ Required folder layout: `MECG-E/`.** Every script that needs
> `models/`, `distill_loss.py` / `distill_loss_v3.py`, `spectrogram_utils.py`,
> or `spectrogram_dataset.py` resolves them via
> `sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))` — i.e. it expects
> those files to live inside a `MECG-E/` subfolder at the repo root (a
> `baseline_model/*.py` script resolves `ROOT_DIR` as *its parent's* parent,
> so `MECG-E/` still means the repo root's `MECG-E/`, not
> `baseline_model/MECG-E/`). This repo currently ships `models/`,
> `distill_loss.py`, `spectrogram_utils.py`, `spectrogram_dataset.py`, and
> `verify_dataset_coverage.py`'s dependency at the **repo root**, not inside
> `MECG-E/` — before running anything, either move those files into a
> `MECG-E/` folder yourself, or symlink one (`ln -s . MECG-E` from the repo
> root also works, since `.` already contains everything `MECG-E/` needs to
> contain). `pipeline_distill_crossarch_v5.py` additionally expects
> `distill_loss_v3.py` specifically (not `distill_loss.py`) inside that same
> `MECG-E/` folder.

## Main files

- `prepare_data.py`: NinaPro DB2 and ECG preprocessing, segmentation, and noisy-mixture generation
- `spectrogram_dataset.py`: **[new]** on-the-fly waveform→spectrogram `Dataset` (`OnTheFlySpectrogramDataset`); reads `prepare_data.py`'s `.npy` output directly, no `.pt` cache needed
- `verify_dataset_coverage.py`: **[new]** diagnostic tool — checks that each split's noisy/clean pairs and subject/exercise/channel coverage match `prepare_data.py`'s intended split
- `convert.py`: *(optional, legacy)* waveform-to-spectrogram `.pt` conversion
- `make_dataset_spectrogram.py`: *(optional, legacy)* noisy-clean spectrogram pairing and tensor dataset construction (produces `dataset/*_spectrogram.pt`)
- `models/SSEMGNet.py`: SSEMG-Net (Teacher) model architecture
- `models/StudentNet.py`: StudentSSEMGNet (Student) model architecture
- `pipeline_spectrogram.py`: Teacher training pipeline
- `pipeline_distill_crossarch.py`: cross-architecture knowledge distillation pipeline (trains the Student)
- `pipeline_distill_crossarch_v5.py`: newer iteration of the above (same CLI/behavior otherwise)
- `distill_loss.py`: KD loss functions, annealing and curriculum schedulers
- `check_loss_balance.py`: sanity-checks whether KD loss terms are balanced
- `inference_demo.py`: Teacher inference, evaluation, and visualization
- `inference_student.py`: Student inference and evaluation *(legacy — still reads a `.pt` bundle via `--dataset`; kept unmodified, prefer `inference_student_patched.py` below)*
- `inference_student_patched.py`: Student inference with optional per-SNR grouped evaluation (`--snr_labels`); use this instead of `inference_student.py` when reproducing the SNR-conditioned results reported in the paper
- `spectrogram_utils.py`: STFT and inverse STFT utilities
- `utils.py`: evaluation metrics
- `config/local_cfg.example.yaml`: example dataset-path configuration
- `config/config_spectrogram_v19_tt_mask.yaml`: SSEMG-Net (Teacher) paper configuration
- `config/student_16ch_1blk.yaml`: edge-deployment Student configuration (16ch, 1 TFConvBlock — used for the reported result)
- `config/local_cfg.example.yaml`: example dataset-path configuration
- `baseline_model/`: baseline model training and inference scripts (FCN, MSEMG, SDEMG) used for comparison against SSEMG-Net

## Data

The raw datasets are not redistributed in this repository.

Required datasets:

- NinaPro DB2 for clean sEMG signals
- MIT-BIH Normal Sinus Rhythm Database for ECG interference

Create the local path configuration:

    cp config/local_cfg.example.yaml config/local_cfg.yaml

Edit `config/local_cfg.yaml`:

    ECG_corpus_dir: /path/to/mit-bih-normal-sinus-rhythm-database
    ECG_storage_dir: /path/to/processed/ecg

    EMG_corpus_dir: /path/to/ninapro-db2
    sEMG_dataset_dir: semg_data/processed

The default pipeline assumes that `sEMG_dataset_dir` contains:

    semg_data/processed/
    ├── train/
    ├── valid/
    └── test/

## 1. Prepare ECG-contaminated sEMG waveforms

Run:

    python prepare_data.py

This script performs the following operations:

1. Reads Channel 1 ECG recordings from the MIT-BIH Normal Sinus Rhythm Database.
2. Resamples ECG from 128 Hz to 1000 Hz.
3. Applies 10 Hz high-pass and 200 Hz low-pass filtering to ECG.
4. Reads NinaPro DB2 sEMG recordings.
5. Applies a 20–500 Hz Butterworth band-pass filter.
6. Downsamples sEMG from 2000 Hz to 1000 Hz.
7. Normalizes and divides sEMG into 10-second segments.
8. Generates ECG-contaminated sEMG signals at multiple SNR levels.

The default data split is:

- Training:
  - NinaPro DB2 subjects 11–40
  - Exercise 1
  - Channel 2
  - SNR: −15, −13, −11, −9, −7, −5 dB

- Validation:
  - NinaPro DB2 subjects 1–10
  - Exercise 3
  - Channel 2
  - SNR: −15, −13, −11, −9, −7, −5 dB

- Testing:
  - NinaPro DB2 subjects 1–10
  - Exercise 2
  - Channels 9–12
  - SNR: −14, −12, −10, −8, −6, −4, −2, 0 dB

The generated waveform structure is:

    semg_data/processed/
    ├── train/
    │   ├── clean/
    │   └── noisy/
    ├── valid/
    │   ├── clean/
    │   └── noisy/
    └── test/
        ├── clean/
        └── noisy/

## 2. *(Optional / legacy)* Convert waveforms to spectrograms

> Skip this step and step 3 if you're using `spectrogram_dataset.py`
> (the default now) — jump straight to step 4 with
> `--data_root semg_data/processed`.

Run the conversion for each split:

    python convert.py --split train
    python convert.py --split valid
    python convert.py --split test

The script converts waveform `.npy` files into compressed magnitude and wrapped-phase spectrogram `.pt` files using:

- sampling rate: 1000 Hz
- FFT size: 512
- window size: 512
- hop size: 128
- magnitude compression exponent: 0.5

The generated directories are:

    spectrogram_cf05/
    spectrogram_cf05_clean/

## 3. *(Optional / legacy)* Build tensor datasets

> Also skippable, same reason as step 2.

Run:

    python make_dataset_spectrogram.py \
        --data_root semg_data/processed \
        --noisy_dir spectrogram_cf05 \
        --clean_dir spectrogram_cf05_clean \
        --output_dir dataset

This produces:

    dataset/
    ├── train_spectrogram.pt
    ├── valid_spectrogram.pt
    └── test_spectrogram.pt

Each dataset contains paired tensors:

    noisy spectrogram: [N, 2, F, T]
    clean spectrogram: [N, 2, F, T]

The two channels represent:

    channel 0: compressed magnitude
    channel 1: wrapped phase

## 4. Train SSEMG-Net

Run:

    python pipeline_spectrogram.py \
        --config config/config_spectrogram_v19_tt_mask.yaml \
        --data_root semg_data/processed

`--data_root` must point at the **waveform** directory produced by
`prepare_data.py` (the one with `train/valid/test/{noisy,clean}/*.npy`).
`pipeline_spectrogram.py` no longer reads the old `dataset/*_spectrogram.pt`
bundle at all — if you still want that route, use the original
(pre-`spectrogram_dataset.py`) version of this file from upstream, or ask
for `load_dataset()` to be wired back into `run_training()`.

The paper configuration uses:

- epochs: 30
- batch size: 4
- gradient accumulation: 4
- effective batch size: 16
- optimizer: AdamW
- learning rate: 3e-4
- dense channels: 64
- TF-Bi-Mamba blocks: 4
- sub-band mask cutoff: 150 Hz

The training objective contains:

- time-domain loss
- complex-domain loss
- STFT consistency loss
- multi-resolution STFT loss

## 5. Evaluate SSEMG-Net

Run:

    python inference_demo.py \
        --config config/config_spectrogram_v19_tt_mask.yaml \
        --weights /path/to/checkpoint.pth \
        --dataset semg_data/processed \
        --batch 64 \
        --index 0

`--dataset` points at the waveform directory (`semg_data/processed`), same
as `--data_root` in step 4.

The evaluation reports:

- SNR improvement
- RMSE
- RMSE of average rectified value
- RMSE of mean frequency

In this repository, `RMSE_MF(Hz)` denotes the error of **mean frequency**, defined as the spectral centroid.

## 6. Knowledge distillation (train the Student)

The Student (`StudentSSEMGNet`) is trained with a Teacher SSEMG-Net checkpoint frozen (`eval()`, no gradient updates), using `spectrogram_dataset.py` to read the same `train`/`valid` waveform splits produced in step 1.

The command below reproduces the reported edge-deployment result (16ch/1blk, Response KD only, SNRimp ≈ 20.15 dB):

    python pipeline_distill_crossarch.py \
        --teacher_config config/config_spectrogram_v19_tt_mask.yaml \
        --teacher_weights <path/to/teacher_checkpoint.pth> \
        --student_config config/student_16ch_1blk.yaml \
        --data_root semg_data/processed \
        --epochs 150 \
        --kd_weight_mode annealed --anneal_schedule linear \
        --w_resp_mask 3.5 --w_resp_mag 5.0 --w_resp_com 2.3 \
        --w_feature 0 \
        --patience 20 \
        --log_csv log_student_16ch_1blk_respKD.csv \
        --model_save model_weight/student_16ch_1blk_respKD.pth

`pipeline_distill_crossarch_v5.py` takes the same CLI and can be used as a
drop-in replacement.

Training stops via early stopping well before reaching `--epochs`; `--epochs` only sets the upper bound and the length of the α annealing schedule (see `--schedule_epochs` below to decouple the two).

Key options:

- `--kd_weight_mode {fixed,annealed}` and `--anneal_schedule {linear,cosine,exponential}`: control how the GT/KD weight α is scheduled over training
- `--schedule_epochs`: decouple the α / noise / layer-curriculum schedule length from the actual training epoch budget (`--epochs`). Defaults to `--epochs` if unset (fully backward compatible) — set this when you want to extend training beyond a schedule that was already validated, without changing its pace
- `--w_feature`, `--w_response`: weight of Feature-based and Response-based KD losses
- `--w_resp_mask`, `--w_resp_mag`, `--w_resp_pha`, `--w_resp_com`: per-component weights of the Response KD loss (Mask, Magnitude, Phase, Complex)
- `--include_relation` / `--w_relation`: enable Relation-based KD loss (self-similarity matrix across time/frequency)
- `--similarity_preserving` / `--w_similarity` / `--similarity_student_idx` / `--similarity_teacher_idx`: enable Similarity-Preserving KD (batch-wise pairwise-similarity matching); unlike Feature/Relation KD it is not limited by student/teacher block-count alignment, so it also works with 1-block students
- `--feature_noise` / `--noise_std_start` / `--noise_std_end` / `--noise_schedule`: Feature Noise Annealing on the Teacher's target features
- `--layer_curriculum`: progressively expand the set of Teacher layers used for Feature KD
- `--two_stage` / `--two_stage_ratio`: hard-switch training into a Teacher-only representation stage followed by a Ground-Truth-only fine-tuning stage
- `--seed`: fix random/numpy/torch seeds for closer run-to-run reproducibility
- `--resume` / `--resume_path`: resume training from a full training-state checkpoint (model + optimizer + scheduler + epoch + best_val + no_improve + RNG state), kept separate from the `--model_save` inference checkpoint
- `--use_amp` / `--num_workers` / `--no_cudnn_benchmark`: pure speed options that do not change training results
- `--swa` / `--swa_dir`: collect per-epoch checkpoints after the schedule ends, for Stochastic Weight Averaging
        
## 7. Evaluate the Student

`inference_student.py` is left unmodified and still expects the legacy `.pt`
bundle (build it via steps 2–3 first if you want to use it):

    python inference_student.py \
        --config config/config_student_crossarch.yaml \
        --weights <path/to/student_checkpoint.pth> \
        --dataset dataset/test_spectrogram.pt \
        --csv-out <path/to/results.csv> \
        --exp-alias "Student_KD"

Prefer `inference_student_patched.py`, which reads waveforms on-the-fly and
supports the per-SNR breakdown:

    python inference_student_patched.py \
        --config config/student_16ch_1blk.yaml \
        --weights <path/to/student_checkpoint.pth> \
        --dataset semg_data/processed \
        --csv-out <path/to/results.csv> \
        --exp-alias "Student_KD"
 
To reproduce the per-SNR breakdown reported in the paper, additionally pass `--snr_labels <path/to/test_snr_labels.json>` (built with `build_snr_labels.py`, **not yet committed to this repo** — regenerate it from the test split definition in step 3, or add the script here for full reproducibility). This writes an extra `*_by_snr.csv` file alongside the main results CSV.

`inference_demo.py` also works with a Student checkpoint and config for single-sample visualization, the same as step 5.

## Repository workflow

The default (no `.pt` cache) pipeline is:

    Raw NinaPro DB2 and ECG data
        ↓
    prepare_data.py
        ↓
    Clean and ECG-contaminated waveform files (semg_data/processed/{split}/{noisy,clean}/*.npy)
        ↓
    spectrogram_dataset.py (OnTheFlySpectrogramDataset — used inside every training/inference script below)
        ↓
    pipeline_spectrogram.py
        ↓
    SSEMG-Net (Teacher) checkpoint
        ↓
    pipeline_distill_crossarch.py / pipeline_distill_crossarch_v5.py  (Teacher frozen, KD training)
        ↓
    StudentSSEMGNet (Student) checkpoint
        ↓
    inference_demo.py / inference_student_patched.py
        ↓
    Evaluation metrics and visualizations

`convert.py` and `make_dataset_spectrogram.py` (producing `spectrogram_cf05*/`
and `dataset/*_spectrogram.pt`) are optional legacy steps — only needed if
you specifically want a precomputed `.pt` cache instead of on-the-fly STFT.

## Baseline comparisons (FCN / MSEMG / SDEMG)

Scripts in `baseline_model/` follow the same convention: `--data_root` /
`--dataset_path` now point at the waveform directory
(`semg_data/processed`), not a `.pt` bundle, e.g.:

    python baseline_model/train_fcn_baseline.py --data_root semg_data/processed ...
    python baseline_model/inference_fcn_baseline_v2.py --dataset_path semg_data/processed ...

(same pattern for `*_msemg_*` and `*_sdemg_*`).

## Sanity-checking the data split

    python verify_dataset_coverage.py --data_root semg_data/processed

Confirms each split's (noisy, clean) pairs resolve correctly and that the
subject/exercise/channel coverage matches `prepare_data.py`'s intended split
(train = subjects 11–40 / exercise 1; valid/test = subjects 1–10 / exercise
3 and 2 respectively).

## Causal student (streaming-oriented variant)

`config/student_16ch_1blk_causal.yaml` is the same student with `model.causal: True`.
With the flag on, `models/StudentNet.py` switches to:

- time-axis convolutions (DenseBlock, TF-ConvBlock temporal DS-Conv) that pad **only the past** side;
- per-frame normalisation (`CausalGroupNorm2d`, `ChannelGroupNorm1d`) instead of GroupNorm/InstanceNorm
  statistics that span the whole time axis;
- no batch-wide input-format sniffing in `forward_spectrogram`.

With the flag off (default) the network is exactly the original, and old checkpoints still load.
A causal student has different layers/statistics, so it **must be trained from scratch** (KD from the
non-causal teacher works unchanged because the STFT frame grid is not modified):

    python pipeline_distill_crossarch_v5.py --student_config config/student_16ch_1blk_causal.yaml ...

Verify causality (future frames must not change past outputs):

    python test_causality.py --config config/student_16ch_1blk_causal.yaml   # expect ✔
    python test_causality.py --config config/student_16ch_1blk.yaml          # expect ✘ (original is non-causal)

**Not yet causal end-to-end:** the STFT front end (`center=True` reflect padding looks ~win/2 ahead) and the
iSTFT overlap-add (needs up to `win_size` samples of look-ahead; 512 ms at 1 kHz with `win_size=512`) still
have look-ahead. Only the *network* is causal. A streaming front end (left-padded `center=False` STFT,
shorter synthesis window / smaller `win_size`) is the next step and requires retraining the teacher on the same grid.

## Installation

SSEMG-Net uses the official `mamba-ssm` package and does not vendor a separate
copy of the Mamba source code.

Install PyTorch first, followed by the remaining dependencies:

    pip install torch==2.3.1
    pip install -r requirements.txt --no-build-isolation

The released environment uses:

- `mamba-ssm==2.2.2`
- `causal-conv1d>=1.4.0`
- `triton==2.3.1`

Linux, an NVIDIA GPU, and a compatible CUDA toolkit are required for the
CUDA-accelerated Mamba implementation used by the **Teacher**.

The **Student** (`StudentSSEMGNet`) uses only standard PyTorch convolutions and does
not import `mamba_ssm` / `causal-conv1d` / `triton`, so once a Student checkpoint is
trained, `inference_student.py` can run on a CPU-only machine.
