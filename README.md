# SSEMG-Net

Research fork of [SSEMG-Net](https://github.com/HowardShih/SSEMG-Net): **A Spectrogram-Based Mamba Network for Surface Electromyography Denoising**, together with a cross-architecture knowledge distillation (KD) pipeline that compresses SSEMG-Net into a lightweight, CPU-deployable student model.

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
> **Folder layout:** current v6 entry points import models from `MECG-E/models/`.
> Keep loss, dataset, and STFT helpers at the repository root. Do not move all
> helpers into `MECG-E/`, replace its existing directory with a symlink, or mix
> the legacy root-level `models/` with the causal model implementation.

## Main files

- `prepare_data.py`: NinaPro DB2 and ECG preprocessing, segmentation, and noisy-mixture generation
- `spectrogram_dataset.py`: **[new]** on-the-fly waveform→spectrogram `Dataset` (`OnTheFlySpectrogramDataset`); reads `prepare_data.py`'s `.npy` output directly, no `.pt` cache needed
- `verify_dataset_coverage.py`: **[new]** diagnostic tool — checks that each split's noisy/clean pairs and subject/exercise/channel coverage match `prepare_data.py`'s intended split
- `convert.py`: *(optional, legacy)* waveform-to-spectrogram `.pt` conversion
- `make_dataset_spectrogram.py`: *(optional, legacy)* noisy-clean spectrogram pairing and tensor dataset construction (produces `dataset/*_spectrogram.pt`)
- `MECG-E/models/SSEMGNet.py`: SSEMG-Net (Teacher) model architecture used by current entry points
- `MECG-E/models/StudentNet.py`: StudentSSEMGNet (Student), including the causal variant
- `pipeline_spectrogram.py`: Teacher training pipeline
- `pipeline_distill_crossarch.py`: cross-architecture knowledge distillation pipeline (trains the Student)
- `pipeline_distill_crossarch_v5.py`: newer iteration of the above (same CLI/behavior otherwise)
- `pipeline_distill_crossarch_v6_256_64.py`: causal 256/64 Student training (Response KD or `--no_kd`)
- `distill_loss_v3_256_64.py` and `cross_resolution_256_64.py`: mask-fixed Student-grid KD targets and waveform/STFT helpers
- `inference_student_patched_v6_256_64.py`: config-driven Student inference; requires the dataset version accepting STFT parameters
- `config/student_16ch_1blk_causal_256_64.yaml`: causal 16ch/1blk Student, n_fft/hop/win=256/64/256
- `distill_loss.py`: KD loss functions, annealing and curriculum schedulers
- `check_loss_balance.py`: sanity-checks whether KD loss terms are balanced
- `inference_teacher_patched_v6.py`: Teacher inference from waveform directories; `inference_demo.py` is a legacy `.pt` entry point
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

    python inference_teacher_patched_v6.py \
        --config config/config_spectrogram_v19_tt_mask.yaml \
        --weights /path/to/checkpoint.pth \
        --dataset semg_data/processed \
        --batch 64 \
        --index 0

`--dataset` in this patched entry point points at the waveform directory
(`semg_data/processed`), same as `--data_root` in step 4. The legacy
`inference_demo.py` expects a `.pt` file, not this directory.

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

Training may stop via early stopping before reaching `--epochs`; `--epochs` only sets the upper bound and the length of the α annealing schedule (see `--schedule_epochs` below to decouple the two).

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
- `--use_amp` / `--num_workers` / `--no_cudnn_benchmark`: speed/backend options; numerical results and exact reproducibility may differ
- `--swa` / `--swa_dir`: collect per-epoch checkpoints after the schedule ends, for Stochastic Weight Averaging
        
## 7. Evaluate the Student

`inference_student.py` is left unmodified and still expects the legacy `.pt`
bundle (build it via steps 2–3 first if you want to use it):

    python inference_student.py \
        --config config/student_16ch_1blk.yaml \
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
 
To report per-SNR scores, additionally pass `--snr_labels <path/to/test_snr_labels.json>`, verified against the exact dataset ordering. `build_snr_labels.py` is included but rebuilds labels from legacy `.pt` data; do not assume it produces waveform-order labels. The option writes an extra `*_by_snr.csv` alongside the main results CSV.

Use the matching Student inference entry point and configuration; do not pass a waveform directory to the legacy `inference_demo.py`.

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
    inference_student_patched.py / inference_student_patched_v6_256_64.py
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
With the flag on, `MECG-E/models/StudentNet.py` switches to:

- time-axis convolutions (DenseBlock, TF-ConvBlock temporal DS-Conv) that pad **only the past** side;
- per-frame normalisation (`CausalGroupNorm2d`, `ChannelGroupNorm1d`) instead of GroupNorm/InstanceNorm
  statistics that span the whole time axis;
- no batch-wide input-format sniffing in `forward_spectrogram`.

With the flag off (default) the network uses the non-causal path; old checkpoint compatibility still depends on parameter names and strict loading.
A causal student has different layers/statistics, so it **must be trained from scratch** (KD from the
non-causal teacher works unchanged because the STFT frame grid is not modified):

    python pipeline_distill_crossarch_v5.py --student_config config/student_16ch_1blk_causal.yaml ...

Verify causality (future frames must not change past outputs):

    python test_causality.py --config config/student_16ch_1blk_causal.yaml   # expect ✔
    python test_causality.py --config config/student_16ch_1blk.yaml          # expect ✘ (original is non-causal)

**Not yet causal end-to-end:** the STFT front end still uses `center=True`, and streaming synthesis/state handling has not been validated. Only the *network* is causal; window/hop durations are not measured end-to-end latency. A different Student grid does not require retraining the Teacher on that grid when using the waveform re-analysis KD described below.

### Cross-resolution experiments (256/64 and 256/128)

The current best causal Student uses Teacher 512/128/512 and Student 256/64/256
(n_fft/hop/win), with the same 16ch/1blk architecture. Use
`pipeline_distill_crossarch_v6_256_64.py`,
`config/student_16ch_1blk_causal_256_64.yaml`, and Teacher weights
`model_weight/v19_tt_mask_v2_mix_nv0_weights.pth`.
Keep the section 6 settings: batch=4, accum=4, epochs/schedule_epochs=150,
patience=20, seed=42, linear annealing, feature/phase KD=0.

Teacher enhanced waveforms are re-analysed on the Student grid before computing
`L_KD = 3.5 L_mask + 5.0 L_mag + 2.3 L_complex`. The maskfix version applies the
Student's actual `low_gate` to the target before full-mask MSE; this is an effective
Student-grid mask, not the Teacher's native mask. Time-KD (0.25) is a separate experiment.

For 256/128 KD, use the separate `pipeline_distill_crossarch_v6_256_128.py`,
`distill_loss_v3_256_128.py`, `cross_resolution_256_128.py`, and
`config/student_16ch_1blk_causal_256_128.yaml`.
Inference can share `inference_student_patched_v6_256_64.py` with the matching
256/128 YAML. Keep every grid/KD/seed's model, CSV and resume paths separate;
never let dev and normal jobs write the same experiment concurrently.

Recorded Test-set Average values (not multi-seed averages):

| Model | Window / Hop | KD | SNRimp ↑ (dB) | RMSE ↓ | RMSE_ARV ↓ | MeanF ↓ (Hz) | MedianF ↓ (Hz) | RMSEM ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Teacher v2 | 512 / 128 | — | 22.1147 | 0.0073 | 0.0009 | 5.6848 | 12.5988 | 0.0068 |
| Causal Student | 512 / 128 | No KD | 19.3322 | 0.0103 | 0.0017 | 9.7434 | 20.4424 | 0.0093 |
| Causal Student | 512 / 128 | Response KD | 20.3197 | 0.0092 | 0.0017 | 8.9362 | 18.7363 | 0.0083 |
| Causal Student | 256 / 128 | No KD | 19.4559 | 0.0101 | 0.0018 | 9.6686 | 19.9868 | 0.0091 |
| Causal Student | 256 / 128 | Response KD | 19.8847 | 0.0097 | 0.0018 | 9.6936 | 19.9983 | 0.0087 |
| Causal Student | 256 / 64 | No KD | 20.5371 | 0.0090 | 0.0016 | 8.3531 | 17.5542 | 0.0081 |
| Causal Student | 256 / 64 | Response KD | **20.7771** | **0.0088** | 0.0016 | 8.6045 | 18.1679 | **0.0079** |
| Causal Student | 256 / 64 | Response KD + Time-KD (0.25) | 20.6001 | 0.0090 | 0.0016 | 8.5908 | 18.2037 | 0.0081 |

256-series Students use seed 42; Teacher/older 512-series seeds are not yet
confirmed. MeanF is the logged RMSE_MF/RMSE_MeanF (one metric, not two).
Historical initialization/resume and concurrent-writer differences mean these
are recorded experiments, not a strictly controlled single-factor ablation.
Use the same metrics implementation and batch=64; printed RMSE is not the mean
of the per-sample CSV RMSE. Inference uses the best-validation `.pth`, not
`.resume_state.pth`; keep the best 256/64 KD checkpoint named
`student_16ch_1blk_respKD_causal_256_64_teacher_v2_maskfix_dev.pth` separate from Time-KD.

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
