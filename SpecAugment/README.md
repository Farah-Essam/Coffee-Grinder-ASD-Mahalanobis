# SpecAugment

**Thesis:** Coffee Grinder Anomalous Sound Detection in Selective Mahalanobis Mode  
**Task:** DCASE 2025 Task 2 — First-shot unsupervised anomalous sound detection  
**Experiment identifier (`export_dir`):** `baseline_specaugment`

---

## 1. Overview

This experiment implements **SpecAugment** as a **training-stage data augmentation** method on log-mel feature patches within the official DCASE Task 2 baseline autoencoder pipeline. The method randomly masks contiguous regions in the frequency and time dimensions of each training batch (by setting selected values to zero) while the autoencoder is trained to reconstruct the **original, unmasked** features.

**Problem addressed.** The baseline autoencoder may overfit fine-grained spectral structure of normal machine sounds. Under **source–target domain shift** (a core condition in DCASE 2025 Task 2), such overfitting can reduce generalisation to the target domain. SpecAugment encourages the model to rely on broader spectro-temporal structure rather than on a fixed pattern of mel bins and frames.

**Component targeted.**

| Category | Targeted by this experiment? |
|----------|:----------------------------:|
| Data augmentation (training input) | **Yes** |
| Model architecture | No |
| Scoring / statistical processing | No |
| Feature extraction (offline) | No |

**Motivation.** By presenting corrupted inputs during training and retaining clean targets for the MSE loss, the experiment tests whether reconstruction-based representations become more robust before they are scored with the unchanged selective Mahalanobis procedure at evaluation time.

---

## 2. Thesis Context

This repository supports the bachelor thesis *Coffee Grinder Anomalous Sound Detection in Selective Mahalanobis Mode*. The present folder extends the **DCASE 2023/2025 Task 2 baseline autoencoder** (`DCASE2023T2-AE` / `AENet`) without introducing alternative encoder designs or modified scoring functions.

**Baseline modified.** Training behaviour only: optional application of `apply_specaugment()` in `DCASE2023T2AE.train()` when `--use_specaugment True`.

**Components that remain unchanged (as implemented in code).**

| Component | Status |
|-----------|--------|
| `AENet` layer structure (`networks/dcase2023t2_ae/network.py`) | Unchanged |
| Offline log-mel extraction (`datasets/loader_common.py`) | Unchanged |
| Training optimiser (Adam) and loss (MSE vs. clean input) | Unchanged |
| Mahalanobis scoring (`networks/criterion/mahala.py`) | Unchanged |
| Selective score `min(score_source, score_target)` at test | Unchanged |
| Gamma-based threshold (`decision_threshold = 0.9`) | Unchanged |

**Independent evaluation.** SpecAugment is trained and reported under a dedicated `export_dir` (`baseline_specaugment`). No other thesis methods (e.g. PCA-based reductions or domain-conditioned autoencoders) are implemented or combined in this codebase.

**Evaluation focus.** Reported results use machine type `DCASE2025T2CoffeeGrinder`, DCASE 2025 **evaluation** data (`--eval`), section `00`, and Mahalanobis scoring (`--score MAHALA`), consistent with the thesis emphasis on selective Mahalanobis mode.

---

## 3. Method Description

### 3.1 Core idea

SpecAugment perturbs log-mel patches during autoencoder training by zeroing random frequency bands and/or short temporal segments. The decoder is still supervised to reproduce the uncorrupted patch. The model therefore learns a denoising-style mapping: reconstruct normal machine spectra from partially observed inputs.

### 3.2 Theoretical motivation

In unsupervised ASD, the autoencoder models the manifold of normal sounds. When train and test domains differ acoustically, representations that memorise domain-specific spectral detail may yield elevated reconstruction error on normal target-domain clips, increasing false alarms. Masking simulates missing or unreliable time–frequency content and discourages dependence on narrow spectral cues.

### 3.3 Expected impact on anomaly detection

Improved training regularisation may:

- reduce overfitting to source-domain idiosyncrasies;
- yield latent representations that transfer more reliably to the target domain;
- indirectly improve Mahalanobis separability if reconstruction errors for normal target sounds decrease.

The effect on anomaly detection is **indirect**: scoring formulas and inference inputs are not augmented.

### 3.4 Relationship to domain shift

DCASE 2025 Coffee Grinder training and test data include **source** and **target** domains (identified in filenames). SpecAugment is applied identically to both domains during training; there is no domain-specific mask policy in the implementation. The hypothesis is that domain-invariant robustness is improved before domain-specific covariances are estimated for Mahalanobis scoring.

### 3.5 Relationship to Mahalanobis scoring

After the final training epoch, an additional epoch estimates `cov_source` and `cov_target` from reconstruction errors on **unaugmented** data. At test time, Mahalanobis distances are computed with the inverse covariances, and the anomaly score is the minimum of the source and target domain scores. SpecAugment does not alter these steps; any benefit arises from changed autoencoder weights learned under augmentation.

---

## 4. Implementation Details

### 4.1 Main Components

| File | Purpose |
|------|---------|
| `train.py` | Entry point; loads `baseline.yaml`, sets random seeds, runs training and/or testing |
| `common.py` | Command-line argument definitions, including SpecAugment flags |
| `baseline.yaml` | Default hyperparameters for training, features, and SpecAugment |
| `networks/models.py` | Maps model name `DCASE2023T2-AE` to `DCASE2023T2AE` |
| `networks/dcase2023t2_ae/dcase2023t2_ae.py` | Training loop, `apply_specaugment()`, validation, test, Mahalanobis integration |
| `networks/dcase2023t2_ae/network.py` | `AENet` autoencoder definition |
| `networks/base_model.py` | Paths for checkpoints, logs, results; device selection |
| `networks/criterion/mahala.py` | Covariance accumulation and Mahalanobis distance |
| `datasets/datasets.py` | Dataset wiring, train/validation split |
| `datasets/dcase_dcase202x_t2_loader.py` | Pickle-based feature dataset loader |
| `datasets/loader_common.py` | Waveform loading and log-mel `file_to_vectors()` |
| `datasets/machine_type_2025_eval.yaml` | Section IDs for DCASE 2025 evaluation machine types |
| `train_ae.sh` | Shell wrapper for training (`--train_only`) |
| `test_ae.sh` | Shell wrapper for testing (`--test_only`, `--score`) |
| `tools/01_train_legacy.sh` | Batch training driver for DCASE 2025 evaluation machines |
| `tools/02b_test_legacy.sh` | Batch testing driver with `score=MAHALA` |

### 4.2 Configuration Parameters

**SpecAugment parameters (defined in `common.py`, defaults in `baseline.yaml`).**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_specaugment` | `False` | Enables SpecAugment during training when `True` |
| `specaug_freq_mask_param` | `12` | Maximum width of a frequency mask (mel bins) |
| `specaug_time_mask_param` | `1` | Maximum width of a time mask (stacked frames) |
| `specaug_num_freq_masks` | `1` | Number of frequency-mask applications per augmented batch |
| `specaug_num_time_masks` | `1` | Number of time-mask applications per augmented batch |
| `specaug_prob` | `0.5` | Probability of applying masking to a training batch |

**Training and feature parameters (`baseline.yaml`).**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `DCASE2023T2-AE` | Baseline autoencoder identifier |
| `epochs` | `100` | Number of training epochs |
| `batch_size` | `256` | Mini-batch size |
| `learning_rate` | `0.001` | Adam learning rate |
| `validation_split` | `0.1` | Fraction of training data used for validation |
| `seed` | `13711` | Random seed for Python, NumPy, and PyTorch |
| `n_mels` | `128` | Number of mel bands |
| `frames` | `5` | Number of consecutive frames per feature vector |
| `n_fft` | `1024` | FFT size |
| `hop_length` | `512` | Hop length for spectrogram computation |
| `frame_hop_length` | `1` | Hop between stacked frame groups |
| `power` | `2.0` | Power for mel spectrogram |
| `export_dir` | `baseline` | Experiment output subdirectory (use `baseline_specaugment` for this study) |
| `score` | `MSE` | Default in yaml; evaluation uses `MAHALA` via CLI |
| `decision_threshold` | `0.9` | Quantile for gamma-fitted anomaly threshold |
| `max_fpr` | `0.1` | Maximum FPR for partial AUC |
| `use_cuda` | `True` | Enable GPU when CUDA is available |
| `gpu_id` | `[0]` | GPU index (default first device) |
| `mono` | `True` (argparse default) | Monaural audio loading |

**Mask sampling (implementation behaviour, not separate CLI flags).** For each mask, width is drawn uniformly from `{0, …, param}`; width `0` skips that mask. Masked values are set to **zero**. One probability draw is made per **batch**; the same mask positions apply to all samples in the batch.

### 4.3 Training Procedure

**Inputs.** Mono waveforms from `data/dcase2025t2/eval_data/raw/CoffeeGrinder/` (evaluation/first-shot setting with `--eval`). Normal clips for section `00`, filtered with `--use_ids 0`.

**Feature extraction.** Offline, via `loader_common.file_to_vectors()`: librosa mel spectrogram, log10 scaling, stacking of `frames` consecutive columns into a vector of size `n_mels × frames` (640 by default). Features are cached in pickle files by `DCASE202XT2Loader`.

**Training process.**

1. Load a mini-batch of clean feature vectors `data`.
2. If `use_specaugment` and not the covariance epoch: `input_data = apply_specaugment(data)`; else `input_data = data`.
3. Forward pass: `recon_batch, z = model(input_data)`.
4. Loss: element-wise MSE between `recon_batch` and **clean** `data` (`loss_fn`).
5. Backpropagation and Adam update.

**Loss function.** `torch.nn.functional.mse_loss` with `reduction="none"`, reduced to a scalar training loss per batch (`networks/dcase2023t2_ae/dcase2023t2_ae.py`).

**Optimisation.** Adam on all `AENet` parameters with learning rate from configuration.

**Additional epoch.** After epoch `epochs`, one further epoch (`epochs + 1`) runs in evaluation mode without SpecAugment to accumulate `cov_source` and `cov_target` for Mahalanobis scoring.

**Validation.** Performed on **unaugmented** data each training epoch; SpecAugment is not applied.

---

## 5. Inference and Scoring Pipeline

1. **Audio input** — Load test waveform (mono) from the evaluation dataset.
2. **Log-mel spectrogram extraction** — Offline mel + log10 transform (already computed in pickle for batched evaluation).
3. **Feature vector formation** — Stack five frames into a 640-dimensional patch (default settings).
4. **Autoencoder inference** — Forward pass on **clean** features; no SpecAugment at test time.
5. **Residual computation** — Difference between input and reconstruction (used inside Mahalanobis block processing).
6. **Covariance-based statistical scoring** — Mahalanobis distance using `inv_cov_source` and `inv_cov_target` (block size `n_mels` = 128).
7. **Selective anomaly score** — `score = min(loss_source, loss_target)`.
8. **Threshold and decision** — Compare score to gamma-distribution quantile (`decision_threshold = 0.9`).
9. **Evaluation metrics** — AUC, pAUC (`max_fpr = 0.1`), precision, recall, and F1 per domain, written to CSV under `results/eval_data/`.

---

## 6. Repository Structure

```
SpecAugment/
├── train.py                          # Main entry point
├── common.py                         # CLI argument definitions
├── baseline.yaml                     # Default configuration
├── requirements.txt                  # Python dependencies
├── train_ae.sh                       # Training shell wrapper
├── test_ae.sh                        # Evaluation shell wrapper
├── networks/
│   ├── models.py                     # Model registry
│   ├── base_model.py                 # Checkpoint and result path management
│   ├── dcase2023t2_ae/
│   │   ├── dcase2023t2_ae.py         # SpecAugment, train/test loops
│   │   └── network.py                # AENet architecture
│   └── criterion/
│       └── mahala.py                 # Mahalanobis scoring
├── datasets/
│   ├── datasets.py                   # Dataset and DataLoader setup
│   ├── dcase_dcase202x_t2_loader.py  # Pickle feature loader
│   ├── loader_common.py              # Log-mel extraction
│   └── machine_type_2025_eval.yaml   # CoffeeGrinder section 00
├── tools/
│   ├── 01_train_legacy.sh            # Batch training script
│   └── 02b_test_legacy.sh            # Batch Mahalanobis testing script
├── models/
│   ├── checkpoint/                   # Training checkpoints (generated)
│   └── saved_model/                  # Final weights and score distributions
├── logs/                             # Training CSV logs (generated)
├── results/
│   └── eval_data/
│       └── baseline_specaugment_MAHALA/   # Evaluation outputs (this experiment)
└── data/
    └── dcase2025t2/eval_data/raw/    # Dataset location (user-provided)
```

---

## 7. Running the Experiment

### Training

Configuration is loaded from `baseline.yaml` and overridden on the command line. SpecAugment must be enabled explicitly; the yaml default is `False`.

```bash
python train.py \
  --dataset DCASE2025T2CoffeeGrinder \
  --eval \
  --use_ids 0 \
  --train_only \
  --mono True \
  --use_specaugment True \
  --export_dir baseline_specaugment
```

The wrapper `train_ae.sh` invokes `python3 train.py` with dataset, eval flag, IDs, and `--train_only`, but **does not** pass SpecAugment flags. Add `--use_specaugment True` and `--export_dir baseline_specaugment` to the Python command if using the shell script.

Legacy batch driver (Coffee Grinder listed for DCASE 2025 evaluation):

```bash
bash tools/01_train_legacy.sh DCASE2025T2 --eval
```

### Evaluation

Requires trained weights under `models/saved_model/baseline_specaugment/`.

```bash
python train.py \
  --dataset DCASE2025T2CoffeeGrinder \
  --eval \
  --use_ids 0 \
  --test_only \
  --mono True \
  --score MAHALA \
  --export_dir baseline_specaugment
```

Wrapper equivalent (Mahalanobis preset in `tools/02b_test_legacy.sh`):

```bash
bash test_ae.sh DCASE2025T2CoffeeGrinder --eval True MAHALA 0
```

Ensure `--export_dir` matches the training run.

### Important Notes

| Requirement | Detail |
|-------------|--------|
| Dataset | DCASE 2025 Coffee Grinder evaluation data under `./data/dcase2025t2/eval_data/raw/CoffeeGrinder/` |
| Configuration | `baseline.yaml` plus CLI overrides |
| Checkpoints | Produced under `models/checkpoint/baseline_specaugment/...` during training |
| Weights for test | `models/saved_model/baseline_specaugment/DCASE2023T2-AE_DCASE2025T2CoffeeGrinder_Eval_seed13711.pth` |
| Section | `00` per `datasets/machine_type_2025_eval.yaml` |
| Download | See `tools/data_download_2025.sh` and `README_legacy.md` |

---

## 8. Experimental Configuration

Settings below reflect `baseline.yaml`, the implemented model, and the saved evaluation run (`baseline_specaugment_MAHALA`).

| Component | Setting |
|-----------|---------|
| Dataset | `DCASE2025T2CoffeeGrinder` |
| Data mode | Evaluation / first-shot (`--eval`) |
| Section | `00` |
| Machine IDs (normal training) | `0` |
| Model | `DCASE2023T2-AE` (`AENet`) |
| Score type (evaluation) | `MAHALA` (selective Mahalanobis) |
| Training loss | MSE (reconstruction vs. clean input) |
| Input dimension | 640 (`128` mels × `5` frames) |
| Latent dimension | 8 |
| SpecAugment | Enabled at runtime (`use_specaugment True`) |
| Export directory | `baseline_specaugment` |
| Epochs | 100 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Optimiser | Adam |
| Random seed | 13711 |
| Mono input | True |
| GPU | `use_cuda True` (CUDA device 0 if available) |

*PCA dimension, domain-adversarial loss weights, and similar parameters are not defined in this codebase.*

---

## 9. Results

Results are taken from:

`results/eval_data/baseline_specaugment_MAHALA/result_DCASE2025T2CoffeeGrinder_test_seed13711_Eval_roc.csv`  
(section `00`, seed `13711`)

| Metric | Value |
|--------|-------|
| AUC (Source) | 0.726 |
| AUC (Target) | 0.417 |
| pAUC | 0.535 |
| pAUC (Source) | 0.509 |
| pAUC (Target) | 0.528 |
| Precision (Source) | 0.500 |
| Precision (Target) | 0.500 |
| Recall (Source) | 1.000 |
| Recall (Target) | 1.000 |
| F1 Score (Source) | 0.667 |
| F1 Score (Target) | 0.667 |

Values rounded to three decimal places. Exact floating-point entries are stored in the CSV. Additional outputs in the same directory include per-file anomaly score CSVs and `DCASE2023T2-AE_DCASE2025T2CoffeeGrinder_Eval_anm_score.png`.

---

## 10. Discussion

**Pipeline aspect improved.** SpecAugment targets **training-time robustness** of the reconstruction model. It does not modify feature extraction, architecture depth, or the Mahalanobis scoring function. Improvements, if any, would manifest through better autoencoder generalisation across source and target domains before selective scoring.

**Strengths.**

- Minimal invasive change: one augmentation function and one conditional call in the training loop.
- Clear separation between augmented forward inputs and clean supervision targets.
- Compatible with the existing selective Mahalanobis evaluation protocol without code changes at inference.

**Limitations.**

- Covariance estimation and all inference use **unaugmented** data; SpecAugment does not regularise second-order statistics directly.
- Mask widths are stochastic; zero-width draws can result in no masking for a given batch.
- Masks are shared across all items in a mini-batch.
- Temporal context per patch is limited to five frames, constraining time masking relative to typical speech-oriented SpecAugment settings.
- `baseline.yaml` disables SpecAugment by default; reproducibility requires explicit CLI activation and consistent `export_dir`.

**When improvement is plausible.** The method is most relevant when performance is limited by overfitting to narrow spectral structure rather than by fundamental domain mismatch that augmentation cannot simulate (e.g. global spectral tilt not well approximated by local zero masking).

---

## 11. Reproducibility

| Item | Specification |
|------|----------------|
| Dataset | DCASE 2025 Task 2 Coffee Grinder, evaluation split (`--eval`) |
| Configuration file | `baseline.yaml` |
| Experiment overrides | `--use_specaugment True`, `--export_dir baseline_specaugment` |
| Random seed | `13711` (`train.py`: Python, NumPy, PyTorch; `cudnn.deterministic=True`) |
| Validation split | 10% via `sklearn.model_selection.train_test_split` (**no `random_state` set in code**) |
| Hardware | GPU optional: `use_cuda=True`, `gpu_id=[0]`; falls back to CPU if CUDA unavailable |
| Dependencies | `requirements.txt`; PyTorch `2.6.0+cu118` listed |
| Training command | See Section 7 (Training) |
| Required checkpoint / weights | `models/saved_model/baseline_specaugment/DCASE2023T2-AE_DCASE2025T2CoffeeGrinder_Eval_seed13711.pth` |
| Evaluation command | See Section 7 (Evaluation) with `--score MAHALA` |
| Serialized run args | `models/checkpoint/baseline_specaugment/DCASE2023T2-AE_DCASE2025T2CoffeeGrinder_Eval_seed13711/args.json` (if present after training) |

**Reproducibility checklist.**

- [ ] Install dependencies from `requirements.txt`
- [ ] Download and place Coffee Grinder evaluation audio under `data/dcase2025t2/eval_data/raw/CoffeeGrinder/`
- [ ] Train with `--use_specaugment True` and `--export_dir baseline_specaugment`
- [ ] Evaluate with `--score MAHALA` and the same `export_dir`
- [ ] Confirm seed `13711` in `baseline.yaml` or CLI
- [ ] Archive `args.json`, weights, and `results/eval_data/baseline_specaugment_MAHALA/*.csv`

---

## 12. Citation

```bibtex
@inproceedings{dcase2025task2,
  title        = {DCASE 2025 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring},
  author       = {{DCASE Community}},
  year         = {2025},
  note         = {Challenge description; insert official URL and authors from dcase.community}
}

@misc{dcase2023_task2_baseline_ae,
  title        = {DCASE 2023 Task 2 Baseline Auto Encoder (PyTorch)},
  author       = {{NTT Corporation et al.}},
  year         = {2023},
  howpublished = {Software repository},
  note         = {Upstream baseline extended in this experiment; insert DOI or URL if available}
}

@article{park2019specaugment,
  title        = {SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition},
  author       = {Park, Daniel S. and Chan, William and Zhang, Yu and Chiu, Chung-Cheng and Zoph, Barret and Le, Quoc V. and Wu, Yonghui},
  journal      = {Proc. Interspeech},
  year         = {2019},
  note         = {Methodological basis for spectrogram masking}
}
```

**Software licence.** See `LICENSE` / `LICENSEv2.1.pdf` in the repository root.

**Upstream documentation.** `README_BASELINE_UPSTREAM.md`, `README_legacy.md`.
