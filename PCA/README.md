# PCA-Based Mahalanobis Scoring

**Thesis context:** *Coffee Grinder Anomalous Sound Detection in Selective Mahalanobis Mode* — DCASE 2025 Task 2 (first-shot unsupervised setting).

**Implementation identifier in code:** `MAHALA_PCA` (`--score MAHALA_PCA`), experiment export directory `baseline_pca`.

---

# 1. Overview

This experiment implements **PCA-based Mahalanobis anomaly scoring** on top of the DCASE Task 2 autoencoder baseline. After a reconstruction autoencoder is trained on normal sounds, anomaly scores are derived from reconstruction residuals. Unlike the standard selective Mahalanobis mode (`MAHALA`), which operates in the full mel-frame residual space, this variant **projects residuals into a PCA subspace** before covariance estimation and Mahalanobis distance computation.

**Problem addressed:** High-dimensional reconstruction residuals can destabilize empirical covariance estimates and Mahalanobis distances. The method seeks more reliable statistical scoring by retaining the dominant variance directions of normal training residuals and scoring anomalies in that reduced space.

**Pipeline component modified:**

| Component | Modified in this experiment? |
|-----------|:----------------------------:|
| Data augmentation | No |
| Network architecture | No |
| Autoencoder training objective | No |
| **Scoring / statistical processing** | **Yes** |
| Decision threshold procedure | No (baseline gamma quantile retained) |

**Motivation:** Mahalanobis distance assumes that second-order statistics of normal data can be estimated reliably. In 128-dimensional mel-frame residual space, weak or noisy directions can distort the covariance matrix. PCA restricts scoring to the leading residual subspace, which is intended to **stabilize** distance-based anomaly detection without altering the reconstruction model itself.

---

# 2. Thesis Context

This repository documents **one independent thesis experiment** within the broader project on Coffee Grinder anomalous sound detection under selective Mahalanobis scoring (DCASE 2025 Task 2).

**Baseline modified:** The official **DCASE Task 2 autoencoder baseline** (`DCASE2023T2-AE`), implemented in PyTorch. The selective Mahalanobis framework (separate source/target statistics and hard-minimum domain combination) is preserved; only the **residual space** in which Mahalanobis distance is computed is changed via PCA.

**Components that remain unchanged:**

- Log-mel feature extraction (`librosa`)
- Autoencoder topology and training (MSE loss, Adam, 100 epochs)
- Domain labeling (source vs. target from filenames)
- Gamma-based decision thresholding (90th percentile)
- Evaluation metrics (AUC, pAUC, and threshold-dependent precision/recall/F1)

**Independent evaluation:** This experiment is **not combined** with other thesis variants that are absent from this codebase (e.g., SpecAugment, domain-conditioned autoencoders). Comparative score modes `MSE` and `MAHALA` exist in the same project for reference but represent separate runs.

---

# 3. Method Description

## Core idea

The autoencoder models normal acoustic patterns in the log-mel domain. At test time, deviations are measured through reconstruction residuals. In **PCA–Mahalanobis scoring**, residuals are projected onto the top-\(k\) principal directions learned from training data before estimating domain-specific covariances and computing Mahalanobis distance.

## Theoretical motivation

Principal Component Analysis orders directions by variance on the training residual distribution. Directions with small variance often correspond to unstable or noisy components that inflate covariance estimation error. Restricting Mahalanobis scoring to the leading \(k\)-dimensional subspace is a form of **variance-driven dimensionality reduction for statistical inference**, not for visualization or storage.

## Expected impact on anomaly detection

- **Potential benefit:** More stable covariance estimates and distance values when \(k \ll 128\), which may improve ranking metrics (AUC, pAUC) especially under domain shift.
- **Trade-off:** Anomalous structure aligned with discarded principal directions may be attenuated in the projected score.

## Relationship to domain shift

DCASE 2025 Task 2 introduces **source** and **target** acoustic domains. The baseline **selective Mahalanobis** mode estimates separate covariances per domain and combines scores with the **minimum** across domains (optimistic normal assumption). This experiment **retains** that selective mechanism; PCA is applied **before** domain-specific covariance estimation, using a **shared** projection matrix \(\mathbf{W}\) and mean \(\boldsymbol{\mu}_{\mathrm{PCA}}\) fitted on all training-split residuals.

## Relationship to Mahalanobis scoring

Standard Mahalanobis scoring uses \(\delta^\top \Sigma^{-1} \delta\) in full residual space. Here \(\delta = (\mathbf{r} - \boldsymbol{\mu}_{\mathrm{PCA}})\mathbf{W}\) with \(\mathbf{r} \in \mathbb{R}^{128}\) per mel frame, and \(\Sigma\) is \(k \times k\) in PCA space. The quadratic form and selective `min(source, target)` aggregation follow the baseline; only the coordinate system changes.

---

# 4. Implementation Details

## 4.1 Main Components

| File | Purpose |
|------|---------|
| `train.py` | Entry point: seeding, loads `baseline.yaml`, runs training and/or testing |
| `common.py` | Argument definitions (`--score`, `--mahala_pca_dim`, features, training hyperparameters) |
| `baseline.yaml` | Default hyperparameters (overridden by CLI) |
| `networks/models.py` | Maps model name `DCASE2023T2-AE` to implementation class |
| `networks/dcase2023t2_ae/dcase2023t2_ae.py` | Training loop, PCA fit (`_run_mahala_pca_covariance_phase`, `_fit_pca_numpy`), scoring (`_mahala_pca_quad_form`), evaluation (`test`, `eval`) |
| `networks/dcase2023t2_ae/network.py` | Autoencoder `AENet` (encoder/decoder, 8-D bottleneck) |
| `networks/criterion/mahala.py` | Covariance helpers (`cov_v_diff`, `cov_v`) and standard Mahalanobis utilities |
| `networks/base_model.py` | Paths, gamma score distribution fit, decision threshold |
| `datasets/datasets.py` | Dataset factory, train/valid/test loaders |
| `datasets/dcase_dcase202x_t2_loader.py` | DCASE T2 data loading, mel feature pickles |
| `datasets/loader_common.py` | `file_to_vectors()` — log-mel and sliding-window features |
| `train_ae.sh` / `test_ae.sh` | Thin wrappers calling `train.py` (do not set `--score` or `--export_dir` by default) |
| `requirements.txt` | Python package dependencies |

## 4.2 Configuration Parameters

Parameters defined in `common.py` and/or `baseline.yaml`, as used in the saved Coffee Grinder run (`args.json`).

| Parameter | Description | Value (final Coffee Grinder run) |
|-----------|-------------|----------------------------------|
| `--score` | Anomaly scoring mode | `MAHALA_PCA` |
| `--mahala_pca_dim` | Number of PCA components retained | `32` |
| `--dataset` | Machine-type dataset identifier | `DCASE2025T2CoffeeGrinder` |
| `--eval` | Use evaluation dataset protocol | enabled |
| `--use_ids` | Section IDs for training subset | `0` (section 00) |
| `--export_dir` | Experiment artifact subdirectory name | `baseline_pca` |
| `--seed` | Random seed | `13711` |
| `--epochs` | Autoencoder training epochs | `100` |
| `--batch_size` | Training batch size | `256` |
| `--learning_rate` | Adam learning rate | `0.001` |
| `--validation_split` | Fraction held out for validation | `0.1` |
| `--decision_threshold` | Gamma quantile for binary decision | `0.9` |
| `--max_fpr` | Maximum FPR for pAUC computation | `0.1` |
| `--n_mels` | Mel filter banks | `128` |
| `--frames` | Frames per feature vector | `5` |
| `--frame_hop_length` | Hop between consecutive vectors | `1` |
| `--n_fft` | FFT size | `1024` |
| `--hop_length` | STFT hop length | `512` |
| `--power` | Mel power exponent | `2.0` |
| `--mono` | Monaural audio loading | `True` |
| `--use_cuda` | Request GPU if available | `True` (runtime may fall back to CPU) |
| `--gpu_id` | GPU device index | `[0]` |
| `--model` | Model class name | `DCASE2023T2-AE` |

*No covariance regularization parameter (\(\lambda\)) is implemented in this codebase.*

## 4.3 Training Procedure

**Inputs:** Mono waveforms from DCASE 2025 Coffee Grinder evaluation data (`./data/dcase2025t2/eval_data/`), section 00, train split.

**Feature extraction:** Log-mel spectrogram via `librosa`; sliding windows of 5 frames × 128 mels → 640-dimensional vectors (`datasets/loader_common.py`).

**Autoencoder training (epochs 1–100):**

- Model: `AENet` — input 640 → hidden 128 → **bottleneck 8** → decode to 640.
- Loss: `F.mse_loss` between input and reconstruction (independent of `--score` during these epochs).
- Optimizer: Adam (`networks/dcase2023t2_ae/dcase2023t2_ae.py`).
- Validation: 10% hold-out split; losses logged to `log.csv`.

**Post-training statistics (epoch 101, when `--score MAHALA_PCA`):**

1. Collect residuals \(\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}\) on training loader; reshape to \((\cdot, 128)\).
2. Fit PCA (`_fit_pca_numpy`): mean, SVD, retain `mahala_pca_dim` components.
3. Project training residuals; accumulate source/target covariances in PCA space.
4. Invert covariances; save `*_mahala_pca.pt`.
5. Compute MAHALA_PCA scores on train + validation loaders; fit gamma distribution (`fit_anomaly_score_distribution`).

---

# 5. Inference and Scoring Pipeline

1. **Audio input** — Test waveform (one file per batch at test time: `batch_size = n_vectors_ea_file`).
2. **Log-mel spectrogram extraction** — `librosa` mel features with configured FFT/hop/mel bands.
3. **Feature vector construction** — 5-frame context vectors (640-D).
4. **Autoencoder inference** — Forward pass; reconstruction \(\hat{\mathbf{x}}\).
5. **Residual computation** — \(\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}\); reshape to \(N \times 128\) rows.
6. **PCA projection** — \(\boldsymbol{\delta} = (\mathbf{r} - \boldsymbol{\mu}_{\mathrm{PCA}})\mathbf{W}\), \(k = 32\).
7. **Mahalanobis scoring per domain** — Quadratic form with \(\Sigma^{-1}_{\mathrm{source}}\) and \(\Sigma^{-1}_{\mathrm{target}}\); aggregate over frame rows (mean of row means of \(\boldsymbol{\delta}\Sigma^{-1}\boldsymbol{\delta}^\top\)).
8. **Selective combination** — Clip score \(s = \min(s_{\mathrm{source}}, s_{\mathrm{target}})\).
9. **Threshold comparison** — Load gamma threshold (90th percentile of train+valid MAHALA_PCA scores); anomaly if \(s > \text{threshold}\).
10. **Outputs** — Anomaly score CSV, binary decision CSV, and metric CSV (AUC/pAUC/F1 when labels are available).

---

# 6. Repository Structure

```
PCA/
├── train.py                 # Main entry point
├── common.py                # CLI and baseline.yaml loader
├── baseline.yaml            # Default hyperparameters
├── requirements.txt         # Python dependencies
├── train_ae.sh              # Training wrapper script
├── test_ae.sh               # Testing wrapper script
├── networks/
│   ├── models.py
│   ├── base_model.py        # Thresholding, paths
│   ├── criterion/
│   │   └── mahala.py
│   └── dcase2023t2_ae/
│       ├── dcase2023t2_ae.py  # MAHALA_PCA implementation
│       └── network.py         # AENet
├── datasets/
│   ├── datasets.py
│   ├── dcase_dcase202x_t2_loader.py
│   ├── loader_common.py
│   └── machine_type_2025_eval.yaml
├── data/
│   └── dcase2025t2/
│       └── eval_data/       # Dataset (not versioned fully)
├── models/
│   ├── checkpoint/
│   │   └── baseline_pca/   # Checkpoints and args.json
│   └── saved_model/
│       └── baseline_pca/   # Weights, *_mahala_pca.pt, score distributions
├── logs/
│   └── baseline_pca/
├── results/
│   └── eval_data/
│       └── baseline_pca_MAHALA_PCA/   # Evaluation CSVs and figures
└── tools/                   # Legacy DCASE scripts and utilities
```

---

# 7. Running the Experiment

## Training

Full training including PCA/covariance fitting (epoch 101) and optional immediate test:

```bash
python train.py \
  --dataset DCASE2025T2CoffeeGrinder \
  --eval \
  --export_dir baseline_pca \
  --score MAHALA_PCA \
  --mahala_pca_dim 32 \
  --use_ids 0 \
  --mono True \
  --seed 13711
```

Training only:

```bash
python train.py \
  --dataset DCASE2025T2CoffeeGrinder \
  --eval \
  --export_dir baseline_pca \
  --score MAHALA_PCA \
  --mahala_pca_dim 32 \
  --use_ids 0 \
  --mono True \
  --seed 13711 \
  --train_only
```

## Evaluation

Requires trained `models/saved_model/baseline_pca/DCASE2023T2-AE_DCASE2025T2CoffeeGrinder_Eval_seed13711.pth` and `..._mahala_pca.pt`:

```bash
python train.py \
  --dataset DCASE2025T2CoffeeGrinder \
  --eval \
  --export_dir baseline_pca \
  --score MAHALA_PCA \
  --use_ids 0 \
  --mono True \
  --seed 13711 \
  --test_only
```

## Important Notes

| Requirement | Detail |
|-------------|--------|
| **Dataset** | DCASE 2025 Coffee Grinder evaluation data under `./data/dcase2025t2/eval_data/` |
| **Configuration** | Defaults in `baseline.yaml`; PCA run overrides `export_dir`, `score`, and uses CLI flags above |
| **Checkpoints** | `*_mahala_pca.pt` must exist before testing with `MAHALA_PCA` |
| **Section filter** | `--use_ids 0` restricts to section 00 |
| **Wrappers** | `train_ae.sh` / `test_ae.sh` do not pass `--export_dir` or `--score MAHALA_PCA`; use `train.py` directly for this experiment |

---

# 8. Experimental Configuration

| Component | Setting |
|-----------|---------|
| Dataset | `DCASE2025T2CoffeeGrinder` |
| Protocol | Evaluation (`--eval`) |
| Section | 00 (`--use_ids 0`) |
| Model | `DCASE2023T2-AE` |
| Score type | `MAHALA_PCA` |
| Export directory | `baseline_pca` |
| Random seed | 13711 |
| Input dimension | 640 (5 × 128 mels) |
| Residual row dimension | 128 (per mel frame) |
| Autoencoder bottleneck (latent) | 8 |
| PCA dimension (\(k\)) | 32 |
| Covariance regularization | Not implemented |
| Domain score fusion | Hard minimum: `min(source, target)` |
| Decision threshold | Gamma 90th percentile (`decision_threshold = 0.9`) |
| Training epochs | 100 (+ 1 statistics epoch) |
| Batch size | 256 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss (AE training) | MSE |
| pAUC max FPR | 0.1 |

---

# 9. Results

**Source:** `results/eval_data/baseline_pca_MAHALA_PCA/result_DCASE2025T2CoffeeGrinder_test_seed13711_Eval_roc.csv` (section 00).

| Metric | Value |
|--------|-------|
| AUC (Source) | 0.727 |
| AUC (Target) | 0.460 |
| pAUC | 0.573 |
| pAUC (Source) | 0.543 |
| pAUC (Target) | 0.592 |
| Precision (Source) | 0.500 |
| Precision (Target) | 0.500 |
| Recall (Source) | 1.000 |
| Recall (Target) | 1.000 |
| F1 Score (Source) | 0.667 |
| F1 Score (Target) | 0.667 |

*Precision, recall, and F1 are computed at the gamma-based operating point (quantile 0.9). DCASE ranking emphasizes threshold-free AUC and pAUC.*

---

# 10. Discussion

**Pipeline aspect improved:** The **statistical scoring stage** after autoencoder reconstruction—specifically covariance-based Mahalanobis distance in residual space—not front-end features or AE training.

**Strengths:**

- Minimal intrusion into the proven DCASE baseline (same AE, same selective Mahalanobis logic, same thresholding).
- Principled variance-based reduction of residual dimensionality before \(\Sigma\) estimation.
- Reproducible artifacts (PCA matrix, inverse covariances, gamma parameters) saved to disk.

**Limitations:**

- PCA subspace is shared across domains; domain shift is handled only after projection via separate \(\Sigma\).
- No explicit regularization on \(\Sigma\); inversion may be numerically fragile.
- PCA is fit on all training-split residuals jointly, not on source-only normal data.
- Frame-level scores are aggregated via mean over a quadratic-form matrix including cross-terms, which differs from a simple per-frame max statistic.

**When improvement is expected:** Conditions where full 128-D residual covariances are poorly conditioned and anomaly energy concentrates in high-variance residual directions. Gains may differ between source and target domains (cf. AUC 0.727 vs. 0.460 in the reported run).

---

# 11. Reproducibility

| Item | Specification |
|------|----------------|
| Dataset | DCASE 2025 Task 2 Coffee Grinder evaluation set (`eval_data`) |
| Configuration file | `baseline.yaml` + CLI overrides |
| Random seed | `13711` |
| Determinism | `torch.backends.cudnn.deterministic = True`, `torch.use_deterministic_algorithms = True` in `train.py` |
| Hardware | GPU optional (`--use_cuda`); CPU fallback if CUDA unavailable |
| Required checkpoints | `..._seed13711.pth`, `..._seed13711_mahala_pca.pt`, `score_distr_*_mahala_pca.pickle` |
| Evaluation procedure | `python train.py ... --test_only` with matching `export_dir`, `score`, `seed`, and dataset flags |
| Saved run record | `models/checkpoint/baseline_pca/DCASE2023T2-AE_DCASE2025T2CoffeeGrinder_Eval_seed13711/args.json` |

---

# 12. Citation

**DCASE 2025 Task 2 (challenge overview)**

```bibtex
@inproceedings{dcase2025t2,
  title        = {{DCASE} 2025 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring},
  author       = {{DCASE Community}},
  year         = {2025},
  note         = {https://dcase.community/challenge2025/}
}
```

**Baseline autoencoder implementation (upstream)**

```bibtex
@misc{dcase2023t2_baseline_ae,
  title        = {DCASE 2023--2026 Task 2 Baseline Auto Encoder},
  author       = {{NTT Corporation and contributors}},
  howpublished = {\url{https://github.com/nttcslab/dcase2023_task2_baseline_ae}},
  note         = {PyTorch implementation; selective Mahalanobis and MSE scoring modes}
}
```

