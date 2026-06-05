# Domain-Aware Convolutional Autoencoder (Domain-CAE)

**Thesis supplementary repository:** *Coffee Grinder Anomalous Sound Detection in Selective Mahalanobis Mode* — DCASE 2025 Task 2 (first-shot unsupervised anomalous sound detection).

---

# 1. Overview

This experiment implements a **Domain-Conditioned Convolutional Autoencoder (Domain-CAE)** with **domain-aware Mahalanobis anomaly scoring** (`DOMAIN_MAHALA`). The system learns to reconstruct normal log-mel feature vectors while modelling **source** and **target** acoustic domains, then detects anomalies from reconstruction residuals using **separate Mahalanobis statistics** per domain.

**Problem addressed.** DCASE 2025 Task 2 requires **unsupervised anomalous sound detection** under **domain shift**: training provides many normal source-domain clips and few normal target-domain clips, while test-time domain identity is unknown. A single unconditioned autoencoder may yield poorly calibrated errors across domains.

**Component targeted by this experiment.**

| Component | Addressed? |
|-----------|------------|
| Data augmentation (e.g. SpecAugment) | No — not implemented in this repository |
| **Architecture** (encoder, conditional decoder, domain head) | **Yes — primary contribution** |
| **Scoring / statistical processing** (`DOMAIN_MAHALA`, `hard_min` / `weighted`) | **Yes — extension of selective Mahalanobis** |
| PCA or other feature transforms | No — not implemented |

**Motivation.** The official baseline trains a domain-agnostic autoencoder and applies selective Mahalanobis scoring only at the **post-reconstruction** stage. This experiment moves **domain awareness into representation learning** (conditional decoding and domain supervision) while retaining **blind inference** at test time (no ground-truth domain labels).

---

# 2. Thesis Context

This repository holds **one independent thesis experiment**. It is evaluated separately from other proposed methods and does **not** combine with techniques that are absent from this codebase.

| Aspect | Description |
|--------|-------------|
| **Baseline modified** | Official DCASE Task 2 PyTorch autoencoder baseline (`DCASE2023T2-AE` in `networks/dcase2023t2_ae/dcase2023t2_ae.py`) |
| **Registered model name** | `DCASE2023T2-Domain-CAE` (`networks/models.py`) |
| **Unchanged from baseline** | Log-mel feature extraction (`datasets/loader_common.py`), dataset loaders, gamma threshold fitting (`networks/base_model.py`), overall train/test entry point (`train.py`), optional `MAHALA` / `MSE` score modes |
| **Changed** | CNN encoder, latent bottleneck (16-D), domain classifier head, concatenation-based conditional decoder, `DOMAIN_MAHALA` scoring with `hard_min` or `weighted` aggregation |
| **Primary thesis machine** | Coffee Grinder — `DCASE2025T2CoffeeGrinder`, evaluation (`--eval`) protocol |
| **Not in this folder** | SpecAugment, PCA, ensemble systems, or other thesis variants |

Multi-year baseline download and shell workflows are documented in [README_legacy.md](README_legacy.md).

---

# 3. Method Description

### Core idea

The method consists of two coupled parts:

1. **Domain-CAE (learning).** A convolutional encoder maps each 640-D log-mel feature vector to a 16-D latent code. A small domain classifier predicts whether the sample is **target-like**. The decoder reconstructs the input from `[latent ; domain signal]`, where the domain signal is the **true** binary domain label during training and the **predicted** target probability at validation and test.

2. **DOMAIN_MAHALA (scoring).** After training, source and target **covariance matrices** (128×128) are estimated from normal-data reconstruction residuals. At inference, each test clip receives **two** Mahalanobis distances; these are combined into one file-level score by **`hard_min`** (minimum per vector, then mean) or **`weighted`** (convex combination using predicted domain probability).

### Theoretical motivation

Domain shift changes the distribution of normal sounds without providing anomaly labels. By conditioning reconstruction on domain information, the autoencoder can learn **domain-specific normal manifolds** rather than a single averaged manifold. The domain BCE term encourages a latent representation that is predictive of domain, which supports routing at inference when filenames do not reveal domain (evaluation set).

Selective Mahalanobis scoring assumes that normal reconstruction errors follow domain-dependent second-order structure. Keeping **separate** `cov_source` and `cov_target` follows the official selective Mahalanobis baseline; Domain-CAE changes **how** residuals are generated, not the existence of dual covariances.

### Expected impact

- **Improved domain-aware reconstruction** for source and target normal sounds.
- **More informative residuals** for Mahalanobis distance when domains differ acoustically.
- **Blind test-time operation**: `hard_min` does not require test domain labels; `weighted` uses only model-predicted probabilities.

### Relationship to domain shift

Training uses filename-derived **source/target** labels (substring `"target"`). Inference does **not**. On the evaluation dataset, test filenames do not contain domain tags; the domain head and decoder must generalise from audio alone.

### Relationship to Mahalanobis scoring

Training optimises **MSE reconstruction + weighted BCE**; Mahalanobis distance is **not** the training loss. After epoch `epochs`, an additional pass (epoch `epochs + 1`) estimates covariances. Anomaly scores at test time are purely **statistical** on residuals, consistent with the selective Mahalanobis family, extended by `DOMAIN_MAHALA` aggregation modes.

---

# 4. Implementation Details

## 4.1 Main Components

| File | Purpose |
|------|---------|
| `train.py` | Entry point: load config, seed RNG, instantiate model, run training (`epochs + 1` iterations) and optional `test()` |
| `common.py` | YAML loading, CLI argument definitions |
| `baseline_domain_cae.yaml` | Default hyperparameters for Domain-CAE experiments |
| `networks/models.py` | Maps `DCASE2023T2-Domain-CAE` → `DCASE2023T2DomainCAE` |
| `networks/dcase2023t2_ae/domain_cae_network.py` | `DomainCAENet` architecture |
| `networks/dcase2023t2_ae/dcase2023t2_domain_cae.py` | Training loop, validation, covariance pass, `DOMAIN_MAHALA` test/scoring |
| `networks/criterion/mahala.py` | Covariance helpers, `calc_inv_cov`, Mahalanobis utilities |
| `networks/base_model.py` | Dataset wiring, checkpoints, gamma score distribution, threshold |
| `networks/dcase2023t2_ae/dcase2023t2_ae.py` | Baseline AE (not used when `model_type=domain_cae`) |
| `networks/dcase2023t2_ae/network.py` | Baseline `AENet` (fully connected) |
| `datasets/datasets.py` | Dataset registry including `DCASE2025T2CoffeeGrinder` |
| `datasets/dcase_dcase202x_t2_loader.py` | Feature pickle loading, train/valid/test loaders |
| `datasets/loader_common.py` | WAV → log-mel → vector extraction |
| `requirements.txt` | Python dependencies (PyTorch, librosa, scipy, etc.) |

## 4.2 Configuration Parameters

Parameters below exist in `common.py` and/or `baseline_domain_cae.yaml`. CLI arguments override YAML on the second `parse_args` pass in `train.py`.

| Parameter | Default (YAML) | Description |
|-----------|----------------|-------------|
| `--config` | `baseline.yaml` (CLI); experiment uses `baseline_domain_cae.yaml` | Path to YAML config |
| `--model` | `DCASE2023T2-Domain-CAE` | Registered model class |
| `--model_type` | `domain_cae` | Maps to `DCASE2023T2-Domain-CAE` |
| `--score` | `DOMAIN_MAHALA` | Anomaly score: `DOMAIN_MAHALA`, `MAHALA`, or `MSE` |
| `--domain_scoring_mode` | `weighted` | `hard_min` or `weighted` (for `DOMAIN_MAHALA`) |
| `--lambda_domain` | `0.05` | Weight of domain BCE in training loss |
| `--domain_prob_clip` | `0.05` | Clip range for weighted scoring: `[clip, 1−clip]` |
| `--seed` | `13711` | Random seed (Python, NumPy, PyTorch) |
| `--epochs` | `100` | Training epochs; covariance pass at epoch `epochs + 1` |
| `-lr` / `--learning_rate` | `0.001` | Adam learning rate |
| `--batch_size` | `256` | Training/validation batch size |
| `--validation_split` | `0.1` | Fraction of training vectors held out for validation |
| `--shuffle` | `True` | Training loader shuffle |
| `--n_mels` | `128` | Mel filter banks |
| `--frames` | `5` | Frames per feature vector (640-D input) |
| `--frame_hop_length` | `1` | Frame hop between consecutive vectors |
| `--n_fft` | `1024` | FFT size |
| `--hop_length` | `512` | STFT hop length |
| `--power` | `2.0` | Mel spectrogram power |
| `--fmin` | `0.0` | Minimum mel frequency |
| `--fmax` | `null` | Maximum mel frequency (optional) |
| `--win_length` | `null` | Window length (optional) |
| `--decision_threshold` | `0.9` | Gamma quantile for binary decision |
| `--max_fpr` | `0.1` | Max FPR for pAUC computation |
| `--export_dir` | `baseline_domain_cae` | Subdirectory tag for results and checkpoints |
| `--dataset_directory` | `./data` | Root data directory |
| `--result_directory` | `./results` | Output root |
| `--use_cuda` | `True` | Enable GPU if available |
| `--gpu_id` | `[0]` | GPU device index |
| `--mono` | `True` | Mono audio loading |
| `--is_auto_download` | `False` | Automatic dataset download |
| `--dev` / `--eval` | flags (off in YAML) | Development vs evaluation data path |
| `--use_ids` | `[]` (CLI example: `0`) | Machine ID filter for training |
| `-tag` / `--model_name_suffix` | `''` (example: `id0`) | Suffix in output filenames |
| `--train_only` | `False` | Train without test |
| `--test_only` | `False` | Test without train |
| `--restart` | `False` | Resume from checkpoint |

**Not present in code:** PCA dimension, SpecAugment parameters, learning-rate scheduler.

## 4.3 Training Procedure

**Inputs.** Mono WAV files from `data/dcase2025t2/eval_data/raw/CoffeeGrinder/train/` (normal source and target) for the Coffee Grinder experiment. Labels: normal only; domain inferred from filename (`"target"` → 1, else 0).

**Feature extraction** (`loader_common.file_to_vectors`):

1. Load audio (librosa).
2. Compute mel spectrogram (`n_fft`, `hop_length`, `n_mels`, `power`, optional `fmin`/`fmax`/`win_length`).
3. Convert to log-mel energies.
4. Slide a window of `frames` consecutive frames with hop `frame_hop_length` → vectors of size `n_mels × frames` = **640**.

**Training process** (`DCASE2023T2DomainCAE.train`):

- Epochs `1 … epochs`: `model.train()`, Adam updates.
- Epoch `epochs + 1`: `model.eval()`, no gradients; accumulate `cov_source` and `cov_target`; fit gamma distributions for Mahalanobis scores on train+valid; save weights and checkpoint.
- Each training epoch ends with validation (`model.eval()`, `domain_label=None`).

**Loss functions.**

$$\mathcal{L} = \mathcal{L}_{\mathrm{recon}} + \lambda_{\mathrm{domain}} \cdot \mathcal{L}_{\mathrm{BCE}}$$

- \(\mathcal{L}_{\mathrm{recon}}\): batch mean of per-sample MSE between `recon` and `x` (`F.mse_loss`, `reduction="none"` then mean over 640 dims, then batch average).
- \(\mathcal{L}_{\mathrm{BCE}}\): `binary_cross_entropy_with_logits` on domain head vs filename-derived labels.

**Optimization.** Adam, fixed learning rate, no scheduler in code. `torch.autograd.set_detect_anomaly(True)` during training.

**Decoder conditioning in training.** `model(data, domain_label=domain_label)` with true labels → teacher forcing on decoder input.

---

# 5. Inference and Scoring Pipeline

1. **Audio input** — One test WAV file per batch (batch size = all vectors in file).
2. **Log-mel spectrogram** — Same parameters as training (`loader_common`).
3. **Feature vectors** — Sequence of 640-D vectors for the clip.
4. **Domain-CAE forward** — `model(data, domain_label=None)` in `eval` mode:
   - Encode → `z` (16-D).
   - Domain head → logit → `p_target = σ(logit)`.
   - Decode with `domain_signal = p_target`.
5. **Residual** — `delta = x - recon` per mel block (128-D blocks, five per vector).
6. **Mahalanobis distances** — `D_source`, `D_target` using `inv_cov_source`, `inv_cov_target` (`_per_sample_mahalanobis_means`).
7. **Domain score selection** (`--score DOMAIN_MAHALA`):
   - **`hard_min`:** per vector \(d_i = \min(D_{\mathrm{source},i}, D_{\mathrm{target},i})\); file score = mean over vectors.
   - **`weighted`:** per vector \((1-p_i') D_{\mathrm{source},i} + p_i' D_{\mathrm{target},i}\) with clipped \(p_i'\); file score = mean.
8. **Final anomaly score** — Scalar written to `anomaly_score_*.csv`.
9. **Binary decision (optional)** — Compare score to gamma threshold (`decision_threshold=0.9`) → `decision_result_*.csv`.
10. **Evaluation metrics** — On development or renamed eval data: AUC, pAUC, precision, recall, F1 per domain (`dcase2023t2_domain_cae.test`).

---

# 6. Repository Structure

```text
domain_cae_mahala/
├── train.py                          # Training and evaluation entry point
├── common.py                         # Configuration and CLI
├── baseline_domain_cae.yaml          # Domain-CAE experiment config
├── baseline.yaml                     # Baseline AE config (reference)
├── requirements.txt
├── README.md
├── README_legacy.md                  # DCASE baseline / legacy scripts
├── networks/
│   ├── models.py
│   ├── base_model.py
│   ├── criterion/
│   │   └── mahala.py
│   └── dcase2023t2_ae/
│       ├── domain_cae_network.py     # DomainCAENet
│       ├── dcase2023t2_domain_cae.py # Train / test / DOMAIN_MAHALA
│       ├── dcase2023t2_ae.py         # Baseline trainer
│       └── network.py                # Baseline AENet
├── datasets/
│   ├── datasets.py
│   ├── dcase_dcase202x_t2_loader.py
│   ├── loader_common.py
│   └── download_path_2025.yaml
├── data/
│   └── dcase2025t2/
│       └── eval_data/raw/CoffeeGrinder/   # Expected data location
├── models/
│   ├── saved_model/                  # Trained .pth weights
│   └── checkpoint/                   # Checkpoints and args.json
├── logs/                             # Training CSV logs
├── results/
│   └── eval_data/
│       └── baseline_domain_cae_DOMAIN_MAHALA/   # Scores and ROC CSVs
└── tools/                            # Legacy download / test shell scripts
```

---

# 7. Running the Experiment

## Training

`train.py` runs **training and testing in one invocation** unless `--train_only` or `--test_only` is set. Training executes `epochs` gradient epochs plus **one** covariance pass (epoch `epochs + 1`).

**Coffee Grinder — evaluation data, machine ID 0, DOMAIN_MAHALA, hard_min** (CLI overrides YAML defaults):

```bash
python train.py \
  --config=baseline_domain_cae.yaml \
  --model_type=domain_cae \
  --dataset=DCASE2025T2CoffeeGrinder \
  --eval \
  --use_ids 0 \
  -tag id0 \
  --score DOMAIN_MAHALA \
  --domain_scoring_mode hard_min \
  --lambda_domain 0.05 \
  --mono=True
```

To train without testing:

```bash
python train.py --config=baseline_domain_cae.yaml --model_type=domain_cae \
  --dataset=DCASE2025T2CoffeeGrinder --eval --use_ids 0 -tag id0 --train_only --mono=True
```

## Evaluation

To run inference only (requires existing checkpoint/weights):

```bash
python train.py --config=baseline_domain_cae.yaml --model_type=domain_cae \
  --dataset=DCASE2025T2CoffeeGrinder --eval --use_ids 0 -tag id0 \
  --score DOMAIN_MAHALA --domain_scoring_mode hard_min --test_only --mono=True
```

Legacy shell scripts (`tools/01_train_legacy.sh`, `tools/02b_test_legacy.sh`) reference DCASE2025 machines including CoffeeGrinder but invoke `train_ae.sh` with `--train_only` only; the **Domain-CAE** experiment is run via `train.py` as above.

## Important Notes

| Requirement | Detail |
|-------------|--------|
| **Dataset** | DCASE 2025 Coffee Grinder eval package under `data/dcase2025t2/eval_data/raw/CoffeeGrinder/` (`train/`, `test/`) |
| **Config** | `baseline_domain_cae.yaml` |
| **Checkpoint** | Written to `models/checkpoint/baseline_domain_cae/.../checkpoint.tar` |
| **Weights** | `models/saved_model/baseline_domain_cae/DCASE2023T2-Domain-CAE_DCASE2025T2CoffeeGrinder_id0_Eval_seed{seed}.pth` |
| **GPU** | Optional; `--use_cuda True` |
| **Scoring mode** | Document `--domain_scoring_mode` for each run; YAML default is `weighted` |

---

# 8. Experimental Configuration

Settings for the documented Coffee Grinder run (inferred from config, command, and result filename). Confirm `domain_scoring_mode` against the run log if reproducing exactly.

| Component | Setting |
|-----------|---------|
| Dataset | `DCASE2025T2CoffeeGrinder` |
| Data protocol | `--eval` (evaluation / first-shot) |
| Machine ID | `0` (`--use_ids 0`, tag `id0`) |
| Model | `DCASE2023T2-Domain-CAE` (`--model_type domain_cae`) |
| Score type | `DOMAIN_MAHALA` |
| Domain scoring mode | `hard_min` (command); YAML default `weighted` |
| Latent dimension | **16** (code default in `DomainCAENet`) |
| Input dimension | **640** (128 mels × 5 frames) |
| Covariance block size | **128** (`n_mels`) |
| `lambda_domain` | **0.05** |
| `domain_prob_clip` | **0.05** (used when `weighted`) |
| Optimiser | Adam, lr **0.001** |
| Epochs | **100** (+ 1 covariance pass) |
| Batch size | **256** |
| Seed | **13711** |
| `decision_threshold` | **0.9** |
| `max_fpr` (pAUC) | **0.1** |
| `export_dir` | `baseline_domain_cae` |
| Mono input | `True` |
| PCA | **Not implemented** |

---

# 9. Results

**Source file:**  
`results/eval_data/baseline_domain_cae_DOMAIN_MAHALA/baseline_domain_cae_DOMAIN_MAHALA/result_DCASE2025T2CoffeeGrinder_test_seed13711_id0_Eval_roc.csv`

**Section 00** (evaluation, seed 13711, suffix `_id0`). The implementation reports precision, recall, and F1 **per domain**; pooled single-domain values are not computed in code.

| Metric | Source | Target |
|--------|-------:|-------:|
| AUC (source) / AUC (target) | **0.721** | **0.423** |
| pAUC (overall) | **0.514** | — |
| pAUC (source) | **0.480** | — |
| pAUC (target) | — | **0.531** |
| Precision | **0.535** | **0.526** |
| Recall | **0.460** | **1.000** |
| F1 score | **0.495** | **0.690** |

*Overall pAUC is computed across both domains in `test()` (standard DCASE baseline behaviour). Binary metrics use the gamma threshold at `decision_threshold = 0.9`.*

### Template (additional runs)

| Metric | Value |
|--------|------:|
| AUC (source) | — |
| AUC (target) | — |
| pAUC | — |
| pAUC (source) | — |
| pAUC (target) | — |
| Precision (source) | — |
| Precision (target) | — |
| Recall (source) | — |
| Recall (target) | — |
| F1 score (source) | — |
| F1 score (target) | — |

---

# 10. Discussion

**Pipeline aspect improved.** This experiment primarily improves the **representation and reconstruction** stage under domain shift, and refines **how** dual Mahalanobis channels are combined (`DOMAIN_MAHALA` vs baseline `min` on legacy `MAHALA` path).

**Strengths.**

- Explicit domain supervision and conditional decoding during training.
- Convolutional encoder over local time–frequency structure (vs baseline fully connected AE).
- Test-time operation without domain labels (`hard_min` or predicted-weighted blending).
- Compatible with the existing DCASE training and evaluation harness.

**Limitations.**

- Mahalanobis covariances remain **post-hoc** and may be unstable with few target training clips.
- `lambda_domain`, `domain_scoring_mode`, and `domain_prob_clip` add hyperparameters not present in the baseline AE.
- Target-domain AUC (0.423 in the recorded run) remains challenging relative to source.
- Architecture change alone does not guarantee improved official ranking; scoring mode and seed matter.

**When improvement is expected.**

- Machine types with pronounced source/target shift where domain-conditioned reconstruction reduces cross-domain residual bias.
- Settings where selective Mahalanobis benefits from more domain-aligned residuals.

**When improvement may be limited.**

- Very few target normals (first-shot regime) limiting covariance and domain-head reliability.
- Evaluation clips where domain prediction or `hard_min` selection fails to match the generating domain.

---

# 11. Reproducibility

| Item | Checklist |
|------|-----------|
| ☐ | Install dependencies from `requirements.txt` (PyTorch 2.6 + CUDA optional) |
| ☐ | Place DCASE 2025 Coffee Grinder data in `data/dcase2025t2/eval_data/raw/CoffeeGrinder/` |
| ☐ | Use config file `baseline_domain_cae.yaml` |
| ☐ | Set random seed `--seed 13711` (or document alternative) |
| ☐ | Set `--dataset DCASE2025T2CoffeeGrinder --eval --use_ids 0 -tag id0` |
| ☐ | Set `--model_type domain_cae --score DOMAIN_MAHALA` |
| ☐ | Record `--domain_scoring_mode` (`hard_min` or `weighted`) |
| ☐ | Record `--lambda_domain` (default 0.05) |
| ☐ | Run `python train.py` with flags in Section 7 |
| ☐ | Verify outputs under `results/eval_data/baseline_domain_cae_DOMAIN_MAHALA/` |
| ☐ | Retain checkpoint in `models/checkpoint/baseline_domain_cae/` for `--test_only` reruns |
| ☐ | Hardware: GPU recommended (`--use_cuda True`); CPU supported if CUDA unavailable |

**Evaluation procedure.** After training, `test()` loads saved `.pth` weights, computes `DOMAIN_MAHALA` scores per test file, writes CSVs, and computes ROC metrics when ground-truth labels are available (development or `test_rename` eval layout).

---

# 12. Citation

```bibtex
@inproceedings{dcase2025task2,
  title        = {Description and Discussion on {DCASE} 2025 Challenge Task 2:
                  First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring},
  author       = {{DCASE Challenge Organisers}},
  year         = {2025},
  note         = {Task description: \url{https://dcase.community/challenge2025/}}
}

@inproceedings{harada2023firstshot,
  title        = {First-Shot Anomaly Sound Detection for Machine Condition Monitoring:
                  A Domain Generalization Baseline},
  author       = {Harada, Noboru and Niizumi, Daisuke and Ohishi, Yasunori and Takeuchi, Daiki and Yasuda, Masahiro},
  booktitle    = {Proc. EUSIPCO},
  year         = {2023}
}
```

**Software reference:** PyTorch implementation extending [dcase2023\_task2\_baseline\_ae](https://github.com/nttcslab/dcase2023_task2_baseline_ae) — Domain-CAE module in `networks/dcase2023t2_ae/`.

---

