# Face-FFT

## Detecting Low Resolution AI Generated Face Videos Using Spectral Analysis

---

## Overview

Face-FFT is a research repository investigating whether low resolution AI generated human face videos can be detected using structured spatiotemporal frequency analysis.

Rather than relying on pixel-level artifacts or large pretrained models, this project analyzes 3D frequency representations derived from video volumes to identify systematic signals introduced during generative modeling.

The focus is strictly on lightweight, interpretable detection in constrained settings.

---

## Motivation

Modern video generation systems have rapidly improved in realism and accessibility. High-end proprietary systems such as Sora and Veo 3 demonstrate strong visual fidelity, while open source models such as CogVideoX 2B and Wan 2.2 enable scalable generation on consumer hardware.

As generation becomes cheaper and more realistic, low resolution synthetic face videos become viable for impersonation, fraud, and misinformation. Even modest resolution content is often sufficient for social media distribution.

This repository investigates whether structured signals in the frequency domain can reliably distinguish synthetic from authentic face videos under low resolution constraints.

---

## Methodology

### 1. Data Construction

The dataset consists of paired real and synthetic face videos.

- Real videos sourced from the Kaggle Selfies and Video dataset
- Synthetic counterparts generated from the first frame of each real video
- Image-to-video generation setup
- Generators:
  - CogVideoX 2B
  - Wan 2.2
- Resolution constraint: 256 × 256
- Matched lighting and background within each real/synthetic pair
- Final dataset stored in PyTorch-compatible format

This paired design ensures controlled comparison between authentic and generated content.

---

### 2. Spectral Transformation

Each video volume is transformed using a 3D Fast Fourier Transform across spatial and temporal dimensions.

The resulting frequency representation captures:

- Spatial frequency structure
- Temporal frequency dynamics
- Joint spatiotemporal interactions

The 3D frequency volume serves as the model input.

---

### 3. Classification

A compact binary convolutional neural network is trained to classify:

- Real
- Synthetic

The architecture is intentionally lightweight to emphasize signal detection rather than large-scale representation learning.

Evaluation metrics include:

- Confusion matrix
- F1 score

---

### 4. Generalization Evaluation

Performance is evaluated beyond the training distribution to measure robustness.

Evaluation includes:

- Subsets of GenVidBench
- Additional open source generators not used during training
- Select foundation models when available

The goal is cross-model generalization rather than optimizing narrowly for a single generator.

---

## Repository Structure

- `apptainer/`
  Container definition files for UVA Rivanna HPC.

- `scripts/`
  Example Slurm job scripts for training and evaluation.

- `src/`
  Source code for preprocessing, spectral transformation, model training, and evaluation.

- `tests/`
  Unit and integration tests.

---

## Development Workflow

- `dev` is the default development branch.
- Direct merges into `dev` are not permitted.
- All changes must be introduced via pull requests.
- Install pre-commit hooks before committing:

```bash
pre-commit install
```

- All CI checks must pass before approval.

## Commit Message Convention

Prefix commit messages with:

- `feat`: new features
- `fix`: bug fixes
- `chore`: tooling or maintenance
- `refactor`: structural improvements
- `test`: test additions or updates
- `docs`: documentation updates

Example:

```txt
feat: add spectral preprocessing module
```

## Scope

This repository is intentionally constrained to:

- Human face videos only
- Low resolution settings
- Binary classification
- Lightweight detection architecture

The objective is focused spectral signal analysis rather than broad multimedia forensic modeling.
