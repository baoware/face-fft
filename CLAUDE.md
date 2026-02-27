# CLAUDE.md

## Project Context

This repository implements a detection pipeline for low resolution AI generated human face videos using spatiotemporal spectral analysis.

The core idea is that efficient video generation models introduce structured artifacts in the 3D frequency domain due to latent compression, patch tokenization, and temporal downsampling. These artifacts may remain detectable even when spatial resolution is low.

The pipeline:

1. Applies a 3D Fast Fourier Transform to video volumes
2. Trains a lightweight CNN classifier on spectral representations
3. Evaluates generalization across generation models

Claude should preserve this conceptual framing when modifying or extending the codebase.

---

## Core Technical Assumptions

Efficient video generation models typically rely on:

- Latent compression
- Patch based tokenization
- Space time flattening
- Temporal downsampling

Expected artifacts:

- Periodic grid aligned frequency spikes
- Harmonic peaks tied to compression stride
- Temporal discontinuities
- Structured spectral anisotropy

When implementing changes, avoid introducing steps that would unintentionally destroy frequency domain structure unless explicitly requested.

---

## Architectural Boundaries

The following constraints define project scope:

- Only human face videos are considered
- Low resolution setting such as 256 by 256
- Binary classification: Real vs Synthetic
- Lightweight detection pipeline
- Emphasis on spectral features rather than pixel domain heuristics

Do not expand scope into:

- Multiclass model attribution
- High resolution forensic pipelines
- Audio based detection
- Large transformer based detection models

---

## Data Pipeline Expectations

Dataset structure assumptions:

- Paired real and synthetic videos
- Identical lighting and background within pairs
- Synthetic videos generated from first frame of real videos
- PyTorch compatible dataset format

When modifying data loading:

- Maintain deterministic pairing
- Avoid data leakage between train and test splits
- Preserve reproducibility

---

## Spectral Processing Guidelines

The detection pipeline relies on:

- 3D FFT across spatial and temporal dimensions
- Structured frequency volume as model input

Guidelines:

- Do not replace FFT with learned spectral transforms unless explicitly requested
- Maintain dimensional consistency across preprocessing steps
- Document normalization choices clearly
- Do not collapse the temporal dimension unless required

---

## Model Design Principles

The classifier should remain:

- Lightweight
- Interpretable in terms of spectral sensitivity
- Computationally feasible for HPC cluster use

Avoid:

- Excessively deep architectures
- Large pretrained vision transformers
- High memory footprint models

The goal is signal detection rather than large scale representation learning.

---

## Evaluation Philosophy

Evaluation should emphasize:

- Cross model generalization
- Robustness to unseen generators
- F1 score and confusion matrix reporting

Avoid optimizing exclusively for in distribution accuracy on a single generator.

Generalization is more important than maximizing training domain performance.

---

## HPC and Environment Awareness

This project targets UVA Rivanna HPC.

When writing or modifying code:

- Assume Slurm scheduling
- Avoid interactive only workflows
- Ensure reproducibility in containerized environments

---

## Modification Guidelines for Claude

When introducing changes:

1. Preserve spectral analysis as the central signal.
2. Avoid scope creep.
3. Prefer simple, well documented implementations.
4. Maintain compatibility with existing dataset structure.
5. Keep experiments modular and reproducible.

If a proposed change weakens the spectral hypothesis, explicitly justify it in comments.
