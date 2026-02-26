# Face-FFT

## Repository Overview

This repository is organized as follows:

- `apptainer/`
  Contains all Apptainer-related definition files and build scripts used to build container images for UVA HPC (Rivanna).

- `scripts/`
  Contains example Slurm job scripts used to run workloads on the HPC environment.

- `src/`
  Contains all source code for the FloodWatch Live project.

- `tests/`
  Contains all unit tests and integration tests for the source code.

---

## Development Rules

Please follow these rules when contributing to the project:

- The dev branch is the default development branch.
- Always create pull requests instead of merging directly into dev.
- Run the following command before committing to enable pre-commit hooks:

```bash
pre-commit install
```

- Ensure all CI checks pass before approving or merging a pull request.
- Use the following commit tags at the start of commit messages:
  - feat: new features
  - fix: bug fixes
  - chore: maintenance or tooling changes
  - refactor: code refactoring
  - test: adding or updating tests
  - docs: adding or updating documentations

Example commit message:

`feat: add real-time data generation pipeline`
