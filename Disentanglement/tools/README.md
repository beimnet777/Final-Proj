# Disentanglement Tools

This folder contains non-core utilities. Core training and model code stays at
the Disentanglement root or in `model/` and `data/`.

## Folders

- `analysis/`: log parsing, plotting, and report generation scripts.
- `diagnostics/`: post-hoc diagnostic jobs that inspect checkpoints or features.
- `smoke/`: local smoke tests that patch heavy dependencies where possible.

These scripts compute the project root from their own location, so they can be
run from the Disentanglement directory or from the repository root.
