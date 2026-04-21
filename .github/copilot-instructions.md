# Project Guidelines

## Code Style
- Target Python 3.8+ and match the existing style in `gammaflow/`, `examples/`, and `tests/`.
- Prefer small, explicit helpers over clever abstractions.
- Keep type hints and docstrings consistent with neighboring files; do not add broad reformatting.
- For direct-run scripts under `examples/`, keep the repo-root `sys.path` bootstrap pattern so imports work from the workspace.

## Architecture
- Core packages live under `gammaflow/core`, `gammaflow/algorithms`, `gammaflow/datasets`, `gammaflow/operations`, `gammaflow/training`, and `gammaflow/visualization`.
- `ARADDetector` is a per-spectrum convolutional autoencoder; `LSTMTemporalDetector` is a causal window model that reconstructs the final spectrum from history.
- The temporal training pipeline consumes preprocessed run files (`run*.pt`) from `preprocessed-data/*` and saves detector checkpoints plus metrics JSON.
- Evaluation scripts may override the model's loss with `score_type`; `jsd`, `corrected_jsd`, `chi2`, `normalized_chi2`, and `reduced_chi2` are supported where implemented.

## Build and Test
- Install editable with dev dependencies: `pip install -e ".[dev]"`.
- Run the full test suite with `pytest tests/`.
- Run a focused temporal test with `pytest tests/test_lstm_temporal.py`.
- Run coverage with `pytest --cov=gammaflow`.
- Prefer the commands documented in `CLAUDE.md` and `tests/README.md` for project workflows.

## Conventions
- Use `gammaflow/training/lstm_temporal_pipeline.py` and `examples/train_lstm_temporal_preprocessed.py` for training from preprocessed RADAI runs.
- When training or evaluating on GPU, pass `--device cuda --require-cuda` so the run fails fast instead of silently falling back to CPU.
- For temporal scoring, remember the sequence warmup window and `seq_stride` change the effective history span.
- Preprocessed RADAI artifacts in `preprocessed-data/` are generated data; avoid editing them unless the task is specifically about regeneration or validation.
- Link to existing docs instead of duplicating them: `README.md`, `CLAUDE.md`, and `tests/README.md`.
