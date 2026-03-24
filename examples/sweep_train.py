"""W&B sweep agent entry-point for LSTMTemporalDetector.

This script is called automatically by `wandb agent`. It does NOT use argparse;
instead it reads hyperparameters from `wandb.config` which the sweep controller
injects before each trial.

Usage (after registering the sweep once):
    wandb sweep examples/sweep.yaml          # prints <entity>/<project>/<sweep_id>
    wandb agent <entity>/<project>/<sweep_id>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import wandb  # must be installed: pip install wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow.training import train_lstm_temporal_from_preprocessed


def sweep_run() -> None:
    with wandb.init() as run:
        cfg = run.config

        # Output path is unique per trial so runs don't clobber each other.
        output_path = REPO_ROOT / "models" / f"sweep-{run.id}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # TemporalLSTMAutoencoder only applies dropout between LSTM layers.
        # PyTorch forces inter-layer dropout to 0.0 when lstm_layers=1, so the
        # config value is irrelevant in that case.  We track the *effective*
        # value so the Bayes controller learns not to waste trials on it.
        effective_dropout = 0.0 if cfg.lstm_layers == 1 else float(cfg.dropout)
        wandb.config.update({"effective_dropout": effective_dropout}, allow_val_change=True)

        def _log_epoch(metrics: Dict[str, float]) -> None:
            wandb.log(
                {
                    "epoch": int(metrics["epoch"]),
                    "train/loss": float(metrics["train_loss"]),
                    "val/loss": float(metrics["val_loss"]),
                    "train/lr": float(metrics["lr"]),
                    "train/best_val_loss": float(metrics["best_val_loss"]),
                },
                step=int(metrics["epoch"]),
            )

        result = train_lstm_temporal_from_preprocessed(
            preprocessed_dir=cfg.preprocessed_dir,
            output_model_path=str(output_path),
            seq_len=cfg.seq_len,
            seq_stride=cfg.get("seq_stride", 1),
            latent_dim=cfg.latent_dim,
            lstm_hidden_dim=cfg.lstm_hidden_dim,
            lstm_layers=cfg.lstm_layers,
            dropout=effective_dropout,
            loss_type=cfg.loss_type,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            val_fraction=cfg.val_fraction,
            seed=cfg.get("seed", 42),
            grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
            latent_mask_pct=cfg.get("latent_mask_pct", 0.0),
            latent_mask_seed=cfg.get("latent_mask_seed", cfg.get("seed", 42)),
            num_workers=cfg.get("num_workers", 0),
            verbose=False,  # keeps logs clean when running many parallel agents
            epoch_end_callback=_log_epoch,
        )

        wandb.summary["best_val_loss"] = float(result["best_val_loss"])
        wandb.summary["model_path"] = str(result["model_path"])

        # Upload the best checkpoint as an artifact so you can retrieve it later.
        artifact = wandb.Artifact(
            name=f"sweep-model-{run.id}",
            type="model",
            metadata={
                "best_val_loss": float(result["best_val_loss"]),
                "loss_type": cfg.loss_type,
                "seq_len": cfg.seq_len,
                "latent_dim": cfg.latent_dim,
                "lstm_hidden_dim": cfg.lstm_hidden_dim,
                "latent_mask_pct": cfg.get("latent_mask_pct", 0.0),
            },
        )
        artifact.add_file(str(result["model_path"]))
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    sweep_run()
