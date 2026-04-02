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

        # TemporalLSTMAutoencoder only applies dropout between LSTM layers.
        # PyTorch forces inter-layer dropout to 0.0 when lstm_layers=1, so the
        # config value is irrelevant in that case.  We track the *effective*
        # value so the Bayes controller learns not to waste trials on it.
        effective_dropout = 0.0 if cfg.lstm_layers == 1 else float(cfg.dropout)
        wandb.config.update({"effective_dropout": effective_dropout}, allow_val_change=True)

        # Build standardized naming for this trial
        arch_type = "attn" if cfg.get("use_attention", False) else "baseline"
        loss_abbrev = cfg.loss_type[:3]  # "jsd" or "chi"
        
        run_name = (
            f"{arch_type}-{loss_abbrev}-"
            f"seq{cfg.seq_len}-lat{cfg.latent_dim}-hid{cfg.lstm_hidden_dim}-"
            f"drop{effective_dropout:.1f}-{cfg.lstm_layers}layer"
        )
        
        model_filename = (
            f"lstm-{arch_type}-{loss_abbrev}-"
            f"seq{cfg.seq_len}-lat{cfg.latent_dim}-hid{cfg.lstm_hidden_dim}-"
            f"drop{effective_dropout:.2f}-lr{cfg.learning_rate:.1e}.pt"
        ).replace("+", "")
        
        output_path = REPO_ROOT / "models" / model_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Update run name and tags for discoverability
        run.name = run_name
        tags = [arch_type, cfg.loss_type, f"seq{cfg.seq_len}", f"lat{cfg.latent_dim}"]
        if effective_dropout > 0:
            tags.append(f"drop{effective_dropout:.1f}")
        if cfg.get("latent_mask_pct", 0.0) > 0:
            tags.append(f"mask{cfg.get('latent_mask_pct', 0.0):.2f}")
        run.tags = tags

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
            use_attention=cfg.get("use_attention", False),
            loss_type=cfg.loss_type,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            scheduler_factor=cfg.get("scheduler_factor", 0.5),
            scheduler_patience=cfg.get("scheduler_patience", 3),
            scheduler_min_lr=cfg.get("scheduler_min_lr", 1e-6),
            scheduler_threshold=cfg.get("scheduler_threshold", 1e-4),
            scheduler_cooldown=cfg.get("scheduler_cooldown", 1),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            min_epochs=cfg.get("min_epochs", 10),
            early_stopping_patience=cfg.get("early_stopping_patience", 8),
            early_stopping_min_delta=cfg.get("early_stopping_min_delta", 1e-4),
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
        wandb.summary["best_epoch"] = int(result.get("best_epoch", 0))
        wandb.summary["epochs_completed"] = int(result.get("epochs_completed", 0))
        wandb.summary["stopped_early"] = bool(result.get("stopped_early", False))
        wandb.summary["stop_reason"] = str(result.get("stop_reason", ""))
        wandb.summary["model_path"] = str(result["model_path"])
        wandb.summary["model_filename"] = model_filename

        # Upload the best checkpoint as an artifact so you can retrieve it later.
        artifact = wandb.Artifact(
            name=f"model-{arch_type}-{loss_abbrev}-seq{cfg.seq_len}-lat{cfg.latent_dim}",
            type="model",
            metadata={
                "run_name": run_name,
                "model_filename": model_filename,
                "best_val_loss": float(result["best_val_loss"]),
                "loss_type": cfg.loss_type,
                "seq_len": cfg.seq_len,
                "latent_dim": cfg.latent_dim,
                "lstm_hidden_dim": cfg.lstm_hidden_dim,
                "lstm_layers": cfg.lstm_layers,
                "use_attention": cfg.get("use_attention", False),
                "dropout": effective_dropout,
                "latent_mask_pct": cfg.get("latent_mask_pct", 0.0),
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "best_epoch": int(result.get("best_epoch", 0)),
                "epochs_completed": int(result.get("epochs_completed", 0)),
                "stopped_early": bool(result.get("stopped_early", False)),
                "stop_reason": str(result.get("stop_reason", "")),
            },
        )
        artifact.add_file(str(result["model_path"]))
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    sweep_run()
