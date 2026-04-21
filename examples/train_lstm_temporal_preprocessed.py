"""Train LSTMTemporalDetector from preprocessed RADAI run tensors.

Example:
    python examples/train_lstm_temporal_preprocessed.py \
      --preprocessed-dir preprocessed-data/no-sources-2.0-1.0 \
      --output-model models/lstm_softmax_jsd_2.0-1.0.pt \
      --epochs 40 --batch-size 256 --seq-len 20 --seq-stride 1 \
      --output-activation softmax --loss-type jsd \
      --mask-target --early-stopping-patience 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure local repo package is importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow.training import train_lstm_temporal_from_preprocessed


def _maybe_import_wandb(enabled: bool):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
        return wandb
    except ImportError as exc:
        raise ImportError(
            "W&B was requested but is not installed. Install with: pip install wandb"
        ) from exc


def _build_wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "preprocessed_dir": args.preprocessed_dir,
        "output_model": args.output_model,
        "seq_len": args.seq_len,
        "seq_stride": args.seq_stride,
        "latent_dim": args.latent_dim,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "mask_target": args.mask_target,
        "loss_type": args.loss_type,
        "output_activation": args.output_activation,
        "count_rate_conditioning": args.count_rate_conditioning,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_min_lr": args.scheduler_min_lr,
        "scheduler_threshold": args.scheduler_threshold,
        "scheduler_cooldown": args.scheduler_cooldown,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "grad_clip_norm": args.grad_clip_norm,
        "latent_mask_pct": args.latent_mask_pct,
        "latent_mask_seed": args.latent_mask_seed,
        "cache_size": args.cache_size,
        "device": args.device,
        "require_cuda": args.require_cuda,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTMTemporalDetector from preprocessed run tensors"
    )

    # --- Data ---
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        required=True,
        help="Directory containing run*.pt and preprocess_stats.json",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        required=True,
        help="Path to save trained detector checkpoint (.pt)",
    )

    # --- Architecture ---
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--mask-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask the target timestep latent embedding to prevent identity shortcut (default: on)",
    )
    parser.add_argument(
        "--output-activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "softmax"],
        help="Output activation for the decoder (default: sigmoid)",
    )
    parser.add_argument(
        "--count-rate-conditioning",
        action="store_true",
        help="Condition the decoder on log1p(total_counts) to reduce count-rate correlation",
    )

    # --- Loss ---
    parser.add_argument("--loss-type", type=str, default="jsd", choices=["mse", "jsd", "chi2", "poisson"])

    # --- Optimiser ---
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    # --- LR scheduler ---
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau multiplicative LR decay factor (default: 0.5)",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="Epochs with no val-loss improvement before LR decay (default: 3)",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=1e-6,
        help="Minimum LR floor for scheduler (default: 1e-6)",
    )
    parser.add_argument(
        "--scheduler-threshold",
        type=float,
        default=1e-4,
        help="Minimum val-loss delta counted as improvement by scheduler (default: 1e-4)",
    )
    parser.add_argument(
        "--scheduler-cooldown",
        type=int,
        default=1,
        help="Cooldown epochs after an LR reduction (default: 1)",
    )

    # --- Training loop ---
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=10,
        help="Minimum epochs before early stopping is allowed (default: 10)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Epochs without meaningful val-loss improvement before stopping (default: 8)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val-loss improvement to reset early-stopping counter (default: 1e-4)",
    )

    # --- Latent masking ---
    parser.add_argument(
        "--latent-mask-pct",
        type=float,
        default=0.0,
        help="Fraction of history latent timesteps to randomly zero during training (default: 0.0)",
    )
    parser.add_argument(
        "--latent-mask-seed",
        type=int,
        default=None,
        help="Optional fixed seed for latent masking RNG (default: uses training seed)",
    )

    # --- Data loading ---
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=2)

    # --- Device ---
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is unavailable instead of falling back to CPU",
    )

    # --- Logging ---
    parser.add_argument("--quiet", action="store_true", help="Disable per-epoch logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gammaflow-lstm")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags (e.g. 'lstm,jsd,softmax')",
    )
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode (default: online; use offline for air-gapped runs)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wandb = _maybe_import_wandb(args.wandb)
    wb_run = None
    epoch_logger = None

    if wandb is not None:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=tags,
            mode=args.wandb_mode,
            job_type="train",
            config=_build_wandb_config(args),
        )

        def _log_epoch(metrics: Dict[str, float]) -> None:
            wandb.log(
                {
                    "epoch": int(metrics["epoch"]),
                    "train/loss": float(metrics["train_loss"]),
                    "val/loss": float(metrics["val_loss"]),
                    "train/lr": float(metrics["lr"]),
                    "train/best_val_loss": float(metrics["best_val_loss"]),
                    "train/best_epoch": int(metrics.get("best_epoch", 0)),
                    "train/epochs_without_improvement": int(
                        metrics.get("epochs_without_improvement", 0)
                    ),
                },
                step=int(metrics["epoch"]),
            )

        epoch_logger = _log_epoch

    result = train_lstm_temporal_from_preprocessed(
        preprocessed_dir=args.preprocessed_dir,
        output_model_path=args.output_model,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        latent_dim=args.latent_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        mask_target=args.mask_target,
        loss_type=args.loss_type,
        output_activation=args.output_activation,
        count_rate_conditioning=args.count_rate_conditioning,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_threshold=args.scheduler_threshold,
        scheduler_cooldown=args.scheduler_cooldown,
        batch_size=args.batch_size,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        latent_mask_pct=args.latent_mask_pct,
        latent_mask_seed=args.latent_mask_seed,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        grad_clip_norm=args.grad_clip_norm,
        cache_size=args.cache_size,
        device=args.device,
        require_cuda=args.require_cuda,
        verbose=not args.quiet,
        epoch_end_callback=epoch_logger,
    )

    if wb_run is not None:
        wandb.summary["best_val_loss"] = float(result["best_val_loss"])
        wandb.summary["best_epoch"] = int(result.get("best_epoch", 0))
        wandb.summary["epochs_completed"] = int(result.get("epochs_completed", 0))
        wandb.summary["stopped_early"] = bool(result.get("stopped_early", False))
        wandb.summary["stop_reason"] = str(result.get("stop_reason", ""))
        wandb.summary["model_path"] = str(result["model_path"])
        wandb.summary["metrics_path"] = str(result["metrics_path"])

        model_artifact = wandb.Artifact(
            name=f"lstm-temporal-{Path(result['model_path']).stem}",
            type="model",
            metadata={
                "best_val_loss": float(result["best_val_loss"]),
                "loss_type": args.loss_type,
                "output_activation": args.output_activation,
                "count_rate_conditioning": args.count_rate_conditioning,
                "seq_len": args.seq_len,
                "seq_stride": args.seq_stride,
                "mask_target": args.mask_target,
                "latent_mask_pct": args.latent_mask_pct,
                "best_epoch": int(result.get("best_epoch", 0)),
                "epochs_completed": int(result.get("epochs_completed", 0)),
                "stopped_early": bool(result.get("stopped_early", False)),
            },
        )
        model_artifact.add_file(str(result["model_path"]))
        model_artifact.add_file(str(result["metrics_path"]))
        wandb.log_artifact(model_artifact)

        with open(result["metrics_path"], "r", encoding="utf-8") as f:
            metrics_payload: Dict[str, Any] = json.load(f)
        history = metrics_payload.get("history", {})
        if history:
            history_table = wandb.Table(columns=["epoch", "train_loss", "val_loss", "lr"])
            n_epochs = len(history.get("train_loss", []))
            for idx in range(n_epochs):
                history_table.add_data(
                    idx + 1,
                    float(history.get("train_loss", [])[idx]),
                    float(history.get("val_loss", [])[idx]),
                    float(history.get("lr", [])[idx]),
                )
            wandb.log({"train/history": history_table})

        wandb.finish()

    print("Training complete")
    print(f"Model:            {result['model_path']}")
    print(f"Metrics:          {result['metrics_path']}")
    print(f"Best val loss:    {result['best_val_loss']:.6f}")
    print(f"Best epoch:       {int(result.get('best_epoch', 0))}")
    print(f"Epochs completed: {int(result.get('epochs_completed', 0))}")
    print(f"Stopped early:    {bool(result.get('stopped_early', False))}")


if __name__ == "__main__":
    main()
