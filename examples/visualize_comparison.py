#!/usr/bin/env python3
"""
Compare ARAD and LSTM reconstructions side-by-side on the same image.
Plots: [Original Spectrum | ARAD Reconstruction | LSTM Reconstruction]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gammaflow.algorithms.arad import ARADDetector
from gammaflow.algorithms.lstm_temporal import LSTMTemporalDetector
from gammaflow.visualization.plotting import plot_spectrum_comparison


def _load_run(run_path):
    """Load a preprocessed run file (.pt)"""
    data = torch.load(run_path, map_location="cpu", weights_only=False)
    spectra = data.get("spectra")
    live_times = data.get("live_times")
    
    if spectra is not None:
        spectra = np.asarray(spectra)
        if spectra.dtype == object:
            spectra = np.array([np.asarray(s) for s in spectra])
    
    if live_times is not None:
        live_times = np.asarray(live_times)
        if live_times.dtype == object:
            live_times = np.array([float(lt) if lt is not None else 1.0 for lt in live_times])
    
    return spectra, live_times


def _to_target_scale_lstm(reconstruction, chi2_loss):
    """
    Denormalize LSTM reconstruction to target scale for comparison.
    LSTM applies max-normalization: x_norm = x / max(x)
    """
    if chi2_loss:
        # chi2: LSTM normalizes by spectrum max, then reconstructs normalized form
        # To plot, scale back up (heuristically use 90th percentile of reconstruction)
        percentile_val = np.percentile(reconstruction, 90) if np.max(reconstruction) > 0 else 1.0
        if percentile_val > 0:
            return reconstruction * percentile_val / 0.9
        return reconstruction
    else:
        # JSD: similar scaling
        percentile_val = np.percentile(reconstruction, 90) if np.max(reconstruction) > 0 else 1.0
        if percentile_val > 0:
            return reconstruction * percentile_val / 0.9
        return reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Compare ARAD and LSTM reconstructions side-by-side"
    )
    parser.add_argument(
        "--arad-model-path", type=str, required=True, help="Path to ARAD model checkpoint"
    )
    parser.add_argument(
        "--lstm-model-path", type=str, required=True, help="Path to LSTM model checkpoint"
    )
    parser.add_argument(
        "--run-file", type=str, required=True, help="Path to preprocessed run file (.pt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="comparison-outputs", help="Output directory for PNGs"
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", help="Specific spectrum indices to visualize"
    )
    parser.add_argument(
        "--log-y", action="store_true", help="Use log scale for y-axis"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device: cpu or cuda"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print(f"Loading ARAD from {args.arad_model_path}...")
    arad = ARADDetector(device=args.device, verbose=False)
    arad.load(args.arad_model_path)
    chi2_loss_arad = arad.loss_type == "chi2"
    
    print(f"Loading LSTM from {args.lstm_model_path}...")
    lstm = LSTMTemporalDetector(device=args.device, verbose=False)
    lstm.load(args.lstm_model_path)
    chi2_loss_lstm = lstm.loss_type == "chi2"
    
    # Load run file
    print(f"Loading run from {args.run_file}...")
    spectra, live_times = _load_run(args.run_file)
    n_spectra = len(spectra)
    print(f"Loaded {n_spectra} spectra")
    
    # Determine indices to visualize
    if args.indices:
        indices = args.indices
    else:
        # Default: all spectra
        indices = list(range(min(10, n_spectra)))
    
    indices = [i for i in indices if i < n_spectra]
    if not indices:
        print("No valid indices found")
        return
    
    print(f"Visualizing {len(indices)} spectra: {indices}")
    
    # Get latent mask for LSTM
    latent_mask = lstm.latent_mask if hasattr(lstm, 'latent_mask') else None
    
    summary = {"indices": [], "arad_scores": [], "lstm_scores": []}
    
    for idx in indices:
        spectrum_data = spectra[idx]
        live_time = live_times[idx] if live_times is not None and live_times.dtype != object else 1.0
        
        # ARAD reconstruction and score
        with torch.no_grad():
            # Normalize to count rate
            spectrum_rate = spectrum_data / live_time if live_time > 0 else spectrum_data
            spectrum_tensor = torch.from_numpy(spectrum_rate).float().unsqueeze(0).to(args.device)
            
            # Score via ARAD
            arad.model_.eval()
            arad_recon_norm = arad.model_(spectrum_tensor).squeeze(0).cpu().numpy()
            
            # Denormalize ARAD reconstruction
            max_val = np.max(spectrum_rate)
            arad_recon = arad_recon_norm * max_val
            
            # Compute ARAD score
            if arad.loss_type == 'jsd':
                x_norm = spectrum_rate / (np.max(spectrum_rate) + 1e-8)
                arad_score_norm = arad_recon_norm / (np.max(arad_recon_norm) + 1e-8)
                # JSD score computation (simplified)
                arad_score = np.sqrt(0.5 * np.sum((x_norm * np.log(x_norm / arad_score_norm + 1e-10) + 
                                                    arad_score_norm * np.log(arad_score_norm / x_norm + 1e-10))))
            else:  # chi2
                eps = 1e-8
                arad_recon_denorm = arad_recon_norm * (max_val + eps)
                arad_score = np.sum((spectrum_rate - arad_recon_denorm) ** 2 / (arad_recon_denorm + eps))
        
        # LSTM reconstruction and score (requires causal window)
        if idx < lstm.seq_len:
            print(f"  Skipping index {idx} (before LSTM seq_len={lstm.seq_len})")
            continue
        
        with torch.no_grad():
            # Build causal window
            window = spectra[idx - lstm.seq_len : idx + 1]  # seq_len + 1 samples
            window_rate = window / (live_times[max(0, idx - lstm.seq_len):idx + 1, np.newaxis] if live_times is not None else 1.0)
            window_tensor = torch.from_numpy(window_rate).float().unsqueeze(0).to(args.device)
            
            # Forward through LSTM
            lstm.model_.eval()
            lstm_out = lstm.model_(window_tensor)  # Output is [batch, n_bins]
            lstm_recon_normalized = lstm_out.squeeze(0).cpu().numpy()  # Get the full reconstruction
            
            # Denormalize LSTM reconstruction
            spectrum_rate = spectrum_data / live_time if live_time > 0 else spectrum_data
            lstm_recon = _to_target_scale_lstm(lstm_recon_normalized, chi2_loss_lstm)
            
            # LSTM score (simplified)
            lstm_score = np.mean(np.abs(spectrum_rate - lstm_recon))

        
        # Create comparison plot with all three on same graph
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot all three on same axes
        if args.log_y:
            ax.semilogy(spectrum_rate, 'k-', linewidth=2, label=f'Original (idx={idx})', alpha=0.8)
            ax.semilogy(arad_recon, 'b--', linewidth=2, label=f'ARAD (score={arad_score:.4f})', alpha=0.7)
            ax.semilogy(lstm_recon, 'r-.', linewidth=2, label=f'LSTM (score={lstm_score:.4f})', alpha=0.7)
        else:
            ax.plot(spectrum_rate, 'k-', linewidth=2, label=f'Original (idx={idx})', alpha=0.8)
            ax.plot(arad_recon, 'b--', linewidth=2, label=f'ARAD (score={arad_score:.4f})', alpha=0.7)
            ax.plot(lstm_recon, 'r-.', linewidth=2, label=f'LSTM (score={lstm_score:.4f})', alpha=0.7)
        
        ax.set_xlabel("Energy Bin", fontsize=12)
        ax.set_ylabel("Count Rate (counts/sec)", fontsize=12)
        ax.set_title(f"Spectrum Reconstruction Comparison - Index {idx}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(args.output_dir, f"comparison_idx{idx}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓ idx={idx}  | ARAD score={arad_score:.4f} | LSTM score={lstm_score:.4f}")
        
        summary["indices"].append(int(idx))
        summary["arad_scores"].append(float(arad_score))
        summary["lstm_scores"].append(float(lstm_score))
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nWrote {len(indices)} comparison images to {args.output_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
