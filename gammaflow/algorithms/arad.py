"""
ARAD (Autoencoder Reconstruction Anomaly Detection) for gamma-ray spectra.

Uses a convolutional autoencoder to learn background spectrum patterns
and detect anomalies via reconstruction error.

References
----------
Ghawaly J, Nicholson A, Archer D, Willis M, Garishvili I, Longmire B, Rowe A, Stewart I,
    Cook M. Characterization of the Autoencoder Radiation Anomaly Detection (ARAD) model.
    Engineering Applications of Artificial Intelligence. 2022 May; 111:104761-. Available from:
    https://linkinghub.elsevier.com/retrieve/pii/S0952197622000550 DOI:10.1016/j.engappai.2022.104761
"""

import numpy as np
from typing import Optional, List, Dict, Any
import warnings
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. ARAD detector requires PyTorch. "
        "Install with: pip install torch"
    )

from gammaflow.algorithms.base import BaseDetector
from gammaflow.core.time_series import SpectralTimeSeries
from gammaflow.core.spectrum import Spectrum


# ======================================================================
# Neural network components
# ======================================================================

class ARADEncoderBlock(nn.Module):
    """Encoder convolutional block with conv -> batchnorm -> maxpool -> dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.mp = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.mp(F.mish(self.bn(self.conv(x)))))


class ARADDecoderBlock(nn.Module):
    """Decoder convolutional block with upsample -> deconv -> activation -> batchnorm -> dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout: float, is_output: bool = False):
        super().__init__()
        self.is_output = is_output
        padding = (kernel_size - 1) // 2

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=padding)

        if is_output:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Mish()
            self.bn = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.deconv(self.upsample(x))
        if self.is_output:
            return self.activation(x)
        else:
            return self.dropout(self.bn(self.activation(x)))


class ARADAutoencoder(nn.Module):
    """Convolutional autoencoder for gamma-ray spectra."""

    def __init__(self, n_bins: int, latent_dim: int = 8, dropout: float = 0.2):
        super().__init__()

        self.n_bins = n_bins
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            ARADEncoderBlock(1, 8, 7, dropout),
            ARADEncoderBlock(8, 8, 5, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            nn.Flatten(),
            nn.Linear(8 * (n_bins // 32), latent_dim),
            nn.Mish(),
            nn.BatchNorm1d(latent_dim)
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 8 * (n_bins // 32)),
            nn.Mish(),
            nn.BatchNorm1d(8 * (n_bins // 32)),
        )

        self.decoder = nn.Sequential(
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 5, dropout),
            ARADDecoderBlock(8, 1, 7, dropout, is_output=True),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

    def _normalize_input(self, x):
        eps = 1e-8
        if x.dim() == 1:
            max_val = torch.max(x)
            normalized_x = x / (max_val + eps)
            normalized_x = normalized_x.view(1, 1, -1)
        else:
            max_val = torch.max(x, dim=1, keepdim=True).values
            normalized_x = x / (max_val + eps)
            normalized_x = normalized_x.view(x.size(0), 1, x.size(1))
        return normalized_x

    def forward(self, x):
        """
        Forward pass through autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input spectra, shape (batch, n_bins) or (n_bins,).

        Returns
        -------
        torch.Tensor
            Reconstructed spectra normalized to [0, 1].
        """
        normalized_x = self._normalize_input(x)
        latent = self.encoder(normalized_x)
        decoded = self.decoder_linear(latent)
        decoded = decoded.view(decoded.size(0), 8, self.n_bins // 32)
        reconstructed = self.decoder(decoded)

        if x.dim() == 1:
            return reconstructed.view(-1)
        return reconstructed.view(reconstructed.size(0), x.size(1))


# ======================================================================
# Detector
# ======================================================================

class ARADDetector(BaseDetector):
    """
    ARAD (Autoencoder Reconstruction Anomaly Detection) for gamma-ray spectra.

    Uses a convolutional autoencoder trained on background spectra to detect
    anomalies via reconstruction error (Jensen-Shannon Divergence or
    Chi-squared).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space (default: 8).
    dropout : float
        Dropout rate for regularization (default: 0.2).
    batch_size : int
        Training batch size (default: 32).
    learning_rate : float
        Initial learning rate (default: 0.01).
    epochs : int
        Maximum training epochs (default: 50).
    l1_lambda : float
        L1 regularization weight (default: 1e-3).
    l2_lambda : float
        L2 regularization weight via AdamW (default: 1e-3).
    early_stopping_patience : int
        Patience for early stopping (default: 6).
    validation_split : float
        Fraction of training data for validation (default: 0.2).
    device : str or None
        PyTorch device. Auto-selects if ``None``.
    threshold : float or None
        Anomaly detection threshold.
    aggregation_gap : float
        Time gap (seconds) for aggregating consecutive alarms (default: 2.0).
    loss_type : str
        Loss function: ``'jsd'`` or ``'chi2'`` (default: ``'jsd'``).
    verbose : bool
        Print training progress (default: True).

    Attributes
    ----------
    model_ : ARADAutoencoder or None
        Trained autoencoder model.
    n_bins_ : int or None
        Number of energy bins (set after ``fit``).
    training_history_ : dict
        Training and validation loss curves.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        dropout: float = 0.2,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 50,
        l1_lambda: float = 1e-3,
        l2_lambda: float = 1e-3,
        early_stopping_patience: int = 6,
        validation_split: float = 0.2,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        aggregation_gap: float = 2.0,
        loss_type: str = 'jsd',
        verbose: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("ARAD requires PyTorch. Install with: pip install torch")

        super().__init__(threshold=threshold, aggregation_gap=aggregation_gap)

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.loss_type = loss_type.lower()
        self.verbose = verbose

        if self.loss_type not in ('jsd', 'chi2'):
            raise ValueError(f"loss_type must be 'jsd' or 'chi2', got '{loss_type}'")

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        if self.verbose:
            print(f"ARAD using device: {self.device}")

        self.model_: Optional[ARADAutoencoder] = None
        self.n_bins_: Optional[int] = None
        self.training_history_: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self.model_ is not None

    def fit(
        self,
        background_training: SpectralTimeSeries,
        validation_data: Optional[SpectralTimeSeries] = None,
        **kwargs,
    ) -> "ARADDetector":
        """
        Train the ARAD detector on background spectra.

        Parameters
        ----------
        background_training : SpectralTimeSeries
            Background spectra for training.
        validation_data : SpectralTimeSeries, optional
            Validation data. If ``None``, uses ``validation_split``.

        Returns
        -------
        self
        """
        counts = background_training.counts
        times = background_training.live_times
        if times is None or times.dtype == object or (
            times.dtype in [np.float32, np.float64] and np.any(np.isnan(times))
        ):
            times = background_training.real_times

        training_spectra = counts / times[:, np.newaxis]
        self.n_bins_ = training_spectra.shape[1]

        if self.n_bins_ % 32 != 0:
            raise ValueError(
                f"Number of bins ({self.n_bins_}) must be divisible by 32 "
                f"for 5 pooling layers. Consider rebinning."
            )

        if validation_data is None:
            n_train = int(len(training_spectra) * (1 - self.validation_split))
            indices = np.random.permutation(len(training_spectra))
            train_data = training_spectra[indices[:n_train]]
            val_data = training_spectra[indices[n_train:]]
        else:
            train_data = training_spectra
            val_counts = validation_data.counts
            val_times = validation_data.live_times
            if val_times is None or val_times.dtype == object or (
                val_times.dtype in [np.float32, np.float64] and np.any(np.isnan(val_times))
            ):
                val_times = validation_data.real_times
            val_data = val_counts / val_times[:, np.newaxis]

        if self.verbose:
            print(f"Training on {len(train_data)} spectra, validating on {len(val_data)}")
            print(f"Loss function: {self.loss_type.upper()}")

        if len(train_data) < 2:
            raise ValueError(
                "ARAD training requires at least 2 training spectra for BatchNorm. "
                "Reduce validation_split or provide more training data."
            )
        if len(val_data) == 0:
            raise ValueError(
                "Validation set is empty. Increase validation_split or provide more data."
            )

        train_tensor = torch.FloatTensor(train_data).to(self.device)
        val_tensor = torch.FloatTensor(val_data).to(self.device)

        train_loader = DataLoader(
            TensorDataset(train_tensor), batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor), batch_size=self.batch_size
        )

        if len(train_loader) == 0:
            raise ValueError(
                "No full training batches were created (likely batch_size > number of "
                "training spectra). Reduce batch_size or provide more training data."
            )

        self.model_ = ARADAutoencoder(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
            eps=0.1,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6
        )

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                reconstructed = self.model_(batch)
                loss = self._compute_loss(batch, reconstructed)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    reconstructed = self.model_(batch)
                    loss = self._compute_loss(batch, reconstructed)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )

            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.training_history_ = {
            'train_loss': train_losses,
            'val_loss': val_losses,
        }

        if self.verbose:
            print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

        return self

    def score_spectrum(self, spectrum: Spectrum) -> float:
        """
        Score a single spectrum (reconstruction error).

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to score.

        Returns
        -------
        float
            Anomaly score (JSD or Chi-squared depending on ``loss_type``).
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be fitted before scoring")

        counts = spectrum.counts
        time = (
            spectrum.live_time
            if spectrum.live_time is not None and not np.isnan(spectrum.live_time)
            else spectrum.real_time
        )
        spectrum_data = counts / time

        if len(spectrum_data) != self.n_bins_:
            raise ValueError(
                f"Spectrum has {len(spectrum_data)} bins, expected {self.n_bins_}"
            )

        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(x)

        if self.loss_type == 'jsd':
            x_norm = self._normalize_spectrum_tensor(x)
            reconstructed_norm = self._normalize_spectrum_tensor(reconstructed)
            return self._jsd_loss(x_norm, reconstructed_norm).item()

        # chi2
        eps = 1e-8
        max_val = torch.max(x)
        reconstructed_denorm = reconstructed * (max_val + eps)
        return self._chi2_loss(x, reconstructed_denorm).item()

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _normalize_spectrum_tensor(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        if x.dim() == 1:
            return x / (torch.max(x) + eps)
        return x / (torch.max(x, dim=1, keepdim=True).values + eps)

    def _jsd_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true = torch.clamp(y_true, min=1e-10, max=1.0)
        y_pred = torch.clamp(y_pred, min=1e-10, max=1.0)
        m = 0.5 * (y_true + y_pred)
        kld_pm = torch.sum(y_true * torch.log(y_true / m), dim=-1)
        kld_qm = torch.sum(y_pred * torch.log(y_pred / m), dim=-1)
        return torch.sqrt(0.5 * (kld_pm + kld_qm)).mean()

    def _chi2_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, min=eps)
        chi2 = torch.sum((y_true - y_pred) ** 2 / y_pred, dim=-1)
        return chi2.mean()

    def _compute_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'jsd':
            y_true_norm = self._normalize_spectrum_tensor(y_true)
            recon_loss = self._jsd_loss(y_true_norm, y_pred)
        elif self.loss_type == 'chi2':
            eps = 1e-8
            if y_true.dim() == 1:
                max_vals = torch.max(y_true)
            else:
                max_vals = torch.max(y_true, dim=1, keepdim=True).values
            y_pred_denorm = y_pred * (max_vals + eps)
            recon_loss = self._chi2_loss(y_true, y_pred_denorm)

        l1_norm = sum(param.abs().sum() for param in self.model_.parameters())
        return recon_loss + self.l1_lambda * l1_norm

    # ------------------------------------------------------------------
    # ARAD-specific public methods
    # ------------------------------------------------------------------

    def reconstruct(self, spectrum: Spectrum) -> np.ndarray:
        """
        Reconstruct a spectrum through the autoencoder.

        Parameters
        ----------
        spectrum : Spectrum
            Input spectrum.

        Returns
        -------
        np.ndarray
            Reconstructed spectrum (count rates).
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be fitted before reconstruction")

        counts = spectrum.counts
        time = (
            spectrum.live_time
            if spectrum.live_time is not None and not np.isnan(spectrum.live_time)
            else spectrum.real_time
        )
        spectrum_data = counts / time
        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(x)

        max_val = np.max(spectrum_data)
        return reconstructed.cpu().numpy().flatten() * max_val

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.

        Returns
        -------
        dict
            Training and validation loss history.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be fitted first")
        return self.training_history_

    def save(self, path: str) -> None:
        """
        Save trained model to file.

        Parameters
        ----------
        path : str
            Path to save model.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save unfitted model")
        save_dict = {
            'model_state': self.model_.state_dict(),
            'n_bins': self.n_bins_,
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'threshold': self.threshold,
            'loss_type': self.loss_type,
            'training_history': self.training_history_,
        }
        torch.save(save_dict, path)
        if self.verbose:
            print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load trained model from file.

        Parameters
        ----------
        path : str
            Path to saved model.
        """
        save_dict = torch.load(path, map_location=self.device)
        self.n_bins_ = save_dict['n_bins']
        self.latent_dim = save_dict['latent_dim']
        self.dropout = save_dict['dropout']
        self.threshold = save_dict['threshold']
        self.loss_type = save_dict.get('loss_type', 'jsd')
        self.training_history_ = save_dict.get('training_history', {})

        self.model_ = ARADAutoencoder(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model_.load_state_dict(save_dict['model_state'])

        if self.verbose:
            print(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    # Saliency / explainability
    # ------------------------------------------------------------------

    def compute_saliency_map(self, spectrum: Spectrum, method: str = 'gradient') -> np.ndarray:
        """
        Compute saliency map showing which energy bins contribute most
        to the anomaly score.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to analyze.
        method : str
            ``'gradient'`` (simple) or ``'integrated'`` (more robust).

        Returns
        -------
        np.ndarray
            Saliency map (same length as spectrum).
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be fitted before computing saliency")
        if method == 'gradient':
            return self._gradient_saliency(spectrum)
        elif method == 'integrated':
            return self._integrated_gradients(spectrum)
        raise ValueError(f"Unknown saliency method: {method}")

    def _spectrum_to_count_rate(self, spectrum: Spectrum) -> np.ndarray:
        time = (
            spectrum.live_time
            if spectrum.live_time is not None and not np.isnan(spectrum.live_time)
            else spectrum.real_time
        )
        return spectrum.counts / time

    def _gradient_saliency(self, spectrum: Spectrum) -> np.ndarray:
        spectrum_data = self._spectrum_to_count_rate(spectrum)
        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)
        x.requires_grad = True

        self.model_.eval()
        reconstructed = self.model_(x)

        x_norm = self._normalize_spectrum_tensor(x)
        loss = self._jsd_loss(x_norm, reconstructed)
        loss.backward()

        return np.abs(x.grad.cpu().numpy().squeeze())

    def _integrated_gradients(
        self,
        spectrum: Spectrum,
        n_steps: int = 50,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        spectrum_data = self._spectrum_to_count_rate(spectrum)
        if baseline is None:
            baseline = np.zeros_like(spectrum_data)

        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)
        baseline_tensor = torch.FloatTensor(baseline).unsqueeze(0).to(self.device)

        integrated_grads = np.zeros_like(spectrum_data)
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            interpolated = baseline_tensor + alpha * (x - baseline_tensor)
            interpolated.requires_grad = True

            self.model_.eval()
            reconstructed = self.model_(interpolated)
            interp_norm = self._normalize_spectrum_tensor(interpolated)
            loss = self._jsd_loss(interp_norm, reconstructed)
            loss.backward()

            integrated_grads += interpolated.grad.cpu().numpy().squeeze()

        integrated_grads = integrated_grads / n_steps
        integrated_grads = integrated_grads * (spectrum_data - baseline)
        return np.abs(integrated_grads)

    def plot_saliency(
        self,
        spectrum: Spectrum,
        method: str = 'gradient',
        figsize: tuple = (14, 8),
        show_reconstruction: bool = True,
    ):
        """
        Plot spectrum with saliency map overlay.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to visualize.
        method : str
            Saliency method (``'gradient'`` or ``'integrated'``).
        figsize : tuple
            Figure size.
        show_reconstruction : bool
            Whether to show reconstruction comparison.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            raise ImportError(
                "Matplotlib required for plotting. Install with: pip install matplotlib"
            )

        count_rate = self._spectrum_to_count_rate(spectrum)
        energy_centers = spectrum.energy_centers
        saliency = self.compute_saliency_map(spectrum, method=method)
        score = self.score_spectrum(spectrum)

        if show_reconstruction:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
            reconstructed = self.reconstruct(spectrum)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]

        ax1 = axes[0]
        ax1.plot(energy_centers, count_rate, 'k-', linewidth=2,
                 label='Original Spectrum', zorder=2)

        saliency_norm = saliency / (np.max(saliency) + 1e-10)
        for i in range(len(energy_centers) - 1):
            ax1.axvspan(
                energy_centers[i], energy_centers[i + 1],
                alpha=0.3 * saliency_norm[i], color='red', zorder=1,
            )

        ax1.set_yscale('log')
        ax1.set_ylabel(r'Count Rate (s$^{-1}$)', fontsize=12)
        ax1.set_title(
            f'Spectrum with Saliency Overlay (Score={score:.4f}, Method={method})',
            fontsize=13, fontweight='bold',
        )
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, zorder=0)

        if show_reconstruction:
            ax2 = axes[1]
            ax2.plot(energy_centers, count_rate, 'k-', linewidth=1.5,
                     label='Original', alpha=0.7)
            ax2.plot(energy_centers, reconstructed, 'r-', linewidth=1.5,
                     label='Reconstructed', alpha=0.7)
            ax2.set_yscale('log')
            ax2.set_xlabel('Energy (keV)', fontsize=12)
            ax2.set_ylabel(r'Count Rate (s$^{-1}$)', fontsize=12)
            ax2.set_title('Reconstruction Comparison', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel('Energy (keV)', fontsize=12)

        plt.tight_layout()
        return fig, axes
