"""Noise-robust spoken digit recognition — PNCC feature experiment.

Ported from baseline.ipynb. Changes vs. the baseline:
  1. extract_feature() uses PNCC instead of MFCC.
  2. Z-score normalisation (fit on training set) is applied to all feature sets.

The LSTM model, training loop, optimizer, LR, and batch size are identical to
the baseline notebook and must not be changed for the main results.

Usage:
    uv run main.py --help
    uv run main.py --gpu-id 1
    uv run main.py --gpu-id 0 --data-root /mnt/data
"""

import copy
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import typer
from sklearn import metrics
from torch.utils.data import DataLoader

from features import extract_pncc

# Must be set before any CUDA operations for deterministic kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """All hyperparameters and paths for one experiment run.

    lr and batch_size are fixed to match the baseline and must not be changed
    for the main results (project requirement).
    """

    data_root: Path = field(default_factory=lambda: Path("data"))
    gpu_id: int = 0
    seed: int = 0
    num_epochs: int = 40

    # Fixed baseline hyperparameters — do not expose as CLI options
    batch_size: int = 32
    lr: float = 3e-4

    # PNCC framing — matched to MFCC baseline (8 kHz audio)
    n_pncc: int = 13
    win_length: int = 200  # 25 ms at 8 kHz
    hop_length: int = 80   # 10 ms at 8 kHz
    n_fft: int = 256

    @property
    def train_dir(self) -> Path:
        return self.data_root / "train_clean"

    @property
    def test_clean_dir(self) -> Path:
        return self.data_root / "test_clean"

    @property
    def test_noisy_5db_dir(self) -> Path:
        return self.data_root / "test_snr_5db_babble"

    @property
    def test_noisy_10db_dir(self) -> Path:
        return self.data_root / "test_snr_10db_babble"


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a wav file and return a mono float array plus its sample rate."""
    audio, fs = torchaudio.load(path)
    return audio.numpy().reshape(-1), int(fs)


def extract_feature(audio: np.ndarray, fs: int, cfg: Config) -> np.ndarray:
    """Extract PNCC features from raw audio; returns shape (n_pncc, T)."""
    return extract_pncc(
        audio,
        sr=fs,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_pncc=cfg.n_pncc,
    )


def get_label(path: Path) -> int:
    """Extract integer digit label from filename (format: {label}_*.wav)."""
    return int(path.stem.split("_")[0])


# ---------------------------------------------------------------------------
# Dataset & DataLoader — unchanged from baseline
# ---------------------------------------------------------------------------


class FeatureDataset(torch.utils.data.Dataset):
    """Wraps a list of (F, T) feature arrays and integer labels."""

    def __init__(self, X_list: list[np.ndarray], y: np.ndarray) -> None:
        self.X = X_list
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (F, T)
        return x, int(self.y[idx]), x.shape[1]


def collate_pad(batch):
    """Pad variable-length sequences to the longest in the batch."""
    xs, ys, lens = zip(*batch)
    B, F, T_max = len(xs), xs[0].shape[0], max(lens)
    xb = torch.zeros(B, 1, F, T_max, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        xb[i, 0, :, : x.shape[1]] = x
    return xb, torch.tensor(ys, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model — identical to baseline, do not modify
# ---------------------------------------------------------------------------


class SimpleLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 10),
        )

    def forward(self, x, lengths):
        # x: (B, 1, F, T) -> (B, T, F)
        x = x.squeeze(1).permute(0, 2, 1).contiguous()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T_max, 256)

        # Mean pooling over valid frames (ignore padding)
        B, T_max, D = out.shape
        mask = torch.arange(T_max, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        out_sum = (out * mask_f).sum(dim=1)           # (B, D)
        denom = mask_f.sum(dim=1).clamp(min=1.0)      # (B, 1)
        return self.classifier(out_sum / denom)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Encapsulates the full training and evaluation pipeline."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._set_seed(cfg.seed)
        self.device = (
            torch.device(f"cuda:{cfg.gpu_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"Device: {self.device}")

        self._znorm_mean: np.ndarray | None = None
        self._znorm_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _load_split(self, data_dir: Path, desc: str) -> tuple[list[np.ndarray], list[int]]:
        """Load all wav files in data_dir and extract PNCC features."""
        files = sorted(data_dir.glob("*.wav"))
        if not files:
            print(f"  No wav files found in {data_dir}")
            return [], []
        feats, labels = [], []
        for wav in files:
            audio, fs = load_audio(wav)
            feats.append(extract_feature(audio, fs, self.cfg))
            labels.append(get_label(wav))
        print(f"  {desc}: {len(feats)} files loaded")
        return feats, labels

    def _znorm_fit(self, feats: list[np.ndarray]) -> None:
        """Fit z-norm statistics from training features (stored on self)."""
        all_frames = np.concatenate(feats, axis=1)  # (F, total_T)
        self._znorm_mean = all_frames.mean(axis=1, keepdims=True)
        self._znorm_std = all_frames.std(axis=1, keepdims=True)

    def _znorm_apply(self, feats: list[np.ndarray]) -> list[np.ndarray]:
        """Apply stored z-norm statistics to a list of (F, T) arrays."""
        return [(f - self._znorm_mean) / (self._znorm_std + 1e-8) for f in feats]

    def _make_loader(
        self,
        feats: list[np.ndarray],
        labels: list[int],
        *,
        shuffle: bool,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> DataLoader | None:
        """Build a DataLoader from feature/label lists, or None if empty."""
        if not feats:
            return None
        return DataLoader(
            FeatureDataset(feats, np.array(labels, dtype=np.int64)),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_pad,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """Run one training epoch; returns average loss."""
        self.net.train()
        total_loss = 0.0
        for xb, yb, lengths in self.train_loader:
            xb, yb, lengths = xb.to(self.device), yb.to(self.device), lengths.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.net(xb, lengths), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader | None) -> float:
        """Return accuracy on a loader, or 0.0 if loader is None."""
        if loader is None:
            return 0.0
        self.net.eval()
        correct = total = 0
        for xb, yb, lengths in loader:
            xb, yb, lengths = xb.to(self.device), yb.to(self.device), lengths.to(self.device)
            preds = self.net(xb, lengths).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def _final_eval(
        self, loader: DataLoader | None, title: str
    ) -> tuple[float, np.ndarray | None]:
        """Evaluate and plot a confusion matrix; returns (acc, cm)."""
        if loader is None:
            return 0.0, None
        self.net.eval()
        all_preds, all_labels = [], []
        correct = total = 0
        for xb, yb, lengths in loader:
            xb, yb, lengths = xb.to(self.device), yb.to(self.device), lengths.to(self.device)
            preds = self.net(xb, lengths).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

        acc = correct / total if total > 0 else 0.0
        cm = metrics.confusion_matrix(
            np.concatenate(all_labels), np.concatenate(all_preds)
        )
        metrics.ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
            values_format="d"
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()
        return acc, cm

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full pipeline: load → normalise → train → evaluate."""
        cfg = self.cfg

        # --- Load all splits ---
        train_feat, train_label = self._load_split(cfg.train_dir, "Train")
        test_feat, test_label = self._load_split(cfg.test_clean_dir, "Test clean")
        noisy5_feat, noisy5_label = self._load_split(cfg.test_noisy_5db_dir, "Test noisy 5dB")
        noisy10_feat, noisy10_label = self._load_split(cfg.test_noisy_10db_dir, "Test noisy 10dB")

        feat_dim = train_feat[0].shape[0]
        print(f"\nFeature dim: {feat_dim}")
        print(
            f"Train: {len(train_feat)}  |  Test clean: {len(test_feat)}  "
            f"|  Test noisy 5dB: {len(noisy5_feat)}  |  Test noisy 10dB: {len(noisy10_feat)}"
        )
        for name, flist in [
            ("Train", train_feat), ("Test clean", test_feat),
            ("Noisy 5dB", noisy5_feat), ("Noisy 10dB", noisy10_feat),
        ]:
            if flist:
                lengths = [f.shape[1] for f in flist]
                print(f"  {name:10s} frames: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

        # --- Z-norm (fit on train, apply to all) ---
        self._znorm_fit(train_feat)
        train_feat = self._znorm_apply(train_feat)
        test_feat = self._znorm_apply(test_feat)
        noisy5_feat = self._znorm_apply(noisy5_feat)
        noisy10_feat = self._znorm_apply(noisy10_feat)

        # --- DataLoaders ---
        loader_g = torch.Generator().manual_seed(cfg.seed)
        self.train_loader = self._make_loader(train_feat, train_label, shuffle=True, batch_size=cfg.batch_size, generator=loader_g)
        self.test_loader = self._make_loader(test_feat, test_label, shuffle=False, batch_size=16)
        self.noisy5_loader = self._make_loader(noisy5_feat, noisy5_label, shuffle=False, batch_size=16)
        self.noisy10_loader = self._make_loader(noisy10_feat, noisy10_label, shuffle=False, batch_size=16)

        # --- Model, loss, optimiser — identical to baseline ---
        self._set_seed(cfg.seed)
        self.net = SimpleLSTM(input_size=feat_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

        best_clean, best_clean_ep = 0.0, -1
        best_5db, best_5db_ep = 0.0, -1
        best_10db, best_10db_ep = 0.0, -1
        saved_checkpoint = None  # saved at best 10 dB accuracy

        for epoch in range(1, cfg.num_epochs + 1):
            avg_loss = self._train_epoch()
            clean_acc = self._evaluate(self.test_loader)
            acc_5db = self._evaluate(self.noisy5_loader)
            acc_10db = self._evaluate(self.noisy10_loader)

            print(
                f"Epoch {epoch:02d}  loss={avg_loss:.4f}  "
                f"clean={clean_acc:.4f}  5dB={acc_5db:.4f}  10dB={acc_10db:.4f}"
            )

            if clean_acc > best_clean:
                best_clean, best_clean_ep = clean_acc, epoch
            if acc_5db > best_5db:
                best_5db, best_5db_ep = acc_5db, epoch
            if acc_10db > best_10db:
                best_10db, best_10db_ep = acc_10db, epoch
                saved_checkpoint = copy.deepcopy(self.net.state_dict())

        # --- Final evaluation with best checkpoint ---
        self.net = SimpleLSTM(input_size=feat_dim).to(self.device)
        self.net.load_state_dict(saved_checkpoint)

        clean_acc, _ = self._final_eval(self.test_loader, "Confusion Matrix — Clean")
        acc_5db, _ = self._final_eval(self.noisy5_loader, "Confusion Matrix — Noisy 5 dB")
        acc_10db, _ = self._final_eval(self.noisy10_loader, "Confusion Matrix — Noisy 10 dB")

        print(f"\nBest checkpoint loaded (epoch {best_10db_ep})")
        print(f"Clean accuracy : {clean_acc:.4f}")
        print(f"5dB  accuracy  : {acc_5db:.4f}")
        print(f"10dB accuracy  : {acc_10db:.4f}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Seed all RNGs for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_root: Path = typer.Option(Path("data"), help="Root directory of M214_project_data"),
    gpu_id: int = typer.Option(0, help="CUDA device index (ignored if no GPU available)"),
    seed: int = typer.Option(0, help="Random seed"),
    epochs: int = typer.Option(40, help="Number of training epochs"),
) -> None:
    cfg = Config(data_root=data_root, gpu_id=gpu_id, seed=seed, num_epochs=epochs)
    Trainer(cfg).run()


if __name__ == "__main__":
    app()
