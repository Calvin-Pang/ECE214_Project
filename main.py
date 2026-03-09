"""Noise-robust spoken digit recognition — PNCC feature experiment.

Ported from baseline.ipynb. The only changes vs. the baseline are:
  1. extract_feature() uses PNCC instead of MFCC.
  2. Z-score normalisation (fit on training set) is applied to all feature sets.

The LSTM model, training loop, optimizer, and all hyperparameters are
identical to the baseline notebook and must not be changed for the main results.

Usage:
    DATA_ROOT=/path/to/M214_project_data uv run main.py
"""

import copy
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn import metrics
from torch.utils.data import DataLoader

from features import extract_pncc

# Must be set before any CUDA operations for deterministic kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 0
BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 3e-4

_DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
TRAIN_DIR = os.path.join(_DATA_ROOT, "train_clean")
TEST_CLEAN_DIR = os.path.join(_DATA_ROOT, "test_clean")
TEST_NOISY_5DB_DIR = os.path.join(_DATA_ROOT, "test_snr_5db_babble")
TEST_NOISY_10DB_DIR = os.path.join(_DATA_ROOT, "test_snr_10db_babble")

# PNCC / framing parameters — matched to the MFCC baseline
N_PNCC = 13
WIN_LENGTH = 200   # 25 ms at 8 kHz
HOP_LENGTH = 80    # 10 ms at 8 kHz
N_FFT = 256


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio(audio_file: str) -> tuple[np.ndarray, int]:
    """Load a wav file and return a mono float array plus its sample rate."""
    audio, fs = torchaudio.load(audio_file)
    audio = audio.numpy().reshape(-1)
    return audio, int(fs)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_feature(audio: np.ndarray, fs: int) -> np.ndarray:
    """Extract PNCC features from raw audio.

    Returns a 2-D array of shape (N_PNCC, T) where T is the number of frames.
    Drop-in replacement for the MFCC baseline's extract_feature().
    """
    return extract_pncc(
        audio,
        sr=fs,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_pncc=N_PNCC,
    )


def extract_feature_from_file(audio_file: str) -> np.ndarray:
    """Load a wav file and return its PNCC feature matrix."""
    audio, fs = load_audio(audio_file)
    return extract_feature(audio, fs)


# ---------------------------------------------------------------------------
# Z-score normalisation (fit on training set, apply to all splits)
# ---------------------------------------------------------------------------


def znorm_fit(feats: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from a list of (F, T) feature arrays.

    Statistics are derived exclusively from the training set to avoid
    data leakage into the evaluation splits.

    Args:
        feats: List of (F, T) numpy arrays.

    Returns:
        mean: Shape (F, 1), per-channel mean across all training frames.
        std:  Shape (F, 1), per-channel std  across all training frames.
    """
    all_frames = np.concatenate(feats, axis=1)  # (F, total_frames)
    mean = all_frames.mean(axis=1, keepdims=True)
    std = all_frames.std(axis=1, keepdims=True)
    return mean, std


def znorm_apply(
    feats: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[np.ndarray]:
    """Apply pre-computed z-norm statistics to a list of (F, T) arrays.

    Args:
        feats: List of (F, T) numpy arrays.
        mean:  Shape (F, 1) mean from znorm_fit().
        std:   Shape (F, 1) std  from znorm_fit().

    Returns:
        List of normalised (F, T) arrays with the same dtypes.
    """
    return [(f - mean) / (std + 1e-8) for f in feats]


# ---------------------------------------------------------------------------
# Dataset & DataLoader
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
    B = len(xs)
    F = xs[0].shape[0]
    T_max = max(lens)
    xb = torch.zeros(B, 1, F, T_max, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        xb[i, 0, :, : x.shape[1]] = x
    return xb, torch.tensor(ys, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_label(file_name: str) -> int:
    """Extract integer digit label from filename (format: {label}_*.wav)."""
    base = os.path.splitext(os.path.basename(file_name))[0]
    return int(base.split("_")[0])


def load_dir(data_dir: str, desc: str = "Loading") -> tuple[list[np.ndarray], list[int]]:
    """Load all wav files from a directory and extract PNCC features.

    Args:
        data_dir: Path to directory containing *.wav files.
        desc: Label used in the progress message.

    Returns:
        feats:  List of (F, T) numpy arrays, one per file.
        labels: Corresponding integer digit labels.
    """
    files = sorted(glob(os.path.join(data_dir, "*.wav")))
    if not files:
        print(f"  No wav files found in {data_dir}")
        return [], []
    feats, labels = [], []
    for wav in files:
        feats.append(extract_feature_from_file(wav))
        labels.append(get_label(wav))
    print(f"  {desc}: {len(feats)} files loaded")
    return feats, labels


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
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 10),
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
        device = out.device

        mask = torch.arange(T_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        out_sum = (out * mask_f).sum(dim=1)           # (B, D)
        denom = mask_f.sum(dim=1).clamp(min=1.0)      # (B, 1)
        mean = out_sum / denom                         # (B, D)

        return self.classifier(mean)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, plot_cm=False, class_names=None, title=None, save_path=None):
    """Evaluate the model and optionally plot a confusion matrix.

    Returns:
        acc (float) when plot_cm is False.
        (acc, cm) when plot_cm is True.
    """
    model.eval()
    if loader is None:
        return (0.0, None) if plot_cm else 0.0

    all_preds, all_labels = [], []
    correct, total = 0, 0

    for xb, yb, lengths in loader:
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
        logits = model(xb, lengths)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        if plot_cm:
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

    acc = correct / total if total > 0 else 0.0

    if not plot_cm:
        return acc

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    cm = metrics.confusion_matrix(y_true, y_pred)

    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(values_format="d")
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return acc, cm


# ---------------------------------------------------------------------------
# Main training / evaluation
# ---------------------------------------------------------------------------


def main() -> None:
    set_seed(SEED)

    # --- Load features ---
    train_feat, train_label = load_dir(TRAIN_DIR, desc="Train")
    test_feat, test_label = load_dir(TEST_CLEAN_DIR, desc="Test clean")
    noisy5_feat, noisy5_label = load_dir(TEST_NOISY_5DB_DIR, desc="Test noisy 5dB")
    noisy10_feat, noisy10_label = load_dir(TEST_NOISY_10DB_DIR, desc="Test noisy 10dB")

    feat_dim = train_feat[0].shape[0]
    print(f"\nFeature dim: {feat_dim}")
    print(
        f"Train: {len(train_feat)}  |  Test clean: {len(test_feat)}  "
        f"|  Test noisy 5dB: {len(noisy5_feat)}  |  Test noisy 10dB: {len(noisy10_feat)}"
    )
    for name, flist in [
        ("Train", train_feat),
        ("Test clean", test_feat),
        ("Noisy 5dB", noisy5_feat),
        ("Noisy 10dB", noisy10_feat),
    ]:
        if flist:
            lengths = [f.shape[1] for f in flist]
            print(f"  {name:10s} frames: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # --- Z-score normalisation (fit on train, apply to all splits) ---
    mean, std = znorm_fit(train_feat)
    train_feat = znorm_apply(train_feat, mean, std)
    test_feat = znorm_apply(test_feat, mean, std)
    noisy5_feat = znorm_apply(noisy5_feat, mean, std)
    noisy10_feat = znorm_apply(noisy10_feat, mean, std)

    # --- Build label arrays ---
    y_train = np.array(train_label, dtype=np.int64)
    y_test = np.array(test_label, dtype=np.int64)
    y_noisy5 = np.array(noisy5_label, dtype=np.int64)
    y_noisy10 = np.array(noisy10_label, dtype=np.int64)

    # --- DataLoaders ---
    loader_g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        FeatureDataset(train_feat, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pad,
        generator=loader_g,
    )
    test_loader = DataLoader(
        FeatureDataset(test_feat, y_test),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_pad,
    )
    noisy5_loader = (
        DataLoader(
            FeatureDataset(noisy5_feat, y_noisy5),
            batch_size=16,
            shuffle=False,
            collate_fn=collate_pad,
        )
        if noisy5_feat
        else None
    )
    noisy10_loader = (
        DataLoader(
            FeatureDataset(noisy10_feat, y_noisy10),
            batch_size=16,
            shuffle=False,
            collate_fn=collate_pad,
        )
        if noisy10_feat
        else None
    )

    # --- Model, loss, optimiser — identical to baseline ---
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleLSTM(input_size=feat_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    best_clean, best_clean_ep = 0.0, -1
    best_5db, best_5db_ep = 0.0, -1
    best_10db, best_10db_ep = 0.0, -1

    # Save the checkpoint with the best 10 dB accuracy.
    saved_checkpoint = None

    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        total_loss = 0.0
        for xb, yb, lengths in train_loader:
            xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = criterion(net(xb, lengths), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        clean_acc = evaluate(net, test_loader, device)
        acc_5db = evaluate(net, noisy5_loader, device)
        acc_10db = evaluate(net, noisy10_loader, device)

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
            saved_checkpoint = copy.deepcopy(net.state_dict())

    # --- Final evaluation with best checkpoint ---
    net = SimpleLSTM(input_size=feat_dim).to(device)
    net.load_state_dict(saved_checkpoint)

    clean_acc, cm_clean = evaluate(
        net, test_loader, device,
        plot_cm=True, class_names=list(range(10)), title="Confusion Matrix — Clean",
    )
    acc_5db, cm_5db = evaluate(
        net, noisy5_loader, device,
        plot_cm=True, class_names=list(range(10)), title="Confusion Matrix — Noisy 5 dB",
    )
    acc_10db, cm_10db = evaluate(
        net, noisy10_loader, device,
        plot_cm=True, class_names=list(range(10)), title="Confusion Matrix — Noisy 10 dB",
    )

    print(f"\nBest checkpoint loaded (epoch {best_10db_ep})")
    print(f"Clean accuracy : {clean_acc:.4f}")
    print(f"5dB  accuracy  : {acc_5db:.4f}")
    print(f"10dB accuracy  : {acc_10db:.4f}")


if __name__ == "__main__":
    main()
