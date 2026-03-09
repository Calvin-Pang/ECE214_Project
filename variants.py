"""Feature variant registry for PNCC experiment comparison.

Each variant is a callable with signature:
    (audio: np.ndarray, sr: int) -> np.ndarray  # shape (F, T)

All variants use the same framing as the 8 kHz baseline so frame counts align
when stacking features across sources.

Usage:
    from variants import VARIANTS
    feat = VARIANTS["pncc_39"](audio, sr)
"""

from __future__ import annotations

from typing import Callable

import librosa
import numpy as np

from features import extract_pncc

# ---------------------------------------------------------------------------
# Shared framing constants — matched to 8 kHz baseline
# ---------------------------------------------------------------------------

_HOP = 80    # 10 ms at 8 kHz
_WIN = 200   # 25 ms at 8 kHz
_N_FFT = 256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_deltas(feat: np.ndarray) -> np.ndarray:
    """Append delta and delta-delta rows to a (F, T) array -> (3F, T)."""
    d1 = librosa.feature.delta(feat, order=1)
    d2 = librosa.feature.delta(feat, order=2)
    return np.concatenate([feat, d1, d2], axis=0)


def _log_energy(audio: np.ndarray) -> np.ndarray:
    """Log RMS frame energy, shape (1, T), float32."""
    rms = librosa.feature.rms(y=audio, frame_length=_WIN, hop_length=_HOP, center=False)
    return np.log(rms + 1e-8).astype(np.float32)


def _rms(audio: np.ndarray) -> np.ndarray:
    """Raw RMS energy per frame, shape (1, T), float32."""
    return librosa.feature.rms(
        y=audio, frame_length=_WIN, hop_length=_HOP, center=False
    ).astype(np.float32)


def _zcr(audio: np.ndarray) -> np.ndarray:
    """Zero-crossing rate per frame, shape (1, T), float32."""
    return librosa.feature.zero_crossing_rate(
        audio, frame_length=_WIN, hop_length=_HOP, center=False
    ).astype(np.float32)


def _trim(*arrs: np.ndarray) -> list[np.ndarray]:
    """Clip all arrays to the minimum T (axis=1) before concatenation."""
    T = min(a.shape[1] for a in arrs)
    return [a[:, :T] for a in arrs]


# ---------------------------------------------------------------------------
# Variant functions
# ---------------------------------------------------------------------------


def _pncc(audio: np.ndarray, sr: int) -> np.ndarray:
    """Bug-fixed PNCC, 13 coefficients. Shape (13, T)."""
    return extract_pncc(
        audio, sr,
        n_fft=_N_FFT, win_length=_WIN, hop_length=_HOP, n_pncc=13,
    )


def _pncc_39(audio: np.ndarray, sr: int) -> np.ndarray:
    """PNCC + delta + delta-delta. Shape (39, T)."""
    return _with_deltas(_pncc(audio, sr))


def _pncc_42(audio: np.ndarray, sr: int) -> np.ndarray:
    """PNCC + delta + delta-delta + log-energy + RMS + ZCR. Shape (42, T)."""
    base = _pncc_39(audio, sr)    # (39, T)
    log_e = _log_energy(audio)    # (1,  T)
    rms   = _rms(audio)           # (1,  T)
    zcr   = _zcr(audio)           # (1,  T)
    return np.concatenate(_trim(base, log_e, rms, zcr), axis=0)


def _mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """MFCC, 13 coefficients — direct comparison anchor. Shape (13, T).

    Uses center=True (librosa default) to match the baseline notebook exactly.
    _trim() handles frame-count alignment when stacking with PNCC (center=False).
    """
    return librosa.feature.mfcc(
        y=audio.astype(np.float32), sr=sr,
        n_mfcc=13, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN,
    )


def _mfcc_39(audio: np.ndarray, sr: int) -> np.ndarray:
    """MFCC + delta + delta-delta — matches teammate's best feature set. Shape (39, T)."""
    return _with_deltas(_mfcc(audio, sr))


def _pncc_mfcc_78(audio: np.ndarray, sr: int) -> np.ndarray:
    """PNCC_39 stacked with MFCC_39. Shape (78, T)."""
    p = _pncc_39(audio, sr)   # (39, T)
    m = _mfcc_39(audio, sr)   # (39, T)
    return np.concatenate(_trim(p, m), axis=0)


def _pncc_n40(audio: np.ndarray, sr: int) -> np.ndarray:
    """PNCC with 40 coefficients + delta + delta-delta. Shape (120, T).

    More coefficients capture finer spectral detail while retaining PNCC's
    noise suppression — no noise-sensitive features added.
    """
    base = extract_pncc(audio, sr, n_fft=_N_FFT, win_length=_WIN, hop_length=_HOP, n_pncc=40)
    return _with_deltas(base)


def _pncc_contrast(audio: np.ndarray, sr: int) -> np.ndarray:
    """PNCC_39 + spectral contrast (with deltas). Shape (60, T).

    Spectral contrast measures peak-vs-valley energy per subband — robust to
    additive noise because babble raises the noise floor (valley) but preserves
    the relative peaks of speech harmonics.
    """
    pncc = _pncc_39(audio, sr)    # (39, T)

    # 7 contrast bands (6 subbands + 1 overall), with deltas -> (21, T)
    contrast = librosa.feature.spectral_contrast(
        y=audio.astype(np.float32), sr=sr,
        n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN, center=False,
    ).astype(np.float32)           # (7, T)
    contrast_d = _with_deltas(contrast)  # (21, T)

    return np.concatenate(_trim(pncc, contrast_d), axis=0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VARIANTS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "pncc":          _pncc,          # 13  — bug-fixed PNCC baseline
    "pncc_39":       _pncc_39,       # 39  — PNCC + Δ + ΔΔ
    "pncc_42":       _pncc_42,       # 42  — pncc_39 + log-energy + RMS + ZCR
    "mfcc":          _mfcc,          # 13  — MFCC anchor (matches teammate baseline)
    "mfcc_39":       _mfcc_39,       # 39  — MFCC + Δ + ΔΔ (matches teammate's best)
    "pncc_mfcc_78":  _pncc_mfcc_78,  # 78  — stacked PNCC_39 + MFCC_39
    "pncc_n40":      _pncc_n40,      # 120 — PNCC 40 coeffs + Δ + ΔΔ
    "pncc_contrast": _pncc_contrast, # 60  — PNCC_39 + spectral contrast + Δ + ΔΔ
}
