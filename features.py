"""PNCC (Power-Normalized Cepstral Coefficients) feature extraction.

Implements the PNCC algorithm from:
    Kim & Stern, "Power-Normalized Cepstral Coefficients (PNCC) for
    Robust Speech Recognition," IEEE/ACM TASLP, 2016, 24(7):1315-1329.

Uses modern librosa (>=0.10) and scipy APIs.
Reference implementation (do not edit): PNCC/pncc.py
"""

from __future__ import annotations

import numpy as np
import librosa
import scipy.signal
from scipy.fft import dct


# ---------------------------------------------------------------------------
# PNCC processing stages — each operates on (T, n_mels) arrays
# ---------------------------------------------------------------------------


def _medium_time_power(power_spec: np.ndarray, M: int = 2) -> np.ndarray:
    """Compute medium-time power via a uniform (2M+1)-frame sliding window.

    Args:
        power_spec: Power spectrum in mel channels, shape (T, n_mels).
        M: Half-width of the smoothing window.

    Returns:
        Smoothed power array of shape (T, n_mels).
    """
    T = power_spec.shape[0]
    padded = np.pad(power_spec, [(M, M), (0, 0)], mode="constant")
    result = np.zeros_like(power_spec)
    weight = 1.0 / (2 * M + 1)
    for i in range(T):
        result[i] = weight * padded[i : i + 2 * M + 1].sum(axis=0)
    return result


def _asymmetric_lowpass(signal: np.ndarray, lm_a: float = 0.999, lm_b: float = 0.5) -> np.ndarray:
    """Asymmetric low-pass filtering to track the signal floor.

    Uses a fast attack (lm_b) when the signal falls and a slow decay (lm_a)
    when the signal rises, forming a lower-envelope tracker.

    Args:
        signal: Input array of shape (T, n_mels).
        lm_a: Smoothing coefficient when signal is rising (slow release).
        lm_b: Smoothing coefficient when signal is falling (fast attack).

    Returns:
        Floor array of shape (T, n_mels).
    """
    T = signal.shape[0]
    floor = np.zeros_like(signal)
    floor[0] = 0.9 * signal[0]
    for m in range(1, T):
        floor[m] = np.where(
            signal[m] >= floor[m - 1],
            lm_a * floor[m - 1] + (1 - lm_a) * signal[m],
            lm_b * floor[m - 1] + (1 - lm_b) * signal[m],
        )
    return floor


def _halfwave_rectify(signal: np.ndarray) -> np.ndarray:
    """Half-wave rectification: clamp negative values to zero.

    Args:
        signal: Input array of any shape.

    Returns:
        Array with the same shape, negative values replaced by zero.
    """
    return np.maximum(signal, 0.0)


def _temporal_masking(
    signal: np.ndarray,
    lam_t: float = 0.85,
    myu_t: float = 0.2,
) -> np.ndarray:
    """Apply online temporal masking to suppress transient noise drops.

    Tracks an online peak power and fills in masked frames with a
    fraction of the peak.

    Args:
        signal: Half-wave rectified signal, shape (T, n_mels).
        lam_t: Peak decay factor per frame.
        myu_t: Fraction of peak used when frame is below threshold.

    Returns:
        Temporally masked signal of shape (T, n_mels).
    """
    T = signal.shape[0]
    masked = np.zeros_like(signal)
    peak = np.zeros_like(signal)
    masked[0] = signal[0]
    peak[0] = signal[0]
    for m in range(1, T):
        peak[m] = np.maximum(lam_t * peak[m - 1], signal[m])
        masked[m] = np.where(
            signal[m] >= lam_t * peak[m - 1],
            signal[m],
            myu_t * peak[m - 1],
        )
    return masked


def _excitation_switch(
    temporal_masked: np.ndarray,
    floor: np.ndarray,
    lower_envelope: np.ndarray,
    medium_time_power: np.ndarray,
    c: float = 2.0,
) -> np.ndarray:
    """Select temporal-masked or floor values based on excitation test.

    Frames where the medium-time power exceeds c * lower_envelope are
    considered voiced (excitation); others fall back to the floor signal.

    Args:
        temporal_masked: Temporally masked signal, shape (T, n_mels).
        floor: Floor signal from second asymmetric low-pass, shape (T, n_mels).
        lower_envelope: First floor estimate (from medium-time power), shape (T, n_mels).
        medium_time_power: Medium-time power, shape (T, n_mels).
        c: Excitation threshold multiplier.

    Returns:
        Combined signal of shape (T, n_mels).
    """
    return np.where(medium_time_power >= c * lower_envelope, temporal_masked, floor)


def _spectral_weight_smoothing(
    signal: np.ndarray,
    medium_time_power: np.ndarray,
    N: int = 4,
) -> np.ndarray:
    """Spectral weight smoothing over a frequency neighbourhood of half-width N.

    Computes a gain weight at each (time, frequency) bin as the mean ratio
    of signal to medium-time power over the 2N neighbouring channels.

    Args:
        signal: Excitation/non-excitation output, shape (T, n_mels).
        medium_time_power: Medium-time power, shape (T, n_mels).
        N: Half-width of the smoothing neighbourhood.

    Returns:
        Spectral weights of shape (T, n_mels).
    """
    T, L = signal.shape
    smoothed = np.zeros_like(signal)
    safe_mtp = np.where(medium_time_power == 0, 1e-30, medium_time_power)
    for m in range(T):
        for l in range(L):
            l1 = max(l - N, 1)
            l2 = min(l + N, L)  # exclusive upper bound — matches range(l1, l2) in reference
            norm = 1.0 / (l2 - l1 + 1)  # reference uses l2-l1+1 even though range has l2-l1 elements
            smoothed[m, l] = norm * (signal[m, l1:l2] / safe_mtp[m, l1:l2]).sum()
    return smoothed


def _mean_power_normalize(
    transfer: np.ndarray,
    lam_myu: float = 0.999,
    k: float = 1.0,
) -> np.ndarray:
    """Online mean power normalization to achieve loudness invariance.

    Tracks a running mean power and divides the transfer function by it,
    making the output independent of absolute signal level.

    Args:
        transfer: Time-frequency transfer function, shape (T, n_mels).
        lam_myu: Exponential smoothing factor for the running mean.
        k: Output scaling constant.

    Returns:
        Normalized transfer function of shape (T, n_mels).
    """
    T, L = transfer.shape
    myu = np.zeros(T)
    myu[0] = 1e-4
    for m in range(1, T):
        myu[m] = lam_myu * myu[m - 1] + (1 - lam_myu) / L * transfer[m, : L - 1].sum()
    safe_myu = np.where(myu == 0, 1e-30, myu)
    return k * transfer / safe_myu[:, None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_pncc(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 256,
    win_length: int = 200,
    hop_length: int = 80,
    n_mels: int = 128,
    n_pncc: int = 13,
    weight_N: int = 4,
    power: int = 2,
) -> np.ndarray:
    """Extract PNCC features from a mono audio signal.

    Implements the full PNCC pipeline: pre-emphasis, mel power spectrum,
    medium-time processing, noise suppression, and DCT.

    Default parameters are matched to the MFCC baseline:
        win_length=200 (25 ms at 8 kHz), hop_length=80 (10 ms at 8 kHz),
        n_fft=256, n_pncc=13.

    Args:
        audio: 1-D mono PCM array (float32 or float64).
        sr: Sample rate in Hz (expected: 8000).
        n_fft: FFT size.
        win_length: Analysis window length in samples (rectangular window).
        hop_length: Frame hop in samples.
        n_mels: Number of mel filterbank channels.
        n_pncc: Number of cepstral coefficients to return.
        weight_N: Half-width of spectral weight smoothing window.
        power: Exponent applied to both the spectrum and mel filter.

    Returns:
        np.ndarray of shape (n_pncc, T), dtype float32.
        Matches the (F, T) convention used by the MFCC baseline.
    """
    # 1. Pre-emphasis filter
    audio = scipy.signal.lfilter([1.0, -0.97], 1.0, audio.astype(np.float64))

    # 2. Power spectrum via STFT (rectangular window, no centering)
    stft_mag = (
        np.abs(
            librosa.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=np.ones(win_length, dtype=np.float32),
                center=False,
            )
        )
        ** power
    )  # shape: (n_fft//2+1, T)

    # 3. Mel filterbank — shape (n_mels, n_fft//2+1)
    mel_fb = np.abs(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)) ** power

    # 4. Map to mel power channels — shape (T, n_mels)
    power_spec = stft_mag.T @ mel_fb.T

    # 5. Medium-time power
    mtp = _medium_time_power(power_spec)

    # 6. Lower envelope via asymmetric low-pass filtering
    lower_env = _asymmetric_lowpass(mtp)

    # 7. Subtract lower envelope and half-wave rectify
    rect = _halfwave_rectify(mtp - lower_env)

    # 8. Track floor of rectified signal
    floor = _asymmetric_lowpass(rect)

    # 9. Temporal masking
    tm = _temporal_masking(rect)

    # 10. Excitation / non-excitation switch
    combined = _excitation_switch(tm, floor, lower_env, mtp)

    # 11. Spectral weight smoothing
    weights = _spectral_weight_smoothing(combined, mtp, N=weight_N)

    # 12. Apply weights and normalize
    tf_norm = power_spec * weights
    mean_norm = _mean_power_normalize(tf_norm)

    # 13. Power-law nonlinearity
    nonlin = np.abs(mean_norm) ** (1.0 / 15.0)

    # 14. DCT-II, keep first n_pncc coefficients — shape (T, n_pncc)
    cepstra = dct(nonlin, type=2, norm="ortho", axis=1)[:, :n_pncc]

    # 15. Transpose to (n_pncc, T) — matches baseline (F, T) convention
    return cepstra.T.astype(np.float32)
