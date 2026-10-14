"""Pitch (F0) estimation used to check that cloning follows the reference voice."""
import numpy as np
from scipy.signal import resample_poly


def f0_track(audio: np.ndarray, sr: int) -> np.ndarray:
    """F0 of voiced 40 ms frames (10 ms hop) by normalized autocorrelation at 16 kHz."""
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr)
        sr = 16000
    n, hop = 640, 160
    lo, hi = sr // 400, sr // 60
    frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    voiced_energy = energy > 0.1 * np.percentile(energy, 95)
    spec = np.fft.rfft(frames, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    lags = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(ac)), lags]
    keep = voiced_energy & (strength > 0.6)
    return sr / lags[keep]


def median_f0(audio, sr) -> float:
    f0 = f0_track(audio, sr)
    return float(np.median(f0)) if len(f0) else float("nan")
