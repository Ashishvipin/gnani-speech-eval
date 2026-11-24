"""
Audio utility functions for the evaluation pipeline.

Handles audio loading, format conversion, noise injection,
duration measurement, and signal analysis.
"""

import io
import wave
import struct
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of an audio file in seconds.
    Supports WAV files directly; for others, tries via scipy or falls back
    to file-size estimation.
    """
    path = Path(audio_path)

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate

    # Try scipy for other formats
    try:
        from scipy.io import wavfile
        rate, data = wavfile.read(str(path))
        return len(data) / rate
    except Exception:
        pass

    # Fallback: rough estimate from file size (16kHz, 16-bit mono)
    file_size = path.stat().st_size
    return file_size / (16000 * 2)  # approximate


def load_wav(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file and return (samples, sample_rate).
    Samples are normalised to [-1.0, 1.0] float range.
    """
    with wave.open(str(audio_path), "r") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Parse raw bytes to numpy array
    if sampwidth == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        max_val = 2147483648.0
    else:
        dtype = np.int16
        max_val = 32768.0

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64) / max_val

    # Convert stereo to mono by averaging channels
    if n_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    return samples, framerate


def save_wav(
    samples: np.ndarray,
    sample_rate: int,
    output_path: str,
    sample_width: int = 2,
) -> str:
    """Save float samples [-1, 1] to a WAV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clip and convert to int16
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())

    return str(path)


def add_noise(
    samples: np.ndarray,
    snr_db: float = 10.0,
    noise_type: str = "white",
) -> np.ndarray:
    """
    Add noise to audio samples at a specified SNR (dB).

    Args:
        samples:    Clean audio samples
        snr_db:     Signal-to-noise ratio in dB (lower = noisier)
        noise_type: 'white' (Gaussian), 'babble' (simulated), or 'pink'

    Returns:
        Noisy audio samples
    """
    signal_power = np.mean(samples ** 2)

    if noise_type == "white":
        noise = np.random.randn(len(samples))
    elif noise_type == "pink":
        # Simple pink noise approximation using 1/f filtering
        white = np.random.randn(len(samples))
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(samples))
        freqs[0] = 1  # avoid division by zero
        fft = fft / np.sqrt(freqs)
        noise = np.fft.irfft(fft, n=len(samples))
    elif noise_type == "babble":
        # Simulate babble: sum of time-shifted random speech-like signals
        noise = np.zeros(len(samples))
        for _ in range(5):
            shifted = np.roll(np.random.randn(len(samples)), random.randint(0, len(samples)))
            noise += shifted * 0.2
    else:
        noise = np.random.randn(len(samples))

    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return samples

    # Scale noise to achieve desired SNR
    # SNR = 10 * log10(signal_power / noise_power)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    return samples + noise


def resample_audio(
    samples: np.ndarray,
    orig_rate: int,
    target_rate: int,
) -> np.ndarray:
    """
    Resample audio with an FFT low-pass anti-aliasing filter when downsampling.
    Uses pure NumPy so no extra runtime dependencies are required.
    """
    if orig_rate == target_rate or len(samples) == 0:
        return samples

    working = samples
    if target_rate < orig_rate:
        # Apply brick-wall low-pass filter below the target Nyquist frequency
        fft = np.fft.rfft(working)
        freqs = np.fft.rfftfreq(len(working), d=1.0 / orig_rate)
        fft[freqs > (target_rate / 2.0)] = 0
        working = np.fft.irfft(fft, n=len(working))

    duration = len(working) / orig_rate
    target_len = max(1, int(round(duration * target_rate)))
    indices = np.linspace(0, len(working) - 1, target_len)
    return np.interp(indices, np.arange(len(working)), working)


def simulate_telephony(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Simulate telephony-quality audio (8kHz, bandpass 300-3400 Hz).

    This is relevant for Gnani's contact-centre use case — many
    customers send narrowband telephony audio.
    """
    # Resample to 8kHz
    if sample_rate != 8000:
        samples = resample_audio(samples, sample_rate, 8000)

    # Simple bandpass filter simulation using FFT
    n = len(samples)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / 8000)

    # Zero out frequencies below 300 Hz and above 3400 Hz
    fft[freqs < 300] = 0
    fft[freqs > 3400] = 0

    filtered = np.fft.irfft(fft, n=n)

    # Add slight noise to simulate phone line
    filtered = add_noise(filtered, snr_db=25.0, noise_type="white")

    return filtered


def compute_spectral_features(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Compute basic spectral features for TTS quality analysis.

    Returns spectral centroid, bandwidth, and energy distribution
    which are useful proxies for voice naturalness.
    """
    n = len(samples)
    fft = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Avoid division by zero
    total_energy = np.sum(fft) + 1e-10

    # Spectral centroid — "center of mass" of the spectrum
    centroid = np.sum(freqs * fft) / total_energy

    # Spectral bandwidth — spread around the centroid
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft) / total_energy)

    # Energy in speech-relevant bands
    low_energy = np.sum(fft[(freqs >= 80) & (freqs < 500)])   # fundamental + 1st formant
    mid_energy = np.sum(fft[(freqs >= 500) & (freqs < 2000)])  # 2nd-3rd formants
    high_energy = np.sum(fft[(freqs >= 2000) & (freqs < 8000)])  # higher harmonics

    return {
        "spectral_centroid_hz": float(centroid),
        "spectral_bandwidth_hz": float(bandwidth),
        "low_band_energy": float(low_energy / total_energy),
        "mid_band_energy": float(mid_energy / total_energy),
        "high_band_energy": float(high_energy / total_energy),
    }
