"""
Audio utility functions for the evaluation pipeline.

Handles audio loading, format conversion, noise injection,
duration measurement, and signal analysis.
"""

import io
import wave
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of an audio file in seconds.
    Supports PCM WAV via stdlib `wave`, IEEE-float WAV via `scipy.io.wavfile`,
    and falls back to file-size estimation.
    """
    path = Path(audio_path)

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
        except wave.Error:
            pass

    # Try scipy for IEEE-float WAV or other PCM variants
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
    Supports standard PCM and IEEE float32 WAVs; samples normalised to [-1.0, 1.0].
    """
    try:
        with wave.open(str(audio_path), "r") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

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
        if n_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        return samples, framerate
    except wave.Error:
        from scipy.io import wavfile
        framerate, data = wavfile.read(str(audio_path))
        if np.issubdtype(data.dtype, np.floating):
            samples = data.astype(np.float64)
        elif data.dtype == np.int16:
            samples = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            samples = data.astype(np.float64) / 2147483648.0
        elif data.dtype == np.uint8:
            samples = (data.astype(np.float64) - 128.0) / 128.0
        else:
            samples = data.astype(np.float64)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        return samples, int(framerate)


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
    seed: Optional[int] = 42,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Add noise to audio samples at a specified SNR (dB).

    Args:
        samples:     Clean audio samples
        snr_db:      Signal-to-noise ratio in dB (lower = noisier)
        noise_type:  'white' (Gaussian), 'babble' (speech-modulated multi-talker), or 'pink'
        seed:        RNG seed for reproducible noise generation (None for random)
        sample_rate: Audio sample rate in Hz (used for speech-band shaping)

    Returns:
        Noisy audio samples
    """
    rng = np.random.default_rng(seed)
    signal_power = np.mean(samples ** 2)
    n = len(samples)

    if noise_type == "white":
        noise = rng.standard_normal(n)
    elif noise_type == "pink":
        # Pink noise approximation using 1/sqrt(f) spectral shaping
        white = rng.standard_normal(n)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1.0  # avoid division by zero
        fft = fft / np.sqrt(freqs)
        noise = np.fft.irfft(fft, n=n)
    elif noise_type == "babble":
        # Multi-talker babble proxy: sum speech-band (300-3400 Hz) noise streams
        # modulated by distinct low-frequency syllabic amplitude envelopes
        noise = np.zeros(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        for _ in range(6):
            raw = rng.standard_normal(n)
            fft = np.fft.rfft(raw)
            fft[(freqs < 250) | (freqs > 3400)] = 0
            band = np.fft.irfft(fft, n=n)
            # Slow syllabic envelope modulation (3-6 Hz)
            t_grid = np.linspace(0, max(1.0 / sample_rate, n / float(sample_rate)), n, endpoint=False)
            mod_freq = rng.uniform(3.0, 6.0)
            phase = rng.uniform(0, 2 * np.pi)
            envelope = 0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t_grid + phase))
            noise += band * envelope
    else:
        noise = rng.standard_normal(n)

    noise_power = np.mean(noise ** 2)
    if noise_power == 0 or signal_power == 0:
        return samples

    # Scale noise to achieve desired SNR: SNR = 10 * log10(signal_power / noise_power)
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


def apply_mulaw_codec(samples: np.ndarray, mu: int = 255) -> np.ndarray:
    """
    Simulate ITU-T G.711 mu-law 8-bit PSTN companding and quantization.
    """
    x = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(((compressed + 1.0) * 0.5) * 255.0) / 255.0 * 2.0 - 1.0
    expanded = np.sign(quantized) * ((1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0))
    return expanded


def simulate_telephony(
    samples: np.ndarray,
    sample_rate: int,
    seed: Optional[int] = 42,
    use_mulaw: bool = True,
) -> np.ndarray:
    """
    Simulate narrowband PSTN telephony audio:
      1. Anti-aliased resample to 8 kHz
      2. Bandpass filter 300-3400 Hz
      3. Optional G.711 mu-law 8-bit companding
      4. Seeded 25 dB SNR channel noise
    """
    if sample_rate != 8000:
        samples = resample_audio(samples, sample_rate, 8000)

    n = len(samples)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / 8000)
    fft[freqs < 300] = 0
    fft[freqs > 3400] = 0
    filtered = np.fft.irfft(fft, n=n)

    if use_mulaw:
        filtered = apply_mulaw_codec(filtered)

    return add_noise(filtered, snr_db=25.0, noise_type="white", seed=seed, sample_rate=8000)


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
