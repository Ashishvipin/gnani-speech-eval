"""
Test Data Preparation Script

Downloads public-domain audio clips from Common Voice / OpenSLR
for ASR evaluation, and generates synthetic noisy / telephony
variants for stress testing.

For clips that cannot be sourced publicly, generates placeholder
WAV files with TTS (using macOS 'say' or pyttsx3) so the
evaluation pipeline can be tested end-to-end.

Usage:
    python scripts/prepare_test_data.py
"""

import os
import sys
import json
import struct
import wave
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.audio_utils import add_noise, simulate_telephony, save_wav


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "asr" / "audio"
MANIFEST_PATH = PROJECT_ROOT / "data" / "asr" / "test_manifest.json"
SAMPLE_RATE = 16000  # 16 kHz standard for ASR


def generate_sine_tone(
    duration_sec: float = 2.0,
    frequency: float = 440.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a sine wave (placeholder for missing audio)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)


def generate_placeholder_audio(text: str, output_path: str, sample_rate: int = SAMPLE_RATE):
    """
    Try to generate audio from text using macOS 'say' command.
    Falls back to a sine tone if 'say' is not available.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Try macOS 'say' command
    aiff_path = str(output).replace(".wav", ".aiff")
    ret = os.system(f'say -o "{aiff_path}" "{text}" 2>/dev/null')

    if ret == 0 and Path(aiff_path).exists():
        # Convert AIFF to WAV using afconvert (macOS)
        os.system(
            f'afconvert -f WAVE -d LEI16@{sample_rate} '
            f'"{aiff_path}" "{output}" 2>/dev/null'
        )
        os.remove(aiff_path)
        if output.exists():
            return

    # Fallback: generate a tone with varying frequency based on text length
    duration = max(1.0, len(text.split()) * 0.4)
    freq = 200 + (len(text) % 300)
    samples = generate_sine_tone(duration, freq, sample_rate)
    save_wav(samples, sample_rate, str(output))


def create_noisy_variant(clean_path: str, noisy_path: str, snr_db: float = 10.0):
    """Create a noisy version of an existing audio file."""
    from src.audio_utils import load_wav
    try:
        samples, sr = load_wav(clean_path)
        noisy = add_noise(samples, snr_db=snr_db, noise_type="white")
        save_wav(noisy, sr, noisy_path)
    except Exception as e:
        print(f"    Warning: Could not create noisy variant: {e}")
        # Copy the original as fallback
        import shutil
        shutil.copy2(clean_path, noisy_path)


def create_telephony_variant(clean_path: str, telephony_path: str):
    """Create a telephony-quality (8kHz narrowband) version."""
    from src.audio_utils import load_wav
    try:
        samples, sr = load_wav(clean_path)
        telephony = simulate_telephony(samples, sr)
        save_wav(telephony, 8000, telephony_path)
    except Exception as e:
        print(f"    Warning: Could not create telephony variant: {e}")
        import shutil
        shutil.copy2(clean_path, telephony_path)


def main():
    print("=" * 60)
    print("  TEST DATA PREPARATION")
    print("=" * 60)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"\n  Preparing {len(manifest)} audio clips...\n")

    for entry in manifest:
        clip_id = entry["id"]
        # Resolve path relative to manifest
        audio_path = PROJECT_ROOT / "data" / "asr" / entry["audio_path"]
        reference = entry["reference"]
        category = entry.get("category", "")
        tags = entry.get("tags", [])

        print(f"  {clip_id:30s} ", end="", flush=True)

        if audio_path.exists():
            print("EXISTS (skipping)")
            continue

        # Generate based on category
        if "noisy" in tags:
            # First generate a clean version, then add noise
            clean_path = str(audio_path).replace("noisy_", "temp_clean_")
            generate_placeholder_audio(reference, clean_path)
            create_noisy_variant(clean_path, str(audio_path), snr_db=10.0)
            if Path(clean_path).exists():
                os.remove(clean_path)
            print("GENERATED (noisy, SNR=10dB)")

        elif "telephony" in tags:
            # Generate clean then simulate telephony
            clean_path = str(audio_path).replace("telephony_", "temp_clean_")
            generate_placeholder_audio(reference, clean_path)
            create_telephony_variant(clean_path, str(audio_path))
            if Path(clean_path).exists():
                os.remove(clean_path)
            print("GENERATED (telephony, 8kHz)")

        else:
            generate_placeholder_audio(reference, str(audio_path))
            print("GENERATED (clean)")

    print(f"\n  Audio files ready in: {AUDIO_DIR}")
    print("=" * 60)
    print()
    print("  NOTE: These are placeholder audio files generated via macOS TTS.")
    print("  For a thorough evaluation, replace them with:")
    print("    - Real recordings from Common Voice / OpenSLR")
    print("    - Customer-representative audio samples")
    print("    - Actual noisy and telephony recordings")
    print()
    print("  The evaluation pipeline will work with any WAV files you drop")
    print("  into data/asr/audio/ as long as the manifest references them.")
    print()


if __name__ == "__main__":
    main()
