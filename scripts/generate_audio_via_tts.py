"""
Generate ASR test audio using Gnani Timbre TTS.

Uses Timbre to synthesize the reference sentences into audio files,
then optionally adds noise and telephony degradation. This creates
realistic test audio that Prisma can actually transcribe — and also
serves as a round-trip fidelity test of the whole pipeline.

Usage:
    python scripts/generate_audio_via_tts.py
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnani_client import GnaniClient
from src.audio_utils import add_noise, simulate_telephony, load_wav, save_wav


PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "asr" / "audio"
MANIFEST_PATH = PROJECT_ROOT / "data" / "asr" / "test_manifest.json"


def main():
    parser = argparse.ArgumentParser(description="Generate ASR test audio via Timbre TTS")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Optional tag filter (e.g., --tags telephony) to regenerate only matching clips",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  GENERATING ASR TEST AUDIO VIA TIMBRE TTS")
    print("=" * 60)

    client = GnaniClient()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.tags:
        manifest = [e for e in manifest if any(t in e.get("tags", []) for t in args.tags)]

    print(f"\n  Generating {len(manifest)} audio clips via Timbre...\n")

    success_count = 0
    fail_count = 0

    for i, entry in enumerate(manifest):
        clip_id = entry["id"]
        reference = entry["reference"]
        language = entry.get("language", "en-IN")
        tags = entry.get("tags", [])
        audio_path = PROJECT_ROOT / "data" / "asr" / entry["audio_path"]

        print(f"  [{i+1}/{len(manifest)}] {clip_id:30s} ", end="", flush=True)

        # Synthesize clean audio via Timbre
        try:
            tts_result = client.synthesize(
                text=reference,
                language=language,
                output_path=str(audio_path),
            )
            print(f"TTS OK ({tts_result['latency_ms']:.0f}ms)", end="")
            success_count += 1
        except Exception as e:
            print(f"TTS FAILED: {e}")
            fail_count += 1
            continue

        # Post-process based on category
        try:
            if "noisy" in tags:
                samples, sr = load_wav(str(audio_path))
                noisy = add_noise(samples, snr_db=10.0, noise_type="white")
                save_wav(noisy, sr, str(audio_path))
                print(" → +noise(10dB)", end="")

            elif "telephony" in tags:
                samples, sr = load_wav(str(audio_path))
                telephony = simulate_telephony(samples, sr)
                save_wav(telephony, 8000, str(audio_path))
                print(" → +telephony(8kHz, 300-3400Hz)", end="")
        except Exception as e:
            print(f" → post-process warning: {e}", end="")

        print()
        time.sleep(0.3)  # Avoid rate limiting

    print(f"\n  Done: {success_count} generated, {fail_count} failed")
    print(f"  Audio files in: {AUDIO_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
