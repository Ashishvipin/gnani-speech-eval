#!/usr/bin/env python3
"""
Main Evaluation Runner

Runs both ASR (Prisma) — across the real-human speech suite and synthetic
smoke suite — and TTS (Timbre) evaluations end-to-end.

Usage:
    # Run everything (default: 3 TTS repeats per sentence)
    python scripts/run_evaluation.py

    # ASR only
    python scripts/run_evaluation.py --asr-only

    # TTS only (with custom repeat count)
    python scripts/run_evaluation.py --tts-only --tts-repeats 3

    # Include optional independent faster-whisper cross-check (if installed)
    python scripts/run_evaluation.py --tts-only --independent-asr
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Gnani Speech Evaluation — Prisma ASR & Timbre TTS"
    )
    parser.add_argument("--asr-only", action="store_true", help="Run ASR evaluation only")
    parser.add_argument("--tts-only", action="store_true", help="Run TTS evaluation only")
    parser.add_argument("--no-roundtrip", action="store_true",
                        help="Skip round-trip ASR check in TTS evaluation")
    parser.add_argument("--tts-repeats", type=int, default=3,
                        help="Number of synthesis repeats per TTS sentence (default: 3)")
    parser.add_argument("--independent-asr", action="store_true",
                        help="Cross-check TTS output with faster-whisper if installed")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gnani API key (or set GNANI_API_KEY env var)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    run_asr = not args.tts_only
    run_tts = not args.asr_only

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          GNANI SPEECH EVALUATION TOOLKIT                ║")
    print("║          Prisma (ASR) + Timbre (TTS)                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    asr_results = None
    tts_results = None

    if run_asr:
        from src.asr_evaluator import run_asr_suites

        suites = {
            "real_human": str(project_root / "data" / "asr" / "real_manifest.json"),
            "real_paired_stress": str(project_root / "data" / "asr" / "real_stress_manifest.json"),
            "synthetic_smoke": str(project_root / "data" / "asr" / "test_manifest.json"),
        }
        output_path = str(project_root / "results" / "asr_results.json")

        print("▶ Starting ASR (Prisma) evaluation (real human + paired stress + synthetic smoke)...\n")
        asr_results = run_asr_suites(
            suites=suites,
            api_key=args.api_key,
            output_path=output_path,
            include_hinglish_ablation=True,
            verbose=True,
        )
        print()

    if run_tts:
        from src.tts_evaluator import run_tts_evaluation

        test_set_path = str(project_root / "data" / "tts" / "test_sentences.json")
        output_dir = str(project_root / "data" / "tts" / "output")
        tts_results_path = str(project_root / "results" / "tts_results.json")

        print("▶ Starting TTS (Timbre) evaluation...\n")
        tts_results = run_tts_evaluation(
            test_set_path=test_set_path,
            api_key=args.api_key,
            output_dir=output_dir,
            output_path=tts_results_path,
            run_roundtrip_asr=not args.no_roundtrip,
            repeats=args.tts_repeats,
            use_independent_asr=args.independent_asr,
            verbose=True,
        )
        print()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║                   EVALUATION COMPLETE                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Results saved in: {project_root / 'results'}/")
    if run_asr:
        print("    - asr_results.json")
    if run_tts:
        print("    - tts_results.json")
    print()

    if run_asr and asr_results and asr_results["metadata"]["successful_clips"] == 0:
        print("  ERROR: All ASR requests failed — check network/API key.")
        sys.exit(1)
    if run_tts and tts_results and tts_results["metadata"]["successful"] == 0:
        print("  ERROR: All TTS requests failed — check network/API key.")
        sys.exit(1)


if __name__ == "__main__":
    main()
