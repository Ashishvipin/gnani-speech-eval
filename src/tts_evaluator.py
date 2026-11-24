"""
TTS Evaluation Pipeline for Gnani Timbre.

Evaluation methodology:
    1. Latency & RTF        — time-to-first-byte vs audio duration
    2. Audio integrity      — silence ratio, clipping, duration sanity
    3. Spectral quality     — centroid, bandwidth, energy distribution
    4. Pronunciation check  — round-trip ASR to detect mispronunciations
    5. Speaking rate         — words-per-minute for pacing analysis

The methodology is designed to be objective and repeatable without
requiring human listeners for every test run. A structured MOS-style
rubric is provided for optional manual scoring.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TTS_MODEL, RESULTS_DIR
from src.gnani_client import GnaniClient, GnaniAPIError
from src.metrics import compute_rtf, compute_speaking_rate, compute_silence_ratio, normalise_text
from src.audio_utils import (
    get_audio_duration, load_wav, compute_spectral_features,
)


def load_tts_test_set(test_set_path: str) -> List[Dict]:
    """
    Load TTS test sentences.

    Expected format (JSON list):
    [
        {
            "id": "num_001",
            "text": "Your order total is ₹1,23,456.",
            "language": "en-IN",
            "category": "numbers",
            "expected_behaviour": "Should read the currency and number correctly",
            "tags": ["numbers", "currency", "indian_format"]
        },
        ...
    ]
    """
    with open(test_set_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single_tts(
    client: GnaniClient,
    entry: Dict,
    output_dir: str,
    run_roundtrip_asr: bool = True,
) -> Dict:
    """
    Synthesize a sentence and evaluate the output.

    Metrics computed:
        - Latency (ms) and RTF
        - Audio duration and speaking rate (WPM)
        - Silence ratio
        - Spectral features
        - Round-trip ASR check (optional): synthesize → transcribe → compare
    """
    text = entry["text"]
    language = entry.get("language", "en-IN")
    clip_id = entry.get("id", "unknown")
    output_path = str(Path(output_dir) / f"{clip_id}.wav")

    # ── Synthesize ─────────────────────────────────────────────────
    try:
        tts_result = client.synthesize(
            text=text,
            language=language,
            output_path=output_path,
        )
        synth_latency_ms = tts_result["latency_ms"]
        audio_bytes = tts_result["audio_bytes"]
        voice = tts_result["voice"]
        status = "success"
        error = None
    except GnaniAPIError as e:
        return {
            "id": clip_id,
            "category": entry.get("category", ""),
            "text": text,
            "language": language,
            "status": "api_error",
            "error": str(e),
        }
    except Exception as e:
        return {
            "id": clip_id,
            "category": entry.get("category", ""),
            "text": text,
            "language": language,
            "status": "error",
            "error": str(e),
        }

    # ── Audio Analysis ─────────────────────────────────────────────
    try:
        audio_duration = get_audio_duration(output_path)
        samples, sample_rate = load_wav(output_path)

        rtf = compute_rtf(audio_duration, synth_latency_ms / 1000.0)
        word_count = len(text.split())
        speaking_rate = compute_speaking_rate(word_count, audio_duration)
        silence_ratio = compute_silence_ratio(samples, sample_rate)
        spectral = compute_spectral_features(samples, sample_rate)

        # Clipping detection: percentage of samples near ±1.0
        import numpy as np
        clipping_ratio = float(np.sum(np.abs(samples) > 0.99) / len(samples))

        audio_analysis = {
            "audio_duration_sec": round(audio_duration, 3),
            "sample_rate": sample_rate,
            "rtf": round(rtf, 4),
            "speaking_rate_wpm": round(speaking_rate, 1),
            "silence_ratio": round(silence_ratio, 4),
            "clipping_ratio": round(clipping_ratio, 6),
            "spectral": {k: round(v, 2) for k, v in spectral.items()},
        }
    except Exception as e:
        audio_analysis = {"error": str(e)}

    # ── Round-trip ASR Check ───────────────────────────────────────
    # Synthesize → transcribe back → compare with original text
    # This catches gross mispronunciations and missing words
    roundtrip_result = None
    if run_roundtrip_asr and status == "success":
        try:
            asr_result = client.transcribe(
                audio_path=output_path,
                language=language,
            )
            roundtrip_transcript = asr_result["transcript"]

            # Normalised multiset word overlap (strips punctuation, preserves repeats)
            ref_tokens = normalise_text(text, language=language[:2]).split()
            hyp_tokens = normalise_text(roundtrip_transcript, language=language[:2]).split()
            if len(ref_tokens) > 0:
                ref_counts = Counter(ref_tokens)
                hyp_counts = Counter(hyp_tokens)
                matched = sum(min(ref_counts[w], hyp_counts[w]) for w in ref_counts)
                overlap = matched / len(ref_tokens)
            else:
                overlap = 1.0

            roundtrip_result = {
                "asr_transcript": roundtrip_transcript,
                "word_overlap": round(overlap, 4),
                "asr_latency_ms": asr_result["latency_ms"],
            }
        except Exception as e:
            roundtrip_result = {"error": str(e)}

    # ── Quality Score ──────────────────────────────────────────────
    # Automated quality score (0-100) based on objective metrics
    quality_score = _compute_quality_score(audio_analysis, roundtrip_result)

    return {
        "id": clip_id,
        "category": entry.get("category", ""),
        "tags": entry.get("tags", []),
        "text": text,
        "language": language,
        "voice": voice,
        "status": status,
        "synth_latency_ms": round(synth_latency_ms, 2),
        "audio_analysis": audio_analysis,
        "roundtrip_asr": roundtrip_result,
        "quality_score": quality_score,
        "output_path": output_path,
        "error": error,
    }


def _compute_quality_score(audio_analysis: Dict, roundtrip: Optional[Dict]) -> float:
    """
    Compute an automated quality score (0-100) from objective metrics.

    Scoring rubric:
        - RTF < 1.0 (real-time capable):         20 points
        - Speaking rate 120-180 WPM (natural):    20 points
        - Silence ratio 0.1-0.4 (natural pauses): 20 points
        - No clipping (< 0.001):                   15 points
        - Round-trip word overlap > 0.7:           25 points

    This is not a replacement for MOS but provides a consistent,
    automated baseline that can run on every evaluation.
    """
    score = 0.0

    if "error" in audio_analysis:
        return 0.0

    # RTF scoring (20 points)
    rtf = audio_analysis.get("rtf", float("inf"))
    if rtf < 0.5:
        score += 20
    elif rtf < 1.0:
        score += 15
    elif rtf < 2.0:
        score += 8
    # else: 0

    # Speaking rate scoring (20 points)
    wpm = audio_analysis.get("speaking_rate_wpm", 0)
    if 120 <= wpm <= 180:
        score += 20
    elif 100 <= wpm <= 200:
        score += 14
    elif 80 <= wpm <= 220:
        score += 8
    # else: 0

    # Silence ratio scoring (20 points)
    silence = audio_analysis.get("silence_ratio", 0)
    if 0.1 <= silence <= 0.4:
        score += 20
    elif 0.05 <= silence <= 0.6:
        score += 12
    # Too much or too little silence: 0

    # Clipping scoring (15 points)
    clipping = audio_analysis.get("clipping_ratio", 1.0)
    if clipping < 0.001:
        score += 15
    elif clipping < 0.01:
        score += 8
    # else: 0

    # Round-trip ASR scoring (25 points)
    if roundtrip and "word_overlap" in roundtrip:
        overlap = roundtrip["word_overlap"]
        if overlap >= 0.9:
            score += 25
        elif overlap >= 0.7:
            score += 18
        elif overlap >= 0.5:
            score += 10
        # else: 0
    else:
        # Can't measure — give partial credit
        score += 10

    return round(score, 1)


def run_tts_evaluation(
    test_set_path: str,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    run_roundtrip_asr: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Run the full TTS evaluation pipeline.

    Args:
        test_set_path:      Path to JSON test sentence file
        api_key:            Gnani API key
        output_dir:         Directory for synthesized audio files
        output_path:        Path to save results JSON (defaults to results/tts_results.json)
        run_roundtrip_asr:  Whether to run ASR on TTS output for validation
        verbose:            Print progress

    Returns:
        Full evaluation results
    """
    client = GnaniClient(api_key=api_key)
    test_set = load_tts_test_set(test_set_path)

    if output_dir is None:
        output_dir = str(Path(test_set_path).parent / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"{'='*60}")
        print(f"  TIMBRE TTS EVALUATION")
        print(f"  Test sentences: {len(test_set)}")
        print(f"  Round-trip ASR: {'enabled' if run_roundtrip_asr else 'disabled'}")
        print(f"{'='*60}\n")

    # ── Evaluate each sentence ─────────────────────────────────────
    results = []
    for i, entry in enumerate(test_set):
        if verbose:
            text_preview = entry["text"][:50] + "..." if len(entry["text"]) > 50 else entry["text"]
            print(f"  [{i+1}/{len(test_set)}] {entry.get('id', '?')} "
                  f"({entry.get('category', '?')}) ... ", end="", flush=True)

        result = evaluate_single_tts(
            client, entry, output_dir, run_roundtrip_asr=run_roundtrip_asr,
        )
        results.append(result)

        if verbose:
            if result["status"] == "success":
                qa = result.get("audio_analysis", {})
                print(f"RTF={qa.get('rtf', '?')}  "
                      f"WPM={qa.get('speaking_rate_wpm', '?')}  "
                      f"Score={result.get('quality_score', '?')}/100  "
                      f"({result.get('synth_latency_ms', 0):.0f}ms)")
            else:
                print(f"ERROR: {result.get('error', 'unknown')}")

        time.sleep(0.6)

    # ── Aggregate by category ──────────────────────────────────────
    successful = [r for r in results if r["status"] == "success"]

    categories = {}
    for r in successful:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_summary = {}
    for cat, cat_results in categories.items():
        analyses = [r["audio_analysis"] for r in cat_results if "error" not in r.get("audio_analysis", {})]
        scores = [r["quality_score"] for r in cat_results]

        category_summary[cat] = {
            "num_sentences": len(cat_results),
            "avg_quality_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "avg_rtf": round(
                sum(a.get("rtf", 0) for a in analyses) / max(len(analyses), 1), 4
            ),
            "avg_latency_ms": round(
                sum(r["synth_latency_ms"] for r in cat_results) / len(cat_results), 2
            ),
            "avg_speaking_rate_wpm": round(
                sum(a.get("speaking_rate_wpm", 0) for a in analyses) / max(len(analyses), 1), 1
            ),
            "avg_silence_ratio": round(
                sum(a.get("silence_ratio", 0) for a in analyses) / max(len(analyses), 1), 4
            ),
        }

    # ── Overall stats ──────────────────────────────────────────────
    all_scores = [r["quality_score"] for r in successful]
    all_rtfs = [r["audio_analysis"].get("rtf", 0) for r in successful
                if "error" not in r.get("audio_analysis", {})]
    all_latencies = [r["synth_latency_ms"] for r in successful]

    evaluation = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": TTS_MODEL,
            "total_sentences": len(test_set),
            "successful": len(successful),
            "failed": len(test_set) - len(successful),
            "roundtrip_asr_enabled": run_roundtrip_asr,
        },
        "overall": {
            "avg_quality_score": round(sum(all_scores) / max(len(all_scores), 1), 1),
            "avg_rtf": round(sum(all_rtfs) / max(len(all_rtfs), 1), 4),
            "avg_latency_ms": round(sum(all_latencies) / max(len(all_latencies), 1), 2),
            "min_latency_ms": round(min(all_latencies) if all_latencies else 0, 2),
            "max_latency_ms": round(max(all_latencies) if all_latencies else 0, 2),
        },
        "by_category": category_summary,
        "per_sentence_results": results,
        "quality_rubric": {
            "description": "Automated quality score (0-100) breakdown",
            "components": {
                "rtf_realtime": "20 pts — RTF < 1.0 (faster than real-time)",
                "speaking_rate": "20 pts — 120-180 WPM (natural pacing)",
                "silence_ratio": "20 pts — 10-40% silence (natural pauses)",
                "no_clipping": "15 pts — Clipping ratio < 0.1%",
                "roundtrip_fidelity": "25 pts — ASR re-transcription overlap > 90%",
            },
        },
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Avg Quality Score : {evaluation['overall']['avg_quality_score']}/100")
        print(f"  Avg RTF           : {evaluation['overall']['avg_rtf']}")
        print(f"  Avg Latency       : {evaluation['overall']['avg_latency_ms']:.0f} ms")
        print(f"\n  By Category:")
        for cat, stats in category_summary.items():
            print(f"    {cat:30s}  Score={stats['avg_quality_score']}/100  "
                  f"RTF={stats['avg_rtf']}  "
                  f"(n={stats['num_sentences']})")
        print(f"{'='*60}\n")

    # ── Save results ───────────────────────────────────────────────
    results_path = output_path or str(RESULTS_DIR / "tts_results.json")
    if len(successful) == 0:
        if verbose:
            print(f"  Skipping save to {results_path} because 0 sentences succeeded.")
    else:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        # Remove audio_bytes from saved results
        save_results = json.loads(json.dumps(evaluation, default=str))
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  Results saved to: {results_path}")

    return evaluation
