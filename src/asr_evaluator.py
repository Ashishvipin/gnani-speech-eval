"""
ASR Evaluation Pipeline for Gnani Prisma.

Orchestrates:
    1. Loading the test manifest (audio paths + ground-truth transcripts)
    2. Calling Prisma ASR on each clip
    3. Computing WER/CER per clip and in aggregate
    4. Breaking down results by test category
    5. Saving structured results to JSON
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASR_MODEL
from src.metrics import compute_wer, compute_cer, aggregate_wer, aggregate_cer
from src.gnani_client import GnaniClient, GnaniAPIError
from src.audio_utils import get_audio_duration


def load_asr_manifest(manifest_path: str) -> List[Dict]:
    """
    Load ASR test manifest.

    Expected format (JSON list):
    [
        {
            "id": "clean_en_001",
            "audio_path": "data/asr/audio/clean_en_001.wav",
            "reference": "the quick brown fox jumps over the lazy dog",
            "language": "en-IN",
            "category": "clean_english",
            "tags": ["clean", "short", "english"]
        },
        ...
    ]
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Resolve relative audio paths
    manifest_dir = Path(manifest_path).parent
    for entry in manifest:
        audio_path = Path(entry["audio_path"])
        if not audio_path.is_absolute():
            entry["audio_path"] = str(manifest_dir / audio_path)

    return manifest


def evaluate_single_clip(
    client: GnaniClient,
    entry: Dict,
    language_override: Optional[str] = None,
) -> Dict:
    """
    Run ASR on a single test clip and compute metrics.

    Returns a result dict with WER, CER, transcript comparison,
    and metadata.
    """
    audio_path = entry["audio_path"]
    reference = entry["reference"]
    language = language_override or entry.get("language", "indic-auto")

    # Get audio duration for context
    try:
        audio_duration = get_audio_duration(audio_path)
    except Exception:
        audio_duration = 0.0

    # Call Prisma ASR
    try:
        asr_result = client.transcribe(
            audio_path=audio_path,
            language=language,
        )
        hypothesis = asr_result["transcript"]
        api_latency_ms = asr_result["latency_ms"]
        detected_language = asr_result["detected_language"]
        status = "success"
        error = None
    except GnaniAPIError as e:
        hypothesis = ""
        api_latency_ms = 0
        detected_language = ""
        status = "api_error"
        error = str(e)
    except Exception as e:
        hypothesis = ""
        api_latency_ms = 0
        detected_language = ""
        status = "error"
        error = str(e)

    # Compute WER and CER
    wer_result = compute_wer(reference, hypothesis, language=language[:2])
    cer_result = compute_cer(reference, hypothesis, language=language[:2])

    return {
        "id": entry.get("id", ""),
        "category": entry.get("category", "uncategorised"),
        "tags": entry.get("tags", []),
        "language": language,
        "detected_language": detected_language,
        "audio_path": audio_path,
        "audio_duration_sec": round(audio_duration, 2),
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": round(wer_result["wer"], 4),
        "cer": round(cer_result["cer"], 4),
        "wer_detail": {
            "substitutions": wer_result["substitutions"],
            "deletions": wer_result["deletions"],
            "insertions": wer_result["insertions"],
            "hits": wer_result["hits"],
            "ref_words": wer_result["ref_length"],
        },
        "cer_detail": {
            "substitutions": cer_result["substitutions"],
            "deletions": cer_result["deletions"],
            "insertions": cer_result["insertions"],
            "hits": cer_result["hits"],
            "ref_chars": cer_result["ref_length"],
        },
        "api_latency_ms": api_latency_ms,
        "status": status,
        "error": error,
    }


def run_asr_evaluation(
    manifest_path: str,
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run the full ASR evaluation pipeline.

    Args:
        manifest_path: Path to JSON test manifest
        api_key:       Gnani API key (or uses env)
        output_path:   Where to save results JSON
        verbose:       Print progress

    Returns:
        Full evaluation results with per-clip and aggregate metrics
    """
    client = GnaniClient(api_key=api_key)
    manifest = load_asr_manifest(manifest_path)

    if verbose:
        print(f"{'='*60}")
        print(f"  PRISMA ASR EVALUATION")
        print(f"  Test clips: {len(manifest)}")
        print(f"  Model: {ASR_MODEL}")
        print(f"{'='*60}\n")

    # ── Run evaluation on each clip ────────────────────────────────
    results = []
    for i, entry in enumerate(manifest):
        if verbose:
            print(f"  [{i+1}/{len(manifest)}] {entry.get('id', '?')} "
                  f"({entry.get('category', '?')}) ... ", end="", flush=True)

        result = evaluate_single_clip(client, entry)
        results.append(result)

        if verbose:
            if result["status"] == "success":
                print(f"WER={result['wer']:.1%}  CER={result['cer']:.1%}  "
                      f"({result['api_latency_ms']:.0f}ms)")
            else:
                print(f"ERROR: {result['error']}")

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # ── Aggregate metrics ──────────────────────────────────────────
    successful = [r for r in results if r["status"] == "success"]
    overall_wer = aggregate_wer([{
        "substitutions": r["wer_detail"]["substitutions"],
        "deletions": r["wer_detail"]["deletions"],
        "insertions": r["wer_detail"]["insertions"],
        "ref_length": r["wer_detail"]["ref_words"],
    } for r in successful])
    overall_cer = aggregate_cer([{
        "substitutions": r["cer_detail"]["substitutions"],
        "deletions": r["cer_detail"]["deletions"],
        "insertions": r["cer_detail"]["insertions"],
        "ref_length": r["cer_detail"]["ref_chars"],
    } for r in successful])

    # ── Category breakdown ─────────────────────────────────────────
    categories = {}
    for r in successful:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_summary = {}
    for cat, cat_results in categories.items():
        cat_wer_data = []
        for r in cat_results:
            cat_wer_data.append({
                "substitutions": r["wer_detail"]["substitutions"],
                "deletions": r["wer_detail"]["deletions"],
                "insertions": r["wer_detail"]["insertions"],
                "ref_length": r["wer_detail"]["ref_words"],
            })
        cat_wer = aggregate_wer(cat_wer_data)
        cat_cer = aggregate_cer([{
            "substitutions": r["cer_detail"]["substitutions"],
            "deletions": r["cer_detail"]["deletions"],
            "insertions": r["cer_detail"]["insertions"],
            "ref_length": r["cer_detail"]["ref_chars"],
        } for r in cat_results])
        category_summary[cat] = {
            "num_clips": len(cat_results),
            "aggregate_wer": round(cat_wer["aggregate_wer"], 4),
            "aggregate_cer": round(cat_cer["aggregate_cer"], 4),
            "avg_latency_ms": round(
                sum(r["api_latency_ms"] for r in cat_results) / len(cat_results), 2
            ),
        }

    # ── Build final report ─────────────────────────────────────────
    evaluation = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": ASR_MODEL,
            "total_clips": len(manifest),
            "successful_clips": len(successful),
            "failed_clips": len(manifest) - len(successful),
        },
        "overall": {
            "wer": round(overall_wer["aggregate_wer"], 4),
            "cer": round(overall_cer["aggregate_cer"], 4),
            "total_ref_words": overall_wer.get("total_ref_words", 0),
            "total_errors": overall_wer.get("total_errors", 0),
        },
        "by_category": category_summary,
        "per_clip_results": results,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Overall WER : {evaluation['overall']['wer']:.2%}")
        print(f"  Overall CER : {evaluation['overall']['cer']:.2%}")
        print(f"\n  By Category:")
        for cat, stats in category_summary.items():
            print(f"    {cat:30s}  WER={stats['aggregate_wer']:.2%}  "
                  f"CER={stats['aggregate_cer']:.2%}  "
                  f"(n={stats['num_clips']})")
        print(f"{'='*60}\n")

    # ── Save results ───────────────────────────────────────────────
    if output_path:
        if len(successful) == 0:
            if verbose:
                print(f"  Skipping save to {output_path} because 0 clips succeeded.")
        else:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"  Results saved to: {output_path}")

    return evaluation
