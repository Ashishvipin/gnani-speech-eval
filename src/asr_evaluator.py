"""
ASR Evaluation Pipeline for Gnani Prisma.

Orchestrates:
    1. Loading test manifests (real human speech and synthetic smoke suites)
    2. Calling Prisma ASR on each clip (tracking latency, retries, elapsed time)
    3. Computing BOTH verbatim and format-normalised WER/CER per clip & aggregate
    4. Computing category breakdowns and p50/p95 latency percentiles
    5. Saving structured results to JSON
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASR_MODEL
from src.audio_utils import get_audio_duration
from src.gnani_client import GnaniAPIError, GnaniClient
from src.metrics import (
    aggregate_cer,
    aggregate_wer,
    compute_cer,
    compute_percentile,
    compute_wer,
)


def load_asr_manifest(manifest_path: str) -> List[Dict]:
    """Load an ASR test manifest and resolve audio paths relative to manifest dir."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

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
    output_format: Optional[str] = None,
) -> Dict:
    """
    Run Prisma ASR on a single clip and compute verbatim + format-normalised WER/CER.
    """
    audio_path = entry["audio_path"]
    reference = entry["reference"]
    reference_formatted = entry.get("reference_formatted", reference)
    language = language_override or entry.get("language", "indic-auto")

    try:
        audio_duration = get_audio_duration(audio_path)
    except Exception:
        audio_duration = 0.0

    try:
        asr_result = client.transcribe(
            audio_path=audio_path,
            language=language,
            output_format=output_format,
        )
        hypothesis = asr_result["transcript"]
        api_latency_ms = asr_result["latency_ms"]
        total_elapsed_ms = asr_result.get("total_elapsed_ms", api_latency_ms)
        retries = asr_result.get("retries", 0)
        detected_language = asr_result["detected_language"]
        status = "success"
        error = None
    except GnaniAPIError as e:
        hypothesis = ""
        api_latency_ms = 0.0
        total_elapsed_ms = 0.0
        retries = 0
        detected_language = ""
        status = "api_error"
        error = str(e)
    except Exception as e:
        hypothesis = ""
        api_latency_ms = 0.0
        total_elapsed_ms = 0.0
        retries = 0
        detected_language = ""
        status = "error"
        error = str(e)

    lang_hint = language[:2]
    wer_verbatim = compute_wer(reference, hypothesis, language=lang_hint)
    cer_verbatim = compute_cer(reference, hypothesis, language=lang_hint)
    if reference_formatted != reference:
        wer_alt = compute_wer(reference_formatted, hypothesis, language=lang_hint)
        wer_formatted = wer_alt if wer_alt["wer"] <= wer_verbatim["wer"] else wer_verbatim
        cer_alt = compute_cer(reference_formatted, hypothesis, language=lang_hint)
        cer_formatted = cer_alt if cer_alt["cer"] <= cer_verbatim["cer"] else cer_verbatim
    else:
        wer_formatted = wer_verbatim
        cer_formatted = cer_verbatim

    return {
        "id": entry.get("id", ""),
        "category": entry.get("category", "uncategorised"),
        "source": entry.get("source", ""),
        "tags": entry.get("tags", []),
        "language": language,
        "detected_language": detected_language,
        "audio_path": audio_path,
        "audio_duration_sec": round(audio_duration, 2),
        "reference": reference,
        "reference_formatted": reference_formatted,
        "hypothesis": hypothesis,
        "wer": round(wer_verbatim["wer"], 4),
        "cer": round(cer_verbatim["cer"], 4),
        "formatted_wer": round(wer_formatted["wer"], 4),
        "formatted_cer": round(cer_formatted["cer"], 4),
        "wer_detail": {
            "substitutions": wer_verbatim["substitutions"],
            "deletions": wer_verbatim["deletions"],
            "insertions": wer_verbatim["insertions"],
            "hits": wer_verbatim["hits"],
            "ref_words": wer_verbatim["ref_length"],
        },
        "formatted_wer_detail": {
            "substitutions": wer_formatted["substitutions"],
            "deletions": wer_formatted["deletions"],
            "insertions": wer_formatted["insertions"],
            "hits": wer_formatted["hits"],
            "ref_words": wer_formatted["ref_length"],
        },
        "cer_detail": {
            "substitutions": cer_verbatim["substitutions"],
            "deletions": cer_verbatim["deletions"],
            "insertions": cer_verbatim["insertions"],
            "hits": cer_verbatim["hits"],
            "ref_chars": cer_verbatim["ref_length"],
        },
        "formatted_cer_detail": {
            "substitutions": cer_formatted["substitutions"],
            "deletions": cer_formatted["deletions"],
            "insertions": cer_formatted["insertions"],
            "hits": cer_formatted["hits"],
            "ref_chars": cer_formatted["ref_length"],
        },
        "api_latency_ms": api_latency_ms,
        "total_elapsed_ms": total_elapsed_ms,
        "retries": retries,
        "status": status,
        "error": error,
    }


def _summarise_asr_results(results: List[Dict]) -> Dict:
    """Compute overall and per-category aggregate metrics (including p50/p95 latencies)."""
    successful = [r for r in results if r["status"] == "success"]

    def _agg_for(items: List[Dict], wer_key: str, cer_key: str) -> Dict:
        w = aggregate_wer([
            {
                "substitutions": r[wer_key]["substitutions"],
                "deletions": r[wer_key]["deletions"],
                "insertions": r[wer_key]["insertions"],
                "ref_length": r[wer_key]["ref_words"],
            }
            for r in items
        ])
        c = aggregate_cer([
            {
                "substitutions": r[cer_key]["substitutions"],
                "deletions": r[cer_key]["deletions"],
                "insertions": r[cer_key]["insertions"],
                "ref_length": r[cer_key]["ref_chars"],
            }
            for r in items
        ])
        return {"wer": w, "cer": c}

    overall_v = _agg_for(successful, "wer_detail", "cer_detail")
    overall_f = _agg_for(successful, "formatted_wer_detail", "formatted_cer_detail")
    latencies = [r["api_latency_ms"] for r in successful]
    elapsed = [r["total_elapsed_ms"] for r in successful]

    categories: Dict[str, List[Dict]] = {}
    for r in successful:
        categories.setdefault(r["category"], []).append(r)

    category_summary = {}
    for cat, cat_results in categories.items():
        cat_v = _agg_for(cat_results, "wer_detail", "cer_detail")
        cat_f = _agg_for(cat_results, "formatted_wer_detail", "formatted_cer_detail")
        cat_lats = [r["api_latency_ms"] for r in cat_results]
        category_summary[cat] = {
            "num_clips": len(cat_results),
            "aggregate_wer": round(cat_v["wer"]["aggregate_wer"], 4),
            "aggregate_cer": round(cat_v["cer"]["aggregate_cer"], 4),
            "formatted_wer": round(cat_f["wer"]["aggregate_wer"], 4),
            "formatted_cer": round(cat_f["cer"]["aggregate_cer"], 4),
            "p50_latency_ms": round(compute_percentile(cat_lats, 50), 2),
            "p95_latency_ms": round(compute_percentile(cat_lats, 95), 2),
            "avg_latency_ms": round(sum(cat_lats) / max(len(cat_lats), 1), 2),
            "retries": sum(r.get("retries", 0) for r in cat_results),
        }

    return {
        "overall": {
            "wer": round(overall_v["wer"]["aggregate_wer"], 4),
            "cer": round(overall_v["cer"]["aggregate_cer"], 4),
            "formatted_wer": round(overall_f["wer"]["aggregate_wer"], 4),
            "formatted_cer": round(overall_f["cer"]["aggregate_cer"], 4),
            "total_ref_words": overall_v["wer"].get("total_ref_words", 0),
            "total_errors": overall_v["wer"].get("total_errors", 0),
            "formatted_ref_words": overall_f["wer"].get("total_ref_words", 0),
            "formatted_errors": overall_f["wer"].get("total_errors", 0),
            "p50_latency_ms": round(compute_percentile(latencies, 50), 2),
            "p95_latency_ms": round(compute_percentile(latencies, 95), 2),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
            "p95_total_elapsed_ms": round(compute_percentile(elapsed, 95), 2),
            "total_retries": sum(r.get("retries", 0) for r in successful),
        },
        "by_category": category_summary,
    }


def run_asr_evaluation(
    manifest_path: str,
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    suite_name: str = "asr_evaluation",
    verbose: bool = True,
) -> Dict:
    """Run the ASR evaluation pipeline on a single manifest."""
    client = GnaniClient(api_key=api_key)
    manifest = load_asr_manifest(manifest_path)

    if verbose:
        print(f"{'='*70}")
        print(f"  PRISMA ASR EVALUATION — {suite_name}")
        print(f"  Test clips: {len(manifest)}  |  Model: {ASR_MODEL}")
        print(f"{'='*70}\n")

    results = []
    for i, entry in enumerate(manifest):
        if verbose:
            print(
                f"  [{i+1}/{len(manifest)}] {entry.get('id', '?')} "
                f"({entry.get('category', '?')}) ... ",
                end="",
                flush=True,
            )

        result = evaluate_single_clip(client, entry)
        results.append(result)

        if verbose:
            if result["status"] == "success":
                retry_tag = f" [retries={result['retries']}]" if result["retries"] else ""
                print(
                    f"WER={result['wer']:.1%} (fmt={result['formatted_wer']:.1%})  "
                    f"CER={result['cer']:.1%}  ({result['api_latency_ms']:.0f}ms){retry_tag}"
                )
            else:
                print(f"ERROR: {result['error']}")

        time.sleep(0.3)

    successful = [r for r in results if r["status"] == "success"]
    summary = _summarise_asr_results(results)

    evaluation = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "suite": suite_name,
            "model": ASR_MODEL,
            "total_clips": len(manifest),
            "successful_clips": len(successful),
            "failed_clips": len(manifest) - len(successful),
        },
        "overall": summary["overall"],
        "by_category": summary["by_category"],
        "per_clip_results": results,
    }

    if verbose:
        ov = evaluation["overall"]
        print(f"\n{'='*70}")
        print(f"  RESULTS SUMMARY — {suite_name}")
        print(f"{'='*70}")
        print(
            f"  Verbatim WER  : {ov['wer']:.2%}  |  Format-Normalised WER : {ov['formatted_wer']:.2%}"
        )
        print(
            f"  Verbatim CER  : {ov['cer']:.2%}  |  Format-Normalised CER : {ov['formatted_cer']:.2%}"
        )
        print(
            f"  Latency       : p50={ov['p50_latency_ms']:.0f}ms  "
            f"p95={ov['p95_latency_ms']:.0f}ms  mean={ov['avg_latency_ms']:.0f}ms  "
            f"retries={ov['total_retries']}"
        )
        print(f"\n  By Category:")
        for cat, stats in evaluation["by_category"].items():
            print(
                f"    {cat:25s}  WER={stats['aggregate_wer']:>6.2%} "
                f"(fmt={stats['formatted_wer']:>6.2%})  "
                f"p50={stats['p50_latency_ms']:>5.0f}ms  (n={stats['num_clips']})"
            )
        print(f"{'='*70}\n")

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


def run_asr_suites(
    suites: Dict[str, str],
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate multiple ASR suites (e.g., real_human primary suite + synthetic_smoke)
    and write a unified structured JSON keeping suite metrics strictly separated.
    """
    suite_outputs = {}
    for suite_name, manifest_path in suites.items():
        suite_outputs[suite_name] = run_asr_evaluation(
            manifest_path=manifest_path,
            api_key=api_key,
            output_path=None,
            suite_name=suite_name,
            verbose=verbose,
        )

    total_clips = sum(s["metadata"]["total_clips"] for s in suite_outputs.values())
    total_ok = sum(s["metadata"]["successful_clips"] for s in suite_outputs.values())

    primary_key = "real_human" if "real_human" in suite_outputs else next(iter(suite_outputs))
    primary = suite_outputs[primary_key]

    combined = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": ASR_MODEL,
            "primary_suite": primary_key,
            "suites_evaluated": list(suite_outputs.keys()),
            "total_clips": total_clips,
            "successful_clips": total_ok,
            "failed_clips": total_clips - total_ok,
        },
        # Primary suite summary at top level (real human speech)
        "overall": primary["overall"],
        "by_category": primary["by_category"],
        "suites": suite_outputs,
    }

    if output_path:
        if total_ok == 0:
            if verbose:
                print(f"  Skipping save to {output_path} because 0 clips succeeded.")
        else:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"  Combined ASR results saved to: {output_path}")

    return combined
