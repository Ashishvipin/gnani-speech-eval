"""
ASR Evaluation Pipeline for Gnani Prisma.

Orchestrates:
    1. Evaluating Clean Real-Human Speech (`real_human`), Paired Clean-vs-Degraded
       Stress Variants (`real_paired_stress`), and Synthetic Smoke (`synthetic_smoke`).
    2. Computing Verbatim WER/CER (primary) and dataset-agnostic Rule-Normalised
       WER/CER (`normalise_for_eval` — zero per-clip output-tuned references).
    3. Breaking out cross-script Latin/Devanagari substitutions cleanly on Hindi.
    4. Computing 95% clip-level bootstrap confidence intervals and p50/p95 latencies.
    5. Running a controlled Hinglish ASR language-parameter ablation (`hi-IN` vs
       `hi-IN,en-IN` vs `indic-auto`).
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
    compute_bootstrap_ci,
    compute_cer,
    compute_percentile,
    compute_rule_cer,
    compute_rule_wer,
    compute_wer,
    count_cross_script_substitutions,
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
    """Run Prisma ASR on a single clip and compute verbatim + rule-normalised WER/CER."""
    audio_path = entry["audio_path"]
    reference = entry["reference"]
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

    lang_hint = language.split("-")[0] if "-" in language else "hi"
    wer_verbatim = compute_wer(reference, hypothesis, language=lang_hint)
    cer_verbatim = compute_cer(reference, hypothesis, language=lang_hint)
    wer_rule = compute_rule_wer(reference, hypothesis, language=lang_hint)
    cer_rule = compute_rule_cer(reference, hypothesis, language=lang_hint)
    script_diag = count_cross_script_substitutions(reference, hypothesis, language=lang_hint)

    return {
        "id": entry.get("id", ""),
        "parent_clip_id": entry.get("parent_clip_id"),
        "condition": entry.get("condition"),
        "category": entry.get("category", "uncategorised"),
        "source": entry.get("source", ""),
        "license": entry.get("license", ""),
        "tags": entry.get("tags", []),
        "language": language,
        "detected_language": detected_language,
        "audio_path": audio_path,
        "audio_duration_sec": round(audio_duration, 2),
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": round(wer_verbatim["wer"], 4),
        "cer": round(cer_verbatim["cer"], 4),
        "rule_normalised_wer": round(wer_rule["wer"], 4),
        "rule_normalised_cer": round(cer_rule["cer"], 4),
        "cross_script_subs": script_diag["cross_script_subs"],
        "wer_detail": {
            "substitutions": wer_verbatim["substitutions"],
            "deletions": wer_verbatim["deletions"],
            "insertions": wer_verbatim["insertions"],
            "hits": wer_verbatim["hits"],
            "ref_words": wer_verbatim["ref_length"],
        },
        "rule_wer_detail": {
            "substitutions": wer_rule["substitutions"],
            "deletions": wer_rule["deletions"],
            "insertions": wer_rule["insertions"],
            "hits": wer_rule["hits"],
            "ref_words": wer_rule["ref_length"],
        },
        "cer_detail": {
            "substitutions": cer_verbatim["substitutions"],
            "deletions": cer_verbatim["deletions"],
            "insertions": cer_verbatim["insertions"],
            "hits": cer_verbatim["hits"],
            "ref_chars": cer_verbatim["ref_length"],
        },
        "rule_cer_detail": {
            "substitutions": cer_rule["substitutions"],
            "deletions": cer_rule["deletions"],
            "insertions": cer_rule["insertions"],
            "hits": cer_rule["hits"],
            "ref_chars": cer_rule["ref_length"],
        },
        "api_latency_ms": api_latency_ms,
        "total_elapsed_ms": total_elapsed_ms,
        "retries": retries,
        "status": status,
        "error": error,
    }


def _agg_metrics_with_ci(items: List[Dict], wer_key: str, cer_key: str) -> Dict:
    """Compute micro-averaged WER/CER + 95% clip-level bootstrap confidence intervals."""
    wer_agg = aggregate_wer([
        {
            "substitutions": r[wer_key]["substitutions"],
            "deletions": r[wer_key]["deletions"],
            "insertions": r[wer_key]["insertions"],
            "ref_length": r[wer_key]["ref_words"],
        }
        for r in items
    ])
    cer_agg = aggregate_cer([
        {
            "substitutions": r[cer_key]["substitutions"],
            "deletions": r[cer_key]["deletions"],
            "insertions": r[cer_key]["insertions"],
            "ref_length": r[cer_key]["ref_chars"],
        }
        for r in items
    ])
    wer_ci = compute_bootstrap_ci(
        [
            {
                "errors": r[wer_key]["substitutions"] + r[wer_key]["deletions"] + r[wer_key]["insertions"],
                "ref_length": r[wer_key]["ref_words"],
            }
            for r in items
        ]
    )
    cer_ci = compute_bootstrap_ci(
        [
            {
                "errors": r[cer_key]["substitutions"] + r[cer_key]["deletions"] + r[cer_key]["insertions"],
                "ref_length": r[cer_key]["ref_chars"],
            }
            for r in items
        ]
    )
    return {"wer": wer_agg, "cer": cer_agg, "wer_ci": wer_ci, "cer_ci": cer_ci}


def _summarise_asr_results(results: List[Dict]) -> Dict:
    """Compute overall and per-category aggregate metrics, CIs, and p50/p95 latencies."""
    successful = [r for r in results if r["status"] == "success"]
    overall_v = _agg_metrics_with_ci(successful, "wer_detail", "cer_detail")
    overall_r = _agg_metrics_with_ci(successful, "rule_wer_detail", "rule_cer_detail")
    latencies = [r["api_latency_ms"] for r in successful]
    elapsed = [r["total_elapsed_ms"] for r in successful]

    categories: Dict[str, List[Dict]] = {}
    for r in successful:
        categories.setdefault(r["category"], []).append(r)

    category_summary = {}
    for cat, cat_results in categories.items():
        cat_v = _agg_metrics_with_ci(cat_results, "wer_detail", "cer_detail")
        cat_r = _agg_metrics_with_ci(cat_results, "rule_wer_detail", "rule_cer_detail")
        cat_lats = [r["api_latency_ms"] for r in cat_results]
        category_summary[cat] = {
            "num_clips": len(cat_results),
            "aggregate_wer": round(cat_v["wer"]["aggregate_wer"], 4),
            "wer_ci_95": [cat_v["wer_ci"]["ci_low"], cat_v["wer_ci"]["ci_high"]],
            "aggregate_cer": round(cat_v["cer"]["aggregate_cer"], 4),
            "rule_normalised_wer": round(cat_r["wer"]["aggregate_wer"], 4),
            "rule_wer_ci_95": [cat_r["wer_ci"]["ci_low"], cat_r["wer_ci"]["ci_high"]],
            "rule_normalised_cer": round(cat_r["cer"]["aggregate_cer"], 4),
            "cross_script_subs": sum(r.get("cross_script_subs", 0) for r in cat_results),
            "p50_latency_ms": round(compute_percentile(cat_lats, 50), 2),
            "p95_latency_ms": round(compute_percentile(cat_lats, 95), 2),
            "avg_latency_ms": round(sum(cat_lats) / max(len(cat_lats), 1), 2),
            "retries": sum(r.get("retries", 0) for r in cat_results),
        }

    return {
        "overall": {
            "wer": round(overall_v["wer"]["aggregate_wer"], 4),
            "wer_ci_95": [overall_v["wer_ci"]["ci_low"], overall_v["wer_ci"]["ci_high"]],
            "cer": round(overall_v["cer"]["aggregate_cer"], 4),
            "cer_ci_95": [overall_v["cer_ci"]["ci_low"], overall_v["cer_ci"]["ci_high"]],
            "rule_normalised_wer": round(overall_r["wer"]["aggregate_wer"], 4),
            "rule_wer_ci_95": [overall_r["wer_ci"]["ci_low"], overall_r["wer_ci"]["ci_high"]],
            "rule_normalised_cer": round(overall_r["cer"]["aggregate_cer"], 4),
            "total_ref_words": overall_v["wer"].get("total_ref_words", 0),
            "total_errors": overall_v["wer"].get("total_errors", 0),
            "rule_ref_words": overall_r["wer"].get("total_ref_words", 0),
            "rule_errors": overall_r["wer"].get("total_errors", 0),
            "cross_script_subs": sum(r.get("cross_script_subs", 0) for r in successful),
            "p50_latency_ms": round(compute_percentile(latencies, 50), 2),
            "p95_latency_ms": round(compute_percentile(latencies, 95), 2),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
            "p95_total_elapsed_ms": round(compute_percentile(elapsed, 95), 2),
            "total_retries": sum(r.get("retries", 0) for r in successful),
        },
        "by_category": category_summary,
    }


def _compute_paired_stress_deltas(
    clean_results: List[Dict],
    stress_results: List[Dict],
) -> List[Dict]:
    """Build exact clean-vs-degraded paired delta rows keyed by parent_clip_id."""
    clean_by_id = {r["id"]: r for r in clean_results if r["status"] == "success"}
    grouped: Dict[str, List[Dict]] = {}
    for r in stress_results:
        pid = r.get("parent_clip_id")
        if pid:
            grouped.setdefault(pid, []).append(r)

    rows = []
    for pid, variants in grouped.items():
        base = clean_by_id.get(pid)
        row = {
            "parent_clip_id": pid,
            "language": base["language"] if base else variants[0]["language"],
            "clean_wer": base["wer"] if base else None,
            "clean_rule_wer": base["rule_normalised_wer"] if base else None,
            "conditions": {},
        }
        for v in variants:
            cond = v.get("condition") or v["category"]
            row["conditions"][cond] = {
                "wer": v["wer"],
                "rule_normalised_wer": v["rule_normalised_wer"],
                "delta_wer": round(v["wer"] - base["wer"], 4) if base else None,
                "delta_rule_wer": round(v["rule_normalised_wer"] - base["rule_normalised_wer"], 4) if base else None,
                "latency_ms": v["api_latency_ms"],
                "hypothesis": v["hypothesis"],
            }
        rows.append(row)
    return rows


def run_hinglish_ablation(
    manifest_path: str,
    api_key: Optional[str] = None,
    languages: tuple = ("hi-IN", "hi-IN,en-IN", "indic-auto"),
    verbose: bool = True,
) -> Dict:
    """Controlled A/B test comparing Prisma language settings on code-mixed Hinglish clips."""
    client = GnaniClient(api_key=api_key)
    manifest = [e for e in load_asr_manifest(manifest_path) if e.get("category") == "code_switching"]
    if not manifest:
        return {}

    if verbose:
        print(f"{'='*70}")
        print(f"  HINGLISH ASR LANGUAGE ABLATION (n={len(manifest)} controlled clips)")
        print(f"{'='*70}\n")

    by_setting: Dict[str, Dict] = {}
    for lang_setting in languages:
        clip_runs = []
        for entry in manifest:
            res = evaluate_single_clip(client, entry, language_override=lang_setting)
            clip_runs.append(res)
            time.sleep(0.25)
        summ = _summarise_asr_results(clip_runs)
        ov = summ["overall"]
        by_setting[lang_setting] = {
            "num_clips": len(clip_runs),
            "verbatim_wer": ov["wer"],
            "rule_normalised_wer": ov["rule_normalised_wer"],
            "cross_script_subs": ov["cross_script_subs"],
            "p50_latency_ms": ov["p50_latency_ms"],
            "clips": [
                {
                    "id": r["id"],
                    "detected_language": r["detected_language"],
                    "wer": r["wer"],
                    "rule_normalised_wer": r["rule_normalised_wer"],
                    "hypothesis": r["hypothesis"],
                }
                for r in clip_runs
            ],
        }
        if verbose:
            print(
                f"  language_code={lang_setting:<11s} -> Verbatim WER={ov['wer']:.2%}  "
                f"Rule-WER={ov['rule_normalised_wer']:.2%}  "
                f"cross_script={ov['cross_script_subs']}  p50={ov['p50_latency_ms']:.0f}ms"
            )
    if verbose:
        print()
    return by_setting


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
                    f"WER={result['wer']:.1%} (rule={result['rule_normalised_wer']:.1%})  "
                    f"CER={result['cer']:.1%}  ({result['api_latency_ms']:.0f}ms){retry_tag}"
                )
            else:
                print(f"ERROR: {result['error']}")

        time.sleep(0.25)

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
        ci_v = ov["wer_ci_95"]
        ci_r = ov["rule_wer_ci_95"]
        print(f"\n{'='*70}")
        print(f"  RESULTS SUMMARY — {suite_name}")
        print(f"{'='*70}")
        print(
            f"  Verbatim WER        : {ov['wer']:.2%}  [95% CI: {ci_v[0]:.2%}–{ci_v[1]:.2%}]"
        )
        print(
            f"  Rule-Normalised WER : {ov['rule_normalised_wer']:.2%}  [95% CI: {ci_r[0]:.2%}–{ci_r[1]:.2%}]"
        )
        print(
            f"  Verbatim CER        : {ov['cer']:.2%}  |  Rule-Normalised CER : {ov['rule_normalised_cer']:.2%}"
        )
        print(
            f"  Latency             : p50={ov['p50_latency_ms']:.0f}ms  "
            f"p95={ov['p95_latency_ms']:.0f}ms  mean={ov['avg_latency_ms']:.0f}ms  "
            f"retries={ov['total_retries']}"
        )
        print(f"\n  By Category:")
        for cat, stats in evaluation["by_category"].items():
            print(
                f"    {cat:25s}  WER={stats['aggregate_wer']:>6.2%} "
                f"(rule={stats['rule_normalised_wer']:>6.2%})  "
                f"p50={stats['p50_latency_ms']:>5.0f}ms  (n={stats['num_clips']})"
            )
        print(f"{'='*70}\n")

    if output_path and len(successful) > 0:
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
    include_hinglish_ablation: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate all ASR suites (clean real_human, real_paired_stress, synthetic_smoke)
    plus Hinglish language ablation, keeping metrics strictly separated per suite.
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

    paired_deltas = []
    if "real_human" in suite_outputs and "real_paired_stress" in suite_outputs:
        paired_deltas = _compute_paired_stress_deltas(
            suite_outputs["real_human"]["per_clip_results"],
            suite_outputs["real_paired_stress"]["per_clip_results"],
        )
        suite_outputs["real_paired_stress"]["paired_clean_vs_degraded"] = paired_deltas

    hinglish_ablation = {}
    if include_hinglish_ablation and "synthetic_smoke" in suites:
        hinglish_ablation = run_hinglish_ablation(
            suites["synthetic_smoke"], api_key=api_key, verbose=verbose
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
            "normaliser": "rule-based dataset-agnostic normalise_for_eval (verbatim WER is primary)",
        },
        "overall": primary["overall"],
        "by_category": primary["by_category"],
        "suites": suite_outputs,
        "hinglish_ablation": hinglish_ablation,
    }

    if output_path and total_ok > 0:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  Combined ASR results saved to: {output_path}")

    return combined
