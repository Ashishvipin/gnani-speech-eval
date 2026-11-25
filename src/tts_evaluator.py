"""
TTS Evaluation Pipeline for Gnani Timbre.

Evaluation methodology (objective, component-level — no arbitrary composite score):
    1. Non-streaming REST HTTP latency (`/api/v1/tts/inference`) & RTF across
       repeated trials (p50 / p95 / mean + retry counts).
    2. Streaming SSE Time-to-First-Byte (`/api/v1/tts/sse`) across representative
       prompts (start event latency, first audio chunk TTFB, first non-silent PCM
       chunk TTFB, and total stream time).
    3. Audio integrity — 25 ms frame-level RMS silence ratio, clipping ratio,
       spectral balance, and speaking rate (WPM).
    4. Round-trip Prisma ASR intelligibility — scored strictly against the raw TTS
       input `text` using Verbatim WER/CER and dataset-agnostic `compute_rule_wer` /
       `compute_rule_cer` (no per-sentence output-tuned `roundtrip_reference`).
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, TTS_MODEL
from src.audio_utils import compute_spectral_features, get_audio_duration, load_wav
from src.gnani_client import GnaniAPIError, GnaniClient
from src.metrics import (
    compute_bootstrap_ci,
    compute_cer,
    compute_percentile,
    compute_rtf,
    compute_rule_cer,
    compute_rule_wer,
    compute_silence_ratio,
    compute_speaking_rate,
    compute_wer,
)


def load_tts_test_set(test_set_path: str) -> List[Dict]:
    """Load TTS test sentences from JSON."""
    with open(test_set_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _transcribe_independent_whisper(audio_path: str, language: str, model=None) -> Optional[Dict]:
    """Optional cross-check with faster-whisper if installed (e.g. on Colab / Python 3.10+)."""
    if model is None:
        return None
    lang_code = language.split("-")[0] if language not in ("hi-en", "indic-auto") else None
    segments, _info = model.transcribe(audio_path, language=lang_code)
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    return {"whisper_transcript": transcript}


def evaluate_single_tts_trial(
    client: GnaniClient,
    entry: Dict,
    output_path: str,
    run_roundtrip_asr: bool = True,
    whisper_model=None,
) -> Dict:
    """Run one TTS synthesis + acoustic analysis + round-trip ASR trial."""
    text = entry["text"]
    language = entry.get("language", "en-IN")
    voice = entry.get("voice")
    clip_id = entry.get("id", "unknown")

    try:
        tts_result = client.synthesize(
            text=text,
            language=language,
            voice=voice,
            output_path=output_path,
        )
        synth_latency_ms = tts_result["latency_ms"]
        total_elapsed_ms = tts_result.get("total_elapsed_ms", synth_latency_ms)
        synth_retries = tts_result.get("retries", 0)
        voice_used = tts_result["voice"]
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
        rtf_with_retries = compute_rtf(audio_duration, total_elapsed_ms / 1000.0)
        word_count = len(text.split())
        speaking_rate = compute_speaking_rate(word_count, audio_duration)
        silence_ratio = compute_silence_ratio(samples, sample_rate)
        spectral = compute_spectral_features(samples, sample_rate)

        import numpy as np
        clipping_ratio = float(np.sum(np.abs(samples) > 0.99) / len(samples))

        audio_analysis = {
            "audio_duration_sec": round(audio_duration, 3),
            "sample_rate": sample_rate,
            "rtf": round(rtf, 4),
            "rtf_with_retries": round(rtf_with_retries, 4),
            "speaking_rate_wpm": round(speaking_rate, 1),
            "silence_ratio": round(silence_ratio, 4),
            "clipping_ratio": round(clipping_ratio, 6),
            "spectral": {k: round(v, 2) for k, v in spectral.items()},
        }
    except Exception as e:
        audio_analysis = {"error": str(e)}

    # ── Round-trip ASR Check (strictly against raw input text) ─────
    roundtrip_result = None
    if run_roundtrip_asr and status == "success":
        try:
            asr_lang = "hi-IN,en-IN" if language == "hi-en" else language
            asr_result = client.transcribe(audio_path=output_path, language=asr_lang)
            rt_transcript = asr_result["transcript"]
            lang_hint = "hi" if language == "hi-en" else language[:2]

            wer_v = compute_wer(text, rt_transcript, language=lang_hint)
            cer_v = compute_cer(text, rt_transcript, language=lang_hint)
            wer_r = compute_rule_wer(text, rt_transcript, language=lang_hint)
            cer_r = compute_rule_cer(text, rt_transcript, language=lang_hint)

            roundtrip_result = {
                "asr_language_param": asr_lang,
                "asr_transcript": rt_transcript,
                "verbatim_wer": round(wer_v["wer"], 4),
                "verbatim_cer": round(cer_v["cer"], 4),
                "rule_normalised_wer": round(wer_r["wer"], 4),
                "rule_normalised_cer": round(cer_r["cer"], 4),
                "verbatim_errors": wer_v["substitutions"] + wer_v["deletions"] + wer_v["insertions"],
                "verbatim_ref_words": wer_v["ref_length"],
                "rule_errors": wer_r["substitutions"] + wer_r["deletions"] + wer_r["insertions"],
                "rule_ref_words": wer_r["ref_length"],
                "asr_latency_ms": asr_result["latency_ms"],
                "asr_retries": asr_result.get("retries", 0),
            }

            if whisper_model is not None:
                w_out = _transcribe_independent_whisper(output_path, language, whisper_model)
                if w_out:
                    w_wer = compute_rule_wer(text, w_out["whisper_transcript"], language=lang_hint)
                    roundtrip_result["whisper_transcript"] = w_out["whisper_transcript"]
                    roundtrip_result["whisper_rule_wer"] = round(w_wer["wer"], 4)
        except Exception as e:
            roundtrip_result = {"error": str(e)}

    return {
        "id": clip_id,
        "category": entry.get("category", ""),
        "tags": entry.get("tags", []),
        "text": text,
        "language": language,
        "voice": voice_used,
        "status": status,
        "synth_latency_ms": round(synth_latency_ms, 2),
        "total_elapsed_ms": round(total_elapsed_ms, 2),
        "retries": synth_retries,
        "audio_analysis": audio_analysis,
        "roundtrip_asr": roundtrip_result,
        "output_path": output_path,
        "error": error,
    }


def evaluate_single_tts(
    client: GnaniClient,
    entry: Dict,
    output_dir: str,
    run_roundtrip_asr: bool = True,
    repeats: int = 1,
    whisper_model=None,
) -> Dict:
    """
    Evaluate a single TTS sentence across `repeats` synthesis trials.

    Saves `<id>.wav` on trial 1 (and `<id>_run{2..k}.wav` for repeats) and computes
    p50/p95 latency, RTF, retries, and round-trip ASR WERs across trials.
    """
    clip_id = entry.get("id", "unknown")
    base_out = str(Path(output_dir) / f"{clip_id}.wav")

    if repeats <= 1:
        return evaluate_single_tts_trial(
            client, entry, base_out, run_roundtrip_asr=run_roundtrip_asr, whisper_model=whisper_model
        )

    trials = []
    for rep in range(repeats):
        trial_out = base_out if rep == 0 else str(Path(output_dir) / f"{clip_id}_run{rep + 1}.wav")
        trial = evaluate_single_tts_trial(
            client, entry, trial_out, run_roundtrip_asr=run_roundtrip_asr, whisper_model=whisper_model
        )
        trials.append(trial)
        if trial["status"] != "success" and rep == 0 and entry.get("text", "").strip(".") == "":
            break
        if rep < repeats - 1:
            time.sleep(0.35)

    ok_trials = [t for t in trials if t["status"] == "success"]
    if not ok_trials:
        failed = dict(trials[0])
        failed["trials"] = trials
        failed["repeats_attempted"] = len(trials)
        return failed

    rep_trial = dict(ok_trials[0])
    lats = [t["synth_latency_ms"] for t in ok_trials]
    elapsed = [t["total_elapsed_ms"] for t in ok_trials]
    rtfs = [t["audio_analysis"]["rtf"] for t in ok_trials if "rtf" in t.get("audio_analysis", {})]
    rt_v_wers = [
        t["roundtrip_asr"]["verbatim_wer"]
        for t in ok_trials
        if t.get("roundtrip_asr") and "verbatim_wer" in t["roundtrip_asr"]
    ]
    rt_r_wers = [
        t["roundtrip_asr"]["rule_normalised_wer"]
        for t in ok_trials
        if t.get("roundtrip_asr") and "rule_normalised_wer" in t["roundtrip_asr"]
    ]
    rt_transcripts = [
        t["roundtrip_asr"]["asr_transcript"]
        for t in ok_trials
        if t.get("roundtrip_asr") and "asr_transcript" in t["roundtrip_asr"]
    ]

    rep_trial["repeats_summary"] = {
        "repeats_requested": repeats,
        "repeats_succeeded": len(ok_trials),
        "p50_latency_ms": round(compute_percentile(lats, 50), 2),
        "p95_latency_ms": round(compute_percentile(lats, 95), 2),
        "mean_latency_ms": round(sum(lats) / len(lats), 2),
        "p95_total_elapsed_ms": round(compute_percentile(elapsed, 95), 2),
        "p50_rtf": round(compute_percentile(rtfs, 50), 4) if rtfs else None,
        "p95_rtf": round(compute_percentile(rtfs, 95), 4) if rtfs else None,
        "total_retries": sum(t.get("retries", 0) for t in ok_trials),
        "mean_roundtrip_verbatim_wer": round(sum(rt_v_wers) / len(rt_v_wers), 4) if rt_v_wers else None,
        "mean_roundtrip_rule_wer": round(sum(rt_r_wers) / len(rt_r_wers), 4) if rt_r_wers else None,
        "all_roundtrip_transcripts": rt_transcripts,
    }
    rep_trial["synth_latency_ms"] = rep_trial["repeats_summary"]["p50_latency_ms"]
    rep_trial["retries"] = rep_trial["repeats_summary"]["total_retries"]
    if rep_trial["repeats_summary"]["p50_rtf"] is not None:
        rep_trial["audio_analysis"]["rtf"] = rep_trial["repeats_summary"]["p50_rtf"]
    rep_trial["trials"] = [
        {
            "run": idx + 1,
            "status": t["status"],
            "synth_latency_ms": t.get("synth_latency_ms"),
            "total_elapsed_ms": t.get("total_elapsed_ms"),
            "retries": t.get("retries", 0),
            "rtf": t.get("audio_analysis", {}).get("rtf"),
            "roundtrip_asr_transcript": (t.get("roundtrip_asr") or {}).get("asr_transcript"),
            "roundtrip_verbatim_wer": (t.get("roundtrip_asr") or {}).get("verbatim_wer"),
            "roundtrip_rule_wer": (t.get("roundtrip_asr") or {}).get("rule_normalised_wer"),
        }
        for idx, t in enumerate(trials)
    ]
    return rep_trial


def run_tts_streaming_benchmark(
    client: GnaniClient,
    test_set: List[Dict],
    sentence_ids: tuple = ("num_en_001", "abbr_en_002", "codemix_001", "long_en_001", "long_hi_001"),
    repeats: int = 3,
    verbose: bool = True,
) -> Dict:
    """Measure true streaming Time-to-First-Byte via `/api/v1/tts/sse`."""
    by_id = {e["id"]: e for e in test_set}
    selected = [by_id[sid] for sid in sentence_ids if sid in by_id]
    if not selected:
        return {}

    if verbose:
        print(f"{'='*72}")
        print(f"  TIMBRE STREAMING SSE TTFB BENCHMARK ({len(selected)} sentences × {repeats} repeats)")
        print(f"{'='*72}\n")

    per_sentence = []
    all_first_chunk = []
    all_first_nonsilent = []
    all_start_events = []
    all_totals = []

    for entry in selected:
        trials = []
        for rep in range(repeats):
            try:
                res = client.synthesize_stream(
                    text=entry["text"],
                    language=entry.get("language", "en-IN"),
                    voice=entry.get("voice"),
                )
                trials.append({
                    "run": rep + 1,
                    "status": "success",
                    "ttfb_start_event_ms": res["ttfb_start_event_ms"],
                    "ttfb_first_chunk_ms": res["ttfb_first_chunk_ms"],
                    "ttfb_first_nonsilent_ms": res["ttfb_first_nonsilent_ms"],
                    "total_stream_ms": res["total_stream_ms"],
                    "num_chunks": res["num_chunks"],
                    "audio_size_bytes": res["audio_size_bytes"],
                })
                if res["ttfb_start_event_ms"] is not None:
                    all_start_events.append(res["ttfb_start_event_ms"])
                if res["ttfb_first_chunk_ms"] is not None:
                    all_first_chunk.append(res["ttfb_first_chunk_ms"])
                if res["ttfb_first_nonsilent_ms"] is not None:
                    all_first_nonsilent.append(res["ttfb_first_nonsilent_ms"])
                all_totals.append(res["total_stream_ms"])
            except Exception as exc:
                trials.append({"run": rep + 1, "status": "error", "error": str(exc)})
            if rep < repeats - 1:
                time.sleep(0.3)

        ok = [t for t in trials if t["status"] == "success"]
        fc = [t["ttfb_first_chunk_ms"] for t in ok if t["ttfb_first_chunk_ms"] is not None]
        fn = [t["ttfb_first_nonsilent_ms"] for t in ok if t["ttfb_first_nonsilent_ms"] is not None]
        tot = [t["total_stream_ms"] for t in ok]
        per_sentence.append({
            "id": entry["id"],
            "language": entry.get("language"),
            "p50_ttfb_first_chunk_ms": round(compute_percentile(fc, 50), 2) if fc else None,
            "p95_ttfb_first_chunk_ms": round(compute_percentile(fc, 95), 2) if fc else None,
            "p50_ttfb_first_nonsilent_ms": round(compute_percentile(fn, 50), 2) if fn else None,
            "p50_total_stream_ms": round(compute_percentile(tot, 50), 2) if tot else None,
            "trials": trials,
        })
        if verbose and fc:
            print(
                f"  {entry['id']:<14s} ({entry['language']:<5s})  "
                f"TTFB chunk p50={compute_percentile(fc, 50):.0f}ms  "
                f"non-silent p50={compute_percentile(fn, 50):.0f}ms  "
                f"total stream p50={compute_percentile(tot, 50):.0f}ms"
            )

    summary = {
        "endpoint": "/api/v1/tts/sse",
        "num_sentences": len(selected),
        "repeats_per_sentence": repeats,
        "total_calls_succeeded": len(all_totals),
        "p50_start_event_ms": round(compute_percentile(all_start_events, 50), 2) if all_start_events else None,
        "p50_ttfb_first_chunk_ms": round(compute_percentile(all_first_chunk, 50), 2) if all_first_chunk else None,
        "p95_ttfb_first_chunk_ms": round(compute_percentile(all_first_chunk, 95), 2) if all_first_chunk else None,
        "p50_ttfb_first_nonsilent_ms": round(compute_percentile(all_first_nonsilent, 50), 2) if all_first_nonsilent else None,
        "p95_ttfb_first_nonsilent_ms": round(compute_percentile(all_first_nonsilent, 95), 2) if all_first_nonsilent else None,
        "p50_total_stream_ms": round(compute_percentile(all_totals, 50), 2) if all_totals else None,
        "p95_total_stream_ms": round(compute_percentile(all_totals, 95), 2) if all_totals else None,
        "per_sentence": per_sentence,
    }
    if verbose and all_first_chunk:
        print(
            f"\n  SSE Overall: TTFB chunk p50={summary['p50_ttfb_first_chunk_ms']:.0f}ms "
            f"(p95={summary['p95_ttfb_first_chunk_ms']:.0f}ms) | "
            f"non-silent p50={summary['p50_ttfb_first_nonsilent_ms']:.0f}ms | "
            f"total stream p50={summary['p50_total_stream_ms']:.0f}ms\n"
        )
    return summary


def _summarise_tts_group(items: List[Dict]) -> Dict:
    """Compute component-level aggregates (p50/p95 latency, RTF, silence, round-trip WER + CIs)."""
    analyses = [r["audio_analysis"] for r in items if "error" not in r.get("audio_analysis", {})]
    lats = [r["synth_latency_ms"] for r in items]
    elapsed = [r.get("total_elapsed_ms", r["synth_latency_ms"]) for r in items]
    rtfs = [a.get("rtf", 0.0) for a in analyses]
    wpms = [a.get("speaking_rate_wpm", 0.0) for a in analyses]
    sils = [a.get("silence_ratio", 0.0) for a in analyses]
    clips = [a.get("clipping_ratio", 0.0) for a in analyses]

    rt_items = [
        r["roundtrip_asr"]
        for r in items
        if r.get("roundtrip_asr") and "rule_normalised_wer" in r["roundtrip_asr"]
    ]
    avg_rt_v_wer = (
        round(sum(x["verbatim_wer"] for x in rt_items) / len(rt_items), 4)
        if rt_items else None
    )
    avg_rt_r_wer = (
        round(sum(x["rule_normalised_wer"] for x in rt_items) / len(rt_items), 4)
        if rt_items else None
    )
    rt_rule_ci = (
        compute_bootstrap_ci([{"errors": x["rule_errors"], "ref_length": x["rule_ref_words"]} for x in rt_items])
        if rt_items else None
    )

    return {
        "num_sentences": len(items),
        "p50_latency_ms": round(compute_percentile(lats, 50), 2),
        "p95_latency_ms": round(compute_percentile(lats, 95), 2),
        "avg_latency_ms": round(sum(lats) / max(len(lats), 1), 2),
        "p95_total_elapsed_ms": round(compute_percentile(elapsed, 95), 2),
        "p50_rtf": round(compute_percentile(rtfs, 50), 4),
        "p95_rtf": round(compute_percentile(rtfs, 95), 4),
        "avg_rtf": round(sum(rtfs) / max(len(rtfs), 1), 4),
        "avg_speaking_rate_wpm": round(sum(wpms) / max(len(wpms), 1), 1),
        "avg_silence_ratio": round(sum(sils) / max(len(sils), 1), 4),
        "max_clipping_ratio": round(max(clips) if clips else 0.0, 6),
        "avg_roundtrip_verbatim_wer": avg_rt_v_wer,
        "avg_roundtrip_rule_wer": avg_rt_r_wer,
        "roundtrip_rule_wer_ci_95": [rt_rule_ci["ci_low"], rt_rule_ci["ci_high"]] if rt_rule_ci else None,
        "total_retries": sum(r.get("retries", 0) for r in items),
    }


def run_tts_evaluation(
    test_set_path: str,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    run_roundtrip_asr: bool = True,
    repeats: int = 1,
    run_streaming_benchmark: bool = True,
    streaming_repeats: int = 3,
    use_independent_asr: bool = False,
    verbose: bool = True,
) -> Dict:
    """Run the full TTS evaluation pipeline (REST + optional SSE streaming benchmark)."""
    client = GnaniClient(api_key=api_key)
    test_set = load_tts_test_set(test_set_path)

    if output_dir is None:
        output_dir = str(Path(test_set_path).parent / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    whisper_model = None
    independent_asr_status = "disabled"
    if use_independent_asr:
        try:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            independent_asr_status = "enabled (faster-whisper base)"
        except ImportError:
            independent_asr_status = "unavailable (faster-whisper not installed)"

    if verbose:
        print(f"{'='*72}")
        print(f"  TIMBRE TTS EVALUATION")
        print(f"  Sentences: {len(test_set)}  |  Repeats: {repeats}  |  Model: {TTS_MODEL}")
        print(f"  Round-trip ASR: {'enabled' if run_roundtrip_asr else 'disabled'}  |  Independent ASR: {independent_asr_status}")
        print(f"{'='*72}\n")

    results = []
    for i, entry in enumerate(test_set):
        if verbose:
            print(
                f"  [{i+1}/{len(test_set)}] {entry.get('id', '?')} "
                f"({entry.get('category', '?')}, {entry.get('language', 'en-IN')}) ... ",
                end="",
                flush=True,
            )

        result = evaluate_single_tts(
            client,
            entry,
            output_dir,
            run_roundtrip_asr=run_roundtrip_asr,
            repeats=repeats,
            whisper_model=whisper_model,
        )
        results.append(result)

        if verbose:
            if result["status"] == "success":
                qa = result.get("audio_analysis", {})
                rt = result.get("roundtrip_asr") or {}
                rt_str = f"RT-RuleWER={rt['rule_normalised_wer']:.1%}" if "rule_normalised_wer" in rt else ""
                retry_tag = f" [retries={result['retries']}]" if result.get("retries") else ""
                print(
                    f"voice={result['voice']}  RTF={qa.get('rtf', '?')}  "
                    f"sil={qa.get('silence_ratio', '?')}  {rt_str}  "
                    f"({result.get('synth_latency_ms', 0):.0f}ms){retry_tag}"
                )
            else:
                print(f"ERROR: {result.get('error', 'unknown')}")

        time.sleep(0.35)

    streaming_bench = {}
    if run_streaming_benchmark:
        streaming_bench = run_tts_streaming_benchmark(
            client, test_set, repeats=streaming_repeats, verbose=verbose
        )

    successful = [r for r in results if r["status"] == "success"]
    categories: Dict[str, List[Dict]] = {}
    for r in successful:
        categories.setdefault(r["category"], []).append(r)

    category_summary = {cat: _summarise_tts_group(items) for cat, items in categories.items()}
    overall_summary = _summarise_tts_group(successful)

    raw_lats = []
    for r in successful:
        if "trials" in r:
            raw_lats.extend(t["synth_latency_ms"] for t in r["trials"] if t["status"] == "success")
        else:
            raw_lats.append(r["synth_latency_ms"])
    overall_summary["min_latency_ms"] = round(min(raw_lats) if raw_lats else 0.0, 2)
    overall_summary["max_latency_ms"] = round(max(raw_lats) if raw_lats else 0.0, 2)

    evaluation = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": TTS_MODEL,
            "latency_type": "full_response_http_ms (REST inference) + streaming_ttfb_ms (SSE)",
            "total_sentences": len(test_set),
            "successful": len(successful),
            "failed": len(test_set) - len(successful),
            "repeats_per_sentence": repeats,
            "roundtrip_asr_enabled": run_roundtrip_asr,
            "independent_asr_status": independent_asr_status,
            "roundtrip_normaliser": "rule-based dataset-agnostic normalise_for_eval (against raw text)",
        },
        "overall": overall_summary,
        "by_category": category_summary,
        "streaming_ttfb_benchmark": streaming_bench,
        "per_sentence_results": results,
    }

    if verbose:
        ov = evaluation["overall"]
        print(f"\n{'='*72}")
        print(f"  RESULTS SUMMARY (Component Metrics)")
        print(f"{'='*72}")
        print(
            f"  REST HTTP Latency : p50={ov['p50_latency_ms']:.0f}ms  "
            f"p95={ov['p95_latency_ms']:.0f}ms  mean={ov['avg_latency_ms']:.0f}ms"
        )
        print(
            f"  RTF               : p50={ov['p50_rtf']:.4f}  p95={ov['p95_rtf']:.4f}  "
            f"mean={ov['avg_rtf']:.4f}"
        )
        print(
            f"  Silence Ratio     : {ov['avg_silence_ratio']:.3f} (25ms frame RMS)  |  "
            f"Max Clipping: {ov['max_clipping_ratio']:.4f}"
        )
        if ov["avg_roundtrip_rule_wer"] is not None:
            print(
                f"  Round-trip WER    : verbatim={ov['avg_roundtrip_verbatim_wer']:.2%}  "
                f"rule-normalised={ov['avg_roundtrip_rule_wer']:.2%}"
            )
        print(f"\n  By Category:")
        for cat, stats in category_summary.items():
            rt_r = stats["avg_roundtrip_rule_wer"]
            rt_disp = f"{rt_r:.1%}" if rt_r is not None else "n/a"
            print(
                f"    {cat:25s}  p50={stats['p50_latency_ms']:>5.0f}ms  "
                f"p50-RTF={stats['p50_rtf']:.3f}  RT-RuleWER={rt_disp:>6s}  (n={stats['num_sentences']})"
            )
        print(f"{'='*72}\n")

    results_path = output_path or str(RESULTS_DIR / "tts_results.json")
    if len(successful) == 0:
        if verbose:
            print(f"  Skipping save to {results_path} because 0 sentences succeeded.")
    else:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        save_results = json.loads(json.dumps(evaluation, default=str))
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  Results saved to: {results_path}")

    return evaluation
