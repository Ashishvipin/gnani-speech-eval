"""
Speech evaluation metrics — implemented from scratch.

WER (Word Error Rate) and CER (Character Error Rate) using the
Levenshtein edit-distance algorithm. No external metric libraries
are used for the core calculations; this is intentional per the
assignment requirements.
"""

from typing import List, Tuple, Dict
import re
import unicodedata


# ════════════════════════════════════════════════════════════════════
#  TEXT NORMALISATION
# ════════════════════════════════════════════════════════════════════

def normalise_text(text: str, language: str = "en") -> str:
    """
    Normalise a transcript for fair comparison.

    Steps:
        1. Unicode NFC normalisation (important for Devanagari / Indic scripts)
        2. Lower-case (for Latin script; Indic scripts are case-less)
        3. Remove punctuation & symbols (including Devanagari danda ।/॥)
           while keeping letters, combining marks (matras), digits, and spaces
        4. Collapse multiple spaces to one
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Keep Unicode letters (L*), combining marks/matras (M*), numbers (N*), and whitespace.
    # This cleanly strips Latin and Indic punctuation (e.g., '.', ',', '।', '॥', '_').
    cleaned = [
        ch if (ch.isspace() or unicodedata.category(ch)[0] in ("L", "M", "N")) else ""
        for ch in text
    ]
    text = "".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ════════════════════════════════════════════════════════════════════
#  LEVENSHTEIN EDIT DISTANCE (DYNAMIC PROGRAMMING)
# ════════════════════════════════════════════════════════════════════

def _levenshtein_matrix(ref: List[str], hyp: List[str]) -> List[List[int]]:
    """
    Compute the full Levenshtein distance matrix between two token lists.

    Uses the standard Wagner-Fischer DP algorithm:
        - dp[i][j] = min edits to transform ref[:i] → hyp[:j]
        - Substitution, insertion, deletion each cost 1

    Returns the full matrix so we can back-trace to get S, I, D counts.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases: transforming from/to empty sequence
    for i in range(n + 1):
        dp[i][0] = i        # delete all ref tokens
    for j in range(m + 1):
        dp[0][j] = j        # insert all hyp tokens

    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]           # match — no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],   # substitution
                    dp[i - 1][j],        # deletion  (ref token has no match)
                    dp[i][j - 1],        # insertion (extra token in hyp)
                )
    return dp


def _backtrace_errors(
    dp: List[List[int]], ref: List[str], hyp: List[str]
) -> Tuple[int, int, int, int]:
    """
    Walk backwards through the DP matrix to count individual
    Substitutions (S), Deletions (D), Insertions (I), and Hits (H).
    """
    i, j = len(ref), len(hyp)
    S = D = I = H = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            H += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1      # substitution
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1      # deletion
            i -= 1
        else:
            I += 1      # insertion
            j -= 1

    return S, D, I, H


# ════════════════════════════════════════════════════════════════════
#  WER — WORD ERROR RATE
# ════════════════════════════════════════════════════════════════════

def compute_wer(reference: str, hypothesis: str, language: str = "en") -> Dict:
    """
    Compute Word Error Rate between a reference and hypothesis transcript.

    WER = (S + D + I) / N

    where:
        S = substitutions, D = deletions, I = insertions
        N = total words in the reference

    Args:
        reference:  Ground-truth transcript
        hypothesis: ASR-predicted transcript
        language:   Language hint for normalisation

    Returns:
        Dictionary with wer, substitutions, deletions, insertions,
        hits, ref_length, and the normalised texts.
    """
    ref_norm = normalise_text(reference, language)
    hyp_norm = normalise_text(hypothesis, language)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    # Edge case: empty reference
    if len(ref_words) == 0:
        return {
            "wer": 0.0 if len(hyp_words) == 0 else float("inf"),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
            "hits": 0,
            "ref_length": 0,
            "ref_normalised": ref_norm,
            "hyp_normalised": hyp_norm,
        }

    dp = _levenshtein_matrix(ref_words, hyp_words)
    S, D, I, H = _backtrace_errors(dp, ref_words, hyp_words)
    N = len(ref_words)

    return {
        "wer": (S + D + I) / N,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
        "hits": H,
        "ref_length": N,
        "ref_normalised": ref_norm,
        "hyp_normalised": hyp_norm,
    }


# ════════════════════════════════════════════════════════════════════
#  CER — CHARACTER ERROR RATE
# ════════════════════════════════════════════════════════════════════

def compute_cer(reference: str, hypothesis: str, language: str = "en") -> Dict:
    """
    Compute Character Error Rate between a reference and hypothesis.

    CER = (S + D + I) / N

    Same edit-distance approach as WER but at the character level.
    Spaces are removed before comparison — we measure purely
    character-level accuracy.
    """
    ref_norm = normalise_text(reference, language)
    hyp_norm = normalise_text(hypothesis, language)

    # Remove spaces — CER operates on character sequences
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    if len(ref_chars) == 0:
        return {
            "cer": 0.0 if len(hyp_chars) == 0 else float("inf"),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_chars),
            "hits": 0,
            "ref_length": 0,
        }

    dp = _levenshtein_matrix(ref_chars, hyp_chars)
    S, D, I, H = _backtrace_errors(dp, ref_chars, hyp_chars)
    N = len(ref_chars)

    return {
        "cer": (S + D + I) / N,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
        "hits": H,
        "ref_length": N,
    }


# ════════════════════════════════════════════════════════════════════
#  AGGREGATE METRICS
# ════════════════════════════════════════════════════════════════════

def aggregate_wer(results: List[Dict]) -> Dict:
    """
    Compute micro-averaged WER across a list of per-utterance results.

    Micro-average: sum all (S+D+I) / sum all N
    This weights longer utterances proportionally, which is the
    standard practice in ASR benchmarking.
    """
    total_errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in results)
    total_ref = sum(r["ref_length"] for r in results)

    if total_ref == 0:
        return {"aggregate_wer": 0.0, "total_errors": 0, "total_ref_words": 0}

    return {
        "aggregate_wer": total_errors / total_ref,
        "total_errors": total_errors,
        "total_ref_words": total_ref,
        "total_substitutions": sum(r["substitutions"] for r in results),
        "total_deletions": sum(r["deletions"] for r in results),
        "total_insertions": sum(r["insertions"] for r in results),
        "num_utterances": len(results),
    }


def aggregate_cer(results: List[Dict]) -> Dict:
    """Micro-averaged CER across a list of per-utterance CER results."""
    total_errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in results)
    total_ref = sum(r["ref_length"] for r in results)

    if total_ref == 0:
        return {"aggregate_cer": 0.0, "total_errors": 0, "total_ref_chars": 0}

    return {
        "aggregate_cer": total_errors / total_ref,
        "total_errors": total_errors,
        "total_ref_chars": total_ref,
        "num_utterances": len(results),
    }


# ════════════════════════════════════════════════════════════════════
#  TTS METRICS
# ════════════════════════════════════════════════════════════════════

def compute_rtf(audio_duration_sec: float, processing_time_sec: float) -> float:
    """
    Real-Time Factor = processing_time / audio_duration.

    RTF < 1.0 means faster than real-time (good for production).
    RTF > 1.0 means slower than real-time (potential latency issue).
    """
    if audio_duration_sec <= 0:
        return float("inf")
    return processing_time_sec / audio_duration_sec


def compute_speaking_rate(num_words: int, audio_duration_sec: float) -> float:
    """Words per minute — a sanity check for natural pacing."""
    if audio_duration_sec <= 0:
        return 0.0
    return (num_words / audio_duration_sec) * 60


def compute_percentile(values: List[float], p: float) -> float:
    """Compute linear-interpolated p-th percentile (0-100) without external deps."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_silence_ratio(
    audio_samples,
    sample_rate: int,
    silence_threshold: float = 0.015,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> float:
    """
    Fraction of 25 ms overlapping frames whose RMS energy falls below threshold.

    Frame-level RMS avoids miscounting voiced-speech zero crossings or quiet
    waveform troughs as silence (unlike raw sample-amplitude thresholding).
    """
    import numpy as np

    samples = np.asarray(audio_samples, dtype=np.float64)
    if len(samples) == 0 or sample_rate <= 0:
        return 0.0

    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))

    if len(samples) < frame_len:
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return 1.0 if rms < silence_threshold else 0.0

    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop_len]
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return float(np.mean(frame_rms < silence_threshold))


# ════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Sanity checks — run with:  python -m src.metrics
    ref = "the cat sat on the mat"
    hyp = "the cat sit on a mat"
    wer_result = compute_wer(ref, hyp)
    print(f"Reference : {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"WER       : {wer_result['wer']:.2%}")
    print(f"  S={wer_result['substitutions']}  D={wer_result['deletions']}  "
          f"I={wer_result['insertions']}  H={wer_result['hits']}")
    print()

    cer_result = compute_cer(ref, hyp)
    print(f"CER       : {cer_result['cer']:.2%}")
    print()

    # Hindi example
    ref_hi = "नमस्ते आप कैसे हैं"
    hyp_hi = "नमस्ते आप कैसे है"
    wer_hi = compute_wer(ref_hi, hyp_hi, language="hi")
    print(f"Hindi WER : {wer_hi['wer']:.2%}")
