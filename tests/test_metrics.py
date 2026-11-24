"""
Unit tests for the metrics module.

Verifies correctness of WER, CER, and aggregation functions
against known test cases.

Run with:
    python -m pytest tests/test_metrics.py -v
    # or simply:
    python tests/test_metrics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    compute_wer, compute_cer, aggregate_wer, aggregate_cer,
    normalise_text, compute_rtf, compute_speaking_rate,
)


def test_perfect_match():
    """Identical reference and hypothesis should give 0% WER and CER."""
    result = compute_wer("hello world", "hello world")
    assert result["wer"] == 0.0
    assert result["substitutions"] == 0
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["hits"] == 2

    cer = compute_cer("hello world", "hello world")
    assert cer["cer"] == 0.0


def test_complete_mismatch():
    """Entirely wrong hypothesis: WER should be 1.0 (all substitutions)."""
    result = compute_wer("the cat sat", "big dog ran")
    assert result["wer"] == 1.0  # 3 substitutions / 3 words
    assert result["substitutions"] == 3


def test_known_wer():
    """
    Reference:  the cat sat on the mat  (6 words)
    Hypothesis: the cat sit on a mat
    Changes:    sat→sit (sub), the→a (sub) = 2 errors
    WER = 2/6 = 0.3333
    """
    result = compute_wer("the cat sat on the mat", "the cat sit on a mat")
    assert abs(result["wer"] - 2 / 6) < 1e-6


def test_deletion():
    """Missing words in hypothesis = deletions."""
    result = compute_wer("one two three four", "one three four")
    assert result["deletions"] >= 1
    assert result["wer"] == 1 / 4  # one deletion out of 4 ref words


def test_insertion():
    """Extra words in hypothesis = insertions."""
    result = compute_wer("one two three", "one two extra three")
    assert result["insertions"] >= 1


def test_empty_reference():
    """Empty reference with non-empty hypothesis should return inf WER."""
    result = compute_wer("", "some words here")
    assert result["wer"] == float("inf")


def test_empty_both():
    """Both empty should give 0% WER."""
    result = compute_wer("", "")
    assert result["wer"] == 0.0


def test_cer_character_level():
    """CER operates at character level, not word level."""
    result = compute_cer("cat", "car")
    # 'cat' vs 'car': 1 substitution (t→r), 2 hits
    assert result["substitutions"] == 1
    assert result["cer"] == 1 / 3


def test_hindi_wer():
    """WER should work with Hindi (Devanagari) text."""
    ref = "नमस्ते आप कैसे हैं"
    hyp = "नमस्ते आप कैसे है"  # हैं → है (1 sub)
    result = compute_wer(ref, hyp, language="hi")
    # After proper normalisation preserving Devanagari matras,
    # हैं and है should be different words → 1 substitution
    assert result["wer"] == 1 / 4, f"Expected 0.25, got {result['wer']}"


def test_normalisation_punctuation():
    """Punctuation should be stripped before comparison."""
    ref = "Hello, world!"
    hyp = "hello world"
    result = compute_wer(ref, hyp)
    assert result["wer"] == 0.0  # Punctuation and case shouldn't matter


def test_normalisation_extra_spaces():
    """Extra whitespace should be normalised."""
    ref = "  hello   world  "
    hyp = "hello world"
    result = compute_wer(ref, hyp)
    assert result["wer"] == 0.0


def test_aggregate_wer():
    """Micro-averaged WER should weight by utterance length."""
    results = [
        {"substitutions": 1, "deletions": 0, "insertions": 0, "ref_length": 10},
        {"substitutions": 0, "deletions": 1, "insertions": 0, "ref_length": 5},
    ]
    agg = aggregate_wer(results)
    # Total errors = 2, total ref = 15
    assert abs(agg["aggregate_wer"] - 2 / 15) < 1e-6


def test_rtf_calculation():
    """RTF = processing_time / audio_duration."""
    assert compute_rtf(10.0, 5.0) == 0.5   # faster than real-time
    assert compute_rtf(10.0, 10.0) == 1.0   # exactly real-time
    assert compute_rtf(10.0, 20.0) == 2.0   # slower than real-time


def test_speaking_rate():
    """Speaking rate should be in words per minute."""
    # 60 words in 60 seconds = 60 WPM
    assert compute_speaking_rate(60, 60.0) == 60.0
    # 150 words in 60 seconds = 150 WPM
    assert compute_speaking_rate(150, 60.0) == 150.0


if __name__ == "__main__":
    # Run all test functions
    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0

    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n  {passed} passed, {failed} failed out of {passed + failed} tests")
    sys.exit(1 if failed > 0 else 0)
