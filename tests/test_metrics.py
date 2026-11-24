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


def test_hindi_danda_punctuation():
    """Devanagari danda (।) and double danda (॥) should be stripped as punctuation."""
    ref = "कृपया मेरा आधार नंबर से लिंक करें"
    hyp = "कृपया मेरा आधार नंबर से लिंक करें।"
    result = compute_wer(ref, hyp, language="hi")
    assert result["wer"] == 0.0, f"Expected 0.0, got {result}"
    assert compute_wer("नमस्ते", "नमस्ते॥", language="hi")["wer"] == 0.0


def test_bengali_danda_punctuation():
    """Bengali sentences ending with danda (।) should normalize cleanly."""
    ref = "আমার একাউন্ট ব্যালেন্স জানতে চাই"
    hyp = "আমার একাউন্ট ব্যালেন্স জানতে চাই।"
    result = compute_wer(ref, hyp, language="bn")
    assert result["wer"] == 0.0, f"Expected 0.0, got {result}"


def test_telephony_audio_simulation():
    """Telephony downsampling + mu-law codec should produce finite 8 kHz audio."""
    import numpy as np
    from src.audio_utils import simulate_telephony, apply_mulaw_codec

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 1000 * t)
    telephony = simulate_telephony(signal, sr, seed=42, use_mulaw=True)
    telephony_repeat = simulate_telephony(signal, sr, seed=42, use_mulaw=True)
    assert len(telephony) == 8000
    assert np.all(np.isfinite(telephony))
    assert np.allclose(telephony, telephony_repeat)  # deterministic with seed
    companded = apply_mulaw_codec(signal)
    assert np.max(np.abs(companded - signal)) < 0.02


def test_frame_silence_ratio_and_percentiles():
    """25 ms frame-level RMS silence ratio should ignore voiced zero-crossings."""
    import numpy as np
    from src.metrics import compute_silence_ratio, compute_percentile

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    voiced = 0.5 * np.sin(2 * np.pi * 220 * t)
    assert compute_silence_ratio(voiced, sr) == 0.0

    half_silent = np.concatenate([np.zeros(sr), voiced])
    sil_ratio = compute_silence_ratio(half_silent, sr)
    assert 0.45 <= sil_ratio <= 0.55, f"Expected ~0.50, got {sil_ratio}"

    vals = [100.0, 200.0, 300.0, 400.0, 1000.0]
    assert compute_percentile(vals, 50) == 300.0
    assert compute_percentile(vals, 95) > 800.0


def test_gnani_client_retry_behavior():
    """GnaniClient should retry transient 500s, track retries, and fail fast on empty-audio 500."""
    from unittest.mock import MagicMock, patch
    from src.gnani_client import GnaniClient, GnaniAPIError

    client = GnaniClient(api_key="test-key", max_retries=2, backoff_sec=0.0)

    transient_resp = MagicMock(status_code=500, text='{"message":"We are facing technical difficulties"}')
    ok_resp = MagicMock(status_code=200, content=b"RIFFdata", headers={})
    empty_resp = MagicMock(status_code=500, text='{"message":"No audio generated during synthesis"}')

    with patch.object(client._session, "post", side_effect=[transient_resp, ok_resp]) as mock_post:
        res = client.synthesize("Hello", language="en-IN")
        assert res["audio_bytes"] == b"RIFFdata"
        assert res["retries"] == 1
        assert "total_elapsed_ms" in res
        assert mock_post.call_count == 2

    with patch.object(client._session, "post", return_value=empty_resp) as mock_post:
        try:
            client.synthesize("...", language="en-IN")
            assert False, "Expected GnaniAPIError"
        except GnaniAPIError as exc:
            assert exc.status_code == 500
            assert mock_post.call_count == 1


def test_format_insensitive_wer_best_of_reference():
    """Format-insensitive WER should accept either verbatim or formatted reference."""
    from unittest.mock import MagicMock
    from src.asr_evaluator import evaluate_single_clip

    mock_client = MagicMock()

    # Case 1: Model outputs spoken word "दस" while reference_formatted uses "10"
    mock_client.transcribe.return_value = {
        "transcript": "मुझे कल सुबह दस बजे की फ्लाइट चाहिए।",
        "latency_ms": 100.0,
        "total_elapsed_ms": 100.0,
        "retries": 0,
        "detected_language": "hi-IN",
    }
    res1 = evaluate_single_clip(
        mock_client,
        {
            "id": "hi_num_word",
            "audio_path": "nonexistent.wav",
            "reference": "मुझे कल सुबह दस बजे की फ्लाइट चाहिए",
            "reference_formatted": "मुझे कल सुबह 10 बजे की फ्लाइट चाहिए",
            "language": "hi-IN",
        },
    )
    assert res1["wer"] == 0.0 and res1["formatted_wer"] == 0.0

    # Case 2: Model outputs ITN digits "5,000" while reference uses words "five thousand"
    mock_client.transcribe.return_value = {
        "transcript": "Please transfer 5,000 rupees.",
        "latency_ms": 100.0,
        "total_elapsed_ms": 100.0,
        "retries": 0,
        "detected_language": "en-IN",
    }
    res2 = evaluate_single_clip(
        mock_client,
        {
            "id": "en_num_digits",
            "audio_path": "nonexistent.wav",
            "reference": "please transfer five thousand rupees",
            "reference_formatted": "please transfer 5,000 rupees",
            "language": "en-IN",
        },
    )
    assert res2["wer"] > 0.0
    assert res2["formatted_wer"] == 0.0


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
