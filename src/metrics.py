"""
Speech evaluation metrics — implemented from scratch.

WER (Word Error Rate) and CER (Character Error Rate) using the
Levenshtein edit-distance algorithm. No external metric libraries
are used for the core calculations; this is intentional per the
assignment requirements.
"""

from typing import List, Tuple, Dict, Optional
import random
import re
import unicodedata


# ════════════════════════════════════════════════════════════════════
#  TEXT NORMALISATION
# ════════════════════════════════════════════════════════════════════

# Standard US/UK spelling & measurement canonicalization table (symmetric, dataset-agnostic)
_CANONICAL_LEXICON: Dict[str, str] = {
    "enquire": "inquire",
    "enquired": "inquired",
    "enquiring": "inquiring",
    "enquiry": "inquiry",
    "notarised": "notarized",
    "normalise": "normalize",
    "normalised": "normalized",
    "organise": "organize",
    "organisation": "organization",
    "recognise": "recognize",
    "recognised": "recognized",
    "customised": "customized",
    "colour": "color",
    "behaviour": "behavior",
    "favour": "favor",
    "centre": "center",
    "metre": "meter",
    "cheque": "check",
    "cancelled": "canceled",
    "travelling": "traveling",
    "programme": "program",
    "paise": "paisa",
    "milligram": "mg",
    "milligrams": "mg",
    "kilogram": "kg",
    "kilograms": "kg",
    "kilometre": "km",
    "kilometres": "km",
    "kilometer": "km",
    "kilometers": "km",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "rupee": "rupees",
    "rs": "rupees",
    "रुपया": "rupees",
    "रुपये": "rupees",
    "रुपए": "rupees",
    "రూపాయలు": "rupees",
    "ரூபாய்": "rupees",
    "টাকা": "rupees",
    "percent": "percent",
    "pct": "percent",
}

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_EN_DIGITS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_EN_UNITS: Dict[str, int] = {
    **_EN_DIGITS,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_ORDINALS: Dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
_EN_SCALES: Dict[str, int] = {
    "hundred": 100,
    "thousand": 1000,
    "lakh": 100000,
    "lakhs": 100000,
    "lac": 100000,
    "lacs": 100000,
    "million": 1000000,
    "crore": 10000000,
    "crores": 10000000,
    "billion": 1000000000,
}

# Standard Hindi number words (0-99 + scales) for algorithmic parsing
_HI_UNITS: Dict[str, int] = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44, "पैंतालीस": 45,
    "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
    "इक्यावन": 51, "बावन": 52, "तिरेपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरेसठ": 63, "चौंसठ": 64, "पैंसठ": 65,
    "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70,
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उन्यासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85,
    "छियासी": 86, "सत्तासी": 87, "अठासी": 88, "नवासी": 89, "नब्बे": 90,
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}
_HI_SCALES: Dict[str, int] = {
    "सौ": 100,
    "हज़ार": 1000,
    "हजार": 1000,
    "लाख": 100000,
    "करोड़": 10000000,
    "करोड": 10000000,
}


def _parse_number_span(tokens: List[str], units: Dict[str, int], scales: Dict[str, int]) -> int:
    """Evaluate a contiguous English or Hindi number-word token list to an integer."""
    total = 0
    current = 0
    for tok in tokens:
        if tok == "and":
            continue
        if tok in units:
            current += units[tok]
        elif tok in scales:
            scale = scales[tok]
            if current == 0:
                current = 1
            if scale >= 1000:
                total += current * scale
                current = 0
            else:
                current *= scale
    return total + current


def _normalise_number_tokens(tokens: List[str]) -> List[str]:
    """Convert contiguous English and Hindi number-word sequences into canonical digits."""
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        # Standalone non-numeric use of "one" (e.g. "no one", "this one") shouldn't become "1"
        if tok == "one" and (i + 1 >= n or (tokens[i + 1] not in _EN_UNITS and tokens[i + 1] not in _EN_SCALES)):
            out.append(tok)
            i += 1
            continue

        # English number phrase / phone-digit sequence
        if tok in _EN_UNITS or tok in _EN_SCALES or tok in _EN_ORDINALS:
            j = i
            while j < n and (
                tokens[j] in _EN_UNITS
                or tokens[j] in _EN_SCALES
                or tokens[j] in _EN_ORDINALS
                or (tokens[j] == "and" and j + 1 < n and (tokens[j + 1] in _EN_UNITS or tokens[j + 1] in _EN_SCALES or tokens[j + 1] in _EN_ORDINALS))
            ):
                j += 1
            while j > i and tokens[j - 1] == "and":
                j -= 1
            span = tokens[i:j]

            # Long runs (>= 4) of single-digit words denote digit-by-digit phone/ID sequences
            if len(span) >= 4 and all(t in _EN_DIGITS for t in span):
                out.append("".join(str(_EN_DIGITS[t]) for t in span))
            elif len(span) == 1 and span[0] in _EN_ORDINALS:
                out.append(str(_EN_ORDINALS[span[0]]))
            else:
                merged_units = {**_EN_UNITS, **_EN_ORDINALS}
                out.append(str(_parse_number_span(span, merged_units, _EN_SCALES)))
            i = j
            continue

        # Hindi number phrase
        if tok in _HI_UNITS or tok in _HI_SCALES:
            j = i
            while j < n and (tokens[j] in _HI_UNITS or tokens[j] in _HI_SCALES):
                j += 1
            span = tokens[i:j]
            out.append(str(_parse_number_span(span, _HI_UNITS, _HI_SCALES)))
            i = j
            continue

        out.append(tok)
        i += 1
    return out


def normalise_text(text: str, language: str = "en") -> str:
    """
    Normalise a transcript strictly for Verbatim WER/CER comparison.

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


def normalise_for_eval(text: str, language: str = "en") -> str:
    """
    Deterministic, dataset-agnostic normalization applied identically to both
    references and hypotheses (never tuned to specific model outputs).

    Rules applied before token comparison:
        1. Unicode NFC, lowercase, Devanagari digits -> ASCII digits.
        2. Standard currency/symbol/unit normalization (₹/Rs/रुपये -> rupees,
           % -> percent, °C -> degrees celsius, ½ -> 0.5, ¾ -> 0.75, 500mg -> 500 mg).
        3. Strip numeric grouping commas (1,23,456 -> 123456) and digit-adjacent hyphens
           (1800-123-4567 -> 18001234567, ABC-1234 -> ABC1234), preserving decimal dots.
        4. Expand dotted titles/company suffixes (`dr.`, `mr.`, `mrs.`, `ltd.`).
        5. Apply symmetric US/UK & unit lexicon + English/Hindi spoken number parsing.
    """
    t = unicodedata.normalize("NFC", text).lower().translate(_DEVANAGARI_DIGITS)

    # Pre-punctuation symbol and fraction expansions
    t = re.sub(r"₹\s*([0-9][0-9,]*(?:\.[0-9]+)?)", r"\1 rupees ", t)
    t = t.replace("₹", " rupees ").replace("°c", " degrees celsius ").replace("℃", " degrees celsius ")
    t = t.replace("%", " percent ").replace("&", " and ")
    t = re.sub(r"(\d)\s*½", r"\1.5", t).replace("½", " 0.5 ")
    t = re.sub(r"(\d)\s*¾", r"\1.75", t).replace("¾", " 0.75 ").replace("3/4", " 0.75 ")
    t = re.sub(r"\bdr\.", " doctor ", t)
    t = re.sub(r"\bmr\.", " mister ", t)
    t = re.sub(r"\bmrs\.", " missus ", t)
    t = re.sub(r"\bltd\.", " limited ", t)
    t = re.sub(r"\brs\.?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", r"\1 rupees ", t)
    t = re.sub(r"\brs\.?", " rupees ", t)
    t = re.sub(r"\bno\.\s*(?=\d)", " number ", t)

    # Numeric commas and digit-adjacent hyphens (e.g. 1800-123-4567, ABC-1234 -> ABC1234)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"(?<=\w)-(?=\d)|(?<=\d)-(?=\w)", "", t)
    # Strip English ordinals on digits (11th -> 11)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", t)
    # Split numbers from attached unit suffixes (500mg -> 500 mg)
    t = re.sub(r"\b(\d+(?:\.\d+)?)(mg|kg|km|ml|gb|tb|am|pm)\b", r"\1 \2", t)

    # Strip non-word punctuation while preserving decimal dots between digits (42.5)
    t = re.sub(r"(?<!\d)\.(?!\d)", " ", t)
    cleaned = [
        ch if (ch == "." or ch.isspace() or unicodedata.category(ch)[0] in ("L", "M", "N")) else " "
        for ch in t
    ]
    tokens = re.sub(r"\s+", " ", "".join(cleaned)).strip().split()
    tokens = [_CANONICAL_LEXICON.get(tok, tok) for tok in tokens]
    tokens = _normalise_number_tokens(tokens)

    # Collapse adjacent duplicate currency tokens (e.g. "₹5,72,300 रुपये" -> "572300 rupees rupees")
    deduped: List[str] = []
    for tok in tokens:
        if tok == "rupees" and deduped and deduped[-1] == "rupees":
            continue
        deduped.append(tok)

    # Normalise trailing ".0" cleanly if integer matches float ("3.0" vs "3")
    deduped = [tok[:-2] if re.fullmatch(r"\d+\.0", tok) else tok for tok in deduped]
    return " ".join(deduped)


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

def compute_wer(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normaliser=None,
) -> Dict:
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
        normaliser: Optional normalisation callable (defaults to normalise_text)

    Returns:
        Dictionary with wer, substitutions, deletions, insertions,
        hits, ref_length, and the normalised texts.
    """
    norm_fn = normaliser or normalise_text
    ref_norm = norm_fn(reference, language)
    hyp_norm = norm_fn(hypothesis, language)

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


def compute_rule_wer(reference: str, hypothesis: str, language: str = "en") -> Dict:
    """Compute WER using deterministic rule-based normalisation (`normalise_for_eval`)."""
    return compute_wer(reference, hypothesis, language=language, normaliser=normalise_for_eval)


# ════════════════════════════════════════════════════════════════════
#  CER — CHARACTER ERROR RATE
# ════════════════════════════════════════════════════════════════════

def compute_cer(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normaliser=None,
) -> Dict:
    """
    Compute Character Error Rate between a reference and hypothesis.

    CER = (S + D + I) / N

    Same edit-distance approach as WER but at the character level.
    Spaces are removed before comparison — we measure purely
    character-level accuracy.
    """
    norm_fn = normaliser or normalise_text
    ref_norm = norm_fn(reference, language)
    hyp_norm = norm_fn(hypothesis, language)

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


def compute_rule_cer(reference: str, hypothesis: str, language: str = "en") -> Dict:
    """Compute CER using deterministic rule-based normalisation (`normalise_for_eval`)."""
    return compute_cer(reference, hypothesis, language=language, normaliser=normalise_for_eval)


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


def compute_bootstrap_ci(
    items: List[Dict],
    errors_key: str = "errors",
    length_key: str = "ref_length",
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute seeded 95% bootstrap confidence intervals for micro-averaged error rates.

    Resamples clip-level `(errors, ref_length)` pairs with replacement across `n_resamples`
    trials and returns exact point estimate + [ci_low, ci_high].
    """
    if not items:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    total_err = sum(x[errors_key] for x in items)
    total_len = sum(x[length_key] for x in items)
    point = (total_err / total_len) if total_len > 0 else 0.0
    if len(items) < 2:
        return {"estimate": round(point, 4), "ci_low": round(point, 4), "ci_high": round(point, 4)}

    rng = random.Random(seed)
    n = len(items)
    draws: List[float] = []
    for _ in range(n_resamples):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        s_err = sum(x[errors_key] for x in sample)
        s_len = sum(x[length_key] for x in sample)
        draws.append((s_err / s_len) if s_len > 0 else 0.0)

    return {
        "estimate": round(point, 4),
        "ci_low": round(compute_percentile(draws, 2.5), 4),
        "ci_high": round(compute_percentile(draws, 97.5), 4),
    }


def count_cross_script_substitutions(
    reference: str,
    hypothesis: str,
    language: str = "hi",
    normaliser=None,
) -> Dict[str, int]:
    """
    Align rule-normalised tokens via DP backtrace and count substitutions where
    one token is Devanagari (U+0900..U+097F) and the aligned token is ASCII Latin.

    Used uniformly on both FLEURS Hindi and code-mixed suites so script-choice
    differences (`ग्रीनलैंड` vs `Greenland`, `अकाउंट` vs `account`) can be broken
    out transparently without altering WER scores.
    """
    norm_fn = normaliser or normalise_for_eval
    ref_words = norm_fn(reference, language).split()
    hyp_words = norm_fn(hypothesis, language).split()
    if not ref_words or not hyp_words:
        return {"cross_script_subs": 0, "other_errors": abs(len(ref_words) - len(hyp_words))}

    dp = _levenshtein_matrix(ref_words, hyp_words)
    i, j = len(ref_words), len(hyp_words)
    cross_script = 0
    other = 0

    def _is_devanagari(tok: str) -> bool:
        return any("\u0900" <= c <= "\u097f" for c in tok)

    def _is_ascii_alpha(tok: str) -> bool:
        return tok.isascii() and any(c.isalpha() for c in tok)

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            rw, hw = ref_words[i - 1], hyp_words[j - 1]
            if (_is_devanagari(rw) and _is_ascii_alpha(hw)) or (_is_ascii_alpha(rw) and _is_devanagari(hw)):
                cross_script += 1
            else:
                other += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            other += 1
            i -= 1
        else:
            other += 1
            j -= 1

    return {"cross_script_subs": cross_script, "other_errors": other}


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
