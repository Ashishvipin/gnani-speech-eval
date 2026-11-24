# Gnani Speech Evaluation Toolkit

A structured evaluation framework for **Gnani Prisma** (ASR) and **Gnani Timbre** (TTS) speech APIs — built to benchmark, measure, and report on speech model quality the way a Solutions Engineer would before recommending these APIs to an enterprise customer.

## What This Does

- **ASR Evaluation (Prisma v3.0):** Runs 30 test clips across 10 categories (clean, noisy, telephony, code-switching, accented, domain-specific, numbers, regional languages, long/short utterances), computing WER and CER per clip and in aggregate.
- **TTS Evaluation (Timbre v2.5):** Synthesizes 28 test sentences across 9 categories (numbers, abbreviations, code-switching, long-form, tricky pronunciation, prosody, special formats, regional languages, edge cases), scoring each on latency, RTF, speaking rate, silence ratio, clipping, and round-trip ASR fidelity.
- **WER/CER implemented from scratch** using the Levenshtein edit-distance algorithm — no external metric libraries.
- **Outputs structured JSON results** broken down by test category, plus a written evaluation report.

## Project Structure

```
gnani-speech-eval/
├── config.py                       # API keys, endpoints, paths
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for API key
│
├── src/
│   ├── metrics.py                  # WER, CER, RTF — built from scratch
│   ├── gnani_client.py             # Prisma ASR + Timbre TTS API wrapper
│   ├── audio_utils.py              # Audio loading, noise injection, spectral analysis
│   ├── asr_evaluator.py            # Full ASR evaluation pipeline
│   └── tts_evaluator.py            # Full TTS evaluation pipeline
│
├── data/
│   ├── asr/
│   │   ├── test_manifest.json      # 30 test clips with ground truth
│   │   └── audio/                  # Test audio files
│   └── tts/
│       ├── test_sentences.json     # 28 test sentences with expected behaviour
│       └── output/                 # Generated TTS audio (created on run)
│
├── scripts/
│   ├── prepare_test_data.py        # Generate/download test audio
│   └── run_evaluation.py           # Main runner for both evaluations
│
├── tests/
│   └── test_metrics.py             # Unit tests for metric correctness
│
├── results/                        # JSON results (created on run)
│   ├── asr_results.json
│   └── tts_results.json
│
└── report/
    └── evaluation_report.md        # Written evaluation and recommendations
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/ashishvipin/gnani-speech-eval.git
cd gnani-speech-eval
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and paste your API key from [app.gnani.ai](https://app.gnani.ai):

```
GNANI_API_KEY=your_api_key_here
```

### 3. Prepare Test Audio

```bash
python scripts/prepare_test_data.py
```

This generates placeholder audio clips using macOS TTS. For a proper evaluation, replace the files in `data/asr/audio/` with real recordings from Common Voice, OpenSLR, or your own domain-specific audio.

### 4. Run

```bash
# Full evaluation (ASR + TTS)
python scripts/run_evaluation.py

# ASR only
python scripts/run_evaluation.py --asr-only

# TTS only (skip round-trip ASR for speed)
python scripts/run_evaluation.py --tts-only --no-roundtrip
```

### 5. Check Results

- `results/asr_results.json` — Per-clip WER/CER, category breakdown, aggregate metrics
- `results/tts_results.json` — Per-sentence quality scores, latency, RTF, category breakdown
- `report/evaluation_report.md` — Written analysis and recommendations

## Test Categories

### ASR (30 clips)

| Category | Focus | Real-world scenario |
|---|---|---|
| Clean English/Hindi | Baseline accuracy | Standard customer calls |
| Noisy Audio | Noise robustness | Callers on the street, in cars |
| Telephony (8 kHz) | Narrowband handling | Contact centre phone lines |
| Numbers & Entities | Numeric accuracy | Banking transactions, OTPs |
| Code-switching | Hinglish handling | "Mera balance check karo" |
| Accented English | Accent diversity | South/North Indian English |
| Domain-specific | Vocabulary coverage | Medical, legal, tech terms |
| Long/Short Utterances | Length robustness | Full queries vs. "yes"/"no" |
| Regional Languages | Language breadth | Telugu, Tamil, Bengali |

### TTS (28 sentences)

| Category | Focus | Real-world scenario |
|---|---|---|
| Numbers | Currency, phone numbers | "₹1,23,456", "1800-123-4567" |
| Abbreviations | Expansion correctness | "Dr.", "EMI", "FY26" |
| Code-switching | Hinglish prosody | Hindi-English mid-sentence |
| Long-form | Pacing consistency | Multi-sentence banking scripts |
| Tricky Pronunciation | Proper nouns, rare words | "Thiruvananthapuram", "quinoa" |
| Punctuation & Prosody | Intonation matching | Questions, exclamations, lists |
| Special Formats | Non-standard text | URLs, emails, fractions |
| Regional Languages | Language breadth | Telugu, Tamil |
| Edge Cases | Failure modes | Ellipsis, repetitions, mixed script |

## Metrics

### ASR Metrics (from scratch)

- **WER (Word Error Rate):** `(Substitutions + Deletions + Insertions) / Reference Words`
- **CER (Character Error Rate):** Same formula at the character level
- Computed using the Wagner-Fischer dynamic programming algorithm for Levenshtein distance
- Micro-averaged across clips (weights longer utterances proportionally)

### TTS Quality Score (0-100)

| Component | Points | Threshold |
|---|---|---|
| RTF < 1.0 | 20 | Faster-than-real-time synthesis |
| Speaking Rate 120-180 WPM | 20 | Natural English pacing |
| Silence Ratio 10-40% | 20 | Natural pauses, not gaps |
| No Clipping | 15 | Clean, undistorted audio |
| Round-trip ASR > 90% | 25 | If ASR can read it back, humans can understand it |

## Running Tests

```bash
python -m pytest tests/ -v
# or
python tests/test_metrics.py
```

## Extending This

- **Add more test clips:** Edit `data/asr/test_manifest.json` — the pipeline picks them up automatically
- **Add more TTS sentences:** Edit `data/tts/test_sentences.json`
- **Add a new language:** Add the language code in `config.py` and create corresponding test entries
- **Custom audio:** Drop WAV files into `data/asr/audio/` and reference them in the manifest

## Built With

- Python 3.10+
- Gnani Vachana API (Prisma v3.0 + Timbre v2.5)
- NumPy (audio processing)
- Requests (API calls)

## License

MIT
