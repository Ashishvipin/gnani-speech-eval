# Gnani Speech Evaluation Toolkit (Prisma ASR + Timbre TTS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end engineering evaluation suite for **Gnani AI Speech APIs** (`api.vachana.ai`):
- **Part 1 — Prisma v3.0 (`prisma-v3.0`, ASR)**: Evaluated across both a **22-clip Real-Human Speech Suite** (AI4Bharat Svarah, Google FLEURS, OpenSLR 66/65/37 + babble/μ-law stress tests) and a **30-clip Synthetic Smoke Suite**, using custom Wagner-Fischer DP `compute_wer` / `compute_cer` with dual **Verbatim** vs **Format-Normalised (ITN)** reporting.
- **Part 2 — Timbre v2.5 (`timbre-v2.5`, TTS)**: Evaluated across **29 test sentences × 3 repeats** using **component metrics** (p50/p95 full-response HTTP latency, retry counts, p50/p95 RTF, 25 ms RMS frame-level silence ratio, peak clipping, and Prisma round-trip WER with an optional `faster-whisper` independent cross-check hook).

---

## Key Observed Results

### 1. Prisma v3.0 ASR (`api.vachana.ai/stt/v3`)

| Evaluation Suite | Clips | Verbatim WER | Format-Normalised WER | Verbatim CER | Latency p50 / p95 | Retries |
|---|---:|---:|---:|---:|---:|---:|
| **Real-Human Speech** (`real_manifest.json`) | 22 | **11.07%** | **10.40%** | 9.95% | 388 / 634 ms | 0 |
| **Synthetic Smoke Test** (`test_manifest.json`) | 30 | 20.00% | 0.37% | 18.62% | 340 / 490 ms | 0 |

- **Real-Human Suite Breakdown (`n=22`)**: Indian English Svarah (`n=5`): 6.25% verbatim / 2.08% formatted WER; Hindi FLEURS (`n=5`): 14.63% WER (driven by Latin script rendering of English loanwords plus two proper-noun slips); Telugu/Tamil/Bengali OpenSLR (`n=8`): 14.55% WER with only **3.17% CER**; 10 dB speech-modulated babble (`n=2`): 2.78% WER; 8 kHz μ-law telephony (`n=2`): 8.33% WER.
- **Why Synthetic Smoke Dropped from 20.00% to 0.37%**: Default Prisma ITN formats spoken numbers/units (`"1,23,456"`, `"9876543210"`, `"500mg"`) and transliterates common Hinglish words into Devanagari (`"अकाउंट"`), masking true accuracy unless scored format-insensitively. Note that synthetic `0.00%` regional/accented results reflect Timbre clean audio, not human acoustic variation.

### 2. Timbre v2.5 TTS (`api.vachana.ai/api/v1/tts/inference`)

- **Reliability & Latency (`28/29` sentences × 3 repeats)**: **p50 852 ms**, **p95 1,321 ms** (full-response non-streaming HTTP from an India link; mean 908 ms, range 485–2,134 ms), **p50 RTF 0.171**, **p95 RTF 0.233**, **0 retries**.
- **Audio & Intelligibility**: 25 ms RMS silence ratio **0.242**, max clipping **0.000**, round-trip Prisma WER **6.21% formatted** (25.71% verbatim).
- **Native Voice Catalog Fixes (`config.py`)**: Mapped native Timbre v2.5 voices (`te-IN` → `Suhana`, `ta-IN` → `Asmita`, `bn-IN` → `Kirra`, `hi-en` → `Poorvi`) instead of falling back to English `Kaveri`.
- **Notable Failure Findings** (diagnostic WAVs included under `data/tts/output/`):
  - `long_hi_001` (`Nalini`): Synthesised `₹2,45,678` as **`2,45,688`** on Run 1 (`long_hi_001.wav`) and omitted **`तीन`** on Run 3.
  - `special_en_001` (`Kaveri`): Mangled `https://www.gnani.ai` (`gany.ai`/`giani.ai`/`nyani.ai`) and `support@gnani.ai` (`"sup. PORT at gyanid dot ai"`) across all 3 runs (`special_en_001.wav`).
  - `edge_empty_001`: Punctuation-only `"..."` deterministically returned HTTP 500 (`No audio generated during synthesis`).

> Full methodology, category breakdowns, failure analysis, and deployment recommendations are in [**report/evaluation_report.md**](report/evaluation_report.md).

---

## Project Structure

```
gnani-speech-eval/
├── config.py                        # API endpoints, native Timbre v2.5 voice mappings & paths
├── LICENSE                          # MIT License
├── requirements.txt                 # Dependencies (requests, numpy, scipy, python-dotenv)
├── evaluation_notebook.ipynb        # Reproducible walkthrough (supports RUN_LIVE_EVAL=True/False)
│
├── src/
│   ├── gnani_client.py              # REST client with capped retry/backoff & retry counters
│   ├── metrics.py                   # Scratch Wagner-Fischer DP WER/CER, danda stripping, percentiles, 25 ms silence RMS
│   ├── audio_utils.py               # IEEE-float/PCM WAV I/O, seeded babble noise, 8 kHz G.711 μ-law telephony
│   ├── asr_evaluator.py             # Real-human & synthetic ASR suites with dual verbatim/formatted scoring
│   └── tts_evaluator.py             # Multi-repeat component TTS benchmark + optional Whisper hook
│
├── scripts/
│   ├── prepare_test_data.py         # Downloads & verifies 22 real-human ASR clips (Svarah/FLEURS/OpenSLR)
│   ├── generate_audio_via_tts.py    # Generates 30 synthetic ASR smoke clips (seeded noise + μ-law telephony)
│   └── run_evaluation.py            # CLI entrypoint (--asr-only, --tts-only, --tts-repeats, --independent-asr)
│
├── data/
│   ├── asr/
│   │   ├── real_manifest.json       # 22 real-human clips + provenance tags
│   │   ├── real_audio/              # Verified 16 kHz / 8 kHz WAVs
│   │   └── test_manifest.json       # 30 synthetic smoke clips with dual references
│   └── tts/
│       ├── test_sentences.json      # 29 TTS test cases with expected spoken roundtrip_reference
│       └── output/                  # Selected failure/diagnostic WAVs (long_hi_001, special_en_001/002, pron_en_001, te_001)
│
├── tests/
│   └── test_metrics.py              # 20 unit tests (DP WER/CER, danda, μ-law, 25 ms silence, client retries, dual ITN)
│
├── results/
│   ├── asr_results.json             # Latest real_human + synthetic_smoke results
│   └── tts_results.json             # Latest 3-repeat TTS component benchmark results
│
└── report/
    └── evaluation_report.md         # Concise engineering report (~900 words)
```

---

## Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/Ashishvipin/gnani-speech-eval.git
cd gnani-speech-eval
pip install -r requirements.txt
cp .env.example .env   # Add GNANI_API_KEY=<your_token>
```

### 2. Run Unit Tests (No API Key Required)
```bash
python3 tests/test_metrics.py
# 20 passed, 0 failed out of 20 tests
```

### 3. Prepare Data & Run Live Evaluation
```bash
# Re-download / verify 22 real human clips (already committed in data/asr/real_audio/)
python3 scripts/prepare_test_data.py

# Run full evaluation (Part 1: real + synthetic ASR; Part 2: 29 TTS sentences × 3 repeats)
python3 scripts/run_evaluation.py

# Options:
python3 scripts/run_evaluation.py --asr-only
python3 scripts/run_evaluation.py --tts-only --tts-repeats 3
python3 scripts/run_evaluation.py --tts-only --independent-asr   # Colab/Python >= 3.10 with faster-whisper
```

---

## Author & License

**Ashish Vipin** — [github.com/Ashishvipin](https://github.com/Ashishvipin) | Licensed under [MIT](LICENSE).
