# Gnani Speech Evaluation Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![ASR: Prisma v3.0](https://img.shields.io/badge/ASR-Prisma%20v3.0-orange.svg)](https://app.gnani.ai)
[![TTS: Timbre v2.5](https://img.shields.io/badge/TTS-Timbre%20v2.5-purple.svg)](https://app.gnani.ai)

A comprehensive, production-grade benchmarking framework evaluating **Gnani Prisma** (ASR) and **Gnani Timbre** (TTS) speech APIs across real-world Indian enterprise conditions: telephony audio, noise, code-switching (Hinglish), domain vocabulary, and regional Indic languages.

Designed and implemented as an AI Solutions Engineer take-home assignment by **Ashish Vipin Korumelli**.

---

## Benchmark Highlights (Empirical Results)

Tested directly against Gnani's live production platform (`api.vachana.ai`):

### 1. Prisma ASR (`prisma-v3.0`)
* **Overall Micro-averaged WER**: **24.67%** (300 reference words, 30 clips across 11 categories)
* **Overall Micro-averaged CER**: **20.47%**
* **Average Inference Latency**: **369.8 ms**
* **Key Category Highlights**:
  - **Accented English**: **0.00% WER** / **0.00% CER** (328.6 ms)
  - **Short Commands**: **0.00% WER** / **0.00% CER** (196.9 ms)
  - **Clean English**: **9.68% WER** / **12.16% CER** (496.4 ms)
  - **Telephony (8 kHz Bandpass)**: **12.50% WER** / **6.25% CER** (314.9 ms)
  - **Regional Languages (Telugu, Tamil, Bengali)**: **12.50% WER** / **5.15% CER** (310.2 ms)
  - **Numbers & Entities (ITN)**: **69.23% WER** nominal, driven by active Inverse Text Normalisation (rendering spelled-out words as standard numeric entities like `"1,23,456"`, `"5,000"`, `"9876543210"`).

### 2. Timbre TTS (`timbre-v2.5`)
* **Average Quality Score**: **84.8 / 100** (28 test sentences across 9 categories)
* **Average Real-Time Factor (RTF)**: **0.1747** (>5.7x faster than real-time playback)
* **Long-Form Telephony Scaling**: RTF drops to **0.0742** (>13x faster than real-time) on long paragraph content
* **Average Latency**: **901.5 ms**
* **High Scoring Categories**:
  - **Abbreviations & Acronyms**: **91.5 / 100** (RTF: 0.1680)
  - **Tricky Indian Pronunciation**: **90.0 / 100** (RTF: 0.1609)
  - **Long-Form Content**: **93.0 / 100** (RTF: 0.0742)

---

## Key Technical Features

- **Custom Metric Engine**: Full Word Error Rate (WER) and Character Error Rate (CER) calculation written from scratch using Wagner-Fischer dynamic programming for Levenshtein edit distance, breaking down Substitutions (S), Deletions (D), and Insertions (I).
- **Indic Script Normalisation**: Preserves Devanagari vowel signs (*matras*) and combining marks while removing non-alphanumeric punctuation.
- **Automated Objective TTS Scoring**: Multi-metric evaluation rubric combining:
  1. Real-Time Factor (RTF)
  2. Speaking rate (WPM)
  3. Silence distribution ratio
  4. Signal clipping detection
  5. Round-trip ASR semantic fidelity
- **Audio Processing Utilities**: Synthetic Gaussian noise injection (SNR scaling), 8 kHz narrowband telephony simulation with 300–3400 Hz bandpass filtering, and raw PCM-to-WAV packaging.
- **Reproducible Harness**: Command-line runners, automated test fixtures, JSON results manifests, and an interactive Jupyter Notebook.

---

## Project Structure

```
gnani-speech-eval/
├── config.py                       # API endpoints, models, voice maps
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for API credentials
├── evaluation_notebook.ipynb       # Interactive Colab / Jupyter evaluation walkthrough
│
├── src/
│   ├── metrics.py                  # Scratch WER/CER DP implementation & TTS metrics
│   ├── gnani_client.py             # Prisma & Timbre REST client wrapper with PCM handling
│   ├── audio_utils.py              # Noise injection, telephony filter, spectral analysis
│   ├── asr_evaluator.py            # ASR batch testing, scoring, and breakdown
│   └── tts_evaluator.py            # TTS synthesis, latency, RTF, and round-trip scoring
│
├── data/
│   ├── asr/
│   │   ├── test_manifest.json      # 30 test utterances with ground-truth transcripts
│   │   └── audio/                  # Audio clips generated for benchmark
│   └── tts/
│       ├── test_sentences.json     # 28 test sentences covering 9 operational categories
│       └── output/                 # Synthesized audio files
│
├── scripts/
│   ├── generate_audio_via_tts.py   # Synthesize audio clips via Timbre with noise/telephony
│   ├── prepare_test_data.py        # Offline fallback audio generator
│   └── run_evaluation.py           # Main CLI runner for full evaluation
│
├── tests/
│   └── test_metrics.py             # 14 unit tests validating metric correctness
│
├── results/
│   ├── asr_results.json            # Full per-clip and categorical ASR benchmark data
│   └── tts_results.json            # Full per-sentence and categorical TTS benchmark data
│
└── report/
    └── evaluation_report.md        # Detailed written findings and architecture recommendations
```

---

## Quick Start & Setup

### 1. Installation

```bash
git clone https://github.com/Ashishvipin/speech-eval.git
cd speech-eval
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and set your Gnani API key:

```bash
cp .env.example .env
```

```env
GNANI_API_KEY=your_gnani_api_key_here
```

### 3. Verify Metrics Implementation

Run the metric verification test suite:

```bash
python3 tests/test_metrics.py
```
```
  ✓ test_perfect_match
  ✓ test_complete_mismatch
  ✓ test_known_wer
  ✓ test_deletion
  ✓ test_insertion
  ✓ test_empty_reference
  ✓ test_empty_both
  ✓ test_cer_character_level
  ✓ test_hindi_wer
  ✓ test_normalisation_punctuation
  ✓ test_normalisation_extra_spaces
  ✓ test_aggregate_wer
  ✓ test_rtf_calculation
  ✓ test_speaking_rate

  14 passed, 0 failed out of 14 tests
```

### 4. Run the Evaluation Suite

```bash
# Execute full evaluation (both Prisma ASR & Timbre TTS)
python3 scripts/run_evaluation.py

# Run ASR evaluation only
python3 scripts/run_evaluation.py --asr-only

# Run TTS evaluation only
python3 scripts/run_evaluation.py --tts-only

# Run TTS without round-trip ASR checking
python3 scripts/run_evaluation.py --tts-only --no-roundtrip
```

### 5. Interactive Notebook

Launch Jupyter or Google Colab to explore the data interactively:

```bash
jupyter notebook evaluation_notebook.ipynb
```

---

## Detailed Written Report

Read the comprehensive technical report covering error analysis, the Inverse Text Normalization (ITN) trade-off, code-switching mechanics, and enterprise telephony architecture in:  
👉 **[`report/evaluation_report.md`](report/evaluation_report.md)**

---

## License

MIT License. Designed and authored by **Ashish Vipin Korumelli** for the Gnani.ai AI Solutions Engineer evaluation.
