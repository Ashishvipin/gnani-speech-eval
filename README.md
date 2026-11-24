# Gnani Speech Evaluation Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![ASR: Prisma v3.0](https://img.shields.io/badge/ASR-Prisma%20v3.0-orange.svg)](https://app.gnani.ai)
[![TTS: Timbre v2.5](https://img.shields.io/badge/TTS-Timbre%20v2.5-purple.svg)](https://app.gnani.ai)

A comprehensive, production-grade benchmarking framework evaluating **Gnani Prisma** (ASR) and **Gnani Timbre** (TTS) speech APIs across real-world Indian enterprise conditions: telephony audio, noise, code-switching (Hinglish), domain vocabulary, and regional Indic languages.

Designed and implemented as an AI Solutions Engineer take-home assignment by **Ashish Vipin**.

---

## Benchmark Highlights (Empirical Results)

Tested directly against Gnani's live production platform (`api.vachana.ai`):

### 1. Prisma ASR (`prisma-v3.0`)
* **Overall Micro-averaged WER**: **20.67%** (300 reference words, 30 clips across 11 categories)
* **Overall Micro-averaged CER**: **18.99%**
* **Average Inference Latency**: **446.6 ms**
* **Key Category Highlights**:
  - **Accented English**: **0.00% WER** / **0.00% CER** (291.1 ms)
  - **Short Commands**: **0.00% WER** / **0.00% CER** (127.6 ms)
  - **Clean Hindi**: **4.55% WER** / **8.05% CER** (1084.6 ms)
  - **Telephony (8 kHz, 300–3400 Hz Bandpass)**: **6.25% WER** / **5.00% CER** (269.3 ms)
  - **Regional Languages (Telugu, Tamil, Bengali)**: **6.25% WER** / **4.12% CER** (258.1 ms)
  - **Clean English**: **9.68% WER** / **12.16% CER** (592.3 ms)
  - **Numbers & Entities (ITN)**: **66.67% WER** nominal, driven by active Inverse Text Normalisation (rendering spelled-out words as standard numeric entities like `"1,23,456"`, `"5,000"`, `"9876543210"`).

### 2. Timbre TTS (`timbre-v2.5`)
* **Average Quality Score**: **88.0 / 100** (27/28 sentences synthesized; 1 expected edge-case failure on `"..."`)
* **Average Real-Time Factor (RTF)**: **0.3257** (~3.1x faster than real-time overall; **0.1000** / 10x real-time on long-form content)
* **Average Latency**: **1501.8 ms** (range: 525.2 ms – 7541.0 ms under tail load spikes)
* **High Scoring Categories**:
  - **Long-Form Content**: **96.5 / 100** (RTF: 0.1000)
  - **Punctuation & Prosody**: **96.0 / 100** (RTF: 0.1611)
  - **Regional Languages (Telugu, Tamil)**: **93.0 / 100** (RTF: 0.1926)
  - **Tricky Indian Pronunciation**: **92.5 / 100** (RTF: 0.5231)
  - **Abbreviations & Acronyms**: **91.8 / 100** (RTF: 0.2399)

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
│   └── test_metrics.py             # 19 unit tests validating metrics, audio & client retries
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
git clone https://github.com/Ashishvipin/gnani-speech-eval.git
cd gnani-speech-eval
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

  ✓ test_hindi_danda_punctuation
  ✓ test_bengali_danda_punctuation
  ✓ test_telephony_audio_simulation
  ✓ test_tts_quality_rubric
  ✓ test_gnani_client_retry_behavior

  19 passed, 0 failed out of 19 tests
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

MIT License. Designed and authored by **Ashish Vipin** for the Gnani.ai AI Solutions Engineer evaluation.
