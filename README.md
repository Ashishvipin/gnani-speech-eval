# Gnani Speech Evaluation Toolkit (Prisma ASR + Timbre TTS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

End-to-end engineering evaluation suite for **Gnani AI Speech APIs** (`api.vachana.ai`):
- **Part 1 — Prisma v3.0 (`prisma-v3.0`, ASR)**: Evaluated across a **18-clip Clean Real-Human Suite** (AI4Bharat Svarah, Google FLEURS, OpenSLR 66/65/37), a **16-clip Paired Clean-vs-Degraded Stress Suite** (20 dB / 10 dB / 5 dB babble noise + 8 kHz μ-law telephony), a **30-clip Synthetic Smoke Suite**, and a **Hinglish language-parameter ablation** (`hi-IN` vs `hi-IN,en-IN` vs `indic-auto`). Uses scratch Wagner-Fischer DP (`src/metrics.py`) with **Verbatim WER** as primary and dataset-agnostic `normalise_for_eval` rules as secondary (zero output-tuned references), plus 95% clip-level bootstrap confidence intervals.
- **Part 2 — Timbre v2.5 (`timbre-v2.5`, TTS)**: Evaluated across **29 sentences × 3 REST repeats** (`/api/v1/tts/inference`) and **5 sentences × 3 SSE streaming repeats** (`/api/v1/tts/sse`) using **component metrics**: streaming TTFB, REST HTTP latency/RTF (`p50`/`p95`), retry counts, 25 ms RMS frame silence ratio, peak clipping, and round-trip Prisma WER scored against raw input text.

---

## Key Observed Results

### 1. Prisma v3.0 ASR (`api.vachana.ai/stt/v3`)

| Evaluation Suite | Clips | Verbatim WER [95% CI] | Rule-Norm WER [95% CI] | Verbatim CER | Latency p50 / p95 | Retries |
|---|---:|---:|---:|---:|---:|---:|
| **Real-Human Clean Speech** (`real_manifest.json`) | 18 | **12.83%** [7.84%–17.74%] | **11.95%** [6.74%–17.47%] | 11.86% | 418 / 720 ms | 0 |
| **Paired Real Stress Variants** (`real_stress_manifest.json`) | 16 | 8.22% [5.66%–10.74%] | 5.59% [3.25%–8.28%] | 8.82% | 465 / 654 ms | 0 |
| **Synthetic Smoke Test** (`test_manifest.json`) | 30 | 20.00% [10.27%–30.69%] | 8.86% [3.31%–15.79%] | 18.62% | 372 / 562 ms | 0 |

- **Real-Human Breakdown (`n=18`)**: Svarah Indian English (`n=5`): 6.25% verbatim / 2.08% rule-normalised WER; FLEURS Hindi (`n=5`): 14.63% WER (14 of 18 word errors are Devanagari→Latin script switches like `Greenland`, `antique`, `television`, `internet`); OpenSLR te/ta/bn (`n=8`): 14.55% WER with **3.17% CER**.
- **Hinglish Ablation (`n=3` clips)**: `hi-IN` and `hi-IN,en-IN` produced identical transcripts (`38.46%` WER, `10` cross-script tokens, `p50=365/368 ms`), whereas `indic-auto` switched Latin words wholly to Devanagari (`84.62%` WER, `22` cross-script tokens).

### 2. Timbre v2.5 TTS (`timbre-v2.5`)

- **Streaming SSE TTFB (`5 sentences × 3 = 15 calls` via `/api/v1/tts/sse`)**: First-audio-chunk **p50 221 ms** (`p95 352 ms`), stream start event `p50 34 ms`, full stream completion `p50 796 ms`.
- **Non-Streaming REST (`28/29 sentences × 3 repeats` via `/api/v1/tts/inference`)**: `p50 838 ms`, `p95 1,254 ms` (mean `882 ms`, range `469–2,091 ms`, from India network path), `p50 RTF 0.169`, `p95 RTF 0.220`, `0` retries.
- **Audio & Intelligibility**: 25 ms RMS silence ratio **0.240**, max clipping **0.000**, round-trip WER against raw text **27.39% verbatim / 18.83% rule-normalised** (`95% CI: 11.40%–25.42%`, inflated on `te-IN`/`ta-IN`/`bn-IN` and Hinglish strictly by verbatim numeral expansion and script differences).
- **Audio Audit (`long_hi_001` & `special_en_001`)**: Direct inspection of [long_hi_001_run2.wav](data/tts/output/long_hi_001_run2.wav) confirmed Timbre `Nalini` pronounced **`₹2,45,678`** accurately while Prisma ASR misheard **`2,45,688`**. Conversely, in [special_en_001.wav](data/tts/output/special_en_001.wav) (`Kaveri`), Timbre letter-spelled `support` (`S-U-P P-O-R-T`) and varied on `gnani.ai`. `edge_empty_001` (`"..."`) deterministically returned HTTP 500.

> Full tables, paired clean-vs-degraded deltas, and failure analysis are in [**report/evaluation_report.md**](report/evaluation_report.md).

---

## Dataset Provenance & Licenses

Code, scripts, and documentation in this repository are **MIT-licensed** ([LICENSE](LICENSE)). Third-party human speech excerpts committed under `data/asr/real_audio/` retain their upstream Creative Commons licenses:

| Dataset | Clips | License | Source & Notes |
|---|---:|---|---|
| **AI4Bharat Svarah** | 5 (`en-IN`) | `CC BY 4.0` | Official repo ([`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah)) is gated behind Hugging Face terms/login; unauthenticated pulls use public mirror [`saisundaresan01/svarah-eval`](https://huggingface.co/datasets/saisundaresan01/svarah-eval) (supports `HF_TOKEN` when configured). Paired stress variants (`real_stress_manifest.json`, 16 clips) adapt only `CC BY 4.0` Svarah & FLEURS clips. |
| **Google FLEURS (`hi_in`)** | 5 (`hi-IN`) | `CC BY 4.0` | Streamed directly from [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (`Conneau et al., SLT 2022`). |
| **OpenSLR 66 / 65 / 37** | 8 (`te`/`ta`/`bn`) | `CC BY-SA 4.0` | Crowdsourced multi-speaker corpora ([SLR66](https://openslr.org/66/), [SLR65](https://openslr.org/65/), [SLR37](https://openslr.org/37/) — `He et al., INTERSPEECH 2020`). Excerpt WAVs remain under `CC BY-SA 4.0`. |

---

## Project Structure

```
gnani-speech-eval/
├── config.py                        # API endpoints (REST + SSE), native Timbre v2.5 voices & paths
├── LICENSE                          # MIT License (code/docs) + upstream CC audio attribution note
├── requirements.txt                 # Dependencies (requests, numpy, scipy, python-dotenv)
├── evaluation_notebook.ipynb        # Executed walkthrough (supports RUN_LIVE_EVAL=True/False)
├── src/
│   ├── gnani_client.py              # REST client + SSE streaming (`synthesize_stream`) + retry tracking
│   ├── metrics.py                   # Scratch DP WER/CER, `normalise_for_eval`, 95% bootstrap CI, 25 ms silence RMS
│   ├── audio_utils.py               # IEEE-float/PCM WAV I/O, seeded babble noise, 8 kHz G.711 μ-law telephony
│   ├── asr_evaluator.py             # Real clean, paired stress deltas, smoke suite & Hinglish ablation
│   └── tts_evaluator.py             # 3-repeat REST + SSE TTFB benchmark + optional Whisper hook
├── scripts/
│   ├── prepare_test_data.py         # Fetches 18 real clips & builds 16 multi-SNR/telephony paired clips
│   ├── generate_audio_via_tts.py    # Generates 30 synthetic ASR smoke clips
│   └── run_evaluation.py            # CLI entrypoint (--asr-only, --tts-only, --tts-repeats, --independent-asr)
├── data/
│   ├── asr/
│   │   ├── real_manifest.json       # 18 clean human clips + provenance & license tags
│   │   ├── real_stress_manifest.json# 16 paired 20/10/5 dB babble + 8 kHz μ-law variants (CC BY 4.0 bases)
│   │   ├── real_audio/              # 34 verified WAV clips
│   │   └── test_manifest.json       # 30 synthetic smoke clips (raw references only)
│   └── tts/
│       ├── test_sentences.json      # 29 TTS prompts (scored against raw text — no tuned references)
│       └── output/                  # Diagnostic WAVs (long_hi_001, long_hi_001_run2, special_en_001/002, pron_en_001, te_001)
├── tests/test_metrics.py            # 21 unit tests (DP WER/CER, normaliser, bootstrap CI, SSE mock, retries)
├── results/                         # Fresh empirical outputs (asr_results.json, tts_results.json)
└── report/evaluation_report.md      # Concise engineering evaluation report
```

---

## Quick Start

```bash
git clone https://github.com/Ashishvipin/gnani-speech-eval.git
cd gnani-speech-eval
pip install -r requirements.txt
cp .env.example .env   # Add GNANI_API_KEY=<your_token>

python3 tests/test_metrics.py                # 21 unit tests (offline)
python3 scripts/prepare_test_data.py         # Download real clips + build stress variants
python3 scripts/run_evaluation.py            # Run full ASR (clean/stress/smoke/ablation) + TTS (REST + SSE)
```

**Author:** Ashish Vipin ([github.com/Ashishvipin](https://github.com/Ashishvipin))
