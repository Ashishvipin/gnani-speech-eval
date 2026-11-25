# Gnani Speech Evaluation Report: Prisma v3.0 (ASR) & Timbre v2.5 (TTS)

**Author:** Ashish Vipin | **Endpoints:** `api.vachana.ai/stt/v3` (`prisma-v3.0`), `api.vachana.ai/api/v1/tts/inference` & `/api/v1/tts/sse` (`timbre-v2.5`)

## 1. Methodology

All measurements ran from an India residential network path, so reported latencies include local transit alongside server inference.

- **Part 1 — Prisma v3.0 ASR (64 clips across three separated suites + Hinglish ablation):**
  - *Clean Real-Human Speech (`n=18`, 247 words):* Public recordings from official AI4Bharat Svarah (`ai4bharat/Svarah`, 5 Indian English across 4 states, `CC BY 4.0`), Google FLEURS `hi_in` (5 Hindi, `CC BY 4.0`), and OpenSLR 66/65/37 (3 Telugu, 3 Tamil, 2 Indian Bengali, `CC BY-SA 4.0`).
  - *Paired Acoustic Stress Set (`n=16`):* Four `CC BY 4.0` human utterances (`svarah_en_001/003`, `fleurs_hi_001/005`) degraded with deterministic 20 dB, 10 dB, and 5 dB speech-modulated babble noise plus 8 kHz G.711 μ-law telephony, scored strictly against paired clean baselines.
  - *Synthetic Smoke Suite (`n=30`):* Timbre-synthesized utterances verifying API formatting; **not** a measure of human acoustic accuracy.
  - *Non-Circular Metrics:* Scratch Wagner-Fischer DP (`src/metrics.py`) reports **Verbatim WER** as primary. A single deterministic, dataset-agnostic normalizer (`normalise_for_eval`: US/UK spelling lexicon, numeric comma/ordinal stripping, English/Hindi spoken number parsing, symbol/unit canonicalization—zero per-clip overrides) produces secondary **Rule-Normalised WER**, alongside 1,000-resample 95% clip-level bootstrap confidence intervals.

- **Part 2 — Timbre v2.5 TTS (29 sentences × 3 REST repeats + 5 × 3 SSE streaming checks):**
  - Native Timbre v2.5 voices (`Kaveri`, `Nalini`, `Poorvi`, `Suhana`, `Asmita`, `Kirra`) are mapped per language in `config.py`.
  - Component metrics report REST HTTP turnaround, **streaming SSE TTFB** (`/api/v1/tts/sse`), RTF, 25 ms RMS frame silence ratio, clipping, and round-trip Prisma WER scored against raw input `text`.

## 2. Results

### Prisma v3.0 ASR — Clean Real-Human Suite (`n=18`) & Controlled Benchmarks

| Suite / Category | Clips | Verbatim WER [95% CI] | Rule-Norm WER [95% CI] | CER | p50 / p95 Latency |
|---|---:|---:|---:|---:|---:|
| `real_indian_english` (Official Svarah) | 5 | 1.45% [0.0%–5.1%] | 1.45% [0.0%–5.1%] | 0.62% | 440 / 690 ms |
| `real_hindi` (FLEURS) | 5 | 14.63% [8.9%–20.0%] | 14.63% [8.9%–20.0%] | 20.35% | 515 / 623 ms |
| `real_regional_indic` (OpenSLR te/ta/bn) | 8 | 14.55% [4.4%–27.9%] | 14.55% [4.4%–27.9%] | 3.17% | 360 / 580 ms |
| **Overall Clean Real-Human** | **18** | **10.93% [5.9%–16.1%]** | **10.93% [5.9%–16.1%]** | **9.79%** | **434 / 656 ms** |
| **Synthetic Smoke Suite** | **30** | **20.00% [10.3%–30.7%]** | **8.86% [3.3%–15.8%]** | **18.62%** | **367 / 511 ms** |

Zero retries occurred across all ASR suites. On FLEURS Hindi (18 errors in 123 words), **14 of 18 errors (77.8%)** were Devanagari→Latin script switches on loanwords/proper nouns (`Greenland`, `antique`, `television`, `internet`), leaving only 4 residual Devanagari errors (`3.25%`). Synthetic smoke similarly dropped to `8.86%` rule-normalised WER because `17` cross-script Hindi/Hinglish switches (`अकाउंट` vs `account`, `कॉल` vs `call`) and `g c p` vs `gcp` remained unmasked by design.

**Paired Clean-vs-Degraded Stress Delta (`n=4` base clips × 4 conditions = 16 clips):**

| Base Clip (`CC BY 4.0`) | Clean Rule WER | 20 dB Babble | 10 dB Babble | 5 dB Babble | 8 kHz μ-law PSTN |
|---|---:|---:|---:|---:|---:|
| `real_svarah_en_001` | 0.00% | +0.00 pp | +4.55 pp | +0.00 pp | +0.00 pp |
| `real_svarah_en_003` | 0.00% | +0.00 pp | +0.00 pp | +0.00 pp | +8.33 pp (`Arunachal→Andhra`) |
| `real_fleurs_hi_001` | 4.17% | +0.00 pp | +0.00 pp | -4.17 pp | +8.33 pp (`अस्थिर→अर्ध`) |
| `real_fleurs_hi_005` | 11.11% | +0.00 pp | +5.56 pp | +0.00 pp | +0.00 pp |

**Hinglish Language Ablation (`n=3` clips):** `hi-IN,en-IN` achieved **38.46% WER** (`10` cross-script errors, `p50=320 ms`) by keeping `codemix_002` in Latin script, outperforming single-language `hi-IN` (**50.00% WER**, `13` errors) and `indic-auto` (**84.62% WER**, `22` errors, full Devanagari transliteration).

### Timbre v2.5 TTS — REST (`28/29` × 3 repeats) & SSE Streaming (`5` × 3 repeats)

- **Non-Streaming REST (`/api/v1/tts/inference`):** `p50=838 ms`, `p95=1,254 ms` (mean `882 ms`, range `469–2,091 ms`), `p50 RTF=0.169`, `p95 RTF=0.220`, 25 ms silence ratio `0.240`, max clipping `0.000`, `0` retries. Round-trip WER against raw `text` was **27.39% verbatim / 18.83% rule-normalised** (`95% CI: 11.4%–25.4%`), led cleanly by `numbers` (`3.6%`), `punctuation` (`3.3%`), and `tricky_pronunciation` (`6.8%`), while `code_switching` (`50.2%`) and `regional_languages` (`35.5%`) reflected cross-script/spoken numeral formatting.
- **SSE Streaming TTFB (`/api/v1/tts/sse`):** First audio chunk arrived at **`p50=221 ms`** (`p95=352 ms`; stream `start` event `p50=34 ms`; stream completion `p50=796 ms`), including `165 ms` p50 TTFB on `long_hi_001` vs `1,107 ms` REST turnaround.

## 3. Concrete Failure & Attribution Findings

1. **Round-Trip Attribution Catch (`long_hi_001`, `Nalini`):** Round-trip Prisma ASR emitted `2,45,688` on Run 2 while emitting `2,45,678` on Runs 1 & 3. Direct listening to [long_hi_001.wav](data/tts/output/long_hi_001.wav) and [long_hi_001_run2.wav](data/tts/output/long_hi_001_run2.wav) confirmed **Timbre pronounced `2,45,678` and `तीन` accurately** in both runs—Prisma misheard the spoken Hindi digit (`678`→`688`). Single-vendor round-trip checking can misattribute ASR slips to TTS unless verified by ear or an independent recognizer.
2. **Text-Normalisation (TN) Breakdown (`special_en_001`, `Kaveri`):** On `"https://www.gnani.ai ... support@gnani.ai"`, listening to [special_en_001.wav](data/tts/output/special_en_001.wav) confirmed Timbre letter-spelled `support` (`S-U-P P-O-R-T`) and rendered `gnani` inconsistently (`giani`/`gyanid` in round-trip ASR, `38.5%` rule-normalised RT WER). On `pron_en_001.wav`, `açaí` also showed slight pronunciation drift.
3. **Strict Empty Input Rejection (`edge_empty_001`):** `"..."` deterministically returned HTTP 500 (`"No audio generated during synthesis"`), correctly failing fast with zero retries.
4. **Prisma Word-Segmentation & Band-Limiting (`real_human` / `real_paired_stress`):** OpenSLR Telugu/Tamil/Bengali had `14.55%` WER alongside just `3.17%` CER (`తటస్థీకరణము→తటస్థీకరణం`, `জিয়াংসু→জিয়াং শু`). On 8 kHz μ-law telephony, narrowband loss caused semantic substitutions (`Arunachal→Andhra`, `अस्थिर केंद्रक→अर्धकेंद्रक`).

## 4. Limitations & Recommendations

**Limitations:** Small sample counts (`n=18` clean human, `n=4` paired stress bases) yield wide confidence intervals (`5.9%–16.1%`). Stress variants simulate additive babble and μ-law codec rather than live PSTN calls. Automated local `faster-whisper` cross-checking was unavailable on this Python 3.9 host without Python ≥ 3.10, though direct human listening verified the key failure WAVs.

**Recommendations:**
- **Interactive Voice Agents:** Use SSE streaming (`/api/v1/tts/sse`, `p50=221 ms` TTFB) rather than full-response REST (`p50=838 ms`) whenever barge-in/low-latency turn-taking matters.
- **Banking / Customer Prompts:** Validate non-empty TTS text client-side and pre-expand raw emails/domains (e.g., `"support at gnani dot ai"`) to avoid awkward letter-spelling.
- **Hinglish & Indic ASR:** Pin native Timbre voices (`Suhana`, `Asmita`, `Kirra`, `Poorvi`) explicitly, pass `hi-IN,en-IN` rather than `hi-IN` or `indic-auto` when Latin transliteration fidelity is needed on Hinglish, and expect mixed Devanagari/Latin outputs on Hindi loanwords.
