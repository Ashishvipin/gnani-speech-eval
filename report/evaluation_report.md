# Gnani Speech Evaluation Report: Prisma v3.0 (ASR) & Timbre v2.5 (TTS)

**Author:** Ashish Vipin | **Endpoints:** `api.vachana.ai/stt/v3` (`prisma-v3.0`), `api.vachana.ai/api/v1/tts/inference` (`timbre-v2.5`)

## 1. Methodology

This evaluation benchmarks Gnani's production ASR and TTS APIs from an India residential network path; reported latency includes local transit alongside server inference.

- **Part 1 — Prisma v3.0 ASR (52 clips across two suites):**
  - *Real-Human Speech (22 clips, 298 words):* Public recordings from AI4Bharat Svarah (5 Indian English), Google FLEURS `hi_in` (5 Hindi), and OpenSLR 66/65/37 (3 Telugu, 3 Tamil, 2 Indian Bengali), plus 4 seeded degradations of real clips (10 dB speech-modulated babble noise `n=2`; 8 kHz 300–3400 Hz G.711 μ-law telephony `n=2`).
  - *Synthetic Smoke Suite (30 clips, 300 words):* Timbre-synthesized controlled utterances probing numerals, domain terms, code-switching, and regional voices. This suite verifies ITN and API behavior only—it does not test real speaker acoustics.
  - *Dual Metrics:* Scratch Wagner-Fischer DP (`src/metrics.py`) with Indic `।`/`॥` punctuation stripping computes both **Verbatim WER** (strict raw reference) and **Format-Normalised WER** (accepting either raw reference or `reference_formatted` for ITN digits/script changes).
- **Part 2 — Timbre v2.5 TTS (29 sentences × 3 repeats):**
  - Voice mappings in `config.py` explicitly select native Timbre v2.5 voices (`Kaveri` en-IN, `Nalini` hi-IN, `Poorvi` hi-en, `Suhana` te-IN, `Asmita` ta-IN, `Kirra` bn-IN) instead of falling back to English.
  - TTS is assessed via **component metrics** (no composite score): `p50`/`p95` full-response non-streaming HTTP latency (not streaming TTFB), retry counts, `p50`/`p95` RTF, 25 ms RMS frame silence ratio (10 ms hop, 0.015 threshold), peak clipping, and round-trip Prisma WER.

## 2. Results

### Prisma v3.0 ASR — Real-Human Suite (`n=22`)

| Category | Clips | Verbatim WER | Formatted WER | CER | p50 / p95 Latency |
|---|---:|---:|---:|---:|---:|
| `real_indian_english` (Svarah) | 5 | 6.25% | 2.08% | 6.85% | 296 / 595 ms |
| `real_hindi` (FLEURS) | 5 | 14.63% | 14.63% | 20.35% | 516 / 621 ms |
| `real_regional_indic` (OpenSLR te/ta/bn) | 8 | 14.55% | 14.55% | 3.17% | 406 / 534 ms |
| `real_noisy_audio` (10 dB babble) | 2 | 2.78% | 2.78% | 1.36% | 305 / 372 ms |
| `real_telephony` (8 kHz μ-law) | 2 | 8.33% | 8.33% | 4.08% | 316 / 424 ms |
| **Overall Real-Human Suite** | **22** | **11.07%** | **10.40%** | **9.95%** | **388 / 634 ms** |

Zero retries occurred (mean 394 ms; formatted CER 9.02%). In `real_svarah_en_001`, Prisma formatted `"eleventh to fourteenth"` → `"11th to 14th"`, dropping Indian-English WER from 6.25% to 2.08%.

On the **30-clip Synthetic Smoke Suite**, Verbatim WER was **20.00%** (60/300 words; CER 18.62%), falling to **0.37%** Format-Normalised WER (1/268 words; CER 0.49%; p50 340 ms, p95 490 ms, 0 retries) once expected ITN/script forms (`"1,23,456"`, `"9876543210"`, `"500mg"`, `"अकाउंट बैलेंस"`) were credited. The single remaining formatted error dropped initial `"please"` (`clean_en_002`). Synthetic `accented_english` and `regional_languages` both scored `0.00%` on clean Timbre audio.

### Timbre v2.5 TTS — Component Metrics (`28/29` succeeded × 3 repeats = 84 calls)

| Category | Sentences | p50 / p95 HTTP | p50 / p95 RTF | 25 ms Silence | RT Fmt WER |
|---|---:|---:|---:|---:|---:|
| `numbers` | 4 | 859 / 954 ms | 0.185 / 0.217 | 0.236 | 0.0% |
| `abbreviations` | 4 | 899 / 980 ms | 0.158 / 0.225 | 0.219 | 0.0% |
| `code_switching` (`Poorvi`) | 3 | 725 / 849 ms | 0.162 / 0.218 | 0.224 | 11.1% |
| `long_form` | 2 | 1,691 / 1,961 ms | 0.082 / 0.086 | 0.259 | 3.9% |
| `tricky_pronunciation` | 4 | 727 / 936 ms | 0.189 / 0.213 | 0.197 | 9.2% |
| `punctuation_and_prosody` | 2 | 946 / 1,086 ms | 0.174 / 0.176 | 0.386 | 0.0% |
| `special_formats` | 3 | 1,073 / 1,179 ms | 0.163 / 0.199 | 0.228 | 29.2% |
| `regional_languages` (`te/ta/bn`) | 3 | 854 / 896 ms | 0.171 / 0.188 | 0.246 | 2.8% |
| `edge_cases` (1 HTTP 500) | 3 ok / 4 | 736 / 916 ms | 0.189 / 0.237 | 0.260 | 0.0% |
| **Overall TTS** | **28 / 29** | **852 / 1,321 ms** | **0.171 / 0.233** | **0.242** | **6.21%** |

All 84 successful synthesis runs had 0 retries, `0.000` clipping, 485–2,134 ms latency range (mean 908 ms), 140.9 mean WPM, and 25.71% verbatim / 6.21% formatted round-trip WER. Native `Suhana`/`Asmita`/`Kirra` voices kept regional RT formatted WER at 0.0–5.6%.

## 3. Concrete Failure Examples

1. **Hindi Digit Hallucination & Drop (`long_hi_001`, `Nalini`):** Prompt had `"₹2,45,678 शेष है। आपकी तीन लेनदेन लंबित हैं।"`. Across 3 runs, Run 1 (`data/tts/output/long_hi_001.wav`) synthesized `2,45,688` (digit substitution); Run 2 read `2,45,678` and `3`; Run 3 dropped `तीन` completely (`"आपकी लेन-देन लंबित है"`). Stochastic numerical instability requires verification before banking deployment.
2. **URL/Email & Loanword TN Breakdown (`special_en_001`, `Kaveri`):** `"https://www.gnani.ai ... support@gnani.ai"` (`special_en_001.wav`) degraded in all 3 repeats (`gany.ai`/`giani.ai`/`nyani.ai`; `"sup. PORT at gyanid dot ai"`, 79.2% RT formatted WER). `pron_en_001.wav` (`Worcestershire`, `açaí`) also drifted (`axai`/`ustusha`/`achate`).
3. **Punctuation-Only Rejection (`edge_empty_001`):** Input `"..."` deterministically failed with HTTP 500 (`"No audio generated during synthesis"`), exiting cleanly without retries.
4. **Prisma Script Mixing & Regional Segmentation:** Real Hindi FLEURS (`14.63%` WER) output Latin loanwords (`Greenland`, `television`, `antique`, `agencies`) alongside recognition slips (`"नॉर्स सगास"` → `"North Sagass"`, `"रेड"` → `"Reed"`). OpenSLR Telugu/Tamil/Bengali (`14.55%` WER) showed low **3.17% CER**, mostly minor suffix/spacing differences (`తటస్థీకరణము` → `తటస్థీకరణం`, `জিয়াংসু` → `জিয়াং শু`). Controlled Hinglish (`codemix_002`: `"Please mera last transaction ka detail bhejo."`) transcribed verbatim (`0.00%` WER).

## 4. Limitations & Recommendations

**Limitations:** Small sample sizes (`n=22` real ASR, `n=2` noisy/telephony, `n=29` TTS), simulated babble/μ-law degradations rather than live PSTN recordings, and single-vendor round-trip circularity (`Prisma` can mask `Timbre` artifacts; `src/tts_evaluator.py` includes a `faster-whisper` hook for Python ≥ 3.10 independent checks).

**Recommendations:**
- **Banking / Fintech IVR:** Validate/reject punctuation-only TTS inputs client-side, keep Hindi financial digit prompts short or spell-confirmed, and normalize URLs/emails into spoken words (`"gnani dot ai"`) pre-TTS.
- **Multilingual Voicebots:** Pin native Timbre voices explicitly (`Suhana`, `Asmita`, `Kirra`, `Poorvi`), use Prisma `hi-IN,en-IN` for Hinglish, and expect mixed Devanagari/Latin orthography downstream.
- **Low-Latency Voice Agents:** Full-response HTTP turnaround (ASR `p95` 634 ms + TTS `p95` 1,321 ms from India) exceeds synchronous barge-in budgets; adopt streaming/WebSocket playback where supported.
