# Technical Evaluation Report: Gnani Prisma (ASR) & Timbre (TTS) APIs

**Author:** Ashish Vipin  
**Role:** AI Solutions Engineer Take-Home Benchmark  
**Date:** November 2026  
**Target Platform:** Gnani Speech Services (`api.vachana.ai`)  
**Evaluated Systems:** Prisma v3.0 (Speech-to-Text) & Timbre v2.5 (Text-to-Speech)

---

## 1. Executive Summary

As an AI Solutions Engineer evaluating Gnani's speech infrastructure for prospective enterprise deployments (particularly across BFSI, contact centers, and customer service automation), I conducted an end-to-end empirical benchmark of:
- **Prisma v3.0 (ASR)** across 30 stress-test utterances divided into 11 real-world operating categories.
- **Timbre v2.5 (TTS)** across 28 test sentences covering 9 operational categories measuring synthesis latency, Real-Time Factor (RTF), speaking rate, silence ratios, and acoustic fidelity.

### Headline Metrics
* **Prisma ASR**: Micro-averaged Word Error Rate (**WER**) of **20.67%** and Character Error Rate (**CER**) of **18.99%** across the full evaluation suite, with an average inference response time of **446.6 ms**.
* **Timbre TTS**: Average automated Quality Score of **88.0 / 100** (27/28 synthesized; 1 expected edge-case failure on `"..."`), with an average Real-Time Factor (**RTF**) of **0.3257** (dropping to **0.1000** / 10x real-time on long-form content) and a mean latency of **1501.8 ms**.

Behind these aggregate figures lies a crucial finding that directly impacts enterprise solutioning: **Prisma's highest apparent WER occurs not from misrecognition, but from active Inverse Text Normalisation (ITN)**—automatically converting spelled-out numbers and spoken currency into digit-formatted entities (`"1,23,456"`, `"5,000"`, `"9876543210"`). When evaluated from an enterprise integration perspective, this behavior is a production asset rather than an accuracy failure.

---

## 2. Part 1 — Exhaustive Evaluation of Prisma ASR

### 2.1 Test-Case Design & Category Rationale

An evaluation harness must test where models actually break in production. Rather than relying solely on clean studio speech, I structured 30 test utterances across 11 critical operational dimensions:

| Category | Utterances | Primary Objective & Production Context |
|:---|:---:|:---|
| **Clean English** | 3 | Baseline accuracy check for standard Indian English conversational audio. |
| **Clean Hindi** | 3 | Core Indic baseline for standard banking queries and customer requests. |
| **Noisy Audio** | 3 | Acoustic robustness under additive Gaussian noise (10 dB SNR) simulating street and transit environments. |
| **Telephony (8 kHz)** | 2 | Bandpass filtered (300 Hz – 3400 Hz) narrowband audio typical of PSTN / IVR customer calls. |
| **Numbers & Entities** | 3 | Spoken currency in Lakhs/Crores, mobile numbers, and transaction figures where exact transcription is non-negotiable. |
| **Code-Switching (Hinglish)** | 3 | Natural bilingual switching (*"मेरा last transaction का detail भेजो"*). |
| **Accented English** | 2 | Dialectal variation across North and South Indian phonology. |
| **Domain-Specific** | 3 | Specialized terminology across Healthcare (drug dosages), Legal (court procedures), and Tech/DevOps. |
| **Long Utterances** | 2 | Complex multi-clause customer inquiries testing context memory and streaming retention. |
| **Short Utterances** | 3 | Single-word commands and confirmations (*"yes"*, *"no cancel it"*, *"हाँ"*). |
| **Regional Languages** | 3 | Multi-script coverage including Telugu (`te-IN`), Tamil (`ta-IN`), and Bengali (`bn-IN`). |

### 2.2 Metric Implementation

To ensure complete transparency and fulfill assignment criteria, both WER and CER were implemented from scratch via the classical Wagner-Fischer dynamic programming algorithm for Levenshtein edit distance:

$$\text{WER} = \frac{S + D + I}{N}$$

Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total reference words.
Micro-averaging was used across aggregate categories so that longer, multi-word queries carry proportionate weight relative to monosyllabic confirmations. Indic text normalization accounts for Unicode NFC forms while preserving Devanagari vowel signs (*matras*) and combining characters.

### 2.3 Empirical Results Breakdown

```
+--------------------------+---------+---------+-------------+-------+
| Category                 | WER (%) | CER (%) | Avg Lat (ms)| Clips |
+--------------------------+---------+---------+-------------+-------+
| Clean English            |   9.68% |  12.16% |    592.3 ms |     3 |
| Clean Hindi              |   4.55% |   8.05% |   1084.6 ms |     3 |
| Noisy Audio (10dB SNR)   |  18.75% |  16.00% |    380.5 ms |     3 |
| Telephony (8 kHz Band)   |   6.25% |   5.00% |    269.3 ms |     2 |
| Numbers & Entities       |  66.67% |  68.10% |    421.5 ms |     3 |
| Code-Switching (Hinglish)|  38.46% |  33.88% |    354.3 ms |     3 |
| Accented English         |   0.00% |   0.00% |    291.1 ms |     2 |
| Domain-Specific          |  24.24% |  13.37% |    342.0 ms |     3 |
| Long Utterance           |   9.84% |  11.11% |    796.8 ms |     2 |
| Short Utterance          |   0.00% |   0.00% |    127.6 ms |     3 |
| Regional Languages       |   6.25% |   4.12% |    258.1 ms |     3 |
+--------------------------+---------+---------+-------------+-------+
| OVERALL BENCHMARK        |  20.67% |  18.99% |    446.6 ms |    30 |
+--------------------------+---------+---------+-------------+-------+
```

### 2.4 Deep-Dive Qualitative Error Analysis

1. **The Inverse Text Normalisation (ITN) Paradox**:
   - **Ground Truth**: *"the total amount is one lakh twenty three thousand four hundred and fifty six rupees"*
   - **Prisma Output**: *"The total amount is 1,23,456 rupees."*
   - **Ground Truth**: *"my mobile number is nine eight seven six five four three two one zero"*
   - **Prisma Output**: *"My mobile number is 9876543210."*
   - *Analysis*: Standard verbatim WER penalizes every numeric token as a substitution or deletion, causing a nominal 66.67% WER. However, for any downstream conversational banking platform or CRM integration, having raw phone numbers formatted as a continuous 10-digit string and currency rendered with Indian grouping commas eliminates custom regex parsing. When pre-normalized for ITN, Prisma's numeric recognition was 100% accurate.

2. **Code-Switching & Dual-Script Rendering**:
   - In Hinglish scenarios, Prisma demonstrated intelligent bilingual script separation:
     - Reference: *"please mera last transaction ka detail bhejo"*
     - Hypothesis: *"Please मेरा last transaction का detail भेजो।"*
   - For English words spoken in a Hindi sentence, Prisma dynamically selected Latin script for loanwords (*"transaction"*, *"detail"*, *"balance"*) while using Devanagari for functional grammar (*"मेरा"*, *"का"*, *"भेजो"*). Normalising Unicode punctuation including Devanagari *danda* (`।`) eliminated false trailing-token errors (`codemix_002` scored 0.00% WER when evaluated with bilingual normalization/script mapping, though cross-script differences remain for fully transliterated words).

3. **Telephony & Acoustic Degradation**:
   - Telephony performance under genuine 8 kHz narrowband 300–3400 Hz bandpass filtering was exceptionally strong at **6.25% WER** and **5.00% CER**, with response latency averaging 269.3 ms.
   - When background noise was introduced at 10 dB SNR, WER rose moderately to 18.75% (CER 16.00%), indicating phonetic intelligibility was preserved even when word boundaries softened.

4. **Accents, Regional Languages, and Test-Set Caveat**:
   - Short commands (*"yes"*, *"no cancel it"*, *"हाँ"*) recorded **0.00% WER** with 127.6 ms mean turnaround.
   - Regional languages (`te-IN`, `ta-IN`, `bn-IN`) recorded **6.25% WER** (`te_001` and `ta_001` at 0.00%, `bn_001` at 20.00% due to a spelling variant `একাউন্ট` vs `অ্যাকাউন্ট`).
   - *Methodology Note*: The 30 ASR test clips in `data/asr/audio/` were synthesized via Timbre TTS (with synthetic noise and 8 kHz bandpass post-processing), meaning `accented_english` reflects Timbre's default `en-IN` voice (`Kaveri`) rather than distinct human regional speakers. Next-stage client POCs should supplement this manifest with customer call recordings or Common Voice / OpenSLR human clips.

---

## 3. Part 2 — Exhaustive Evaluation of Timbre TTS

### 3.1 Test-Case Design & Operational Scope

Evaluating Text-to-Speech requires moving beyond subjective impression into repeatable acoustic and algorithmic metrics. I designed 28 sentences testing:
- **Number Formats**: Indian currency notation (`₹1,23,456`), toll-free patterns (`1800-123-4567`), temperatures and percentages.
- **Abbreviations & Acronyms**: Corporate financial titles (*"CEO of HDFC Ltd."*, *"YoY growth in Q3 FY26"*), medical titles (*"Dr. R.K. Sharma from AIIMS"*).
- **Code-Switching**: Hinglish sentences requiring seamless phonetic transitions between languages.
- **Long-Form Telephony Scripts**: Full contact-center opening statements (60+ words).
- **Tricky Pronunciation**: Multi-syllable South Indian proper nouns (*"Venkatasubramaniam"*, *"Thiruvananthapuram"*), culinary loanwords (*"Worcestershire"*, *"quinoa"*).
- **Punctuation & Prosodic Variation**: Em-dashes, parenthetical lists, question intonations.
- **Special Formatting**: URLs, email addresses, fractions (`2½ cups`), train PNR codes.
- **Edge Conditions**: Rapid word repetitions, mixed scripts, minimal input.

### 3.2 Evaluation Methodology & Objective Quality Scoring

I developed an automated 5-pillar objective scoring rubric (0–100 scale) to grade every generated clip repeatably:
1. **Real-Time Factor (20 pts)**: Ratio of synthesis time to audio duration. Full points for $\text{RTF} < 0.5$, scaled down to 0 for $\text{RTF} > 2.0$.
2. **Speaking Rate (20 pts)**: Words per minute (WPM). Optimal conversational pacing defined as 120–180 WPM.
3. **Silence Distribution (20 pts)**: Non-speech audio percentage. Target range of 10%–40% ensuring natural phrasing without unnatural pauses.
4. **Signal Integrity / No Clipping (15 pts)**: Proportion of samples exceeding $\pm 0.99$ amplitude threshold (clipping distortion).
5. **Round-Trip Transcription Fidelity (25 pts)**: Synthesized audio is fed back directly into Prisma ASR to evaluate semantic preservation and phoneme intelligibility.

### 3.3 Empirical Findings Breakdown

```
+--------------------------+-------------+--------+-------------+-----------+
| Category                 | Score (/100)|  RTF   | Avg Lat (ms)| Rate (WPM)|
+--------------------------+-------------+--------+-------------+-----------+
| Numbers                  |        78.8 | 0.6547 |   2556.5 ms |  90.3 WPM |
| Abbreviations            |        91.8 | 0.2399 |   1243.6 ms | 133.4 WPM |
| Code-Switching           |        82.7 | 0.2595 |   1010.0 ms | 179.7 WPM |
| Long-Form                |        96.5 | 0.1000 |   2005.5 ms | 149.1 WPM |
| Tricky Pronunciation     |        92.5 | 0.5231 |   1852.3 ms | 170.4 WPM |
| Punctuation & Prosody    |        96.0 | 0.1611 |    876.7 ms | 150.6 WPM |
| Special Formats          |        88.7 | 0.2653 |   1675.5 ms | 110.0 WPM |
| Regional Languages       |        93.0 | 0.1926 |    737.5 ms | 125.2 WPM |
| Edge Cases               |        79.3 | 0.2134 |    881.4 ms | 206.8 WPM |
+--------------------------+-------------+--------+-------------+-----------+
| OVERALL BENCHMARK        |        88.0 | 0.3257 |   1501.8 ms | 145.4 WPM |
+--------------------------+-------------+--------+-------------+-----------+
```

### 3.4 Key Observations & Engineering Insights

1. **Synthesis Efficiency, Scaling & Tail Latency**:
   - Overall mean RTF was **0.3257** (~3.1x faster than real-time), with typical requests synthesizing in **525–1200 ms** (RTF `0.14–0.24`), and long-form paragraph scripts achieving **0.1000 RTF** (10x real-time).
   - Two tail-latency outliers (`num_hi_001` at 7541 ms / RTF `2.0492` and `pron_en_001` at 4699 ms / RTF `1.4686`) plus intermittent transient HTTP 500s (`"We are facing technical difficulties"`) highlighted the need for exponential-backoff retries in client SDKs (implemented in `GnaniClient`).
   
2. **Abbreviation Expansion & Pronunciation Strengths**:
   - Financial and administrative acronyms (*"HDFC Ltd."*, *"YoY"*, *"Q3 FY26"*, *"AIIMS"*) scored **91.8 / 100**. Timbre expanded fiscal designations naturally rather than spelling them awkwardly.
   - Multi-syllable Indic proper nouns (*"Venkatasubramaniam"*, *"Thiruvananthapuram"*, *"विश्वनाथन अय्यर"*) achieved **100 / 100** scores, while rare foreign loanwords (*"Worcestershire"*) and domain URLs (`"gnani.ai"`) remain areas where custom phonetic hints/lexicons improve fidelity.

3. **Critical Edge-Case Finding (API Hardening)**:
   - When presented with `edge_empty_001` (input: `"..."` containing punctuation without alphanumeric characters), the Timbre API failed deterministically with:  
     `HTTP 500 Internal Server Error: {"success":false,"message":"No audio generated during synthesis"}`.
   - *Solutions Engineering Recommendation*: In enterprise voicebots, empty or non-vocal prompts can occur due to upstream LLM guardrails or empty tool payloads. The API layer should catch this with an HTTP 400 Bad Request error or return a neutral 200 ms comfort pause, avoiding unhandled 500 exceptions in production monitoring.

4. **Prosody on Numeric Pacing**:
   - In numeric test cases, speaking rate dropped to ~90.3 WPM because reading complex currency amounts (*"One Lakh Twenty Three Thousand..."*) naturally requires deliberate syllabic cadence. This is desirable behavior for callers noting down verification numbers.

---

## 4. Solutions Architecture & Enterprise Recommendations

Based on these findings, here is how I would position and architect Gnani's speech stack for prospective enterprise clients:

### 1. Ingestion & Pre-Processing
* **Narrowband Telephony Ready**: Gnani's models demonstrate strong resilience on 8 kHz bandpass-filtered (300–3400 Hz) streams (**6.25% WER**). Clients do not need compute-heavy upsampling pipelines before calling Prisma.
* **Unified API Formatting**: Timbre returns clean 16-bit PCM streams (`x-audio-container: raw`). For web and mobile clients requiring direct media element playback, an edge proxy or lightweight wrapper should encapsulate headers into standard RIFF WAV or Opus/Ogg containers.

### 2. Conversational Voicebot Latency Budget
For interactive telephony voice agents, human conversation tolerates a maximum turn-taking budget of ~1500–1800 ms.
* **Prisma ASR**: ~447 ms mean turnaround (sub-300 ms on short commands and telephony).
* **LLM Reasoning & Orchestration**: ~500–700 ms budget.
* **Timbre TTS (Streaming/SSE + Client Retries)**: With typical non-outlier RTF around `0.15–0.24` (and `0.10` on long-form), time-to-first-audio chunk via Server-Sent Events (`/api/v1/tts/sse`) stays responsive, while client-side exponential backoff protects against transient 500s.

### 3. Continuous Quality Assurance Framework
Enterprises should not treat speech quality as static. I recommend setting up an automated weekly regression harness using the test manifest developed here:
- Automatically route 500 randomized production calls through the evaluation suite.
- Monitor ITN alignment and acoustic drift.
- Set automated alerting thresholds if aggregate WER degrades by more than 3% following upstream model updates.

---

## 5. Repository & Reproducibility Guide

The complete benchmarking suite, test audio, JSON manifests, and unit tests are hosted at:  
**GitHub Repository:** [https://github.com/Ashishvipin/gnani-speech-eval](https://github.com/Ashishvipin/gnani-speech-eval)

### Running the Suite Locally
```bash
# Clone repository and install dependencies
git clone https://github.com/Ashishvipin/gnani-speech-eval.git
cd gnani-speech-eval
pip install -r requirements.txt

# Run metric verification unit tests
python tests/test_metrics.py

# Execute full evaluation against live Gnani APIs
python scripts/run_evaluation.py

# Review interactive breakdown
jupyter notebook evaluation_notebook.ipynb
```
