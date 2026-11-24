# Evaluation Report: Gnani Prisma (ASR) & Timbre (TTS)

**Author:** Ashish Vipin Korumelli  
**Date:** November 2026

---

## 1. Objective

This report presents the design, methodology, and findings from an independent evaluation of two speech APIs from Gnani's platform:

- **Prisma v3.0** — Automatic Speech Recognition (ASR)
- **Timbre v2.5** — Text-to-Speech (TTS)

The goal was to evaluate both systems the way I would before recommending them to an enterprise customer — by defining what "good" looks like for their use case, building test cases around that definition, and measuring systematically against it. I focused on the Indian voice landscape: multilingual, code-switched, noisy, and telephony-grade audio that reflects what real customers actually throw at these APIs.

---

## 2. Prisma ASR Evaluation

### 2.1 Test Design

I designed **30 test clips** across **10 categories**, each chosen to stress a different axis that matters in production:

| Category | # Clips | Why It Matters |
|---|---|---|
| Clean English | 3 | Baseline — if this fails, nothing else matters |
| Clean Hindi | 3 | Most-used Indic language, sets the floor for regional quality |
| Noisy Audio | 3 | Real calls have background noise, street sounds, crowd chatter |
| Telephony (8 kHz) | 2 | Contact centres compress audio to 8 kHz narrowband — huge portion of enterprise traffic |
| Numbers & Entities | 3 | Currency, phone numbers, dates — these must be exact for banking/insurance use cases |
| Code-switching (Hinglish) | 3 | Indians switch between Hindi and English mid-sentence constantly |
| Accented English | 2 | South Indian vs. North Indian English accents differ significantly |
| Domain-specific | 3 | Medical, legal, and tech vocabulary that general models often butcher |
| Long Utterances | 2 | Real customer queries can be 30+ words; tests for context maintenance |
| Short Utterances | 3 | Single-word confirmations ("yes", "haan") — often missed or hallucinated |
| Regional Languages | 3 | Telugu, Tamil, Bengali — testing breadth beyond Hindi-English |

**Data sources:** A mix of publicly available datasets (Common Voice, OpenSLR) and self-generated clips using macOS TTS with added noise and telephony simulation for stress conditions. For a production evaluation, I would source actual call recordings from the customer's domain.

### 2.2 Metrics

I implemented **WER (Word Error Rate)** and **CER (Character Error Rate)** from scratch using the Levenshtein edit-distance algorithm — no library calls for the core computation. The implementation breaks down errors into:

- **Substitutions (S):** A word in the reference was replaced with a different word
- **Deletions (D):** A word in the reference was dropped entirely
- **Insertions (I):** An extra word appeared that is not in the reference

**WER = (S + D + I) / N**, where N is the total words in the reference.

I used **micro-averaged WER** for aggregate numbers, which sums all errors and all reference words across clips before dividing. This prevents short utterances from disproportionately skewing the aggregate.

CER uses the same algorithm at the character level (spaces removed), which is especially useful for agglutinative Indic languages where word boundaries are ambiguous.

### 2.3 Key Findings

> *Note: The numbers below are from placeholder audio. When you run this against the live Prisma API with real recordings, these will be replaced with actual measurements.*

**Where I expect Prisma to be strong:**
- Clean Hindi and English in quiet conditions — Gnani trains on real Indian data, so baseline accuracy should be competitive
- Indian number formats (lakhs, crores) — a common differentiator vs. English-first ASR
- Multi-language detection with `indic-auto` — a genuinely useful feature for mixed-language call centres

**Where to watch carefully:**
- **Code-switching accuracy:** Hinglish is hard for every ASR — the model needs to handle "mera account balance check karo" without tripping up at the language boundary
- **Telephony audio quality:** 8 kHz narrowband audio with compression artifacts is where many ASR systems lose 10-15% WER compared to wideband
- **Domain vocabulary:** Medical terms (paracetamol, ophthalmologist), legal terms (affidavit, magistrate), and tech acronyms need to be in the vocabulary
- **Short utterance reliability:** Single-word confirmations ("yes", "haan") are easy to miss or hallucinate as longer phrases

### 2.4 What I Would Recommend to a Customer

Before going live, I would run Prisma against 200-500 clips sampled from their actual call recordings — not generic test sets. The categories above tell you *what* to test, but the data has to come from *their* domain. I would also set up a weekly spot-check pipeline that samples production transcriptions and flags when WER drifts above the agreed threshold.

---

## 3. Timbre TTS Evaluation

### 3.1 Test Design

I designed **28 test sentences** across **9 categories**:

| Category | # Sentences | What I Am Testing |
|---|---|---|
| Numbers | 4 | Currency (₹), phone numbers, decimals, percentages — must read correctly, not literally |
| Abbreviations | 4 | Dr., EMI, Rs., HDFC, FY26 — should expand naturally |
| Code-switching | 3 | Hinglish sentences where the voice needs to switch pronunciation mid-sentence |
| Long-form | 2 | Multi-sentence banking scripts — tests for pacing and stamina |
| Tricky Pronunciation | 4 | Worcestershire, Venkatasubramaniam, Thiruvananthapuram — real-world names people use |
| Punctuation & Prosody | 2 | Questions, exclamations, lists — intonation should match the punctuation |
| Special Formats | 3 | URLs, email addresses, fractions, PNR codes |
| Regional Languages | 2 | Telugu and Tamil — breadth check |
| Edge Cases | 3 | Ellipsis-only input, repeated words, mixed Devanagari-Latin script |

### 3.2 Evaluation Methodology

TTS evaluation is harder than ASR because "good speech" is subjective. I designed a methodology that is as objective and repeatable as possible without requiring a panel of listeners for every run:

**Automated Metrics (scored out of 100):**

| Component | Weight | What It Measures |
|---|---|---|
| RTF (Real-Time Factor) | 20 pts | Is synthesis faster than real-time? RTF < 1.0 is the production bar |
| Speaking Rate | 20 pts | 120-180 WPM is natural English pacing; outside this feels rushed or sluggish |
| Silence Ratio | 20 pts | 10-40% silence is normal (pauses between phrases); too high means unnatural gaps |
| Clipping | 15 pts | Audio peaks hitting ±1.0 means distortion — should be near zero |
| Round-trip Fidelity | 25 pts | Synthesize text → transcribe back with Prisma → compare. If the ASR can't understand the TTS output, a human will struggle too |

**Why these metrics:**
- **RTF** directly determines whether the TTS can be used in real-time IVR flows — if it cannot keep up with the conversation pace, it is a non-starter
- **Speaking rate** catches the two most common TTS failure modes: rushing through text so fast it becomes unintelligible, or dragging so slowly the caller hangs up
- **Round-trip ASR** is the most valuable automated check — it catches gross mispronunciations, skipped words, and garbled output without needing a human in the loop
- **Silence ratio and clipping** are audio-engineering fundamentals that catch synthesis artifacts

**For the final recommendation,** I would add a structured MOS (Mean Opinion Score) listening test with 5-10 listeners rating naturalness on a 1-5 scale. But for rapid iteration and regression testing, the automated pipeline is enough to catch most issues.

### 3.3 Key Findings

> *Note: Actual scores will be populated when run against the live Timbre API.*

**Where I expect Timbre to do well:**
- Hindi and English in straightforward sentences — Gnani has focused heavily on Indian voice quality
- Indian-format number reading (lakhs, crores, rupee amounts) — this is table-stakes for banking IVR
- Reasonable latency for batch synthesis

**Where to watch:**
- **Abbreviation expansion consistency:** "Dr." should become "Doctor" in medical context but "Drive" in an address. Context-dependent expansion is hard
- **Code-switching prosody:** When the voice switches from Hindi to English mid-sentence, the transition should sound natural, not like two different recordings spliced together
- **Long-form pacing:** Over 100+ words, TTS systems tend to either rush or develop monotone intonation — both are problems for banking scripts
- **Edge cases like URLs and mixed-script text:** These are common in modern customer communications and many TTS systems either skip special characters or read them literally

### 3.4 Latency Considerations

For enterprise voice bots, the latency budget is tight:
- **IVR / telephony:** Total round-trip (STT → processing → TTS) needs to be under 2 seconds for the conversation to feel natural
- **TTS alone should target RTF < 0.5** to leave headroom for ASR and business logic

I measure both wall-clock latency (time from API call to full audio response) and RTF (latency normalized by output duration) to separate network overhead from actual synthesis speed.

---

## 4. Overall Recommendation

Prisma and Timbre are built for the Indian market — that is their structural advantage over global ASR/TTS providers. The language coverage, Indian number formatting, and multilingual detection are features that most competitors either lack or handle as an afterthought.

**For a customer evaluation, I would focus the conversation on three questions:**

1. **Does accuracy hold up on their actual audio?** Generic benchmarks are encouraging but not sufficient. Run 200+ clips from their production traffic and measure WER in the conditions that matter to them (their accents, their noise profiles, their domain vocabulary).

2. **Is the latency budget feasible for their architecture?** Real-time telephony has much tighter requirements than batch transcription. Measure end-to-end, including network round-trip to the API, not just model inference.

3. **What is the regression story?** Models get updated. The evaluation pipeline I have built can be run on a schedule to catch quality changes before they hit production. This is not a one-time check — it is an ongoing quality gate.

---

## 5. Reproducing This Evaluation

```bash
# 1. Clone and install
git clone https://github.com/ashishvipin/gnani-speech-eval.git
cd gnani-speech-eval
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your GNANI_API_KEY from app.gnani.ai

# 3. Prepare test audio (placeholder clips — replace with real recordings)
python scripts/prepare_test_data.py

# 4. Run the full evaluation
python scripts/run_evaluation.py

# Results will be saved to results/asr_results.json and results/tts_results.json
```

The pipeline is designed to be reusable: swap in different audio files, modify the test manifest, and re-run. All metrics are computed from scratch and all results are saved as structured JSON for downstream analysis.
