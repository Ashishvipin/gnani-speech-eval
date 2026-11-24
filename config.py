"""
Configuration for Gnani Speech Evaluation.
Loads API key from environment or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# ── API Configuration ──────────────────────────────────────────────
GNANI_API_KEY = os.getenv("GNANI_API_KEY", "")
GNANI_BASE_URL = "https://api.vachana.ai"

# ── ASR (Prisma) Configuration ─────────────────────────────────────
ASR_ENDPOINT = f"{GNANI_BASE_URL}/stt/v3"
ASR_MODEL = "prisma-v3.0"

# Supported language codes for ASR
ASR_LANGUAGES = {
    "hi-IN": "Hindi",
    "en-IN": "English (Indian)",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "indic-auto": "Auto-detect",
}

# ── TTS (Timbre) Configuration ─────────────────────────────────────
TTS_ENDPOINT = f"{GNANI_BASE_URL}/api/v1/tts/inference"
TTS_STREAMING_ENDPOINT = f"{GNANI_BASE_URL}/api/v1/tts/sse"
TTS_MODEL = "timbre-v2.5"

# Available voices per language
TTS_VOICES = {
    "hi-IN": {"female": "Nalini", "male": "Deepak"},
    "en-IN": {"female": "Kaveri", "male": "Pranav"},
    "hi-en": {"female": "Poorvi"},  # Hinglish / code-mixed
}

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ASR_DATA_DIR = DATA_DIR / "asr"
ASR_AUDIO_DIR = ASR_DATA_DIR / "audio"
TTS_DATA_DIR = DATA_DIR / "tts"
TTS_OUTPUT_DIR = TTS_DATA_DIR / "output"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Evaluation Defaults ────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 16000  # 16 kHz — standard for telephony ASR
TTS_SAMPLE_RATE = 48000      # 48 kHz — Timbre default output
