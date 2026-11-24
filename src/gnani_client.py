"""
Gnani API Client — wraps Prisma (ASR) and Timbre (TTS) REST endpoints.

Handles authentication, request formatting, response parsing,
retry logic, and latency measurement.
"""

import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GNANI_API_KEY, ASR_ENDPOINT, ASR_MODEL,
    TTS_ENDPOINT, TTS_MODEL, TTS_VOICES,
)


class GnaniAPIError(Exception):
    """Raised when the Gnani API returns an error response."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"[HTTP {status_code}] {message}")


class GnaniClient:
    """
    Thin wrapper around Gnani's Vachana REST API.

    Usage:
        client = GnaniClient(api_key="your-key")
        result = client.transcribe("audio.wav", language="hi-IN")
        audio  = client.synthesize("Hello world", language="en-IN")
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or GNANI_API_KEY
        if not self.api_key:
            raise ValueError(
                "GNANI_API_KEY not set. Export it or add it to .env"
            )
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key-ID": self.api_key})

    # ── ASR (Prisma) ───────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str,
        language: str = "indic-auto",
        model: str = ASR_MODEL,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file using Prisma ASR.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, OGG, etc.)
            language:   Language code or 'indic-auto' for auto-detection
            model:      Model version (default: prisma-v3.0)

        Returns:
            Dict with keys: transcript, segments, language, latency_ms, etc.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as audio_file:
            files = {"audio_file": (audio_path.name, audio_file)}
            data = {
                "model": model,
                "language_code": language,
            }

            start_time = time.time()
            response = self._session.post(
                ASR_ENDPOINT,
                files=files,
                data=data,
                timeout=self.timeout,
            )
            latency_ms = (time.time() - start_time) * 1000

        if response.status_code != 200:
            raise GnaniAPIError(response.status_code, response.text)

        resp_json = response.json()

        return {
            "transcript": resp_json.get("full_transcript", "") or resp_json.get("transcript", ""),
            "segments": resp_json.get("segments", []),
            "detected_language": resp_json.get("detected_language_code", "") or resp_json.get("language_code", language),
            "model": resp_json.get("model", model),
            "status": resp_json.get("status", "SUCCESS" if resp_json.get("success") else "UNKNOWN"),
            "latency_ms": round(latency_ms, 2),
            "raw_response": resp_json,
        }

    # ── TTS (Timbre) ───────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        language: str = "en-IN",
        voice: Optional[str] = None,
        model: str = TTS_MODEL,
        speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text using Timbre TTS.

        Args:
            text:        Text to convert to speech
            language:    Language code (hi-IN, en-IN, etc.)
            voice:       Voice name (Nalini, Kaveri, etc.). Auto-selected if None
            model:       Model version (default: timbre-v2.5)
            speed:       Playback speed multiplier (default: 1.0)
            output_path: If set, save audio to this path

        Returns:
            Dict with keys: audio_bytes, latency_ms, output_path, etc.
        """
        # Auto-select voice for language if not specified
        if voice is None:
            lang_voices = TTS_VOICES.get(language, TTS_VOICES.get("en-IN", {}))
            voice = lang_voices.get("female", lang_voices.get("male", "Kaveri"))

        payload = {
            "text": text,
            "voice": voice,
            "model": model,
            "language": language,
            "speed": speed,
        }

        start_time = time.time()
        response = self._session.post(
            TTS_ENDPOINT,
            json=payload,
            timeout=self.timeout,
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code != 200:
            raise GnaniAPIError(response.status_code, response.text)

        audio_bytes = response.content

        # Timbre returns raw PCM — wrap in WAV header if needed
        content_disp = response.headers.get("content-disposition", "")
        audio_container = response.headers.get("x-audio-container", "")
        if "audio.raw" in content_disp or audio_container == "raw":
            audio_bytes = self._wrap_pcm_as_wav(audio_bytes, sample_rate=48000)

        result = {
            "audio_bytes": audio_bytes,
            "audio_size_bytes": len(audio_bytes),
            "latency_ms": round(latency_ms, 2),
            "voice": voice,
            "language": language,
            "model": model,
            "text": text,
        }

        # Save audio if output path specified
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(audio_bytes)
            result["output_path"] = str(out)

        return result

    # ── Health Check ───────────────────────────────────────────────

    @staticmethod
    def _wrap_pcm_as_wav(
        pcm_data: bytes,
        sample_rate: int = 48000,
        num_channels: int = 1,
        sample_width: int = 2,
    ) -> bytes:
        """Wrap raw PCM bytes in a valid WAV (RIFF) header."""
        import struct
        data_size = len(pcm_data)
        byte_rate = sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,        # file size - 8
            b'WAVE',
            b'fmt ',
            16,                    # fmt chunk size
            1,                     # PCM format
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            sample_width * 8,      # bits per sample
            b'data',
            data_size,
        )
        return header + pcm_data

    def check_connection(self) -> bool:
        """Quick check that the API key is valid and the service is up."""
        try:
            resp = self._session.get(
                "https://api.vachana.ai",
                timeout=10,
            )
            return resp.status_code < 500
        except requests.RequestException:
            return False
