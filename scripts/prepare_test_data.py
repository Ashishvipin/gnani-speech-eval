#!/usr/bin/env python3
"""
Real-Human Speech Dataset Preparation Script

Fetches a compact, multi-lingual test set of real human speech recordings
with verified ground-truth transcripts from public datasets:
  - AI4Bharat Svarah (Indian-accented English)
  - Google FLEURS (Hindi hi_in test set via streaming tar extraction)
  - OpenSLR 66 (Telugu male & female native speakers via HTTP Range ZIP)
  - OpenSLR 65 (Tamil male & female native speakers via HTTP Range ZIP)
  - OpenSLR 37 (Indian Bengali native speakers via HTTP Range ZIP)

Also constructs deterministic acoustic stress variants (10 dB babble noise and
8 kHz bandpass + G.711 mu-law PSTN telephony) directly from the real speech
recordings, and writes data/asr/real_manifest.json.

Usage:
    python scripts/prepare_test_data.py
"""

import io
import json
import os
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.audio_utils import add_noise, load_wav, save_wav, simulate_telephony

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
REAL_AUDIO_DIR = PROJECT_ROOT / "data" / "asr" / "real_audio"
REAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "asr" / "real_manifest.json"
REAL_STRESS_MANIFEST_PATH = PROJECT_ROOT / "data" / "asr" / "real_stress_manifest.json"


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable file-like wrapper using HTTP Range requests for remote ZIPs."""

    def __init__(self, url: str, tail_bytes: int = 524288):
        self.url = url
        self.session = requests.Session()
        head = self._request("head", url, allow_redirects=True, timeout=30)
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        tail = min(self.size, tail_bytes)
        self.tail_start = self.size - tail
        resp = self._request(
            "get",
            url,
            headers={"Range": f"bytes={self.tail_start}-{self.size - 1}"},
            timeout=45,
        )
        self.tail_buf = resp.content

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(4):
            try:
                resp = getattr(self.session, method)(url, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                if attempt == 3:
                    raise
                import time
                time.sleep(1.0 * (attempt + 1))

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n)
        if self.pos >= self.tail_start:
            chunk = self.tail_buf[self.pos - self.tail_start : end - self.tail_start]
        else:
            resp = self._request(
                "get",
                self.url,
                headers={"Range": f"bytes={self.pos}-{end - 1}"},
                timeout=45,
            )
            chunk = resp.content
        self.pos += len(chunk)
        return chunk


def fetch_svarah_english() -> List[Dict]:
    """
    Download 5 real Indian-accented English clips from AI4Bharat Svarah (`CC BY 4.0`).

    When `HF_TOKEN` is configured (after accepting dataset terms at
    https://huggingface.co/datasets/ai4bharat/Svarah), pulls directly from
    the official `ai4bharat/Svarah` repository via the Hugging Face Datasets
    Server API, selecting 5 complete, regionally diverse speaker utterances.
    Falls back to the public `saisundaresan01/svarah-eval` mirror only if
    `HF_TOKEN` is unset.
    """
    print("  Fetching AI4Bharat Svarah Indian-English samples...")
    hf_token = os.getenv("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    if hf_token:
        rows_url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=ai4bharat%2FSvarah&config=default&split=test&offset=0&limit=100"
        )
        resp = requests.get(rows_url, headers=headers, timeout=30)
        resp.raise_for_status()
        by_idx = {item["row_idx"]: item["row"] for item in resp.json()["rows"]}

        # Diverse native states/languages & genders with complete sentences
        # (keeping rows 0 & 4 as en_001 & en_003 for paired stress continuity):
        #   0: Female Kannada (Karnataka), 5: Female Bengali (West Bengal),
        #   4: Male Maithili (Bihar), 16: Male Nepali (West Bengal),
        #   29: Female Tamil (Kerala)
        selected_indices = [0, 5, 4, 16, 29]
        entries = []
        for idx, row_idx in enumerate(selected_indices, start=1):
            row = by_idx[row_idx]
            audio_info = row["audio_filepath"][0]
            wav_resp = requests.get(audio_info["src"], headers=headers, timeout=30)
            wav_resp.raise_for_status()

            clip_id = f"real_svarah_en_{idx:03d}"
            out_path = REAL_AUDIO_DIR / f"{clip_id}.wav"
            out_path.write_bytes(wav_resp.content)
            samples, sr = load_wav(str(out_path))
            save_wav(samples, sr, str(out_path))

            lang = row.get("primary_language", "")
            state = row.get("native_place_state", "")
            gender = row.get("gender", "")
            ref_text = row["text"].strip()
            entries.append({
                "id": clip_id,
                "audio_path": f"real_audio/{clip_id}.wav",
                "reference": ref_text,
                "language": "en-IN",
                "category": "real_indian_english",
                "source": f"AI4Bharat Svarah (official ai4bharat/Svarah, {gender}/{lang}/{state})",
                "license": "CC BY 4.0",
                "tags": ["real_human", "indian_english", "svarah", "official"],
            })
            print(f"    + {clip_id} ({gender}, {lang}/{state}): {ref_text[:60]}")
        return entries

    print("    HF_TOKEN not set; falling back to public saisundaresan01/svarah-eval mirror...")
    meta_url = "https://huggingface.co/datasets/saisundaresan01/svarah-eval/resolve/main/data/test.jsonl"
    tar_url = "https://huggingface.co/datasets/saisundaresan01/svarah-eval/resolve/main/data/audio.tar.gz"

    rows = {}
    for line in requests.get(meta_url, timeout=30).content.decode("utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            rows[obj["audio_filepath"]] = obj["reference_transcription"]

    tar_bytes = requests.get(tar_url, timeout=60).content
    entries = []
    idx = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            fname = Path(member.name).name
            if fname not in rows:
                continue
            idx += 1
            clip_id = f"real_svarah_en_{idx:03d}"
            out_path = REAL_AUDIO_DIR / f"{clip_id}.wav"
            out_path.write_bytes(tf.extractfile(member).read())
            entries.append({
                "id": clip_id,
                "audio_path": f"real_audio/{clip_id}.wav",
                "reference": rows[fname],
                "language": "en-IN",
                "category": "real_indian_english",
                "source": "AI4Bharat Svarah (via saisundaresan01/svarah-eval mirror, CC BY 4.0)",
                "license": "CC BY 4.0",
                "tags": ["real_human", "indian_english", "svarah"],
            })
            print(f"    + {clip_id}: {rows[fname][:60]}")
    return entries


def fetch_fleurs_hindi(limit: int = 5) -> List[Dict]:
    """Stream the first few real Hindi clips from Google FLEURS hi_in test split."""
    print("  Fetching Google FLEURS Hindi (hi_in) test samples...")
    tsv_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hi_in/test.tsv"
    tar_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hi_in/audio/test.tar.gz"

    tsv_map = {}
    for line in requests.get(tsv_url, timeout=30).content.decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            wav_name = parts[1].strip()
            raw_text = parts[2].strip()
            gender = parts[6].strip() if len(parts) >= 7 else "UNKNOWN"
            if wav_name.endswith(".wav") and raw_text:
                tsv_map[wav_name] = (raw_text, gender)

    entries = []
    seen_texts = set()
    # Stream gzipped tar incrementally until we have `limit` valid distinct Hindi clips
    resp = requests.get(tar_url, stream=True, timeout=60)
    resp.raise_for_status()
    decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = bytearray()
    offset = 0

    try:
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            buf.extend(decomp.decompress(chunk))
            # Parse complete 512-byte tar entries from buf[offset:]
            while len(buf) - offset >= 512:
                header = bytes(buf[offset : offset + 512])
                if header == b"\x00" * 512:
                    offset += 512
                    continue
                name = header[0:100].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                size_oct = header[124:136].split(b"\x00", 1)[0].strip()
                size = int(size_oct, 8) if size_oct else 0
                typeflag = chr(header[156]) if header[156] else "0"
                padded = ((size + 511) // 512) * 512
                if len(buf) - offset < 512 + padded:
                    break
                payload = bytes(buf[offset + 512 : offset + 512 + size])
                offset += 512 + padded

                fname = Path(name).name
                if typeflag in ("0", "\x00") and fname in tsv_map:
                    ref_text, gender = tsv_map[fname]
                    # Prefer distinct utterances across speakers
                    if ref_text in seen_texts:
                        continue
                    seen_texts.add(ref_text)
                    idx = len(entries) + 1
                    clip_id = f"real_fleurs_hi_{idx:03d}"
                    out_path = REAL_AUDIO_DIR / f"{clip_id}.wav"
                    out_path.write_bytes(payload)
                    # Convert IEEE-float32 FLEURS WAV to standard 16-bit PCM WAV
                    samples, sr = load_wav(str(out_path))
                    save_wav(samples, sr, str(out_path))
                    entries.append({
                        "id": clip_id,
                        "audio_path": f"real_audio/{clip_id}.wav",
                        "reference": ref_text,
                        "language": "hi-IN",
                        "category": "real_hindi",
                        "source": f"Google FLEURS hi_in ({gender.lower()})",
                        "tags": ["real_human", "hindi", "fleurs"],
                    })
                    print(f"    + {clip_id} ({gender}): {ref_text[:55]}")
                    if len(entries) >= limit:
                        return entries
    finally:
        resp.close()
    return entries


def fetch_openslr_clips(
    label: str,
    language: str,
    category: str,
    zip_url: str,
    tsv_url: str,
    prefix_id: str,
    limit: int = 2,
    skip_first: int = 0,
    internal_tsv: bool = False,
) -> List[Dict]:
    """Extract a few real speaker clips from a remote OpenSLR ZIP via HTTP Range."""
    print(f"  Fetching {label} samples via HTTP Range ZIP...")
    with zipfile.ZipFile(HTTPRangeFile(zip_url)) as zf:
        if internal_tsv:
            tsv_lines = zf.read("line_index.tsv").decode("utf-8").splitlines()
        else:
            tsv_lines = requests.get(tsv_url, timeout=30).content.decode("utf-8").splitlines()

        available = {Path(n).name: n for n in zf.namelist() if n.endswith(".wav")}
        entries = []
        matched = 0
        for line in tsv_lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            ref_text = parts[-1].strip()
            # Filter out OpenSLR annotation artifacts like '_pronoun' or empty lines
            if not ref_text or "_" in ref_text or len(ref_text.split()) < 3:
                continue
            wav_key = key if key.endswith(".wav") else f"{key}.wav"
            if wav_key not in available:
                continue
            matched += 1
            if matched <= skip_first:
                continue

            idx = len(entries) + 1
            clip_id = f"{prefix_id}_{idx:03d}"
            out_path = REAL_AUDIO_DIR / f"{clip_id}.wav"
            out_path.write_bytes(zf.read(available[wav_key]))
            entries.append({
                "id": clip_id,
                "audio_path": f"real_audio/{clip_id}.wav",
                "reference": ref_text,
                "language": language,
                "category": category,
                "source": f"{label} (human)",
                "tags": ["real_human", language[:2], "openslr"],
            })
            print(f"    + {clip_id}: {ref_text[:55]}")
            if len(entries) >= limit:
                break
    return entries


def create_real_stress_variants(clean_entries: List[Dict]) -> List[Dict]:
    """
    Derive deterministic paired stress variants (20 dB, 10 dB, 5 dB speech-modulated
    babble noise + 8 kHz G.711 mu-law PSTN telephony) from 4 CC-BY-4.0 real speech
    utterances (2 Indian English Svarah, 2 Hindi FLEURS) for clean-vs-degraded deltas.
    """
    print("  Generating paired multi-SNR & telephony stress variants from real speech...")
    variants = []

    target_ids = [
        "real_svarah_en_001",
        "real_svarah_en_003",
        "real_fleurs_hi_001",
        "real_fleurs_hi_005",
    ]
    by_id = {e["id"]: e for e in clean_entries}
    base_clips = [by_id[cid] for cid in target_ids if cid in by_id]

    for idx, base in enumerate(base_clips):
        base_id = base["id"]
        samples, sr = load_wav(str(PROJECT_ROOT / "data" / "asr" / base["audio_path"]))

        for snr_db in (20, 10, 5):
            seed = 42 + idx * 10 + snr_db
            noisy = add_noise(samples, snr_db=float(snr_db), noise_type="babble", seed=seed, sample_rate=sr)
            cid = f"{base_id}_babble_{snr_db}db"
            save_wav(noisy, sr, str(REAL_AUDIO_DIR / f"{cid}.wav"))
            variants.append({
                "id": cid,
                "parent_clip_id": base_id,
                "condition": f"babble_{snr_db}db",
                "audio_path": f"real_audio/{cid}.wav",
                "reference": base["reference"],
                "language": base["language"],
                "category": f"paired_babble_{snr_db}db",
                "source": f"{base_id} + {snr_db} dB babble (seed={seed})",
                "license": base.get("license", "CC BY 4.0"),
                "tags": ["paired_stress", f"babble_{snr_db}db", base["language"][:2]],
            })

        tel_seed = 100 + idx
        tel = simulate_telephony(samples, sr, seed=tel_seed, use_mulaw=True)
        cid_tel = f"{base_id}_telephony"
        save_wav(tel, 8000, str(REAL_AUDIO_DIR / f"{cid_tel}.wav"))
        variants.append({
            "id": cid_tel,
            "parent_clip_id": base_id,
            "condition": "telephony_8khz_mulaw",
            "audio_path": f"real_audio/{cid_tel}.wav",
            "reference": base["reference"],
            "language": base["language"],
            "category": "paired_telephony_8khz",
            "source": f"{base_id} -> 8 kHz bandpass + mu-law (seed={tel_seed})",
            "license": base.get("license", "CC BY 4.0"),
            "tags": ["paired_stress", "telephony", "8khz", "mulaw", base["language"][:2]],
        })

    return variants


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare real human ASR & paired stress manifests")
    parser.add_argument(
        "--stress-only",
        action="store_true",
        help="Reuse already-downloaded clean real clips and rebuild paired stress variants only",
    )
    parser.add_argument(
        "--refresh-svarah",
        action="store_true",
        help="Re-fetch the 5 Svarah clips (from official ai4bharat/Svarah when HF_TOKEN is set) and rebuild stress set",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPARING REAL HUMAN ASR TEST DATASET")
    print("=" * 60)
    REAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Remove legacy 2-clip stress artifacts if present
    for old_name in ("real_noisy_en_001.wav", "real_noisy_hi_001.wav", "real_telephony_en_001.wav", "real_telephony_hi_001.wav"):
        old_path = REAL_AUDIO_DIR / old_name
        if old_path.exists():
            old_path.unlink()

    if args.refresh_svarah and REAL_MANIFEST_PATH.exists():
        with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
            existing = [
                e for e in json.load(f)
                if not e["id"].startswith("real_svarah_")
            ]
        manifest = fetch_svarah_english() + existing
    elif args.stress_only and REAL_MANIFEST_PATH.exists():
        with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = [
                e for e in json.load(f)
                if e.get("category") not in ("real_noisy_audio", "real_telephony")
            ]
    else:
        manifest = []
        manifest.extend(fetch_svarah_english())
        manifest.extend(fetch_fleurs_hindi(limit=5))
        manifest.extend(fetch_openslr_clips(
            label="OpenSLR 66 Telugu Female",
            language="te-IN",
            category="real_regional_indic",
            zip_url="https://openslr.trmal.net/resources/66/te_in_female.zip",
            tsv_url="https://openslr.trmal.net/resources/66/line_index_female.tsv",
            prefix_id="real_openslr_te_f",
            limit=2,
        ))
        manifest.extend(fetch_openslr_clips(
            label="OpenSLR 66 Telugu Male",
            language="te-IN",
            category="real_regional_indic",
            zip_url="https://openslr.trmal.net/resources/66/te_in_male.zip",
            tsv_url="https://openslr.trmal.net/resources/66/line_index_male.tsv",
            prefix_id="real_openslr_te_m",
            limit=1,
        ))
        manifest.extend(fetch_openslr_clips(
            label="OpenSLR 65 Tamil Female",
            language="ta-IN",
            category="real_regional_indic",
            zip_url="https://openslr.trmal.net/resources/65/ta_in_female.zip",
            tsv_url="https://openslr.trmal.net/resources/65/line_index_female.tsv",
            prefix_id="real_openslr_ta_f",
            limit=2,
        ))
        manifest.extend(fetch_openslr_clips(
            label="OpenSLR 65 Tamil Male",
            language="ta-IN",
            category="real_regional_indic",
            zip_url="https://openslr.trmal.net/resources/65/ta_in_male.zip",
            tsv_url="https://openslr.trmal.net/resources/65/line_index_male.tsv",
            prefix_id="real_openslr_ta_m",
            limit=1,
        ))
        manifest.extend(fetch_openslr_clips(
            label="OpenSLR 37 Bengali",
            language="bn-IN",
            category="real_regional_indic",
            zip_url="https://openslr.trmal.net/resources/37/bn_in.zip",
            tsv_url="",
            prefix_id="real_openslr_bn",
            limit=2,
            internal_tsv=True,
        ))
        for e in manifest:
            if e["id"].startswith("real_fleurs_"):
                e["license"] = "CC BY 4.0"
            elif e["id"].startswith("real_openslr_"):
                e["license"] = "CC BY-SA 4.0"

    stress_manifest = create_real_stress_variants(manifest)

    with open(REAL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(REAL_STRESS_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(stress_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  Prepared {len(manifest)} clean human clips -> {REAL_MANIFEST_PATH}")
    print(f"  Prepared {len(stress_manifest)} paired stress clips -> {REAL_STRESS_MANIFEST_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
