"""Runtime configuration loaded from .env / environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _opt_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


def _autodetect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class Config:
    hf_token: str | None
    model: str
    device: str
    compute_type: str
    language: str | None
    min_speakers: int | None
    max_speakers: int | None
    diarize_model: str | None
    incoming_dir: Path
    transcripts_dir: Path
    models_dir: Path

    @classmethod
    def load(cls) -> "Config":
        device = os.getenv("DEVICE", "").strip() or _autodetect_device()
        compute_type = os.getenv("COMPUTE_TYPE", "").strip() or (
            "float16" if device == "cuda" else "int8"
        )
        incoming = Path(os.getenv("INCOMING_DIR", "incoming"))
        transcripts = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts"))
        models = Path(os.getenv("MODELS_DIR", "models"))
        for p in (incoming, transcripts, models):
            (ROOT / p).mkdir(parents=True, exist_ok=True)
        return cls(
            hf_token=os.getenv("HF_TOKEN") or None,
            model=os.getenv("WHISPER_MODEL", "large-v3").strip(),
            device=device,
            compute_type=compute_type,
            language=(os.getenv("LANGUAGE", "").strip() or None),
            min_speakers=_opt_int("MIN_SPEAKERS"),
            max_speakers=_opt_int("MAX_SPEAKERS"),
            diarize_model=(os.getenv("DIARIZE_MODEL", "").strip() or None),
            incoming_dir=ROOT / incoming,
            transcripts_dir=ROOT / transcripts,
            models_dir=ROOT / models,
        )
