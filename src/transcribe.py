"""WhisperX transcription + diarization pipeline."""
from __future__ import annotations

import gc
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import whisperx

from .config import Config

log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}

# Pipeline stage names — kept here so the UI and pipeline agree on labels.
STAGE_LOAD = "LOAD"
STAGE_ASR = "ASR"
STAGE_ALIGN = "ALIGN"
STAGE_DIARIZE = "DIARIZE"
STAGE_WRITE = "WRITE"
STAGES: tuple[str, ...] = (STAGE_LOAD, STAGE_ASR, STAGE_ALIGN, STAGE_DIARIZE, STAGE_WRITE)

StageCallback = Callable[[str, dict[str, Any]], None]
ProgressCallback = Callable[[str, float], None]  # (stage, fraction in 0..1)


def _srt_ts(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0
    td = timedelta(seconds=float(seconds))
    total_ms = int(td.total_seconds() * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_outputs(result: dict[str, Any], out_base: Path) -> None:
    segments = result.get("segments", [])

    out_base.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_??")
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{_srt_ts(seg.get('start', 0))}] {spk}: {text}")
    out_base.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("speaker", "SPEAKER_??")
        srt_lines.append(str(i))
        srt_lines.append(f"{_srt_ts(seg.get('start', 0))} --> {_srt_ts(seg.get('end', 0))}")
        srt_lines.append(f"{spk}: {text}")
        srt_lines.append("")
    out_base.with_suffix(".srt").write_text("\n".join(srt_lines), encoding="utf-8")


def _noop_stage(stage: str, info: dict[str, Any]) -> None:
    pass


def _noop_progress(stage: str, fraction: float) -> None:
    pass


def transcribe_file(
    audio_path: Path,
    cfg: Config,
    *,
    batch_size: int = 16,
    on_stage: StageCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Run ASR + alignment + diarization on `audio_path`.

    `on_stage(stage, info)` fires at the start of each pipeline stage.
    `on_progress(stage, fraction)` fires repeatedly with 0..1 progress for the
    stages WhisperX/pyannote expose callbacks for (ASR, ALIGN, DIARIZE).

    Returns the path of the produced JSON transcript.
    """
    cb = on_stage or _noop_stage
    pcb = on_progress or _noop_progress
    audio_path = Path(audio_path).resolve()
    if audio_path.suffix.lower() not in AUDIO_EXTS:
        raise ValueError(f"Unsupported file extension: {audio_path.suffix}")
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    cb(STAGE_LOAD, {})
    log.info("Loading audio: %s", audio_path.name)
    audio = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / 16000.0  # whisperx resamples to 16k

    cb(STAGE_ASR, {"audio_duration": audio_duration})
    log.info("Loading ASR model (%s, %s/%s)", cfg.model, cfg.device, cfg.compute_type)
    asr_model = whisperx.load_model(
        cfg.model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        download_root=str(cfg.models_dir),
        language=cfg.language,
    )
    result = asr_model.transcribe(
        audio,
        batch_size=batch_size,
        language=cfg.language,
        progress_callback=lambda p: pcb(STAGE_ASR, max(0.0, min(1.0, p / 100.0))),
    )
    detected_lang = result.get("language") or cfg.language or "en"
    log.info("Transcribed; language=%s segments=%d", detected_lang, len(result.get("segments", [])))

    del asr_model
    gc.collect()
    try:
        import torch
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    cb(STAGE_ALIGN, {"language": detected_lang})
    try:
        log.info("Loading alignment model (%s)", detected_lang)
        align_model, align_meta = whisperx.load_align_model(
            language_code=detected_lang, device=cfg.device
        )
        result = whisperx.align(
            result["segments"], align_model, align_meta, audio,
            device=cfg.device, return_char_alignments=False,
            progress_callback=lambda p: pcb(STAGE_ALIGN, max(0.0, min(1.0, p / 100.0))),
        )
        del align_model
        gc.collect()
    except Exception as e:
        log.warning("Alignment unavailable for %r (%s) — continuing without word-level timestamps.",
                    detected_lang, e)

    cb(STAGE_DIARIZE, {})
    if cfg.hf_token:
        try:
            log.info("Running diarization")
            diarize_kwargs_init: dict[str, Any] = {"token": cfg.hf_token, "device": cfg.device}
            if cfg.diarize_model:
                diarize_kwargs_init["model_name"] = cfg.diarize_model
            diarize_pipeline = whisperx.diarize.DiarizationPipeline(**diarize_kwargs_init)
            diarize_kwargs: dict[str, Any] = {}
            if cfg.min_speakers is not None:
                diarize_kwargs["min_speakers"] = cfg.min_speakers
            if cfg.max_speakers is not None:
                diarize_kwargs["max_speakers"] = cfg.max_speakers
            diarize_kwargs["progress_callback"] = lambda p: pcb(STAGE_DIARIZE, max(0.0, min(1.0, p / 100.0 if p > 1 else p)))
            diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            log.error("Diarization failed: %s — output will lack speaker labels.", e)
    else:
        log.warning("HF_TOKEN not set; skipping diarization.")

    cb(STAGE_WRITE, {})
    out_dir = cfg.transcripts_dir / audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / audio_path.stem
    _write_outputs(result, out_base)
    log.info("Wrote %s.{json,txt,srt}", out_base)
    return out_base.with_suffix(".json")
