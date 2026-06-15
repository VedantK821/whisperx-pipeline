"""WhisperX transcription + diarization pipeline."""
from __future__ import annotations

import gc
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import whisperx
import whisperx.diarize  # ensure the diarize submodule is loaded (whisperx does not auto-import it)

from .config import Config

log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}

# Domain vocabulary primer — biases Whisper toward the team's names/products/tools so it
# stops mis-hearing jargon (e.g. "Micro Insurance" -> "microwave insurance"). Overridable
# via the INITIAL_PROMPT env var. Kept well under Whisper's ~224-token prompt budget.
DEFAULT_INITIAL_PROMPT = (
    "Meeting of the acme health-insurance product team in India. "
    "People: Vedant, Alice, Bob, Morgan, Taylor, Jordan, Sam, Casey, Riley, "
    "Alex, Priya, Jesse, Robin, Northwind. "
    "Tools and products: Adobe XD, Figma, Micro Insurance, SafeCare, SafeCare Duo, "
    "SafeCare Max, Acme Health, Wellsprings, ShieldPlus, personal accident insurance, "
    "telemedicine, sum insured, EMI, premium, policy, brochure, claim."
)

# Pipeline stage names — kept here so the UI and pipeline agree on labels.
STAGE_LOAD = "LOAD"
STAGE_ASR = "ASR"
STAGE_ALIGN = "ALIGN"
STAGE_DIARIZE = "DIARIZE"
STAGE_REVIEW = "REVIEW"
STAGE_WRITE = "WRITE"
STAGES: tuple[str, ...] = (
    STAGE_LOAD, STAGE_ASR, STAGE_ALIGN, STAGE_DIARIZE, STAGE_REVIEW, STAGE_WRITE,
)

StageCallback = Callable[[str, dict[str, Any]], None]
ProgressCallback = Callable[[str, float], None]  # (stage, fraction in 0..1)
# (examples, audio_path) -> mapping (or None to skip)
ReviewCallback = Callable[[list[Any], Path], "dict[str, str] | None"]


def _srt_ts(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0
    td = timedelta(seconds=float(seconds))
    total_ms = int(td.total_seconds() * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(result: dict[str, Any], out_base: Path) -> None:
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


def _load_align_model_with_fallback(language: str, device: str):
    """Load the wav2vec2 alignment model for `language`; if none exists, retry
    with English before giving up.

    Whisper's language detection misfires on Indian-accent / code-switched
    audio (one real meeting came back as 'jw' — Javanese), and whisperx has no
    align model for such languages. Losing alignment entirely costs word-level
    timing, which downstream breaks speaker-pure rename samples. The transcript
    text in these cases is English anyway, so the English aligner is correct.
    """
    try:
        return whisperx.load_align_model(language_code=language, device=device)
    except Exception as e:
        if language == "en":
            raise
        log.warning(
            "No alignment model for %r (%s) — retrying with 'en'.", language, e
        )
        return whisperx.load_align_model(language_code="en", device=device)


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
    on_review: ReviewCallback | None = None,
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
        asr_options={"initial_prompt": (os.getenv("INITIAL_PROMPT", "").strip() or DEFAULT_INITIAL_PROMPT)},
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
        align_model, align_meta = _load_align_model_with_fallback(
            detected_lang, cfg.device
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
    speaker_embeddings: "dict[str, list[float]] | None" = None
    if cfg.hf_token:
        try:
            log.info("Running diarization")
            diarize_kwargs_init: dict[str, Any] = {"token": cfg.hf_token, "device": cfg.device}
            if cfg.diarize_model:
                diarize_kwargs_init["model_name"] = cfg.diarize_model
            diarize_pipeline = whisperx.diarize.DiarizationPipeline(**diarize_kwargs_init)
            diarize_kwargs: dict[str, Any] = {"return_embeddings": True}
            if cfg.min_speakers is not None:
                diarize_kwargs["min_speakers"] = cfg.min_speakers
            if cfg.max_speakers is not None:
                diarize_kwargs["max_speakers"] = cfg.max_speakers
            diarize_kwargs["progress_callback"] = lambda p: pcb(STAGE_DIARIZE, max(0.0, min(1.0, p / 100.0 if p > 1 else p)))
            diarize_segments, speaker_embeddings = diarize_pipeline(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            log.error("Diarization failed: %s — output will lack speaker labels.", e)
    else:
        log.warning("HF_TOKEN not set; skipping diarization.")

    cb(STAGE_REVIEW, {})
    out_dir = cfg.transcripts_dir / audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Local import to avoid a top-level circular import (rename imports from
    # transcribe via local imports inside its functions).
    from . import rename

    segments = result.get("segments") or []
    persisted = rename.load_persisted_mapping(out_dir, audio_path.stem)
    if persisted:
        rename.apply_speaker_mapping(segments, persisted)
        log.info("Applied %d persisted name(s) for %s", len(persisted), audio_path.name)

    remaining = rename.unmapped_speakers(segments)
    new_mapping: dict[str, str] = {}
    if remaining and on_review is not None:
        examples = rename.build_speaker_examples(segments)
        examples = [e for e in examples if e.label in remaining]
        try:
            review_result = on_review(examples, audio_path)
        except Exception:
            log.exception("on_review callback raised; continuing without rename")
            review_result = None
        if review_result:
            rename.apply_speaker_mapping(segments, review_result)
            new_mapping = review_result

    if new_mapping:
        merged = dict(persisted or {})
        merged.update(new_mapping)
        rename.save_persisted_mapping(out_dir, audio_path.stem, audio_path.name, merged)

    cb(STAGE_WRITE, {})
    out_base = out_dir / audio_path.stem
    write_outputs(result, out_base)
    log.info("Wrote %s.{json,txt,srt}", out_base)

    # Phase-1 auto-ID artifacts for the iMac: embeddings (from diarization) + clips.
    # Must never fail the transcript itself.
    if cfg.hf_token:
        try:
            from .export_speakers import export_speaker_artifacts
            export_speaker_artifacts(segments, speaker_embeddings, audio_path, out_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("Speaker-artifact export failed for %s: %s", audio_path.name, e)

    return out_base.with_suffix(".json")
