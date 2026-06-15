"""Phase-1 auto-ID artifacts written next to a transcript:
<stem>.embeddings.json (speaker embeddings from diarization + clip filenames)
and clips/<label>.ogg (best speaker-pure sample).

Clip selection reuses rename.build_speaker_examples(). Embeddings are passed in
from transcribe.py — they come free from the diarization call with
return_embeddings=True, so this module does no model inference of its own.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from . import rename

log = logging.getLogger(__name__)


def ffmpeg_clip_cmd(audio: Path, start: float, dur: float, out: Path) -> list[str]:
    """ffmpeg command that cuts [start, start+dur] of `audio` to a mono Opus clip."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "quiet", "-y",
        "-ss", str(start), "-t", str(dur), "-i", str(audio),
        "-c:a", "libopus", "-b:a", "32k", "-ac", "1", str(out),
    ]


def export_clips(segments: list[dict], audio_path: Path, clips_dir: Path) -> dict[str, str]:
    """Cut each speaker's best ~6s sample to clips_dir/<label>.ogg.

    Returns {label: clip_filename}. Skips speakers with no usable snippet, and
    skips (with a warning) any speaker whose clip cut fails.
    """
    clips_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for ex in rename.build_speaker_examples(segments):
        if not ex.snippets:
            continue
        snip = ex.snippets[0]  # already speaker-pure, ranked toward the 3-6s sweet spot
        dur = max(0.5, min(6.0, snip.end - snip.start))
        clip = clips_dir / f"{ex.label}.ogg"
        try:
            subprocess.run(ffmpeg_clip_cmd(audio_path, snip.start, dur, clip), check=True)
            out[ex.label] = clip.name
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.warning("clip cut failed for %s: %s", ex.label, e)
    return out


def export_speaker_artifacts(
    segments: list[dict],
    embeddings: "dict[str, list[float]] | None",
    audio_path: Path,
    out_dir: Path,
) -> None:
    """Write <out_dir>/<stem>.embeddings.json and <out_dir>/clips/<label>.ogg.

    `embeddings` is the dict returned by the diarization call (keyed by the same
    SPEAKER_XX labels used on the segments); may be None if diarization did not
    return embeddings.
    """
    stem = audio_path.stem
    clips = export_clips(segments, audio_path, out_dir / "clips")
    payload = {"embeddings": embeddings or {}, "clips": clips}
    (out_dir / f"{stem}.embeddings.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    log.info(
        "Exported %d embeddings, %d clips for %s",
        len(payload["embeddings"]), len(clips), stem,
    )
