# WhisperX transcription + diarization

Local audio → speaker-labelled transcripts using [WhisperX](https://github.com/m-bain/whisperx).
Built for three workflows:

1. **Manual** — pass a file to `src.app`, get JSON / TXT / SRT back.
2. **Folder-watch** — point Syncthing (or anything else) at `incoming/`; the watcher transcribes new files as they finish syncing.
3. **Phone ingest** — auto-pull stock Pixel Recorder recordings off an Android phone via a Tasker → Syncthing bridge. See [docs/runbooks/pixel-recorder-ingest.md](docs/runbooks/pixel-recorder-ingest.md).

## Layout

```
.
├── .env                 # your local config (copy from .env.example)
├── incoming/            # drop / sync audio files here
├── transcripts/         # outputs land here ({name}/{name}.json, .txt, .srt)
├── models/              # whisper + alignment model cache
├── scripts/             # ops helpers (e.g. register the background watcher)
├── requirements.txt
└── src/
    ├── config.py        # env-loaded settings
    ├── transcribe.py    # ASR + alignment + diarization pipeline
    ├── watcher.py       # incoming/ filesystem-watch helpers
    ├── rename.py        # interactive speaker naming
    ├── ui.py            # terminal dashboard rendering
    └── app.py           # TUI entrypoint (scan / transcribe / watch)
```

## Setup

Prereqs (already verified on this box): Python 3.13, ffmpeg on PATH, NVIDIA GPU with recent driver.

```powershell
# already done by the bootstrap:
#   python -m venv .venv
#   .venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
#   .venv\Scripts\python -m pip install -r requirements.txt

Copy-Item .env.example .env
```

Then edit `.env` and fill in `HF_TOKEN`. You also need to visit and click "Agree" on these gated models — without that, diarization will silently fail:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Manual transcription

```powershell
# transcribe a specific file
.venv\Scripts\python -m src.app --files "C:\path\to\recording.m4a"
# or scan everything sitting in incoming/
.venv\Scripts\python -m src.app
```

Outputs go to `transcripts\<basename>\<basename>.{json,txt,srt}`.

## Folder watcher (Syncthing target)

```powershell
# process anything already in incoming/, then keep watching for new files:
.venv\Scripts\python -m src.app -y --watch --no-review

# one-shot drain (process new files, then exit):
.venv\Scripts\python -m src.app -y --no-review
```

Files already present at startup are processed first. New arrivals are queued, then the watcher waits 5s of file-size stability before transcribing — this lets Syncthing finish writing before we touch the file. Syncthing's `.syncthing.*.tmp` placeholders are ignored. Files that already have a transcript folder are skipped, so recordings may safely accumulate in `incoming/`.

### Syncthing setup (when you're ready)

On the recorder side, share the recordings folder. On this machine, accept the share and set its destination to `C:\Projects\Whisperx\incoming` (or wherever you point `INCOMING_DIR` in `.env`).

The stock **Pixel Recorder** hides recordings in Android scoped storage that Syncthing can't read directly — use the Tasker bridge in [the phone-ingest runbook](docs/runbooks/pixel-recorder-ingest.md) to lift them into a syncable folder.

Recommended Syncthing folder settings on the receiver:

- **Folder Type:** Receive Only (the recorder is the source of truth)
- **Ignore Permissions:** on (Windows ↔ Android compatibility)
- **File Versioning:** Trash Can or Staggered if you want a recoverable copy

Once a file finishes syncing, the watcher picks it up automatically.

### Run the watcher in the background (optional)

Register a logon Scheduled Task with the bundled helper:

```powershell
.\scripts\register-watcher-task.ps1
Start-ScheduledTask -TaskName "WhisperX Watcher"   # start now without re-logon
```

This launches `python -m src.app -y --watch --no-review` in a **minimized console**. (The Rich TUI can't run under console-less `pythonw.exe`, so we use a minimized window rather than a hidden process.)

## Configuration (`.env`)

| Var | Default | Notes |
|---|---|---|
| `HF_TOKEN` | — | Required for diarization. Read-only token is fine. |
| `WHISPER_MODEL` | `large-v3` | `tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3` |
| `DEVICE` | auto | `cuda` or `cpu`; auto-detected if blank |
| `COMPUTE_TYPE` | `float16` on GPU, `int8` on CPU | Lower precision = less VRAM |
| `LANGUAGE` | auto | ISO code (`en`, `de`, …) or blank to auto-detect |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | — | Hints for diarization; blank lets pyannote decide |
| `INCOMING_DIR` | `incoming` | Watched folder |
| `TRANSCRIPTS_DIR` | `transcripts` | Output folder |
| `MODELS_DIR` | `models` | Local model cache |

## Outputs

For each `recording.ext`, three files land in `transcripts\recording\`:

- **`recording.json`** — full WhisperX result, including per-word timestamps and speaker tags.
- **`recording.txt`** — readable transcript: `[hh:mm:ss,mmm] SPEAKER_00: text`
- **`recording.srt`** — subtitle file with speaker prefixes.

Speaker labels start as `SPEAKER_00`, `SPEAKER_01`, … Run `python -m src.app` and press **`r`** to name them interactively; names persist and rewrite the outputs.

## Troubleshooting

- **"Diarization failed: 401 / gated"** → you didn't accept the pyannote model agreements with the same HF account whose token you're using.
- **CUDA OOM** → drop to `WHISPER_MODEL=medium` or set `COMPUTE_TYPE=int8_float16`.
- **cuDNN errors with faster-whisper** → reinstall `ctranslate2` (`pip install --force-reinstall ctranslate2`); some versions ship incompatible cuDNN bindings.
- **Alignment skipped** → the language has no published wav2vec2 alignment model. The transcript is still written, just without word-level timestamps.
