# WhisperX transcription + diarization

Local audio → speaker-labelled transcripts using [WhisperX](https://github.com/m-bain/whisperx).
Built for two workflows:

1. **Manual** — drop a file path on the CLI, get JSON / TXT / SRT back.
2. **Folder-watch** — point Syncthing (or anything else) at `incoming/`; the watcher transcribes new files as they finish syncing.

## Layout

```
.
├── .env                 # your local config (copy from .env.example)
├── incoming/            # drop / sync audio files here
├── transcripts/         # outputs land here ({name}.json, .txt, .srt)
├── models/              # whisper + alignment model cache
├── requirements.txt
└── src/
    ├── config.py        # env-loaded settings
    ├── transcribe.py    # ASR + alignment + diarization pipeline
    ├── cli.py           # one-shot CLI
    └── watch.py         # incoming/ folder watcher
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
.venv\Scripts\python -m src.cli "C:\path\to\recording.m4a"
# or a whole folder, recursively
.venv\Scripts\python -m src.cli .\incoming
```

Outputs go to `transcripts\<basename>.{json,txt,srt}`.

## Folder watcher (Syncthing target)

```powershell
# process anything already in incoming/, then keep watching for new files:
.venv\Scripts\python -m src.watch

# one-shot drain:
.venv\Scripts\python -m src.watch --once
```

Files already present at startup are processed first. New arrivals are queued, then the watcher waits 5s of file-size stability before transcribing — this lets Syncthing finish writing before we touch the file. Syncthing's `.syncthing.*.tmp` placeholders are ignored.

### Syncthing setup (when you're ready)

On the recorder side, share the recordings folder. On this machine, accept the share and set its destination to `C:\Projects\Whisperx\incoming` (or wherever you point `INCOMING_DIR` in `.env`). Recommended Syncthing folder settings on the receiver:

- **Folder Type:** Receive Only (the recorder is the source of truth)
- **Ignore Permissions:** on (Windows ↔ Android compatibility)
- **File Versioning:** Trash Can or Staggered if you want a recoverable copy

Once a file finishes syncing, the watcher picks it up automatically.

### Run the watcher in the background (optional)

Easiest path: a Scheduled Task that runs on logon. Quick version:

```powershell
$task = "WhisperX Watcher"
$action = New-ScheduledTaskAction `
  -Execute "C:\Projects\Whisperx\.venv\Scripts\pythonw.exe" `
  -Argument "-m src.watch" `
  -WorkingDirectory "C:\Projects\Whisperx"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -RunLevel Limited
```

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

For each `recording.ext`, three files land in `transcripts/`:

- **`recording.json`** — full WhisperX result, including per-word timestamps and speaker tags.
- **`recording.txt`** — readable transcript: `[hh:mm:ss,mmm] SPEAKER_00: text`
- **`recording.srt`** — subtitle file with speaker prefixes.

## Troubleshooting

- **"Diarization failed: 401 / gated"** → you didn't accept the pyannote model agreements with the same HF account whose token you're using.
- **CUDA OOM** → drop to `WHISPER_MODEL=medium` or set `COMPUTE_TYPE=int8_float16`.
- **cuDNN errors with faster-whisper** → reinstall `ctranslate2` (`pip install --force-reinstall ctranslate2`); some versions ship incompatible cuDNN bindings.
- **Alignment skipped** → the language has no published wav2vec2 alignment model. The transcript is still written, just without word-level timestamps.
