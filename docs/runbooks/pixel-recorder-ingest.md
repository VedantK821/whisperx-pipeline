# Runbook: Pixel Recorder → WhisperX auto-ingest

How phone recordings made with the stock **Pixel Recorder** reach this machine
and get transcribed automatically.

## The chain

```
Pixel Recorder ──(1 tap: ⋮ → Audio → Share → Tasker)──► Tasker "WhisperX Ingest"
   writes WhisperX_<timestamp>.m4a ──► /sdcard/Download/WhisperXQueue/
   ──► Syncthing-Fork (Send Only) ──► PC Syncthing (Receive Only)
   ──► C:\Projects\Whisperx\incoming\ ──► src.app --watch ──► transcripts\
```

Nothing scrapes `recorder.google.com`. Tasker pushes; Syncthing carries; the
existing watcher transcribes each file exactly once (dedup keys on the filename
stem, so files may safely pile up in `incoming/`).

---

## 1. Phone — Tasker "WhisperX Ingest" task

**Goal:** one Share tap drops a uniquely-named `.m4a` into
`/sdcard/Download/WhisperXQueue/`.

### Import (fast path)

Import `docs/tasker/whisperx-ingest.tsk.xml` into Tasker (Tasks tab → long-press →
Import), then re-check the file path and the share variable as below.

### Build from scratch

1. Create the folder `/sdcard/Download/WhisperXQueue` (any file manager).
2. (Optional, recommended) Enable Tasker → **Preferences → "Direct Share Targets
   Enabled"** so your named share trigger shows up directly in Android's share
   sheet. Without it the flow still works — you just tap **Tasker** first, then
   pick your trigger. The trigger's *name* comes from the Received Share profile
   in the next step, not from a preferences screen.
3. **Profile → Event → Received Share** (optionally add a Pixel Recorder App
   context to scope it). Free, no plugin.
   - *Fallback:* the **AutoShare** plugin (paid, one-time) exposes the shared
     file as a real path `%asfile` — use it only if the `content://` copy below
     misbehaves.
4. Discover the shared-file variable: add **Alert → Flash `%rs_all_extras`**, do a
   test share, and read which variable holds the audio `content://` URI (current
   Tasker: `%rs_stream`; AutoShare: `%asfile`).
5. Task body:
   1. **Variable Set** `%ts` = `%TIMES` (epoch seconds — unique + sortable).
   2. **Variable Convert** `%ts`, *Seconds to Date Time*, format
      `yyyy-MM-dd_HH-mm-ss`.
   3. **Variable Set** `%dest` = `/sdcard/Download/WhisperXQueue/WhisperX_%ts.m4a`.
   4. **File → Copy File**: *From* = the share URI (`%rs_stream` / `%asfile`),
      *To* = `%dest`.
   5. **Alert → Flash** `Queued for WhisperX ✓`.

### Use it

Pixel Recorder → open a recording → **⋮ → Audio → Share → Tasker**. You should
see the `Queued for WhisperX ✓` flash, and a new `WhisperX_<timestamp>.m4a` in
`WhisperXQueue/`.

---

## 2. Sync — Syncthing

### Phone (Syncthing-Fork by Catfriend1)

The original Syncthing-Android is discontinued; use the maintained fork.

- Add Folder → path `/storage/emulated/0/Download/WhisperXQueue`,
  **Folder Type: Send Only**, share with the PC device.
- Exempt Syncthing-Fork from **battery optimization** (Settings → Apps →
  Syncthing-Fork → Battery → Unrestricted) so it syncs in the background.

### PC (Syncthing for Windows, Web UI at http://127.0.0.1:8384)

- Accept the shared folder → local path `C:\Projects\Whisperx\incoming`,
  **Folder Type: Receive Only**, Advanced → **Ignore Permissions: on**.

Files queue on the phone while the PC is off and drain when it returns.

---

## 3. PC — background watcher

The watcher is the "listener that goes back to sleep": it idles on `incoming/`
and transcribes on arrival.

### Register it (runs at logon, minimized)

```powershell
.\scripts\register-watcher-task.ps1
Start-ScheduledTask -TaskName "WhisperX Watcher"   # start now without re-logon
```

This runs `python -m src.app -y --watch --no-review` in a **minimized console**.

> **Why a console window?** `src.app` is a Rich TUI and cannot run under the
> console-less `pythonw.exe`. The minimized console is the trade-off for keeping
> the PC side code-free. A future `--headless` log-only watch mode would remove
> the window.

### Name speakers later (decoupled)

Transcription does **not** block on naming. When you want to label speakers:

```powershell
.venv\Scripts\python -m src.app
```

Press **`r`** to open the rename-pending picker and name speakers for any
transcript that still has `SPEAKER_00`-style labels.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| File never reaches `incoming/` | Syncthing paused, phone on battery-restricted Syncthing, or folder not shared/accepted. Check both Syncthing UIs. |
| `WhisperXQueue` file is 0 bytes / wrong audio | Wrong share variable in the Tasker Copy File step — re-run the `%rs_all_extras` flash and fix *From*. |
| Two recordings collide / one is skipped | Filenames not unique — confirm the task uses `%TIMES`-derived `%ts`. |
| Watcher not transcribing | Is "WhisperX Watcher" running? `Get-ScheduledTask -TaskName "WhisperX Watcher"`. Diarization 401 → see README HF-token notes. |
| Half-synced file picked up early | Handled: `.syncthing.*.tmp` ignored + 5s size-stability wait before transcription. |
