#!/usr/bin/env pwsh
# Scan incoming/ and transcribe new audio files. Pass any flags through to src.app.
Push-Location $PSScriptRoot
try {
    & .\.venv\Scripts\python.exe -m src.app @args
}
finally {
    Pop-Location
}
