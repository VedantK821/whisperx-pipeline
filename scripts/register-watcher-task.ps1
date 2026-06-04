# Registers "WhisperX Watcher" to run at logon: a minimized console running the
# unattended watcher. Re-run any time to update (uses -Force).
#
# Why a minimized console and not pythonw.exe?
#   src.app is a Rich TUI. Under pythonw.exe there is no console, which breaks
#   Rich's console I/O. So we launch real python.exe inside a minimized console
#   window via `cmd /c start "" /min`. --no-review keeps it from waiting on
#   interactive speaker-naming keypresses; name speakers later with
#   `python -m src.app` then press `r`.
$ErrorActionPreference = "Stop"

$root   = "C:\Projects\Whisperx"
$python = "$root\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python — create it before registering the task."
}

# cmd /c start "" /min  ->  launch the TUI in a minimized console window.
$cmdArgs = '/c start "" /min "' + $python + '" -m src.app -y --watch --no-review'

$action  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmdArgs -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "WhisperX Watcher" -Action $action -Trigger $trigger `
    -Settings $settings -RunLevel Limited -Force | Out-Null

Write-Host "Registered 'WhisperX Watcher' (runs at logon, minimized)."
Write-Host "Start it now with:  Start-ScheduledTask -TaskName 'WhisperX Watcher'"
Write-Host "Remove it with:     Unregister-ScheduledTask -TaskName 'WhisperX Watcher' -Confirm:`$false"
