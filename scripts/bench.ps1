# Frame benchmark on this machine's GPU. Windows PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\bench.ps1
# Optional: -Presets low,medium -Scenes lap -Width 2560 -Height 1440
# Results: bench-results\<preset>-<scene>.json and bench-results\summary.txt.
# Send summary.txt (or the whole folder) back in the project thread.
param(
    [string[]]$Presets = @("low", "medium", "high"),
    [string[]]$Scenes = @("lap", "niva"),
    [int]$Width = 1920,
    [int]$Height = 1080,
    [int]$Frames = 1200
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
cargo build --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
New-Item -ItemType Directory -Force bench-results | Out-Null
$summary = "bench-results\summary.txt"
"Silt bench $(Get-Date -Format s)  ${Width}x${Height}" | Set-Content $summary
foreach ($scene in $Scenes) {
    foreach ($preset in $Presets) {
        Write-Host "== $preset / $scene =="
        $out = "bench-results\$preset-$scene.json"
        $lines = & .\target\release\silt.exe bench --quality $preset --scene $scene `
            --width $Width --height $Height --frames $Frames --out $out 2>$null
        $lines | Where-Object { $_ -like "BENCH*" -or $_ -like "GPU_*" } | Tee-Object -FilePath $summary -Append
        "" | Add-Content $summary
    }
}
Write-Host "Done: $summary"
