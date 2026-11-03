param(
    [string]$OutputPath = 'logs/local-resource-samples-q4-20261103.csv',
    [int]$IntervalMilliseconds = 500
)

$ErrorActionPreference = 'Stop'
$output = Join-Path (Get-Location) $OutputPath
$parent = Split-Path -Parent $output
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}
'timestamp_utc,gpu_used_mib,gpu_free_mib,system_free_ram_mib,llama_working_set_mib,llama_peak_working_set_mib' |
    Set-Content -LiteralPath $output -Encoding utf8
Write-Host "Sampling local inference resources to $output (Ctrl+C to stop)."

while ($true) {
    $timestamp = [DateTime]::UtcNow.ToString('o')
    $gpuValues = (& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits).Trim() -split ',\s*'
    $freeRam = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 1)
    $server = Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue |
        Sort-Object WorkingSet64 -Descending | Select-Object -First 1
    $workingSet = if ($server) { [math]::Round($server.WorkingSet64 / 1MB, 1) } else { 0 }
    $peakWorkingSet = if ($server) { [math]::Round($server.PeakWorkingSet64 / 1MB, 1) } else { 0 }
    "$timestamp,$($gpuValues[0].Trim()),$($gpuValues[1].Trim()),$freeRam,$workingSet,$peakWorkingSet" |
        Add-Content -LiteralPath $output -Encoding utf8
    Start-Sleep -Milliseconds $IntervalMilliseconds
}
