param(
    [string]$ModelPath = "$env:LOCALAPPDATA\models\muse-glimmer\Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
    [int]$GpuLayers = 99,
    [int]$ParallelSlots = 1,
    [int]$Port = 8080,
    [int]$ContextSize = 8192,
    [int]$ReasoningBudget = 512,
    [ValidateSet('low', 'off')]
    [string]$ReasoningMode = 'low'
)

$ErrorActionPreference = 'Stop'
$llamaRoot = "$env:LOCALAPPDATA\llama.cpp\b12058"
$serverExe = Join-Path $llamaRoot 'server\llama-server.exe'
$runtimeDir = Join-Path $llamaRoot 'runtime'
$logDir = Join-Path (Get-Location) 'logs'
$serverLog = Join-Path $logDir "llama-server-q4-ngl$GpuLayers-np$ParallelSlots-$ReasoningMode-20261103.log"

if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
    throw "llama-server.exe not found: $serverExe"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Model file not found: $ModelPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $runtimeDir 'cudart64_13.dll') -PathType Leaf)) {
    throw "CUDA runtime DLL not found: $(Join-Path $runtimeDir 'cudart64_13.dll')"
}
if (-not (Test-Path -LiteralPath $logDir -PathType Container)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$serverArgs = @(
    '--model', $ModelPath,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--n-gpu-layers', "$GpuLayers",
    '--ctx-size', "$ContextSize",
    '--parallel', "$ParallelSlots",
    '--flash-attn', 'on',
    '--cache-type-k', 'q8_0',
    '--cache-type-v', 'q8_0'
)

# Windows PowerShell 5.1 strips embedded quotes while marshalling arguments to
# native executables. Escape the JSON quotes for its Windows command-line parser;
# PowerShell 7's standard native argument passing preserves the JSON directly.
$chatTemplateKwargs = '{"reasoning_strength":"low"}'
if ($PSVersionTable.PSVersion.Major -lt 7) {
    $chatTemplateKwargs = '{\"reasoning_strength\":\"low\"}'
}

if ($ReasoningMode -eq 'off') {
    $serverArgs += @('--reasoning', 'off')
}

$serverArgs += @(
    '--reasoning-effort', 'low',
    '--reasoning-budget', "$ReasoningBudget",
    '--chat-template-kwargs', $chatTemplateKwargs,
    '--metrics',
    '--verbosity', '4'
)

Write-Host "Starting llama-server on 127.0.0.1:$Port"
Write-Host "Model: $ModelPath"
Write-Host "GPU layers requested: $GpuLayers; context: $ContextSize; parallel slots: $ParallelSlots; K/V cache: q8_0; flash attention: on"
Write-Host "Reasoning mode: $ReasoningMode; effort: low; reasoning budget: $ReasoningBudget; log verbosity: trace"
Write-Host "Server log: $serverLog"
$env:PATH = "$runtimeDir;$env:PATH"
# llama.cpp writes ordinary startup and request logs to stderr. Windows
# PowerShell 5.1 wraps those lines as NativeCommandError records; with Stop,
# that can interrupt this wrapper even while the server is starting. Convert
# stderr records back to plain log lines and preserve the server's exit code.
$ErrorActionPreference = 'Continue'
& $serverExe @serverArgs 2>&1 |
    ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $_.Exception.Message
        } else {
            [string]$_
        }
    } |
    Tee-Object -FilePath $serverLog -Append
$serverExitCode = $LASTEXITCODE
exit $serverExitCode
