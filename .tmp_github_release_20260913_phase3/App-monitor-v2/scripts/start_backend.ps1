param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$backendRoot = Join-Path $projectRoot "backend"
$python = Join-Path $backendRoot ".demo-venv\Scripts\python.exe"
$guardSource = Join-Path (Split-Path $projectRoot -Parent) "guard-connection-ai\src"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Demo environment is missing. Run scripts\setup_demo_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $guardSource)) {
    throw "Front AI source directory was not found: $guardSource"
}

$previousGuardSource = $env:GUARD_AI_SRC
try {
    $env:GUARD_AI_SRC = $guardSource
    $arguments = @(
        "-m", "uvicorn", "main:app",
        "--host", $HostAddress,
        "--port", $Port.ToString()
    )
    if (-not $NoReload) {
        $arguments += "--reload"
    }
    Push-Location $backendRoot
    try {
        & $python @arguments
    } finally {
        Pop-Location
    }
} finally {
    $env:GUARD_AI_SRC = $previousGuardSource
}
