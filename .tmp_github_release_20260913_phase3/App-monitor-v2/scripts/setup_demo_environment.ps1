param(
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$backendRoot = Join-Path $projectRoot "backend"
$environmentRoot = Join-Path $backendRoot ".demo-venv"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"

if (-not $PythonExecutable) {
    $siblingPython = Join-Path (Split-Path $projectRoot -Parent) "guard-connection-ai\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $siblingPython) {
        $PythonExecutable = $siblingPython
    } else {
        $PythonExecutable = "python"
    }
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    Write-Host "Creating the isolated demo environment: $environmentRoot"
    & $PythonExecutable -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python environment." }
}

Push-Location $backendRoot
try {
    & $environmentPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Failed to install Python dependencies." }
    & $environmentPython -c "import fastapi, numpy, pydantic, scipy, sqlalchemy, websockets; print('backend dependencies: OK')"
    if ($LASTEXITCODE -ne 0) { throw "Failed to import Python dependencies." }
    $previousGuardSource = $env:GUARD_AI_SRC
    try {
        $env:GUARD_AI_SRC = Join-Path (Split-Path $projectRoot -Parent) "guard-connection-ai\src"
        & $environmentPython -c "import main; print('backend application import: OK')"
        if ($LASTEXITCODE -ne 0) { throw "Failed to import the backend application." }
    } finally {
        $env:GUARD_AI_SRC = $previousGuardSource
    }
} finally {
    Pop-Location
}

Write-Host "Demo environment is ready."
Write-Host "Use scripts\start_backend.ps1 to start the backend."
