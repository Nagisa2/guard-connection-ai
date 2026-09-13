param(
    [ValidateSet("none", "patient_ok", "patient_unwell", "patient_help")]
    [string]$PatientAction = "patient_unwell",
    [switch]$Loop,
    [switch]$NoWait,
    [switch]$ControlPanel
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$backendRoot = Join-Path $projectRoot "backend"
$frontendRoot = Join-Path $projectRoot "frontend"
$python = Join-Path $backendRoot ".demo-venv\Scripts\python.exe"
$logRoot = Join-Path $backendRoot "logs"
$guardSource = Join-Path (Split-Path $projectRoot -Parent) "guard-connection-ai\src"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Demo environment is missing. Run scripts\setup_demo_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $guardSource)) {
    throw "Front AI source directory was not found: $guardSource"
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm.cmd was not found. Install or configure Node.js."
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$backendOut = Join-Path $logRoot "backend.stdout.log"
$backendErr = Join-Path $logRoot "backend.stderr.log"
$frontendOut = Join-Path $logRoot "frontend.stdout.log"
$frontendErr = Join-Path $logRoot "frontend.stderr.log"
$simulatedEcgOut = Join-Path $logRoot "simulated-ecg.stdout.log"
$simulatedEcgErr = Join-Path $logRoot "simulated-ecg.stderr.log"
$backendProcess = $null
$frontendProcess = $null
$simulatedEcgProcess = $null
$previousGuardSource = $env:GUARD_AI_SRC
$previousJwtSecret = $env:GUARD_JWT_SECRET
$previousDownstreamApiKey = $env:GUARD_DOWNSTREAM_API_KEY

function Wait-HttpReady([string]$Url, [int]$TimeoutSeconds) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1 | Out-Null
            return
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    throw "Timed out while waiting for: $Url"
}

function Assert-DemoPatients {
    $loginBody = @{ login_id = "yamada"; password = "password123" } | ConvertTo-Json
    $login = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/auth/login" `
        -ContentType "application/json" -Body $loginBody -TimeoutSec 5
    $headers = @{ Authorization = "Bearer $($login.token)" }
    $patientResponse = Invoke-WebRequest -Method Get -Uri "http://127.0.0.1:8000/patients" `
        -Headers $headers -UseBasicParsing -TimeoutSec 5
    $patients = ConvertFrom-Json -InputObject $patientResponse.Content
    $patientCount = @($patients).Count
    if ($patientCount -lt 2) {
        throw "Demo patient initialization failed: expected at least 2 patients, got $patientCount."
    }
    Write-Host "Demo patients ready: $patientCount"
}

function Assert-PortAvailable([int]$Port, [string]$Label) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connected = $client.ConnectAsync("127.0.0.1", $Port).Wait(300)
        if ($connected -and $client.Connected) {
            throw "$Label is already using port $Port. Stop the existing process before starting the unified demo."
        }
    } finally {
        $client.Dispose()
    }
}

function Stop-OwnedProcessTree($Process) {
    if ($Process -and -not $Process.HasExited) {
        & taskkill.exe /PID $Process.Id /T /F | Out-Null
    }
}

try {
    Assert-PortAvailable 8000 "Backend"
    Assert-PortAvailable 5173 "Frontend"
    $env:GUARD_AI_SRC = $guardSource
    $env:GUARD_JWT_SECRET = "guard-connection-local-contest-demo-only"
    if (-not $env:GUARD_DOWNSTREAM_API_KEY) {
        $randomKey = [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
        $env:GUARD_DOWNSTREAM_API_KEY = [Convert]::ToBase64String($randomKey)
    }
    $backendProcess = Start-Process -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $backendRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr
    Wait-HttpReady "http://127.0.0.1:8000/openapi.json" 30
    Assert-DemoPatients

    $frontendProcess = Start-Process -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") `
        -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr
    Wait-HttpReady "http://127.0.0.1:5173" 30

    $simulatedEcgProcess = Start-Process -FilePath $python `
        -ArgumentList @(
            "test_sender.py",
            "--user-ids", "test_user_01", "test_user_02",
            "--signal-types", "ECG",
            "--ws-base-url", "ws://127.0.0.1:8000"
        ) `
        -WorkingDirectory $backendRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $simulatedEcgOut -RedirectStandardError $simulatedEcgErr

    Write-Host "Doctor app: http://127.0.0.1:5173"
    Write-Host "Patient demo: http://127.0.0.1:5173/?mode=patient&user_id=test_user_02"
    Write-Host "Family demo: http://127.0.0.1:5173/?mode=family&user_id=test_user_02"
    Write-Host "Demo control: http://127.0.0.1:5173/?mode=control"
    Write-Host "Demo login: yamada / password123"
    Write-Host "AF alerts are explicitly labeled as demo stubs, not model results."
    Write-Host "The doctor app combines supplemental simulated ECG with realtime PPG and non-diagnostic pseudo ECG."
    Write-Host "Press Ctrl+C to stop."

    if ($ControlPanel) {
        Start-Process "http://127.0.0.1:5173"
        Start-Process "http://127.0.0.1:5173/?mode=control"
        Write-Host "The doctor app and control panel were opened. Press Ctrl+C here to stop all demo processes."
        while ($true) { Start-Sleep -Seconds 1 }
    }

    $scenarioArguments = @(
        "contest_demo_scenario.py",
        "--base-url", "http://127.0.0.1:8000",
        "--patient-action", $PatientAction
    )
    if ($Loop) { $scenarioArguments += "--loop" }
    if ($NoWait) { $scenarioArguments += "--no-wait" }
    Push-Location $backendRoot
    try {
        & $python @scenarioArguments
    } finally {
        Pop-Location
    }
} finally {
    Stop-OwnedProcessTree $simulatedEcgProcess
    Stop-OwnedProcessTree $frontendProcess
    Stop-OwnedProcessTree $backendProcess
    $env:GUARD_AI_SRC = $previousGuardSource
    $env:GUARD_JWT_SECRET = $previousJwtSecret
    $env:GUARD_DOWNSTREAM_API_KEY = $previousDownstreamApiKey
    Write-Host "Stopped the demo processes started by this script."
}
