#requires -Version 5.1
<#
  Локальный запуск бэкенда и фронтенда «Сканер российских вин».

  Использование (из корня репозитория):
    powershell -ExecutionPolicy Bypass -File local\run-local.ps1 restart
    powershell -ExecutionPolicy Bypass -File local\run-local.ps1 restart -DemoOcr "Фанагория 100 оттенков красного. Каберне"
    $env:MISTRAL_API_KEY = "ключ"; powershell -ExecutionPolicy Bypass -File local\run-local.ps1 restart
    $env:MINERU_API_KEY = "ключ"; powershell -ExecutionPolicy Bypass -File local\run-local.ps1 restart
    powershell -ExecutionPolicy Bypass -File local\run-local.ps1 status
    powershell -ExecutionPolicy Bypass -File local\run-local.ps1 stop

  Логи: local\logs. PID: local\run. Файл локальный, в git не попадает.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'restart', 'stop', 'status')]
    [string]$Command = 'restart',

    [string]$DemoOcr
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
# Bundle layout ships backend/; the dev checkout runs wineid from the repo root.
$BackendDir = Join-Path $RepoRoot 'backend'
if (-not (Test-Path $BackendDir)) { $BackendDir = $RepoRoot }
$WebDir     = Join-Path $RepoRoot 'web'
$LogsDir    = Join-Path $PSScriptRoot 'logs'
$RunDir     = Join-Path $PSScriptRoot 'run'
$Python     = Join-Path $BackendDir '.venv\Scripts\python.exe'
$BackendPid = Join-Path $RunDir 'backend.pid'
$FrontendPid = Join-Path $RunDir 'frontend.pid'
$BackendOut = Join-Path $LogsDir 'backend.out.log'
$BackendErr = Join-Path $LogsDir 'backend.err.log'
$FrontendOut = Join-Path $LogsDir 'frontend.out.log'
$FrontendErr = Join-Path $LogsDir 'frontend.err.log'

$BackendHost = '127.0.0.1'
$BackendPort = 8080
$FrontendPort = 5173

New-Item -ItemType Directory -Force -Path $LogsDir, $RunDir | Out-Null

function Get-Pid($file) {
    if (-not (Test-Path $file)) { return $null }
    $value = (Get-Content $file -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($value -match '^\d+$') { return [int]$value }
    return $null
}

function Test-Alive($processId) {
    if ($null -eq $processId) { return $false }
    return $null -ne (Get-Process -Id $processId -ErrorAction SilentlyContinue)
}

function Test-Port($port) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect('127.0.0.1', $port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Stop-ProcessTree($processId) {
    if ($null -eq $processId) { return }
    & taskkill.exe /PID $processId /T /F 2>$null | Out-Null
}

function Stop-Stack {
    $backend = Get-Pid $BackendPid
    $frontend = Get-Pid $FrontendPid
    if (Test-Alive $backend) { Stop-ProcessTree $backend }
    if (Test-Alive $frontend) { Stop-ProcessTree $frontend }
    Remove-Item $BackendPid, $FrontendPid -ErrorAction SilentlyContinue
    # Подчистить возможные осиротевшие процессы на портах.
    foreach ($port in @($BackendPort, $FrontendPort)) {
        Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-ProcessTree $_.OwningProcess }
    }
}

function Start-Backend {
    if (-not (Test-Path $Python)) {
        throw "Нет venv: $Python. Создай его в $BackendDir`: python -m venv .venv; .venv\Scripts\python -m pip install -c constraints.txt -r requirements-api.txt"
    }

    # ROI-режим: demo/real OCR читают кадр целиком (экспериментальная абляция),
    # обычный режим честно отказывается (automatic_roi_unavailable).
    $roiMode = 'refuse'
    if ($DemoOcr) {
        $env:WINE_MOCK_OCR = $DemoOcr
        $env:WINE_ENABLE_MOCK_OCR = '1'
        $env:WINE_OCR_PROVIDER = 'mock'
        $roiMode = 'full_frame_experimental'
        Write-Host "Демо-OCR: `"$DemoOcr`""
    } else {
        Remove-Item Env:WINE_MOCK_OCR -ErrorAction SilentlyContinue
        if ($env:MINERU_TOKEN -or $env:MINERU_API_KEY) {
            if (-not $env:MINERU_TOKEN) { $env:MINERU_TOKEN = $env:MINERU_API_KEY }
            $env:WINE_OCR_PROVIDER = 'mineru'
            $roiMode = 'full_frame_experimental'
            Write-Host 'Реальный OCR: MinerU cloud настроен'
        } elseif ($env:MISTRAL_API_KEY) {
            $env:WINE_OCR_PROVIDER = 'mistral'
            $roiMode = 'full_frame_experimental'
            Write-Host 'Реальный OCR: MISTRAL_API_KEY найден'
        }
    }
    # Реальный OCR-провайдер думает дольше: расширяем общий дедлайн бэкенда
    # и таймаут скана на фронте (MinerU асинхронный и может идти десятки секунд).
    if ($roiMode -ne 'refuse') {
        if ($env:MINERU_TOKEN) {
            if (-not $env:WINE_REQUEST_TIMEOUT) { $env:WINE_REQUEST_TIMEOUT = '60' }
            if (-not $env:VITE_SCAN_TIMEOUT_MS) { $env:VITE_SCAN_TIMEOUT_MS = '70000' }
        } else {
            if (-not $env:WINE_REQUEST_TIMEOUT) { $env:WINE_REQUEST_TIMEOUT = '30' }
            if (-not $env:VITE_SCAN_TIMEOUT_MS) { $env:VITE_SCAN_TIMEOUT_MS = '40000' }
        }
    } else {
        Remove-Item Env:WINE_REQUEST_TIMEOUT -ErrorAction SilentlyContinue
        Remove-Item Env:VITE_SCAN_TIMEOUT_MS -ErrorAction SilentlyContinue
    }
    if ($env:WINE_ROI_MODE) { $roiMode = $env:WINE_ROI_MODE }
    $env:WINE_ROI_MODE = $roiMode
    $env:WINE_HOST = $BackendHost
    $env:WINE_PORT = "$BackendPort"

    $p = Start-Process -FilePath $Python -ArgumentList '-m', 'wineid.api' `
        -WorkingDirectory $BackendDir -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $BackendOut -RedirectStandardError $BackendErr
    Set-Content $BackendPid $p.Id
    Write-Host "Бэкенд PID $($p.Id) -> http://$BackendHost`:$BackendPort (roi_mode=$roiMode)"
}

function Start-Frontend {
    $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (-not $npm) { $npm = 'npm.cmd' }
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "npm run dev -- --host 127.0.0.1 --port $FrontendPort" `
        -WorkingDirectory $WebDir -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $FrontendOut -RedirectStandardError $FrontendErr
    Set-Content $FrontendPid $p.Id
    Write-Host "Фронтенд PID $($p.Id) -> http://127.0.0.1`:$FrontendPort"
}

function Wait-Endpoint($url, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return $true }
        } catch {
            Start-Sleep -Milliseconds 400
        }
    }
    return $false
}

function Write-Status {
    $backend = Get-Pid $BackendPid
    $frontend = Get-Pid $FrontendPid
    $backendUp = (Test-Alive $backend) -or (Test-Port $BackendPort)
    $frontendUp = (Test-Alive $frontend) -or (Test-Port $FrontendPort)
    Write-Host ''
    Write-Host ("Бэкенд  : " + $(if ($backendUp) { "работает (PID $backend)" } else { 'остановлен' }) + "  http://$BackendHost`:$BackendPort")
    Write-Host ("Фронтенд: " + $(if ($frontendUp) { "работает (PID $frontend)" } else { 'остановлен' }) + "  http://127.0.0.1`:$FrontendPort")
    if ($backendUp) {
        try {
            $health = Invoke-RestMethod -Uri "http://$BackendHost`:$BackendPort/api/health" -TimeoutSec 3
            Write-Host ("health  : ready=$($health.ready) recognition_ready=$($health.recognition_ready) roi_mode=$($health.roi_mode) ocr=$($health.ocr_configured) media=$($health.media_configured) wines=$($health.wine_count)")
        } catch {
            Write-Host 'health  : недоступен'
        }
    }
    Write-Host ("Логи    : " + $LogsDir)
    Write-Host ''
}

switch ($Command) {
    'start' {
        if (-not (Test-Port $BackendPort)) { Start-Backend } else { Write-Host 'Бэкенд уже слушает порт' }
        if (-not (Test-Port $FrontendPort)) { Start-Frontend } else { Write-Host 'Фронтенд уже слушает порт' }
        [void](Wait-Endpoint "http://$BackendHost`:$BackendPort/api/health" 30)
        [void](Wait-Endpoint "http://127.0.0.1`:$FrontendPort/" 60)
        Write-Status
    }
    'restart' {
        Stop-Stack
        Start-Sleep -Milliseconds 500
        Start-Backend
        Start-Frontend
        [void](Wait-Endpoint "http://$BackendHost`:$BackendPort/api/health" 30)
        [void](Wait-Endpoint "http://127.0.0.1`:$FrontendPort/" 60)
        Write-Status
        Write-Host "Открой http://127.0.0.1`:$FrontendPort  (панель метрик: /scan?debug=1)"
    }
    'stop' { Stop-Stack; Write-Host 'Остановлено.' }
    'status' { Write-Status }
}
