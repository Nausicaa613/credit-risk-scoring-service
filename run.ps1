#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Windows-friendly equivalent of the Makefile.

.DESCRIPTION
    GNU make is not installed by default on Windows, so every Makefile target has
    an identical twin here. The behaviour and default values match.

.EXAMPLE
    .\run.ps1 data
    .\run.ps1 train
    .\run.ps1 run
    .\run.ps1 test
    .\run.ps1 smoke
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'data', 'train', 'run', 'install', 'test', 'test-quiet', 'smoke', 'bench', 'lint', 'check', 'clean', 'clean-data', 'all')]
    [string]$Target = 'help',

    [int]$Rows = 6000,
    [int]$Seed = 20260101,
    [string]$Data = 'data/applications.jsonl',
    [string]$Model = 'models/model.json',
    [string]$HostName = '127.0.0.1',
    [int]$Port = 8080,
    [string]$DbPath = 'data/riskscore.db',
    [string]$Threshold = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Invoke-Setup {
    foreach ($directory in @('data', 'models', 'reports')) {
        $path = Join-Path $repoRoot $directory
        if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path | Out-Null }
    }
}

function Invoke-Data {
    Invoke-Setup
    Invoke-Python 'scripts/generate_dataset.py' '--rows' "$Rows" '--seed' "$Seed" '--output' $Data
}

function Invoke-Train {
    $arguments = @('scripts/train_model.py', '--data', $Data, '--model', $Model)
    if ($Threshold) { $arguments += @('--threshold', $Threshold, '--threshold-mode', 'fixed') }
    Invoke-Python @arguments
}

function Invoke-Run {
    $env:RISKSCORE_HOST = $HostName
    $env:RISKSCORE_PORT = "$Port"
    $env:RISKSCORE_DB_PATH = $DbPath
    $env:RISKSCORE_MODEL_PATH = $Model
    # The package lives under src/ and is not installed by default, so the import
    # path is set here. That keeps `run` working on a fresh clone with no pip step.
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    Invoke-Python '-m' 'riskscore.server'
}

function Invoke-Install {
    Invoke-Python '-m' 'pip' 'install' '-e' '.'
}

function Invoke-Tests {
    param([switch]$Quiet)
    $arguments = @('-m', 'unittest', 'discover', '-s', 'tests', '-t', '.')
    if (-not $Quiet) { $arguments += '-v' }
    Invoke-Python @arguments
}

function Invoke-Lint {
    Invoke-Python '-m' 'compileall' '-q' 'src' 'scripts' 'tests'
}

function Invoke-Clean {
    foreach ($pattern in @('reports/*.json', 'reports/*.md')) {
        Get-ChildItem -Path (Join-Path $repoRoot $pattern) -ErrorAction SilentlyContinue |
            Remove-Item -Force
    }
    Get-ChildItem -Path $repoRoot -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
    Get-ChildItem -Path $repoRoot -Recurse -File -Filter '*.pyc' -ErrorAction SilentlyContinue |
        Remove-Item -Force
}

function Invoke-CleanData {
    foreach ($directory in @('data', 'models', 'reports')) {
        $path = Join-Path $repoRoot $directory
        if (Test-Path $path) { Remove-Item -Recurse -Force $path }
    }
    Write-Host 'removed data/, models/ and reports/'
}

function Show-Help {
    @'
  setup        Create the output directories
  data         Generate the synthetic dataset (Rows, Seed)
  train        Train the scorecard (needs `data` first)
  run          Start the HTTP service
  install      Install the package into the current environment (editable)
  test         Run the unit and integration test suite
  test-quiet   Run the test suite with a summary only
  smoke        End-to-end check in a temporary directory
  bench        Time training and 1000 in-process scorings
  lint         Byte-compile everything to catch syntax errors
  check        Lint plus the end-to-end smoke check
  clean        Remove caches and generated reports
  clean-data   Remove data/, models/ and reports/
  all          data + train + check
'@ | Write-Host
}

Push-Location $repoRoot
try {
    switch ($Target) {
        'help' { Show-Help }
        'setup' { Invoke-Setup; Write-Host 'directories ready' }
        'data' { Invoke-Data }
        'train' { Invoke-Train }
        'run' { Invoke-Run }
        'install' { Invoke-Install; Write-Host 'installed; `python -m riskscore.server` now works anywhere' }
        'test' { Invoke-Tests }
        'test-quiet' { Invoke-Tests -Quiet }
        'smoke' { Invoke-Python 'scripts/smoke_check.py' }
        'bench' { Invoke-Python 'scripts/bench.py' }
        'lint' { Invoke-Lint; Write-Host 'compile check passed' }
        'check' { Invoke-Lint; Invoke-Python 'scripts/smoke_check.py'; Write-Host 'all checks passed' }
        'clean' { Invoke-Clean; Write-Host 'cleaned' }
        'clean-data' { Invoke-CleanData }
        'all' { Invoke-Data; Invoke-Train; Invoke-Lint; Invoke-Python 'scripts/smoke_check.py' }
    }
}
finally {
    Pop-Location
}
