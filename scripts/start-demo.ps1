<# Start the PrismDesk local demo. Open http://127.0.0.1:8000/docs after startup. #>
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\ProgramData\anaconda3\envs\agent\python.exe"
$logDir = Join-Path $root "logs"
$stdout = Join-Path $logDir "demo-api.out.log"
$stderr = Join-Path $logDir "demo-api.err.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Demo Python environment was not found: $python"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Use a separate Redis port to avoid conflicts with other local projects.
$redisRunning = docker ps --filter "name=^/prismdesk-redis-demo$" --format "{{.Names}}"
if (-not $redisRunning) {
    docker start prismdesk-redis-demo | Out-Null
}

# load_dotenv does not override these explicit local demo settings.
$env:PYTHONPATH = "$root\.demo-lib;$root"
$env:REDIS_URL = "redis://:prismdesk123@127.0.0.1:6380/0"
$env:CHROMA_HOST = "127.0.0.1"
$env:CHROMA_PORT = "8001"
$env:CHROMA_PERSIST_DIRECTORY = Join-Path $root "data\chroma-demo"
$env:EVAL_BASELINE_PATH = Join-Path $root "data\eval\baseline-demo.json"
$env:PRISMDESK_SKILLS_DIR = Join-Path $root "skills"
$env:PRISMDESK_LOAD_DEFAULT_KB = "false"

$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($existing.OwningProcess)"
    if ($process.CommandLine -match "uvicorn api\.main:app") {
        Write-Host "PrismDesk is already running at http://127.0.0.1:8000"
        exit 0
    }
    throw "Port 8000 is occupied by another process (PID $($existing.OwningProcess))."
}

Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

Write-Host "PrismDesk is starting. Open http://127.0.0.1:8000/docs in a few seconds."
