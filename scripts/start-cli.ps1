<# Prepare local dependencies and run PrismDesk's built-in CLI. #>
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\ProgramData\anaconda3\envs\agent\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Demo Python environment was not found: $python"
}

$redisRunning = & docker ps --filter "name=^/prismdesk-redis$" --format "{{.Names}}"
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not available. Start Docker Desktop, then launch PrismDesk again."
}
if (-not $redisRunning) {
    & docker start prismdesk-redis | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "The PrismDesk Redis container could not be started."
    }
}

$chromaRunning = & docker ps --filter "name=^/prismdesk-chromadb$" --format "{{.Names}}"
if (-not $chromaRunning) {
    & docker start prismdesk-chromadb | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "The PrismDesk ChromaDB container could not be started."
    }
}

$env:PYTHONPATH = $root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# CLI 面向终端对话，隐藏技能加载与 HTTP 请求的常规诊断日志；错误和告警仍会显示。
$env:LOG_LEVEL = "WARNING"
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = $env:NO_PROXY
$env:REDIS_URL = "redis://:prismdesk123@127.0.0.1:6379/0"
$env:CHROMA_HOST = "127.0.0.1"
$env:CHROMA_PORT = "8001"
$env:CHROMA_PERSIST_DIRECTORY = Join-Path $root "data\chroma-demo"
$env:PRISMDESK_SKILLS_DIR = Join-Path $root "skills"
$env:PRISMDESK_LOAD_DEFAULT_KB = "false"

$chromaReady = $false
for ($attempt = 1; $attempt -le 15; $attempt++) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:8001/api/v2/heartbeat"
        if ($response.StatusCode -eq 200) {
            $chromaReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $chromaReady) {
    throw "The PrismDesk ChromaDB container did not become ready."
}

Push-Location $root
try {
    & $python "api\main.py" "--cli"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
