<# Prepare local dependencies and run PrismDesk's built-in CLI. #>
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\ProgramData\anaconda3\envs\agent\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Demo Python environment was not found: $python"
}

$redisRunning = & docker ps --filter "name=^/prismdesk-redis-demo$" --format "{{.Names}}"
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not available. Start Docker Desktop, then launch PrismDesk again."
}
if (-not $redisRunning) {
    & docker start prismdesk-redis-demo | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "The PrismDesk Redis container could not be started."
    }
}

$env:PYTHONPATH = $root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:REDIS_URL = "redis://:prismdesk123@127.0.0.1:6380/0"
$env:CHROMA_HOST = "127.0.0.1"
$env:CHROMA_PORT = "8001"
$env:CHROMA_PERSIST_DIRECTORY = Join-Path $root "data\chroma-demo"
$env:PRISMDESK_SKILLS_DIR = Join-Path $root "skills"
$env:PRISMDESK_LOAD_DEFAULT_KB = "false"

Push-Location $root
try {
    & $python "api\main.py" "--cli"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
