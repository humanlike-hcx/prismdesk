<# Launch the local API and enter the interactive PrismDesk CLI. #>
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$startScript = Join-Path $PSScriptRoot "start-demo.ps1"
$healthUrl = "http://127.0.0.1:8000/health"
$python = "C:\ProgramData\anaconda3\envs\agent\python.exe"
$cli = Join-Path $root "cli.py"

try {
    & $startScript

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $healthUrl
            if ($response.StatusCode -eq 200) {
                & $python $cli
                exit $LASTEXITCODE
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    throw "The service did not pass its health check. Review logs\\demo-api.err.log."
}
catch {
    Write-Host "PrismDesk startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
