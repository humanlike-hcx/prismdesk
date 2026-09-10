<# Launch the local demo and open its API documentation after a health check. #>
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$startScript = Join-Path $PSScriptRoot "start-demo.ps1"
$healthUrl = "http://127.0.0.1:8000/health"
$docsUrl = "http://127.0.0.1:8000/docs"

try {
    & $startScript

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $healthUrl
            if ($response.StatusCode -eq 200) {
                Start-Process $docsUrl
                exit 0
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    throw "The service did not pass its health check. Review logs\\demo-api.err.log."
}
catch {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        $_.Exception.Message,
        "PrismDesk startup failed",
        "OK",
        "Error"
    ) | Out-Null
    exit 1
}
