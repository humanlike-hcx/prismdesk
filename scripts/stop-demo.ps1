<# Stop the PrismDesk demo API. The Redis data container remains for next time. #>

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($process.CommandLine -match "uvicorn api\.main:app") {
        Stop-Process -Id $listener.OwningProcess -Force
        Write-Host "PrismDesk demo API stopped."
    } else {
        throw "Port 8000 is not owned by PrismDesk; no process was stopped."
    }
} else {
    Write-Host "PrismDesk demo API is not running."
}
