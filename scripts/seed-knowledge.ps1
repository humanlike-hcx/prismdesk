<# Idempotently import the versioned PrismDesk demo knowledge pack. #>
param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$knowledgeFile = Join-Path $root "knowledge\standard_customer_service.json"

if (-not (Test-Path -LiteralPath $knowledgeFile)) {
    throw "Knowledge pack was not found: $knowledgeFile"
}

$documents = Get-Content -LiteralPath $knowledgeFile -Raw -Encoding UTF8 | ConvertFrom-Json
$payload = @{ documents = $documents } | ConvertTo-Json -Depth 6
$response = Invoke-RestMethod `
    -Method Post `
    -Uri "$($BaseUrl.TrimEnd('/'))/knowledge/add" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($payload))

$response | ConvertTo-Json -Depth 5
