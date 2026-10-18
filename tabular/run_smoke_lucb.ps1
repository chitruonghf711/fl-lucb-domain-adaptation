<#
.SYNOPSIS
Runs a lightweight five-client Flower smoke test for tabular LUCB feedback.

.DESCRIPTION
The server forces feedback after round one. Clients train one epoch on their
own shard, skip TVAE setup/generation, and validate abstract feedback in round
two. Each invocation creates a new output directory and does not touch result_*
artefacts.
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$LogDir = (Join-Path "logs" ("tabular_lucb_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDirectory = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot $LogDir))
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$serverStdout = Join-Path $outputDirectory "server.stdout.log"
$serverStderr = Join-Path $outputDirectory "server.stderr.log"
$serverArguments = @(
    "server_tabular.py",
    "--server-address", "127.0.0.1:9000",
    "--num-rounds", "2",
    "--min-clients", "5",
    "--selection-mode", "lucb",
    "--feedback-protocol", "abstract",
    "--lucb-epsilon", "1.0",
    "--weak-class-accuracy", "1.0",
    "--test-force-feedback",
    "--log-dir", $outputDirectory
)

$server = Start-Process -FilePath $Python -ArgumentList $serverArguments -WorkingDirectory $scriptRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $serverStdout -RedirectStandardError $serverStderr
$clients = @()

try {
    Start-Sleep -Seconds 2
    foreach ($clientNumber in 1..5) {
        $clientId = "client$clientNumber"
        $clientStdout = Join-Path $outputDirectory "$clientId.stdout.log"
        $clientStderr = Join-Path $outputDirectory "$clientId.stderr.log"
        $clientArguments = @(
            "client_tabular.py",
            "--client-id", $clientId,
            "--train-csv", "dataset/7030/train$clientNumber.csv",
            "--server-address", "127.0.0.1:9000",
            "--log-dir", $outputDirectory,
            "--epochs", "1",
            "--smoke-no-generation"
        )
        $clients += Start-Process -FilePath $Python -ArgumentList $clientArguments -WorkingDirectory $scriptRoot `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $clientStdout -RedirectStandardError $clientStderr
    }

    foreach ($client in $clients) {
        $client.WaitForExit()
        $client.Refresh()
        if ($null -ne $client.ExitCode -and $client.ExitCode -ne 0) {
            throw "Client smoke process failed with exit code $($client.ExitCode)."
        }
    }
    $server.WaitForExit()
    $server.Refresh()
    if ($null -ne $server.ExitCode -and $server.ExitCode -ne 0) {
        throw "Server smoke process failed with exit code $($server.ExitCode)."
    }

    $auditPath = Join-Path $outputDirectory "server\audit\feedback_round_1.json"
    if (-not (Test-Path -LiteralPath $auditPath)) {
        throw "Expected LUCB audit file was not created: $auditPath"
    }
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    if ($audit.model_sample_evaluations -gt $audit.budget) {
        throw "LUCB model-sample evaluations exceeded the recorded budget."
    }
    $auditText = Get-Content -LiteralPath $auditPath -Raw
    if ($auditText -match '"reference"|"raw"|"feature_value"') {
        throw "Sanitized audit contains a prohibited raw-content field."
    }
    $feedbackDirectory = Join-Path $outputDirectory "server\feedback_logs"
    $feedbackFiles = @(Get-ChildItem -LiteralPath $feedbackDirectory -Filter "*.json" -File -ErrorAction SilentlyContinue)
    if ($feedbackFiles.Count -eq 0) {
        throw "Expected at least one abstract feedback payload, but none was written."
    }
    foreach ($feedbackFile in $feedbackFiles) {
        $feedbackText = Get-Content -LiteralPath $feedbackFile.FullName -Raw
        if ($feedbackText -match '"(?:reference|raw|feature_value|row_data|rows|values|samples|inputs|payload|token|text|pixel|image_data|tensor)"\s*:') {
            throw "Abstract feedback payload contains a prohibited raw-content field: $($feedbackFile.FullName)"
        }
        $feedbackBundle = $feedbackText | ConvertFrom-Json
        if ($feedbackBundle.feedback.protocol -ne "abstract" -or $feedbackBundle.feedback.domain -ne "tabular") {
            throw "Feedback payload did not use the expected abstract tabular protocol: $($feedbackFile.FullName)"
        }
    }
    Write-Output "Tabular LUCB Flower smoke passed. Output: $outputDirectory"
}
finally {
    foreach ($client in $clients) {
        if (-not $client.HasExited) {
            Stop-Process -Id $client.Id -Force
        }
    }
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
