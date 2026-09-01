param(
    [string]$BaseUrl = "http://localhost:8080",
    [string]$ApiKey = $env:AGENTHUB_API_KEY
)

$ErrorActionPreference = "Stop"
$headers = @{}
if ($ApiKey) {
    $headers["X-API-Key"] = $ApiKey
}

$runId = [Guid]::NewGuid().ToString("N")
$tempPath = Join-Path ([IO.Path]::GetTempPath()) "agenthub-smoke-$runId.md"
$fileId = $null

try {
    [IO.File]::WriteAllText(
        $tempPath,
        "# AgentHub smoke test`n`nThe service identifier is $runId. It uses Qdrant as its vector database."
    )

    $ready = Invoke-RestMethod -Uri "$BaseUrl/api/health/ready" -Method Get
    if ($ready.status -ne "ok") {
        throw "API readiness check did not return ok"
    }
    $baselineStats = Invoke-RestMethod -Uri "$BaseUrl/api/admin/stats" -Method Get

    $curlArgs = @("-sS", "-f")
    if ($ApiKey) {
        $curlArgs += @("-H", "X-API-Key: $ApiKey")
    }
    $curlArgs += @("-F", "file=@$tempPath;type=text/markdown", "$BaseUrl/api/ingest/upload")
    $upload = (& curl.exe @curlArgs) | ConvertFrom-Json
    $fileId = $upload.file_id
    if (-not $fileId) {
        throw "Upload response did not include file_id"
    }

    $duplicate = (& curl.exe @curlArgs) | ConvertFrom-Json
    if (-not $duplicate.duplicate -or $duplicate.file_id -ne $fileId) {
        throw "Content-hash idempotency check failed"
    }

    if ($upload.status -eq "review_required") {
        $reviewBody = @{ thread_id = $upload.thread_id; approved = $true } | ConvertTo-Json
        Invoke-RestMethod -Uri "$BaseUrl/api/ingest/review" -Method Post -Headers $headers `
            -ContentType "application/json" -Body $reviewBody | Out-Null
    }

    $documents = Invoke-RestMethod -Uri "$BaseUrl/api/documents" -Method Get
    if (-not ($documents | Where-Object { $_.file_id -eq $fileId })) {
        throw "Persistent document registry did not return the uploaded file"
    }

    $questionBody = @{ question = "Which vector database does service $runId use?" } | ConvertTo-Json
    $answer = Invoke-RestMethod -Uri "$BaseUrl/api/qa/ask" -Method Post `
        -ContentType "application/json" -Body $questionBody
    if (-not $answer.answer) {
        throw "QA response was empty"
    }

    Invoke-RestMethod -Uri "$BaseUrl/api/documents/$fileId" -Method Delete -Headers $headers | Out-Null
    $documentsAfterDelete = Invoke-RestMethod -Uri "$BaseUrl/api/documents" -Method Get
    if ($documentsAfterDelete | Where-Object { $_.file_id -eq $fileId }) {
        throw "Deleted document is still present in the registry"
    }
    $finalStats = Invoke-RestMethod -Uri "$BaseUrl/api/admin/stats" -Method Get
    if (
        $finalStats.vector_store.total_vectors -ne $baselineStats.vector_store.total_vectors -or
        $finalStats.knowledge_graph.total_entities -ne $baselineStats.knowledge_graph.total_entities -or
        $finalStats.knowledge_graph.total_relations -ne $baselineStats.knowledge_graph.total_relations
    ) {
        throw "Storage counts did not return to their pre-test baseline"
    }
    $fileId = $null
    Write-Host "E2E smoke passed: readiness, upload, dedupe, registry, QA, and delete."
}
finally {
    if ($fileId) {
        try {
            Invoke-RestMethod -Uri "$BaseUrl/api/documents/$fileId" -Method Delete -Headers $headers | Out-Null
        }
        catch {
            Write-Warning "Cleanup failed for $fileId"
        }
    }
    Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
}
