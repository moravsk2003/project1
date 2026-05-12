$ErrorActionPreference = "Stop"

$localConfig = Join-Path (Get-Location) ".llm_supervisor.local.ps1"
if (Test-Path $localConfig) {
    . $localConfig
}

function Write-Decision($decision) {
    $decision | ConvertTo-Json -Depth 40 -Compress
}

function Parse-ModelJson([string]$text) {
    if ([string]::IsNullOrWhiteSpace($text)) {
        return [pscustomobject]@{ Success = $false; Decision = $null; Error = "returned empty text" }
    }
    try {
        return [pscustomobject]@{ Success = $true; Decision = ($text | ConvertFrom-Json); Error = $null }
    } catch {
        $match = [regex]::Match($text, "\{[\s\S]*\}")
        if ($match.Success) {
            try {
                return [pscustomobject]@{ Success = $true; Decision = ($match.Value | ConvertFrom-Json); Error = $null }
            } catch {
            }
        }
    }
    return [pscustomobject]@{ Success = $false; Decision = $null; Error = "returned non-JSON content" }
}

function Add-FallbackReason($decision, [string]$primaryModel, [string]$primaryError, [string]$fallbackModel) {
    $reason = "no reason"
    if ($decision.PSObject.Properties.Name -contains "reason" -and -not [string]::IsNullOrWhiteSpace($decision.reason)) {
        $reason = $decision.reason
    }
    $prefixedReason = "Fallback $fallbackModel used after $primaryModel failed ($primaryError). $reason"
    if ($decision.PSObject.Properties.Name -contains "reason") {
        $decision.reason = $prefixedReason
    } else {
        $decision | Add-Member -NotePropertyName reason -NotePropertyValue $prefixedReason
    }
    return $decision
}

function Invoke-GeminiSupervisorModel([string]$modelName, [string]$baseUrl, [string]$apiKey, [string]$body) {
    $endpoint = "$baseUrl/models/$modelName`:generateContent"
    try {
        $response = Invoke-RestMethod -Uri $endpoint -Method Post -Headers @{ "x-goog-api-key" = $apiKey } -ContentType "application/json" -Body $body -TimeoutSec 30
    } catch {
        $statusCode = $null
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
        $message = if ($statusCode) { "$modelName request failed with HTTP $statusCode" } else { "$modelName request failed: $($_.Exception.Message)" }
        return [pscustomobject]@{ Success = $false; Decision = $null; Error = $message }
    }

    $text = ""
    if ($response.candidates -and $response.candidates.Count -gt 0) {
        $parts = $response.candidates[0].content.parts
        if ($parts -and $parts.Count -gt 0) {
            $text = ($parts | ForEach-Object { $_.text }) -join "`n"
        }
    }

    $parsed = Parse-ModelJson $text
    if ($parsed.Success) {
        return [pscustomobject]@{ Success = $true; Decision = $parsed.Decision; Error = $null }
    }
    return [pscustomobject]@{ Success = $false; Decision = $null; Error = "$modelName $($parsed.Error)" }
}

$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) {
    Write-Decision @{ action = "KEEP_GOING"; reason = "No observation received." }
    exit 0
}

$apiKey = $env:GEMINI_API_KEY
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    $apiKey = $env:GOOGLE_API_KEY
}
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Decision @{ action = "KEEP_GOING"; reason = "No GEMINI_API_KEY/GOOGLE_API_KEY configured." }
    exit 0
}

$model = if ([string]::IsNullOrWhiteSpace($env:GEMINI_MODEL)) { "gemini-2.5-flash" } else { $env:GEMINI_MODEL }
$fallbackModel = if ([string]::IsNullOrWhiteSpace($env:GEMINI_FALLBACK_MODEL)) { "gemma-4-31b-it" } else { $env:GEMINI_FALLBACK_MODEL }
$baseUrl = if ([string]::IsNullOrWhiteSpace($env:GEMINI_BASE_URL)) { "https://generativelanguage.googleapis.com/v1beta" } else { $env:GEMINI_BASE_URL.TrimEnd("/") }

$observation = $raw | ConvertFrom-Json
$systemPrompt = @"
You supervise reinforcement learning for a Rust base builder.
Return exactly one JSON object and no markdown.
Use only actions listed in observation.allowedActions.
Prefer KEEP_GOING unless the observation gives clear evidence.
Use SET_EPSILON for small exploration adjustments.
Use REPLACE_REWARD_CONFIG sparingly and only with small numeric changes or safe arithmetic rewardTerms.
Use STOP_TRAINING only when the run is clearly wasting the remaining budget.
Use START_NEW_RUN only when observation.trainingContext.trainingRunning is false and provide modelName.
You MUST provide a unique modelName when using START_NEW_RUN.
You can optionally toggle 2D vs 3D CNN architecture via use2dCnn when starting.
Use REQUEST_HISTORICAL_REPORT to request aggregated epoch logs for a specific past model. If you do this, you will immediately receive a new observation with `historicalReport` populated. Provide reportModelName, reportStartEpoch, and reportEndEpoch.
Use PROMOTE_BRANCH only after observation.trendMetrics.latestBranchComparison shows a candidate that should replace the live reward config.
Use JUMP_TO_BRANCH only with a modelName from observation.trendMetrics.knownBranchModelNames.
Never invent fields outside rewardConfig or rewardTerms.
Formula terms may use only variables and functions from observation.rewardFormulaContract.
Use trainingRemainingSeconds and trainingDeadline to avoid disruptive changes near the end of a run.
"@

$userPayload = @{
    task = "Review this compact RL observation and return one supervisor decision."
    observation = $observation
    decision_schema = @{
        action = "one of observation.allowedActions"
        epsilon = "optional number for SET_EPSILON"
        rewardConfig = "optional numeric patch for RLRewardConfig fields"
        rewardTerms = "optional array of {name, scope: STEP|FINAL, expression}"
        use2dCnn = "optional boolean to use 2D CNN (true) or 3D CNN (false)"
        modelName = "a unique string for the model (e.g. run_2d_base_1)"
        reportModelName = "optional string name of the model to query"
        reportStartEpoch = "optional start epoch for historical report"
        reportEndEpoch = "optional end epoch for historical report"
        reason = "short explanation"
    }
}

$schema = @{
    type = "OBJECT"
    properties = @{
        action = @{
            type = "STRING"
            enum = @("KEEP_GOING", "SET_EPSILON", "REPLACE_REWARD_CONFIG", "REQUEST_PROMOTION_CHECK", "START_NEW_RUN", "STOP_TRAINING", "REQUEST_HISTORICAL_REPORT", "PROMOTE_BRANCH", "JUMP_TO_BRANCH")
        }
        epsilon = @{ type = "NUMBER" }
        rewardConfig = @{ type = "OBJECT" }
        rewardTerms = @{
            type = "ARRAY"
            items = @{
                type = "OBJECT"
                properties = @{
                    name = @{ type = "STRING" }
                    scope = @{ type = "STRING"; enum = @("STEP", "FINAL") }
                    expression = @{ type = "STRING" }
                }
            }
        }
        use2dCnn = @{ type = "BOOLEAN" }
        modelName = @{ type = "STRING" }
        reportModelName = @{ type = "STRING" }
        reportStartEpoch = @{ type = "INTEGER" }
        reportEndEpoch = @{ type = "INTEGER" }
        reason = @{ type = "STRING" }
    }
    propertyOrdering = @("action", "epsilon", "rewardConfig", "rewardTerms", "use2dCnn", "modelName", "reportModelName", "reportStartEpoch", "reportEndEpoch", "reason")
}

$body = @{
    systemInstruction = @{
        parts = @(@{ text = $systemPrompt })
    }
    contents = @(
        @{
            role = "user"
            parts = @(@{ text = ($userPayload | ConvertTo-Json -Depth 40 -Compress) })
        }
    )
    generationConfig = @{
        temperature = 0.2
        responseMimeType = "application/json"
        responseSchema = $schema
    }
} | ConvertTo-Json -Depth 60 -Compress

$primary = Invoke-GeminiSupervisorModel $model $baseUrl $apiKey $body
if ($primary.Success) {
    Write-Decision $primary.Decision
    exit 0
}

if (-not [string]::IsNullOrWhiteSpace($fallbackModel) -and $fallbackModel -ne $model) {
    $fallback = Invoke-GeminiSupervisorModel $fallbackModel $baseUrl $apiKey $body
    if ($fallback.Success) {
        Write-Decision (Add-FallbackReason $fallback.Decision $model $primary.Error $fallbackModel)
        exit 0
    }
    Write-Decision @{ action = "KEEP_GOING"; reason = "Gemini primary and fallback failed: $($primary.Error); $($fallback.Error)" }
    exit 0
}

Write-Decision @{ action = "KEEP_GOING"; reason = "Gemini request failed: $($primary.Error)" }
