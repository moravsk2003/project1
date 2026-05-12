$ErrorActionPreference = "Stop"

function Write-Decision($decision) {
    $decision | ConvertTo-Json -Depth 20 -Compress
}

$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) {
    Write-Decision @{ action = "KEEP_GOING"; reason = "No observation received." }
    exit 0
}

$observation = $raw | ConvertFrom-Json
$invalidRate = if ($null -ne $observation.invalidActionRate) { [double]$observation.invalidActionRate } else { 0.0 }
$epsilon = if ($null -ne $observation.epsilon) { [double]$observation.epsilon } else { 1.0 }
$bestBlocks = if ($null -ne $observation.bestBaseBlocks) { [int]$observation.bestBaseBlocks } else { 0 }

if ($invalidRate -gt 0.35) {
    Write-Decision @{
        action = "SET_EPSILON"
        epsilon = [Math]::Min(1.0, $epsilon + 0.10)
        reason = "Mock policy raised exploration because invalid action rate is high."
    }
} elseif ($bestBlocks -gt 20 -and $epsilon -gt 0.15) {
    Write-Decision @{
        action = "SET_EPSILON"
        epsilon = [Math]::Max(0.15, $epsilon - 0.05)
        reason = "Mock policy reduced exploration after finding a non-trivial base."
    }
} else {
    Write-Decision @{
        action = "KEEP_GOING"
        reason = "Mock policy sees no safe change."
    }
}
