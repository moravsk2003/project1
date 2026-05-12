#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request


SYSTEM_PROMPT = """You supervise reinforcement learning for a Rust base builder.
Return exactly one JSON object and no markdown.
Use only actions listed in observation.allowedActions.
Use REPLACE_REWARD_CONFIG sparingly. Prefer small bounded changes.
Use STOP_TRAINING only when the run is clearly wasting the remaining budget.
Use START_NEW_RUN only when observation.trainingContext.trainingRunning is false and provide modelName.
Use REQUEST_HISTORICAL_REPORT only when idle and you need prior epoch trends before starting a run.
Use PROMOTE_BRANCH only after observation.trendMetrics.latestBranchComparison shows a candidate that should replace the live reward config.
Use JUMP_TO_BRANCH only with a modelName from observation.trendMetrics.knownBranchModelNames.
Never invent fields outside rewardConfig or rewardTerms.
Formula terms may use only variables and functions from observation.rewardFormulaContract.
Use trainingRemainingSeconds and trainingDeadline to avoid disruptive changes near the end of a run.
"""


def main():
    observation = json.load(sys.stdin)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

    if not api_key:
        print(json.dumps({
            "action": "KEEP_GOING",
            "reason": "No OPENAI_API_KEY/LLM_API_KEY configured."
        }))
        return

    user_prompt = {
        "task": "Review the compact RL observation and return one supervisor decision.",
        "observation": observation,
        "decision_schema": {
            "action": "one of observation.allowedActions",
            "epsilon": "optional number for SET_EPSILON",
            "rewardConfig": "optional object of numeric RLRewardConfig fields",
            "rewardTerms": "optional array of {name, scope: STEP|FINAL, expression}",
            "modelName": "required safe unique name for START_NEW_RUN",
            "reportModelName": "required for REQUEST_HISTORICAL_REPORT",
            "reportStartEpoch": "optional non-negative integer",
            "reportEndEpoch": "optional non-negative integer",
            "reason": "short explanation"
        }
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, separators=(",", ":"))},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "action": "KEEP_GOING",
            "reason": "LLM request failed: " + str(exc),
        }))
        return

    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "action": "KEEP_GOING",
            "reason": "LLM returned non-JSON content.",
        }

    print(json.dumps(parsed, separators=(",", ":")))


if __name__ == "__main__":
    main()
