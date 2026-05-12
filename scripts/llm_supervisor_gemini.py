#!/usr/bin/env python3
import json
import os
import re
import sys
import urllib.error
import urllib.request


def load_local_env():
    path = os.path.join(os.getcwd(), ".llm_supervisor.local.env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


SYSTEM_PROMPT = """You supervise reinforcement learning for a Rust base builder.
Return exactly one JSON object and no markdown.
Allowed actions: KEEP_GOING, SET_EPSILON, REPLACE_REWARD_CONFIG, REQUEST_PROMOTION_CHECK.
Prefer KEEP_GOING unless the observation gives clear evidence.
Use SET_EPSILON for small exploration adjustments.
Use REPLACE_REWARD_CONFIG sparingly and only with small numeric changes or safe arithmetic rewardTerms.
Never invent fields outside rewardConfig or rewardTerms.
Formula terms may use only arithmetic variables supplied in the observation contract.
Use trainingRemainingSeconds and trainingDeadline to avoid disruptive changes near the end of a run.
"""


DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": [
                "KEEP_GOING",
                "SET_EPSILON",
                "REPLACE_REWARD_CONFIG",
                "REQUEST_PROMOTION_CHECK",
            ],
        },
        "epsilon": {"type": "NUMBER"},
        "rewardConfig": {"type": "OBJECT"},
        "rewardTerms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "scope": {"type": "STRING", "enum": ["STEP", "FINAL"]},
                    "expression": {"type": "STRING"},
                },
            },
        },
        "reason": {"type": "STRING"},
    },
    "propertyOrdering": ["action", "epsilon", "rewardConfig", "rewardTerms", "reason"],
}


def keep_going(reason):
    print(json.dumps({"action": "KEEP_GOING", "reason": reason}, separators=(",", ":")))


def extract_text(response):
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(texts).strip()


def parse_model_json(text):
    if not text:
        return None, "returned empty text"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), None
            except json.JSONDecodeError:
                pass
    return None, "returned non-JSON content"


def build_request_body(observation):
    user_payload = {
        "task": "Review this compact RL observation and return one supervisor decision.",
        "observation": observation,
        "decision_schema": {
            "action": "KEEP_GOING | SET_EPSILON | REPLACE_REWARD_CONFIG | REQUEST_PROMOTION_CHECK",
            "epsilon": "optional number for SET_EPSILON",
            "rewardConfig": "optional numeric patch for RLRewardConfig fields",
            "rewardTerms": "optional array of {name, scope: STEP|FINAL, expression}",
            "reason": "short explanation",
        },
    }

    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": json.dumps(user_payload, separators=(",", ":"))}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": DECISION_SCHEMA,
        },
    }


def call_gemini_model(model, base_url, api_key, request_body):
    endpoint = base_url.rstrip("/") + "/models/" + model + ":generateContent"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, f"{model} request failed with HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"{model} request failed: {exc}"

    decision, parse_error = parse_model_json(extract_text(body))
    if parse_error:
        return None, f"{model} {parse_error}"
    return decision, None


def mark_fallback_decision(decision, primary_model, primary_error, fallback_model):
    if not isinstance(decision, dict):
        return decision
    reason = decision.get("reason") or "no reason"
    decision["reason"] = (
        f"Fallback {fallback_model} used after {primary_model} failed "
        f"({primary_error}). {reason}"
    )
    return decision


def main():
    load_local_env()
    observation = json.load(sys.stdin)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        keep_going("No GEMINI_API_KEY/GOOGLE_API_KEY configured.")
        return

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    fallback_model = os.environ.get("GEMINI_FALLBACK_MODEL", "gemma-4-31b-it")
    base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

    request_body = build_request_body(observation)
    decision, primary_error = call_gemini_model(model, base_url, api_key, request_body)
    if decision is not None:
        print(json.dumps(decision, separators=(",", ":")))
        return

    if fallback_model and fallback_model != model:
        fallback_decision, fallback_error = call_gemini_model(fallback_model, base_url, api_key, request_body)
        if fallback_decision is not None:
            print(json.dumps(
                mark_fallback_decision(fallback_decision, model, primary_error, fallback_model),
                separators=(",", ":")))
            return
        keep_going(f"Gemini primary and fallback failed: {primary_error}; {fallback_error}")
        return

    keep_going("Gemini request failed: " + primary_error)


if __name__ == "__main__":
    main()
