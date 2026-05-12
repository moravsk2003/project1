# LLM Supervisor

The RL supervisor is an optional control layer around `RLTrainingService`.
It receives compact training observations every N episodes and can return a
validated structured decision. The default UI interval is 1000 episodes.

Built-in call limits protect the configured API key:

- `10` requests per minute.
- `1,500` requests per day.
- `1,000,000` estimated tokens per minute.

When a limit is reached, training pauses before the next supervisor call instead
of failing the run.

## UI

Open the RL dialog and use the `LLM Supervisor` section:

- `Enable LLM supervisor`: enables the hook between episodes.
- `Call every episodes`: frequency of supervisor calls. Default: `1000`.
- `Time limit`: optional wall-clock duration for the training run. Examples:
  `30m`, `2h`, `01:30`. Empty or `0` means no time limit.
- `Apply mode`: `AUTO_APPLY` applies validated decisions; `LOG_ONLY` and
  `MANUAL_APPROVAL` record decisions without changing training.
- `API key`: optional key for the external command. It is passed only to the
  launched process as `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `LLM_API_KEY`;
  it is not saved in model metadata.
- `Command`: optional external command. Leave empty for a safe no-op supervisor.
- `Apply Pending`: applies the most recent validated decision when using
  `MANUAL_APPROVAL`.

## External Command Contract

The command receives one JSON object on stdin and writes one JSON object to
stdout. The app does not run the command through a shell; quote paths with
spaces in the command field.

Ready-to-edit examples:

- `powershell -ExecutionPolicy Bypass -File scripts/llm_supervisor_mock.ps1`
- `powershell -ExecutionPolicy Bypass -File scripts/llm_supervisor_gemini.ps1`
- `python scripts/llm_supervisor_mock.py`
- `python scripts/llm_supervisor_gemini.py`
- `python scripts/llm_supervisor_openai_compatible.py`

Gemini wrapper:

```powershell
$env:GEMINI_API_KEY = "<your key>"
$env:GEMINI_MODEL = "gemini-2.5-flash"
$env:GEMINI_FALLBACK_MODEL = "gemma-4-31b-it"
```

You can skip setting `GEMINI_API_KEY` in PowerShell when you paste the key into
the UI `API key` field. The wrapper receives it from the app environment.
The same Google AI Studio / Gemini API key is used for the fallback model when
Gemma is available through the Gemini API for that project.

By default, `scripts/llm_supervisor_gemini.ps1` and
`scripts/llm_supervisor_gemini.py` call `gemini-2.5-flash` first. If the primary
model fails, times out, returns empty text, returns non-JSON content, or hits an
HTTP error such as a rate-limit response, the wrapper retries once with
`gemma-4-31b-it`. Set `GEMINI_FALLBACK_MODEL` to another model name to override
that fallback, or to the same value as `GEMINI_MODEL` to disable fallback.

Alternatively, create a local ignored config file in the project root:

```powershell
Copy-Item scripts/llm_supervisor.local.example.ps1 .llm_supervisor.local.ps1
notepad .llm_supervisor.local.ps1
```

Put the real key in `.llm_supervisor.local.ps1`. This file is ignored by git.

Then use this UI command:

```text
powershell -ExecutionPolicy Bypass -File scripts/llm_supervisor_gemini.ps1
```

Example stdout:

```json
{
  "action": "SET_EPSILON",
  "epsilon": 0.25,
  "reason": "Invalid action rate is stable, reduce exploration."
}
```

Reward config patch with a formula term:

```json
{
  "action": "REPLACE_REWARD_CONFIG",
  "rewardConfig": {
    "blockGrowthReward": 0.08,
    "penaltyCollision": -0.35
  },
  "rewardTerms": [
    {
      "name": "socket_shape",
      "scope": "STEP",
      "expression": "clamp(socket_connections * 0.1, 0, 0.5)"
    }
  ],
  "reason": "Encourage connected growth without changing baseline branch."
}
```

Allowed actions:

- `KEEP_GOING`
- `SET_EPSILON`
- `REPLACE_REWARD_CONFIG`
- `STOP_TRAINING`
- `REQUEST_PROMOTION_CHECK`

Formula expressions are arithmetic-only and support variables, `min`, `max`,
`clamp`, `abs`, and `sqrt`. Invalid formulas evaluate to `0`.

Supervisor observations include training time context:

- `trainingStartTime`
- `currentTime`
- `trainingDeadline`
- `trainingElapsedSeconds`
- `trainingRemainingSeconds`
- `trainingTimeLimitEnabled`
- `trainingTimeLimitReached`

Each supervisor response is logged in two files under `models_rl`:

- `<model>_supervisor_decisions.jsonl`
- `<model>_supervisor_decisions.csv`

The CSV includes the parsed LLM decision, whether it was applied, the reason,
proposed epsilon when present, reward-config-change flag, key training metrics,
and elapsed/remaining training time. It does not include the API key.
